"""Flagship timing-stage execution with sanitized request rows.

The executor records one fixed backend context per row group.  Timing jobs perform a
fixed number of attempts: warmup and measured failures are retained and no failed
attempt is retried or replaced.  The default backend performs real Concrete/TenSEAL
calls; tests may inject a backend implementing :class:`TimingBackend`.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from unseen_loop.flagship.manifest import FlagshipManifest, PlannedJob, canonical_json
from unseen_loop.timing import TIMING_ROW_SCHEMA, summarize_timing_rows

if TYPE_CHECKING:
    from unseen_loop.crypto.ckks import CKKSParameters

TIMING_ARTIFACT_SCHEMA = "unseen-loop/flagship-timing-job-v1"
CONTEXT_RECEIPT_SCHEMA = "unseen-loop/flagship-timing-context-v1"
COLOCATED_TRUST_SCOPE = (
    "client and server timing endpoints run in the same container; these measurements "
    "cover colocated process and serialization cost and make no claim about remote "
    "network transport, network isolation, or deployment end-to-end privacy"
)
RETRY_POLICY = "zero retries; failed warmup and measured attempts are retained and not replaced"
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class UnsupportedTimingCell(RuntimeError):
    """A backend cannot honestly execute a requested fixed timing context."""

    def __init__(self, reason_code: str) -> None:
        if _SAFE_CODE.fullmatch(reason_code) is None:
            raise ValueError("reason_code must be a sanitized code")
        super().__init__(reason_code)
        self.reason_code = reason_code


class TimingCallError(RuntimeError):
    """A sanitized per-attempt failure that remains in the timing rows."""

    def __init__(self, failure_code: str) -> None:
        if _SAFE_CODE.fullmatch(failure_code) is None:
            raise ValueError("failure_code must be a sanitized code")
        super().__init__(failure_code)
        self.failure_code = failure_code


@dataclass(frozen=True, slots=True)
class TimingContextRequest:
    workload: str
    implementation: str
    trajectories: int | None
    horizon: int | None
    container_id: str
    configured_clients: int
    seed: int


@dataclass(frozen=True, slots=True)
class BackendContextReceipt:
    """Public context identity supplied by a timing backend, never private inputs."""

    backend: str
    backend_version: str
    execution_label: str
    implementation_id: str
    circuit_digest: str
    server_artifact_digest: str
    client_artifact_digest: str
    hardware_digest: str
    compile_ns: int
    key_setup_ns: int
    evaluation_key_bytes: int

    def __post_init__(self) -> None:
        if self.execution_label not in {
            "REAL FHE",
            "REAL FHE (approximate arithmetic)",
            "FHE SIMULATED",
        }:
            raise ValueError("execution_label must explicitly identify real or simulated FHE")
        for field in (
            "circuit_digest",
            "server_artifact_digest",
            "client_artifact_digest",
            "hardware_digest",
        ):
            value = cast(str, getattr(self, field))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if not self.backend or not self.backend_version or not self.implementation_id:
            raise ValueError("backend context labels must not be empty")
        if self.compile_ns < 0 or self.key_setup_ns < 0 or self.evaluation_key_bytes < 0:
            raise ValueError("backend context sizes and durations must be non-negative")


@dataclass(frozen=True, slots=True)
class TimingObservation:
    """One sanitized backend result compatible with ``summarize_timing_rows``."""

    timing_ns: Mapping[str, int]
    byte_metrics: Mapping[str, int]
    success: bool = True
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.success == (self.failure_code is not None):
            raise ValueError("success and failure_code are inconsistent")
        for family in (self.timing_ns, self.byte_metrics):
            if any(type(value) is not int or value < 0 for value in family.values()):
                raise ValueError("timing observations require non-negative integer metrics")
        if self.success and not self.timing_ns:
            raise ValueError("successful timing observations require timing metrics")


class TimingSession(Protocol):
    @property
    def context_receipt(self) -> BackendContextReceipt: ...

    def measure(self, request_index: int) -> TimingObservation: ...


class TimingBackend(Protocol):
    def open_context(self, request: TimingContextRequest) -> TimingSession: ...


class _RealTimingBackend:
    """Lazy optional-dependency backend used by Modal timing workers."""

    def open_context(self, request: TimingContextRequest) -> TimingSession:
        if request.workload == "shield":
            return _RealShieldSession(request)
        if request.workload == "ope" and request.implementation == "POLYNOMIAL_APPROX_OPE_V1":
            return _RealCKKSSession(request)
        raise UnsupportedTimingCell("timing.unsupported_context")


class _RealShieldSession:
    def __init__(self, request: TimingContextRequest) -> None:
        import importlib.metadata

        import numpy as np

        from unseen_loop.shield.fhe import (
            ShieldFHEClient,
            ShieldFHEServer,
            ShieldIntegerSpec,
            compile_shield,
        )

        self._np = np
        self._request = request
        self._temporary = tempfile.TemporaryDirectory(prefix="unseen-loop-shield-timing-")
        compiled = compile_shield(ShieldIntegerSpec(), self._temporary.name)
        self._compiled = compiled
        self._client = ShieldFHEClient.from_path(compiled.client_specs_path, compiled.spec)
        self._server = ShieldFHEServer(compiled.server_path)
        key_setup_ns, self._evaluation_keys = self._client.generate_keys()
        try:
            backend_version = importlib.metadata.version("concrete-python")
        except importlib.metadata.PackageNotFoundError:
            backend_version = compiled.receipt.concrete_python_version
        self._receipt = BackendContextReceipt(
            backend="Concrete-Python TFHE",
            backend_version=backend_version,
            execution_label="REAL FHE",
            implementation_id="shield_exact_margin_tensor_v1",
            circuit_digest=compiled.receipt.spec_digest,
            server_artifact_digest=compiled.receipt.server_artifact_sha256,
            client_artifact_digest=compiled.receipt.client_specs_sha256,
            hardware_digest=_hardware_digest(),
            compile_ns=compiled.receipt.compile_ns,
            key_setup_ns=key_setup_ns,
            evaluation_key_bytes=len(self._evaluation_keys),
        )

    @property
    def context_receipt(self) -> BackendContextReceipt:
        return self._receipt

    def measure(self, request_index: int) -> TimingObservation:
        rng = random.Random(self._request.seed + request_index)
        quantized = self._np.asarray([rng.randrange(-2, 3) for _ in range(6)], dtype=self._np.int64)
        started = time.perf_counter_ns()
        phase = time.perf_counter_ns()
        serialized_request = self._client.encrypt(quantized)
        encrypt_ns = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        serialized_response = self._server.evaluate(serialized_request, self._evaluation_keys)
        server_ns = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        observed = self._client.decrypt_margin_tensor(serialized_response)
        decrypt_ns = time.perf_counter_ns() - phase
        end_to_end_ns = time.perf_counter_ns() - started
        expected = self._compiled.clear(quantized)
        if not self._np.array_equal(observed, expected):
            return TimingObservation({}, {}, False, "backend.conformance_mismatch")
        return TimingObservation(
            {
                "client_encrypt": encrypt_ns,
                "server_evaluate": server_ns,
                "client_decrypt": decrypt_ns,
                "end_to_end": end_to_end_ns,
            },
            {
                "evaluation_keys": len(self._evaluation_keys),
                "request": len(serialized_request),
                "response": len(serialized_response),
            },
        )


class _RealCKKSSession:
    def __init__(self, request: TimingContextRequest) -> None:
        if request.trajectories is None or request.horizon is None:
            raise UnsupportedTimingCell("timing.ope_shape_missing")

        from unseen_loop.crypto.ckks import CKKSClient, CKKSServer
        from unseen_loop.ope.ckks import (
            POLYNOMIAL_APPROX_OPE_V1,
            OPECKKSClient,
            OPECKKSServer,
            PolynomialApproxOPESpec,
            generate_ope_contexts,
        )
        from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

        parameters = _ckks_parameters(request.trajectories, request.horizon)
        shape = TrajectorySpec(
            request.trajectories,
            request.horizon,
            6,
            5,
            state_min=(-1.0,) * 6,
            state_max=(1.0,) * 6,
            reward_min=-1.0,
            reward_max=1.0,
        )
        policy = PolynomialPolicySpec(
            action_count=5,
            state_dim=6,
            degree=1,
            coefficients=tuple((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(5)),
        )
        spec = PolynomialApproxOPESpec(
            trajectories=shape,
            target_policy=policy,
            minimum_behavior_propensity=1.0,
            identifier=POLYNOMIAL_APPROX_OPE_V1,
        )
        try:
            contexts = generate_ope_contexts(spec, parameters)
        except (RuntimeError, ValueError) as error:
            raise UnsupportedTimingCell("timing.ckks_context_unsupported") from error
        self._client = OPECKKSClient(
            CKKSClient.from_serialized(contexts.ckks.client_context, parameters=parameters), spec
        )
        self._server = OPECKKSServer(
            CKKSServer.from_serialized(contexts.ckks.server_context, parameters=parameters), spec
        )
        zero_state = (0.0,) * 6
        self._batch = TrajectoryBatch(
            shape,
            tuple(
                tuple(zero_state for _ in range(shape.horizon)) for _ in range(shape.trajectories)
            ),
            tuple(tuple(0 for _ in range(shape.horizon)) for _ in range(shape.trajectories)),
            tuple(tuple(1.0 for _ in range(shape.horizon)) for _ in range(shape.trajectories)),
            tuple(tuple(1.0 for _ in range(shape.horizon)) for _ in range(shape.trajectories)),
        )
        ckks_receipt = contexts.ckks.receipt
        self._receipt = BackendContextReceipt(
            backend=ckks_receipt.backend,
            backend_version=ckks_receipt.tenseal_version,
            execution_label=ckks_receipt.mode,
            implementation_id=POLYNOMIAL_APPROX_OPE_V1,
            circuit_digest=contexts.computation.digest,
            server_artifact_digest=ckks_receipt.server_context_sha256,
            client_artifact_digest=ckks_receipt.client_context_sha256,
            hardware_digest=_hardware_digest(),
            compile_ns=0,
            key_setup_ns=ckks_receipt.keygen_ns,
            evaluation_key_bytes=ckks_receipt.server_context_bytes,
        )

    @property
    def context_receipt(self) -> BackendContextReceipt:
        return self._receipt

    def measure(self, request_index: int) -> TimingObservation:
        del request_index
        started = time.perf_counter_ns()
        encrypted_request, encrypt = self._client.encrypt_batch(self._batch)
        encrypted_response, evaluate = self._server.evaluate(encrypted_request)
        _, decrypt = self._client.decrypt_statistics(encrypted_response, "clipped_wpdis")
        end_to_end_ns = time.perf_counter_ns() - started
        return TimingObservation(
            {
                "client_encrypt": encrypt.elapsed_ns,
                "server_evaluate": evaluate.elapsed_ns,
                "client_decrypt": decrypt.elapsed_ns,
                "end_to_end": end_to_end_ns,
            },
            {
                "evaluation_keys": self._receipt.evaluation_key_bytes,
                "request": encrypt.output_bytes,
                "response": evaluate.output_bytes,
            },
        )


def _ckks_parameters(trajectories: int, horizon: int) -> CKKSParameters:
    """Choose a tc128-compatible chain or reject a cell before claiming execution."""

    from unseen_loop.crypto.ckks import CKKSParameters

    if trajectories < 1 or horizon < 1 or horizon > 64:
        raise UnsupportedTimingCell("timing.ckks_shape_unsupported")
    required_depth = horizon + 6
    scale_bits = 24
    modulus_bits = 80 + required_depth * scale_bits
    security_limits = ((8192, 218), (16384, 438), (32768, 881))
    for degree, limit in security_limits:
        if trajectories <= degree // 2 and modulus_bits <= limit:
            return CKKSParameters(
                poly_modulus_degree=degree,
                coeff_mod_bit_sizes=(40, *((scale_bits,) * required_depth), 40),
                global_scale=float(2**scale_bits),
            )
    raise UnsupportedTimingCell("timing.ckks_depth_unsupported")


def _hardware_digest() -> str:
    identity = "\0".join((platform.system(), platform.machine(), platform.processor()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


ManifestInput: TypeAlias = FlagshipManifest | Mapping[str, object]
JobInput: TypeAlias = PlannedJob | Mapping[str, object]


class _InvalidTimingJob(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _member(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _positive_manifest_int(value: object, section: str, name: str) -> int:
    raw = _member(_member(value, section), name)
    if type(raw) is not int or raw < 1:
        raise _InvalidTimingJob(f"timing.invalid_{section}_{name}")
    return raw


def _manifest_sequence(value: object, section: str, name: str) -> tuple[int, ...]:
    raw = _member(_member(value, section), name)
    if not isinstance(raw, (tuple, list)) or not raw:
        raise _InvalidTimingJob(f"timing.invalid_{section}_{name}")
    if any(type(item) is not int or item < 1 for item in raw):
        raise _InvalidTimingJob(f"timing.invalid_{section}_{name}")
    return tuple(cast(list[int] | tuple[int, ...], raw))


def _manifest_context_digest(manifest: ManifestInput) -> str:
    if isinstance(manifest, FlagshipManifest):
        return manifest.digest
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def _job_parts(job: JobInput) -> tuple[str, int, Mapping[str, str | int | float]]:
    job_id = _member(job, "job_id")
    stage = _member(job, "stage")
    seed = _member(job, "seed")
    if (
        not isinstance(job_id, str)
        or not job_id
        or len(job_id) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in job_id)
    ):
        raise _InvalidTimingJob("timing.invalid_job_id")
    if stage != "timing":
        raise _InvalidTimingJob("timing.wrong_stage")
    if type(seed) is not int or seed < 0:
        raise _InvalidTimingJob("timing.invalid_seed")
    coordinates: object = (
        job.coordinate_dict() if isinstance(job, PlannedJob) else _member(job, "coordinates")
    )
    if not isinstance(coordinates, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, (str, int, float))
        for key, value in coordinates.items()
    ):
        raise _InvalidTimingJob("timing.invalid_coordinates")
    return job_id, seed, cast(Mapping[str, str | int | float], coordinates)


def _validate_coordinates(
    manifest: ManifestInput, coordinates: Mapping[str, str | int | float]
) -> str:
    kind = coordinates.get("kind")
    expected: set[str]
    if kind == "shield":
        expected = {"kind", "container"}
        upper = _positive_manifest_int(manifest, "systems", "shield_timing_containers")
        if _coordinate_int(coordinates, "container") >= upper:
            raise _InvalidTimingJob("timing.container_out_of_range")
    elif kind == "ope":
        expected = {"kind", "container"}
        upper = _positive_manifest_int(manifest, "systems", "ope_timing_containers")
        if _coordinate_int(coordinates, "container") >= upper:
            raise _InvalidTimingJob("timing.container_out_of_range")
    elif kind == "scale":
        expected = {"kind", "trajectories", "horizon", "container"}
        trajectories = _coordinate_int(coordinates, "trajectories")
        horizon = _coordinate_int(coordinates, "horizon")
        container = _coordinate_int(coordinates, "container")
        if trajectories not in _manifest_sequence(manifest, "systems", "scale_trajectory_counts"):
            raise _InvalidTimingJob("timing.trajectories_out_of_range")
        if horizon not in _manifest_sequence(manifest, "systems", "scale_horizons"):
            raise _InvalidTimingJob("timing.horizon_out_of_range")
        if container >= _positive_manifest_int(manifest, "systems", "scale_containers_per_cell"):
            raise _InvalidTimingJob("timing.container_out_of_range")
    elif kind == "concurrent_client":
        expected = {"kind", "client"}
        if _coordinate_int(coordinates, "client") >= _positive_manifest_int(
            manifest, "systems", "concurrent_clients"
        ):
            raise _InvalidTimingJob("timing.client_out_of_range")
    else:
        raise _InvalidTimingJob("timing.unknown_kind")
    if set(coordinates) != expected:
        raise _InvalidTimingJob("timing.unexpected_coordinates")
    return kind


def _ope_shape(manifest: ManifestInput) -> tuple[int, int]:
    challenge = _member(_member(manifest, "ope"), "fhe_challenge")
    trajectories = _member(challenge, "trajectories_per_batch")
    horizon = _member(challenge, "horizon")
    if type(trajectories) is not int or trajectories < 1:
        raise _InvalidTimingJob("timing.invalid_ope_trajectories")
    if type(horizon) is not int or horizon < 1:
        raise _InvalidTimingJob("timing.invalid_ope_horizon")
    return trajectories, horizon


def _context_request(
    manifest: ManifestInput,
    job_seed: int,
    *,
    workload: str,
    container_id: str,
    trajectories: int | None = None,
    horizon: int | None = None,
) -> TimingContextRequest:
    implementation = (
        "shield_exact_margin_tensor_v1" if workload == "shield" else "POLYNOMIAL_APPROX_OPE_V1"
    )
    return TimingContextRequest(
        workload=workload,
        implementation=implementation,
        trajectories=trajectories,
        horizon=horizon,
        container_id=container_id,
        configured_clients=_positive_manifest_int(manifest, "systems", "concurrent_clients"),
        seed=job_seed,
    )


def _context_payload(
    manifest: ManifestInput,
    job_id: str,
    request: TimingContextRequest,
    receipt: BackendContextReceipt,
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "schema_version": CONTEXT_RECEIPT_SCHEMA,
        "manifest_context_digest": _manifest_context_digest(manifest),
        "job_id": job_id,
        "workload": request.workload,
        "implementation": request.implementation,
        "trajectories": request.trajectories,
        "horizon": request.horizon,
        "container_id": request.container_id,
        "configured_clients": request.configured_clients,
        "backend": receipt.backend,
        "backend_version": receipt.backend_version,
        "execution_label": receipt.execution_label,
        "implementation_id": receipt.implementation_id,
        "circuit_digest": receipt.circuit_digest,
        "server_artifact_digest": receipt.server_artifact_digest,
        "client_artifact_digest": receipt.client_artifact_digest,
        "hardware_digest": receipt.hardware_digest,
        "compile_ns": receipt.compile_ns,
        "key_setup_ns": receipt.key_setup_ns,
        "evaluation_key_bytes": receipt.evaluation_key_bytes,
        "trust_scope": COLOCATED_TRUST_SCOPE,
        "retry_policy": RETRY_POLICY,
    }
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return {**payload, "context_digest": digest}


def _failure_observation(error: Exception) -> TimingObservation:
    code = error.failure_code if isinstance(error, TimingCallError) else "backend.call_failed"
    return TimingObservation({}, {}, False, code)


def _run_attempts(
    *,
    manifest: ManifestInput,
    job_id: str,
    backend: TimingBackend,
    request: TimingContextRequest,
    warmups: int,
    measured: int,
    request_prefix: str,
) -> dict[str, JSONValue]:
    session = backend.open_context(request)
    context = _context_payload(manifest, job_id, request, session.context_receipt)
    context_digest = cast(str, context["context_digest"])
    rows: list[dict[str, object]] = []
    for index in range(warmups + measured):
        try:
            observation = session.measure(index)
        except Exception as error:  # backend exception text is deliberately never retained
            observation = _failure_observation(error)
        row: dict[str, object] = {
            "schema_version": TIMING_ROW_SCHEMA,
            "container_id": request.container_id,
            "trial_id": job_id,
            "context_digest": context_digest,
            "request_id": f"{request_prefix}-{index:04d}",
            "is_warmup": index < warmups,
            "success": observation.success,
            "timing_ns": dict(observation.timing_ns),
            "byte_metrics": dict(observation.byte_metrics),
        }
        if observation.failure_code is not None:
            row["failure_code"] = observation.failure_code
        rows.append(row)
    summary = summarize_timing_rows(rows)
    return {
        "context": context,
        "rows": cast(list[JSONValue], rows),
        "summary": summary,
    }


def _success_envelope(
    root: Path, path: PurePosixPath, payload: Mapping[str, object]
) -> dict[str, object]:
    encoded = canonical_json(payload)
    destination = root / Path(*path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing to replace an existing timing artifact") from error
    return {
        "status": "succeeded",
        "artifact_path": str(path),
        "artifact_digest": hashlib.sha256(encoded).hexdigest(),
        "reason_code": None,
    }


def _rejected(reason_code: str) -> dict[str, object]:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason_code,
    }


def execute_flagship_job(
    manifest: ManifestInput,
    job: JobInput,
    evidence_root: str | Path,
    *,
    backend: TimingBackend | None = None,
) -> dict[str, object]:
    """Execute one timing job and write its sole canonical artifact.

    Modal supplies canonical mapping payloads; direct callers may supply the immutable
    manifest/job models. ``backend`` is a deterministic test seam. Production calls
    omit it and lazily construct actual Concrete/TenSEAL contexts.
    """

    try:
        job_id, job_seed, coordinates = _job_parts(job)
        kind = _validate_coordinates(manifest, coordinates)
    except _InvalidTimingJob as error:
        return _rejected(error.reason_code)
    selected_backend = backend or _RealTimingBackend()
    root = Path(evidence_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence_root must be an existing non-symlink directory")

    groups: dict[str, JSONValue] = {}
    try:
        if kind == "shield":
            container = _coordinate_int(coordinates, "container")
            request = _context_request(
                manifest, job_seed, workload="shield", container_id=f"shield-{container:04d}"
            )
            groups["shield"] = _run_attempts(
                manifest=manifest,
                job_id=job_id,
                backend=selected_backend,
                request=request,
                warmups=_positive_manifest_int(manifest, "systems", "shield_warmups_per_container"),
                measured=_positive_manifest_int(
                    manifest, "systems", "shield_measured_per_container"
                ),
                request_prefix="shield",
            )
        elif kind == "ope":
            container = _coordinate_int(coordinates, "container")
            trajectories, horizon = _ope_shape(manifest)
            request = _context_request(
                manifest,
                job_seed,
                workload="ope",
                container_id=f"ope-{container:04d}",
                trajectories=trajectories,
                horizon=horizon,
            )
            groups["ope"] = _run_attempts(
                manifest=manifest,
                job_id=job_id,
                backend=selected_backend,
                request=request,
                warmups=_positive_manifest_int(manifest, "systems", "ope_warmups_per_container"),
                measured=_positive_manifest_int(manifest, "systems", "ope_measured_per_container"),
                request_prefix="ope",
            )
        elif kind == "scale":
            container = _coordinate_int(coordinates, "container")
            trajectories = _coordinate_int(coordinates, "trajectories")
            horizon = _coordinate_int(coordinates, "horizon")
            request = _context_request(
                manifest,
                job_seed,
                workload="ope",
                container_id=f"scale-{trajectories}x{horizon}-{container:04d}",
                trajectories=trajectories,
                horizon=horizon,
            )
            groups["ope"] = _run_attempts(
                manifest=manifest,
                job_id=job_id,
                backend=selected_backend,
                request=request,
                warmups=_positive_manifest_int(manifest, "systems", "scale_warmups_per_container"),
                measured=_positive_manifest_int(
                    manifest, "systems", "scale_measured_per_container"
                ),
                request_prefix="scale",
            )
        else:
            client = _coordinate_int(coordinates, "client")
            trajectories, horizon = _ope_shape(manifest)
            container_id = f"concurrent-{client:04d}"
            shield_request = _context_request(
                manifest, job_seed, workload="shield", container_id=container_id
            )
            ope_request = _context_request(
                manifest,
                job_seed,
                workload="ope",
                container_id=container_id,
                trajectories=trajectories,
                horizon=horizon,
            )
            groups["shield"] = _run_attempts(
                manifest=manifest,
                job_id=job_id,
                backend=selected_backend,
                request=shield_request,
                warmups=0,
                measured=_positive_manifest_int(
                    manifest, "systems", "concurrent_shield_calls_per_client"
                ),
                request_prefix="concurrent-shield",
            )
            groups["ope"] = _run_attempts(
                manifest=manifest,
                job_id=job_id,
                backend=selected_backend,
                request=ope_request,
                warmups=0,
                measured=_positive_manifest_int(
                    manifest, "systems", "concurrent_ope_calls_per_client"
                ),
                request_prefix="concurrent-ope",
            )
    except (UnsupportedTimingCell, _InvalidTimingJob) as error:
        return _rejected(error.reason_code)

    payload: dict[str, object] = {
        "schema_version": TIMING_ARTIFACT_SCHEMA,
        "job_id": job_id,
        "stage": "timing",
        "kind": kind,
        "coordinates": coordinates,
        "retry_policy": RETRY_POLICY,
        "trust_scope": COLOCATED_TRUST_SCOPE,
        "groups": groups,
    }
    relative = PurePosixPath("timing", f"{job_id}.json")
    return _success_envelope(root, relative, payload)


def _coordinate_int(coordinates: Mapping[str, str | int | float], name: str) -> int:
    value = coordinates.get(name)
    if type(value) is not int or value < 0:
        raise _InvalidTimingJob(f"timing.invalid_{name}")
    return value
