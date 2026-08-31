"""Flagship REAL-FHE shield challenge executor.

The persisted job receipt contains systems/conformance accounting only.  In
particular, neither the derived integer state nor either margin tensor is
written to evaluator evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from unseen_loop.shield.certificate import ErrorBuffer
from unseen_loop.shield.fhe import (
    DOMAIN_POINTS,
    MARGIN_SHAPE,
    QMAX,
    STATE_SHAPE,
    ShieldCallReceipt,
    ShieldFHEClient,
    ShieldFHEMode,
    ShieldFHEServer,
    ShieldIntegerSpec,
    clear_margin_tensor,
    compile_shield,
)
from unseen_loop.shield.fhe import (
    SCHEMA_VERSION as SHIELD_SCHEMA_VERSION,
)
from unseen_loop.shield.types import Action

ARTIFACT_SCHEMA_VERSION = "unseen-loop/flagship-shield-fhe-job-v1"
_STAGE = "shield_fhe_challenge"
_CATEGORIES = ("occupancy", "extrema", "threshold", "tie", "canary")
_JOB_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SERVER_NAME = "shield-server.zip"
_CLIENT_NAME = "shield-client-specs.bin"
_RECEIPT_NAME = "shield-receipt.json"


@dataclass(frozen=True, slots=True)
class _Cache:
    directory: Path
    server_path: Path
    client_path: Path
    receipt: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _challenge(manifest: Mapping[str, object]) -> Mapping[str, object]:
    shield = _mapping(manifest.get("shield"), "manifest.shield")
    output_shape = shield.get("output_shape")
    if not isinstance(output_shape, (list, tuple)) or tuple(output_shape) != MARGIN_SHAPE:
        raise ValueError("manifest shield output_shape does not match the frozen FHE circuit")
    challenge = _mapping(shield.get("fhe_challenge"), "manifest.shield.fhe_challenge")
    if _integer(challenge.get("security_level"), "security_level") != 128:
        raise ValueError("the shield FHE challenge requires 128-bit security")
    global_p_error = challenge.get("global_p_error")
    if isinstance(global_p_error, bool) or not isinstance(global_p_error, (int, float)):
        raise ValueError("global_p_error must be numeric")
    if not 0.0 < float(global_p_error) <= 1e-3:
        raise ValueError("global_p_error must lie in (0, 1e-3]")
    return challenge


def _validate_job(
    manifest: Mapping[str, object], job: Mapping[str, object]
) -> tuple[str, str, Mapping[str, object], Mapping[str, object]]:
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job_id is not a safe canonical identifier")
    if job.get("stage") != _STAGE:
        raise ValueError("job stage is not shield_fhe_challenge")
    seed = _integer(job.get("seed"), "job.seed")
    if seed < 0:
        raise ValueError("job.seed must be non-negative")
    coordinates = _mapping(job.get("coordinates"), "job.coordinates")
    kind = coordinates.get("kind")
    if kind not in {"valid", "invalid"}:
        raise ValueError("shield FHE job kind must be valid or invalid")
    challenge = _challenge(manifest)
    return job_id, str(kind), coordinates, challenge


def _coordinate_index(coordinates: Mapping[str, object], name: str, upper: int) -> int:
    value = _integer(coordinates.get(name), f"coordinates.{name}")
    if value < 0 or value >= upper:
        raise ValueError(f"coordinates.{name} is outside the planned denominator")
    return value


def _base_five_point(namespace: str, state_index: int) -> npt.NDArray[np.int64]:
    digest = hashlib.sha256(f"{namespace}\0{state_index}".encode()).digest()
    number = int.from_bytes(digest, "big")
    values: list[int] = []
    for _ in range(STATE_SHAPE[0]):
        number, digit = divmod(number, 2 * QMAX + 1)
        values.append(digit - QMAX)
    return np.asarray(values, dtype=np.int64)


def _valid_state(category: str, state_index: int) -> npt.NDArray[np.int64]:
    """Derive a stable qmax=2 challenge point without using encryption index/seed."""

    if category == "extrema":
        return np.asarray(
            [QMAX if (state_index >> dimension) & 1 else -QMAX for dimension in range(6)],
            dtype=np.int64,
        )
    point = _base_five_point(category, state_index)
    if category == "threshold":
        # Every threshold row has one exact center coordinate and varied neighbours.
        point[state_index % STATE_SHAPE[0]] = 0
    elif category == "tie":
        # Symmetric position/velocity wires exercise equal directional inputs.
        point[1] = point[0]
        point[3] = point[2]
    if point.shape != STATE_SHAPE or np.any(np.abs(point) > QMAX):
        raise AssertionError("derived valid shield point escaped the frozen domain")
    return point


def _invalid_state(case: int) -> npt.NDArray[np.int64]:
    point = _base_five_point("invalid", case)
    dimension = case % STATE_SHAPE[0]
    point[dimension] = (QMAX + 1 + case // STATE_SHAPE[0]) * (-1 if case & 1 else 1)
    if not np.any(np.abs(point) > QMAX):
        raise AssertionError("derived invalid shield point remained in domain")
    return point


def _requested_action(category: str, state_index: int) -> Action:
    digest = hashlib.sha256(f"requested-action\0{category}\0{state_index}".encode()).digest()
    return Action(int.from_bytes(digest[:8], "big") % len(Action))


def _file_digest(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("compiled shield cache is missing a regular artifact")
    return _sha256(path.read_bytes())


def _load_cache(
    directory: Path, spec: ShieldIntegerSpec, challenge: Mapping[str, object]
) -> _Cache:
    server_path = directory / _SERVER_NAME
    client_path = directory / _CLIENT_NAME
    receipt_path = directory / _RECEIPT_NAME
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RuntimeError("compiled shield cache is missing its receipt")
    raw = json.loads(receipt_path.read_text())
    if not isinstance(raw, dict):
        raise RuntimeError("compiled shield receipt is not a JSON object")
    required = {
        "schema_version": SHIELD_SCHEMA_VERSION,
        "spec_digest": spec.spec_digest,
        "qmax": QMAX,
        "domain_points": DOMAIN_POINTS,
        "security_level": _integer(challenge.get("security_level"), "security_level"),
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise RuntimeError("compiled shield cache does not match the frozen challenge")
    if (
        tuple(raw.get("input_shape", ())) != STATE_SHAPE
        or tuple(raw.get("output_shape", ())) != MARGIN_SHAPE
    ):
        raise RuntimeError("compiled shield cache has the wrong tensor contract")
    requested = raw.get("requested_global_p_error")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or float(requested)
        != float(cast(int | float, challenge["global_p_error"]))
    ):
        raise RuntimeError("compiled shield cache has the wrong global_p_error")
    markers = raw.get("server_secret_key_markers")
    if not isinstance(markers, list) or markers:
        raise RuntimeError("compiled shield server artifact failed the secret-marker audit")
    server_digest = _file_digest(server_path)
    client_digest = _file_digest(client_path)
    if (
        raw.get("server_artifact_sha256") != server_digest
        or raw.get("client_specs_sha256") != client_digest
    ):
        raise RuntimeError("compiled shield cache artifact digest mismatch")
    if (
        raw.get("server_artifact_bytes") != server_path.stat().st_size
        or raw.get("client_specs_bytes") != client_path.stat().st_size
    ):
        raise RuntimeError("compiled shield cache artifact size mismatch")
    return _Cache(directory, server_path, client_path, raw)


def _ensure_compiled_cache(
    evidence_root: Path, spec: ShieldIntegerSpec, challenge: Mapping[str, object]
) -> _Cache:
    shared = evidence_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    cache_dir = shared / "shield-fhe"
    lock_path = shared / "shield-fhe.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+b", closefd=True) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_dir.exists():
                if not cache_dir.is_dir() or cache_dir.is_symlink():
                    raise RuntimeError("compiled shield cache path is not a regular directory")
                return _load_cache(cache_dir, spec, challenge)
            staging = Path(tempfile.mkdtemp(prefix=".shield-fhe-build-", dir=shared))
            try:
                compile_shield(
                    spec,
                    staging,
                    global_p_error=float(cast(int | float, challenge["global_p_error"])),
                    security_level=_integer(challenge.get("security_level"), "security_level"),
                )
                compiled = _load_cache(staging, spec, challenge)
                os.replace(staging, cache_dir)
                return _Cache(
                    cache_dir,
                    cache_dir / compiled.server_path.name,
                    cache_dir / compiled.client_path.name,
                    compiled.receipt,
                )
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except BaseException:
        # fdopen owns the descriptor after successful construction.
        raise


def _call_receipt(
    *,
    started: int,
    keygen_ns: int,
    encrypt_ns: int,
    evaluate_ns: int,
    decrypt_ns: int,
    evaluation_keys: bytes,
    request: bytes,
    response: bytes,
) -> ShieldCallReceipt:
    return ShieldCallReceipt(
        mode=ShieldFHEMode.REAL,
        input_shape=STATE_SHAPE,
        output_shape=MARGIN_SHAPE,
        keygen_ns=keygen_ns,
        encrypt_ns=encrypt_ns,
        server_evaluate_ns=evaluate_ns,
        decrypt_ns=decrypt_ns,
        end_to_end_ns=time.perf_counter_ns() - started,
        evaluation_key_bytes=len(evaluation_keys),
        request_bytes=len(request),
        response_bytes=len(response),
        evaluation_key_sha256=_sha256(evaluation_keys),
        request_sha256=_sha256(request),
        response_sha256=_sha256(response),
        output_matches_clear=True,
        server_secret_key_marker_present=False,
    )


def _persist_success(
    evidence_root: Path, job_id: str, payload: Mapping[str, object]
) -> dict[str, str | None]:
    relative = Path(_STAGE) / f"{job_id}.json"
    destination = evidence_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(payload)
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "succeeded",
        "artifact_path": relative.as_posix(),
        "artifact_digest": _sha256(encoded),
        "reason_code": None,
    }


def _rejected(reason_code: str) -> dict[str, str | None]:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason_code,
    }


def execute_flagship_job(
    manifest: Mapping[str, object], job: Mapping[str, object], evidence_root: str | Path
) -> dict[str, str | None]:
    """Execute one planned shield challenge call with serialized REAL FHE."""

    try:
        job_id, kind, coordinates, challenge = _validate_job(manifest, job)
        if kind == "invalid":
            if set(coordinates) != {"kind", "case"}:
                return _rejected("shield-fhe.invalid-job")
            case = _coordinate_index(
                coordinates,
                "case",
                _integer(challenge.get("invalid_domain_rejections"), "invalid_domain_rejections"),
            )
            _invalid_state(case)
            # The domain violation is decided before cache loading, key generation, or encryption.
            return _rejected("shield-fhe.invalid-domain")

        if set(coordinates) != {"kind", "category", "state", "encryption"}:
            return _rejected("shield-fhe.invalid-job")
        category = coordinates.get("category")
        if not isinstance(category, str) or category not in _CATEGORIES:
            return _rejected("shield-fhe.invalid-category")
        states = _integer(challenge.get(f"{category}_states"), f"{category}_states")
        state_index = _coordinate_index(coordinates, "state", states)
        encryptions = (
            _integer(
                challenge.get("canary_encryptions_per_state"),
                "canary_encryptions_per_state",
            )
            if category == "canary"
            else 1
        )
        encryption_index = _coordinate_index(coordinates, "encryption", encryptions)
    except (KeyError, TypeError, ValueError):
        return _rejected("shield-fhe.invalid-job")

    quantized = _valid_state(category, state_index)
    requested_action = _requested_action(category, state_index)
    spec = ShieldIntegerSpec()
    root = Path(evidence_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence_root must be an existing regular directory")
    cache = _ensure_compiled_cache(root, spec, challenge)

    started = time.perf_counter_ns()
    client = ShieldFHEClient.from_path(cache.client_path, spec)
    server = ShieldFHEServer(cache.server_path)
    keygen_ns, evaluation_keys = client.generate_keys()
    phase = time.perf_counter_ns()
    request = client.encrypt(quantized)
    encrypt_ns = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    response = server.evaluate(request, evaluation_keys)
    evaluate_ns = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    decrypted = client.decrypt_margin_tensor(response)
    decrypt_ns = time.perf_counter_ns() - phase

    clear = clear_margin_tensor(spec, quantized)
    if decrypted.shape != MARGIN_SHAPE or clear.shape != MARGIN_SHAPE:
        raise RuntimeError("shield margin accounting shape mismatch")
    matches = int(np.count_nonzero(decrypted == clear))
    output_count = int(np.prod(MARGIN_SHAPE))
    if matches != output_count:
        raise RuntimeError("REAL FHE shield margins disagree with the exact clear circuit")
    encrypted_selection = client.select_action(
        decrypted, requested_action, error_buffer=ErrorBuffer()
    )
    clear_selection = client.select_action(clear, requested_action, error_buffer=ErrorBuffer())
    if encrypted_selection.action != clear_selection.action:
        raise RuntimeError("REAL FHE shield client action disagrees with clear selection")

    call = _call_receipt(
        started=started,
        keygen_ns=keygen_ns,
        encrypt_ns=encrypt_ns,
        evaluate_ns=evaluate_ns,
        decrypt_ns=decrypt_ns,
        evaluation_keys=evaluation_keys,
        request=request,
        response=response,
    )
    canary: dict[str, object] | None = None
    if category == "canary":
        pair_payload = _canonical_bytes({"category": category, "state": state_index})
        canary = {
            "pair_id": _sha256(pair_payload),
            "encryption_index": encryption_index,
            "ciphertext_sha256": call.request_sha256,
        }
    artifact: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "job_id": job_id,
        "stage": _STAGE,
        "category": category,
        "compile": cache.receipt,
        "execution": {
            "backend": "Concrete-Python TFHE",
            "mode": ShieldFHEMode.REAL.value,
            "privacy_evidence": False,
            "privacy_claim": "none",
            "trust_scope": (
                "colocated client and server in one Modal worker; no remote-server secrecy claim"
            ),
            "server_selected_action": False,
            "client_selected_action": True,
        },
        "accounting": {
            "valid_calls": 1,
            "decoded_margins": output_count,
            "margin_matches": matches,
            "margin_mismatches": output_count - matches,
            "action_matches": 1,
            "action_mismatches": 0,
            "invalid_domain_rejections": 0,
        },
        "call": call.to_dict(),
        "canary": canary,
    }
    return _persist_success(root, job_id, artifact)
