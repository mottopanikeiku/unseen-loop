"""Closed, nonsecret evidence bundle for a complete nonlinear REAL-FHE challenge."""

from __future__ import annotations

import hashlib
import math
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from unseen_loop.artifacts import ArtifactLedger, dataclass_dict
from unseen_loop.fhe_backend import (
    CompiledPolicy,
    RoundTripMeasurement,
    _import_fhe,
    compile_policy,
    server_artifact_secret_markers,
)
from unseen_loop.policy import PolynomialPolicy
from unseen_loop.specs import PolicySpec, QuantizerSpec

CHALLENGE_SCHEMA = "unseen-loop/nonlinear-fhe-challenge-v1"
ROW_SCHEMA = "unseen-loop/nonlinear-fhe-row-v1"
SUMMARY_FILE = "summary.json"
RAW_ROWS_FILE = "raw.jsonl"
POLICY_FILE = "policy.json"
RECEIPT_FILE = "receipt.json"
SERVER_FILE = "server.zip"
CLIENT_SPECS_FILE = "client-specs.bin"
CHECKSUM_FILE = "checksums.sha256"
LEDGERED_FILES = (
    CLIENT_SPECS_FILE,
    POLICY_FILE,
    RAW_ROWS_FILE,
    RECEIPT_FILE,
    SERVER_FILE,
    SUMMARY_FILE,
)

Compiler = Callable[..., CompiledPolicy]
_TIMING_FIELDS = (
    "encrypt_ns",
    "server_evaluate_ns",
    "decrypt_ns",
    "end_to_end_ns",
)
_SIZE_FIELDS = ("evaluation_key_bytes", "request_bytes", "response_bytes")


@dataclass(frozen=True)
class TimingDistribution:
    """Explicitly qualified empirical timing distribution in nanoseconds."""

    samples: int
    p50_ns: float | None
    p95_ns: float | None
    percentile_method: str = "linear interpolation between ordered observations"
    p50_minimum_samples: int = 2
    p95_minimum_samples: int = 20


@dataclass(frozen=True)
class ChallengeSummary:
    """Completion record for the exhaustive circuit-family challenge."""

    policy_digest: str
    qmax: int
    domain_points: int
    simulation_rows: int
    real_domain_rows: int
    canary_codes: int
    canary_repetitions_per_code: int
    canary_rows: int
    domain_order_sha256: str
    real_fhe_rows: int
    quadratic_feature_products_per_inference: int
    simulation_all_match: bool
    real_fhe_all_match: bool
    canary_distinct_request_hashes: int
    canary_randomness_passed: bool
    single_client_key_context: bool
    client_keygen_ns: int
    evaluation_key_sha256: str
    client_context_sha256: str
    server_secret_key_marker_present: bool
    timing_distributions: dict[str, TimingDistribution]
    backend: str = "REAL FHE"
    circuit_family: str = "degree-2 two-feature two-action integer polynomial"
    calibration_strategy: str = "exhaustive signed integer domain"
    execution_boundary: str = "serialized Concrete client/server"
    claim: str = "encrypted-encrypted multiplication exercised over the complete declared domain"
    schema_version: str = CHALLENGE_SCHEMA


def challenge_policy_spec(*, qmax: int = 2) -> PolicySpec:
    """Return the fixed, deterministic nonlinear policy used by the challenge."""

    if isinstance(qmax, bool) or not isinstance(qmax, int) or not 1 <= qmax <= 2:
        raise ValueError("challenge qmax must be the integer 1 or 2")
    integer_coefficients = (
        (3, -2, 1, 1, 2, -1),
        (-1, 1, -2, -2, 1, 2),
    )
    return PolicySpec(
        name="nonlinear-real-fhe-exhaustive-challenge",
        env_id="SyntheticDegree2-v0",
        degree=2,
        actions=2,
        quantizer=QuantizerSpec(center=(0.0, 0.0), step=(1.0, 1.0), qmax=qmax),
        float_coefficients=tuple(
            tuple(float(value) for value in row) for row in integer_coefficients
        ),
        integer_coefficients=integer_coefficients,
        coefficient_scale=1.0,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[int], probability: float, minimum_samples: int) -> float | None:
    if len(values) < minimum_samples:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _timing_distribution(values: Sequence[int]) -> TimingDistribution:
    return TimingDistribution(
        samples=len(values),
        p50_ns=_percentile(values, 0.50, 2),
        p95_ns=_percentile(values, 0.95, 20),
    )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class ChallengeSession(Protocol):
    """One client key and serialized evaluator context reused for all challenge calls."""

    client_keygen_ns: int
    evaluation_key_sha256: str
    client_context_sha256: str

    def run(self, quantized: npt.ArrayLike) -> RoundTripMeasurement: ...


SessionFactory = Callable[[CompiledPolicy], ChallengeSession]


def _serialized_bytes(value: bytes | str) -> bytes:
    return value.encode() if isinstance(value, str) else bytes(value)


class _SerializedChallengeSession:
    """REAL-FHE session that re-encrypts each input under one fixed client key."""

    def __init__(self, compiled: CompiledPolicy) -> None:
        fhe = _import_fhe()
        client_specs_payload = compiled.client_specs_path.read_bytes()
        client_specs = fhe.ClientSpecs.deserialize(client_specs_payload)
        self._client = fhe.Client(client_specs)
        started = time.perf_counter_ns()
        self._client.keys.generate()
        self.client_keygen_ns = time.perf_counter_ns() - started
        self._evaluation_keys = _serialized_bytes(self._client.evaluation_keys.serialize())
        self._server_evaluation_keys = fhe.EvaluationKeys.deserialize(self._evaluation_keys)
        self._server = fhe.Server.load(str(compiled.server_path))
        self._server_secret_marker_present = bool(
            server_artifact_secret_markers(compiled.server_path)
        )
        self._fhe = fhe
        self._policy = compiled.policy
        self.evaluation_key_sha256 = _sha256(self._evaluation_keys)
        self.client_context_sha256 = _sha256(client_specs_payload)

    def run(self, quantized: npt.ArrayLike) -> RoundTripMeasurement:
        values = np.asarray(quantized, dtype=np.int64)
        started = time.perf_counter_ns()
        phase = time.perf_counter_ns()
        encrypted_input = self._client.encrypt(values)
        serialized_input = _serialized_bytes(encrypted_input.serialize())
        encrypt_ns = time.perf_counter_ns() - phase

        phase = time.perf_counter_ns()
        server_input = self._fhe.Value.deserialize(serialized_input)
        encrypted_output = self._server.run(
            server_input,
            evaluation_keys=self._server_evaluation_keys,
        )
        serialized_output = _serialized_bytes(encrypted_output.serialize())
        server_evaluate_ns = time.perf_counter_ns() - phase

        phase = time.perf_counter_ns()
        client_output = self._fhe.Value.deserialize(serialized_output)
        decrypted = np.atleast_1d(np.asarray(self._client.decrypt(client_output), dtype=np.int64))
        decrypt_ns = time.perf_counter_ns() - phase
        clear = np.atleast_1d(self._policy.integer_scores_from_quantized(values))
        return RoundTripMeasurement(
            input_shape=tuple(int(value) for value in values.shape),
            output_shape=tuple(int(value) for value in decrypted.shape),
            output_matches_clear=bool(np.array_equal(decrypted, clear)),
            keygen_ns=0,
            encrypt_ns=encrypt_ns,
            server_evaluate_ns=server_evaluate_ns,
            decrypt_ns=decrypt_ns,
            end_to_end_ns=time.perf_counter_ns() - started,
            evaluation_key_bytes=len(self._evaluation_keys),
            request_bytes=len(serialized_input),
            response_bytes=len(serialized_output),
            request_sha256=_sha256(serialized_input),
            response_sha256=_sha256(serialized_output),
            server_secret_key_marker_present=self._server_secret_marker_present,
        )


def _validate_measurement(measurement: RoundTripMeasurement) -> None:
    if measurement.backend != "REAL FHE":
        raise RuntimeError("challenge received a measurement not labelled REAL FHE")
    if measurement.input_shape != (2,) or measurement.output_shape != (2,):
        raise RuntimeError("REAL FHE measurement has an unexpected input or output shape")
    if not measurement.output_matches_clear:
        raise RuntimeError("REAL FHE output disagrees with exact integer semantics")
    if measurement.server_secret_key_marker_present:
        raise RuntimeError("serialized server artifact reports a client secret-key marker")
    for field in _TIMING_FIELDS + _SIZE_FIELDS:
        value = getattr(measurement, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"REAL FHE measurement has invalid {field}")
    if not _valid_sha256(measurement.request_sha256) or not _valid_sha256(
        measurement.response_sha256
    ):
        raise RuntimeError("REAL FHE measurement has an invalid ciphertext digest")


def _measurement_row(
    measurement: RoundTripMeasurement,
    *,
    case_index: int,
    phase: str,
    repetition: int | None,
    simulation_matches_integer_clear: bool,
    session: ChallengeSession,
) -> dict[str, Any]:
    row = {
        "schema_version": ROW_SCHEMA,
        "phase": phase,
        "case_index": case_index,
        "repetition": repetition,
        "simulation_matches_integer_clear": simulation_matches_integer_clear,
        "real_fhe_matches_integer_clear": measurement.output_matches_clear,
        "input_shape": list(measurement.input_shape),
        "output_shape": list(measurement.output_shape),
        "evaluation_key_sha256": session.evaluation_key_sha256,
        "client_context_sha256": session.client_context_sha256,
        "request_sha256": measurement.request_sha256,
        "response_sha256": measurement.response_sha256,
        "server_secret_key_marker_present": measurement.server_secret_key_marker_present,
    }
    for field in _TIMING_FIELDS + _SIZE_FIELDS:
        row[field] = getattr(measurement, field)
    return row


def _validate_compilation(
    compiled: CompiledPolicy,
    policy: PolynomialPolicy,
    domain: npt.NDArray[np.int64],
) -> tuple[bytes, bytes]:
    receipt = compiled.receipt
    if (
        compiled.policy.spec.digest != policy.spec.digest
        or receipt.policy_digest != policy.spec.digest
    ):
        raise RuntimeError("compiled policy digest does not match the challenge policy")
    if receipt.calibration_strategy != "exhaustive signed integer domain":
        raise RuntimeError("degree-2 challenge was not compiled with exhaustive calibration")
    if receipt.domain_points != len(domain) or receipt.calibration_rows != len(domain):
        raise RuntimeError("compiler receipt does not cover the complete declared domain")
    if (
        receipt.input_min != (-policy.spec.quantizer.qmax,) * 2
        or receipt.input_max != (policy.spec.quantizer.qmax,) * 2
    ):
        raise RuntimeError("compiler receipt input range does not match the declared domain")
    if receipt.server_secret_key_markers:
        raise RuntimeError("serialized server artifact contains a client secret-key marker")

    server_payload = compiled.server_path.read_bytes()
    client_specs_payload = compiled.client_specs_path.read_bytes()
    if receipt.calibration_sha256 != _sha256(domain.tobytes(order="C")):
        raise RuntimeError("compiler receipt calibration digest does not match domain order")
    if server_artifact_secret_markers(compiled.server_path):
        raise RuntimeError("serialized server artifact contains a client secret-key marker")
    if (
        len(server_payload) != receipt.server_artifact_bytes
        or _sha256(server_payload) != receipt.server_artifact_sha256
    ):
        raise RuntimeError("serialized server artifact does not match its compiler receipt")
    if (
        len(client_specs_payload) != receipt.client_specs_bytes
        or _sha256(client_specs_payload) != receipt.client_specs_sha256
    ):
        raise RuntimeError("client specs do not match their compiler receipt")
    return server_payload, client_specs_payload


def _require_empty_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("challenge output must be a directory")
        if any(destination.iterdir()):
            raise RuntimeError(
                "challenge output directory must be empty; refusing unledgered output"
            )


def run_fhe_challenge(
    output: str | Path,
    *,
    qmax: int = 2,
    canary_repetitions: int = 5,
    global_p_error: float = 1e-6,
    security_level: int = 128,
    compiler: Compiler | None = None,
    session_factory: SessionFactory | None = None,
) -> ChallengeSummary:
    """Compile and execute the complete nonlinear domain, then close its evidence ledger.

    ``compiler`` and ``session_factory`` are explicit failure-injection seams for focused
    tests. Production callers leave both unset to use Concrete's serialized REAL-FHE path.
    """

    if (
        isinstance(canary_repetitions, bool)
        or not isinstance(canary_repetitions, int)
        or canary_repetitions < 2
    ):
        raise ValueError("canary_repetitions must be an integer of at least two")
    destination = Path(output)
    _require_empty_destination(destination)

    policy = PolynomialPolicy(challenge_policy_spec(qmax=qmax))
    values = range(-qmax, qmax + 1)
    domain = np.asarray(tuple(product(values, repeat=2)), dtype=np.int64)
    selected_canaries = (
        np.asarray((-qmax, -qmax), dtype=np.int64),
        np.asarray((-qmax, qmax), dtype=np.int64),
        np.asarray((qmax, qmax), dtype=np.int64),
    )
    compile_challenge = compiler or compile_policy
    create_session = session_factory or _SerializedChallengeSession

    rows: list[dict[str, Any]] = []
    canary_request_hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="unseen-loop-fhe-challenge-") as temporary:
        compiled = compile_challenge(
            policy,
            domain,
            Path(temporary),
            global_p_error=global_p_error,
            security_level=security_level,
        )
        server_payload, client_specs_payload = _validate_compilation(compiled, policy, domain)
        session = create_session(compiled)
        if not _valid_sha256(session.evaluation_key_sha256) or not _valid_sha256(
            session.client_context_sha256
        ):
            raise RuntimeError("serialized FHE session has an invalid context digest")
        evaluation_key_sha256 = session.evaluation_key_sha256
        client_context_sha256 = session.client_context_sha256
        if (
            isinstance(session.client_keygen_ns, bool)
            or not isinstance(session.client_keygen_ns, int)
            or session.client_keygen_ns < 0
        ):
            raise RuntimeError("serialized FHE session has an invalid keygen timing")

        for case_index, code in enumerate(domain):
            clear = np.atleast_1d(policy.integer_scores_from_quantized(code))
            simulated = np.atleast_1d(compiled.simulate(code))
            simulation_match = bool(np.array_equal(simulated, clear))
            if not simulation_match:
                raise RuntimeError("compiled simulation disagrees with exact integer semantics")
            measurement = session.run(code)
            _validate_measurement(measurement)
            rows.append(
                _measurement_row(
                    measurement,
                    case_index=case_index,
                    phase="exhaustive-domain",
                    repetition=None,
                    simulation_matches_integer_clear=simulation_match,
                    session=session,
                )
            )

        for case_index, code in enumerate(selected_canaries):
            clear = np.atleast_1d(policy.integer_scores_from_quantized(code))
            simulated = np.atleast_1d(compiled.simulate(code))
            simulation_match = bool(np.array_equal(simulated, clear))
            if not simulation_match:
                raise RuntimeError("canary simulation disagrees with exact integer semantics")
            group_hashes: list[str] = []
            for repetition in range(canary_repetitions):
                measurement = session.run(code)
                _validate_measurement(measurement)
                group_hashes.append(measurement.request_sha256)
                canary_request_hashes.append(measurement.request_sha256)
                rows.append(
                    _measurement_row(
                        measurement,
                        case_index=case_index,
                        phase="fresh-ciphertext-canary",
                        repetition=repetition,
                        simulation_matches_integer_clear=simulation_match,
                        session=session,
                    )
                )
            if len(set(group_hashes)) != canary_repetitions:
                raise RuntimeError(
                    "fresh encryptions of a canary produced repeated ciphertext hashes"
                )

        expected_domain_points = (2 * qmax + 1) ** 2
        expected_canary_rows = len(selected_canaries) * canary_repetitions
        if (
            len(domain) != expected_domain_points
            or len(rows) != expected_domain_points + expected_canary_rows
        ):
            raise RuntimeError("challenge row accounting is incomplete")
        evaluation_contexts = {str(row["evaluation_key_sha256"]) for row in rows}
        client_contexts = {str(row["client_context_sha256"]) for row in rows}
        if evaluation_contexts != {evaluation_key_sha256} or client_contexts != {
            client_context_sha256
        }:
            raise RuntimeError("challenge calls did not reuse one client key context")
        rows.sort(key=lambda row: str(row["request_sha256"]))
        timing_distributions = {
            field: _timing_distribution([int(row[field]) for row in rows])
            for field in _TIMING_FIELDS
        }
        summary = ChallengeSummary(
            policy_digest=policy.spec.digest,
            qmax=qmax,
            domain_points=expected_domain_points,
            simulation_rows=expected_domain_points,
            real_domain_rows=expected_domain_points,
            canary_codes=len(selected_canaries),
            canary_repetitions_per_code=canary_repetitions,
            canary_rows=expected_canary_rows,
            real_fhe_rows=len(rows),
            domain_order_sha256=compiled.receipt.calibration_sha256,
            quadratic_feature_products_per_inference=policy.encrypted_multiplications,
            simulation_all_match=True,
            real_fhe_all_match=True,
            canary_distinct_request_hashes=len(set(canary_request_hashes)),
            canary_randomness_passed=(len(set(canary_request_hashes)) == expected_canary_rows),
            single_client_key_context=True,
            client_keygen_ns=session.client_keygen_ns,
            evaluation_key_sha256=evaluation_key_sha256,
            client_context_sha256=client_context_sha256,
            server_secret_key_marker_present=False,
            timing_distributions=timing_distributions,
        )
        if not summary.canary_randomness_passed:
            raise RuntimeError("canary ciphertext hashes collide across selected inputs")

        ledger = ArtifactLedger(destination)
        ledger.write_json(POLICY_FILE, policy.spec.to_dict())
        ledger.write_json(RECEIPT_FILE, dataclass_dict(compiled.receipt))
        ledger.write_bytes(SERVER_FILE, server_payload)
        ledger.write_bytes(CLIENT_SPECS_FILE, client_specs_payload)
        ledger.write_jsonl(RAW_ROWS_FILE, rows)
        ledger.write_json(SUMMARY_FILE, asdict(summary))
        ledger.finalize()
        verified, failures = ledger.verify()
        if not verified:
            raise RuntimeError(f"challenge artifact verification failed: {failures}")
        checksum_rows = (destination / CHECKSUM_FILE).read_text().splitlines()
        ledgered = tuple(sorted(line.partition("  ")[2] for line in checksum_rows))
        if ledgered != tuple(sorted(LEDGERED_FILES)):
            raise RuntimeError("challenge checksum ledger is not closed over the expected files")
        return summary
