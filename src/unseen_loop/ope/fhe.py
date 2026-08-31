"""Exact Concrete-Python canary for the frozen integer OPE circuit.

The server circuit performs the hard-clipped integer arithmetic in
:class:`OPECircuitSpec` and returns three encrypted horizon vectors.  Reciprocal
creation, decryption, fixed-point decoding, and estimator division remain
client operations.  Compilation is deliberately restricted to the documented
N=4, H=4, state-dimension-6 canary: accepting arbitrary shapes would turn a
compiler input set into unsupported domain evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from unseen_loop.fhe_backend import (
    FHEUnavailableError,
    _import_fhe,
    server_artifact_secret_markers,
)
from unseen_loop.ope.circuit import (
    CircuitOperationCounts,
    EstimatorName,
    IntegerSufficientStatistics,
    OPECircuitSpec,
    QuantizedTrajectoryTensors,
    _quantize,
    _quantize_reciprocal,
)
from unseen_loop.ope.circuit import (
    CircuitReceipt as IntegerCircuitReceipt,
)
from unseen_loop.ope.types import SufficientStatistics, TrajectoryBatch

CANARY_TRAJECTORIES = 4
CANARY_HORIZON = 4
CANARY_STATE_DIM = 6
MINIMAL_CANARY_SHAPE = (1, 2, 1)
MAX_CALIBRATION_ROWS = 1_000_000
MAX_COMPILED_INTEGER_BITS = 63
MAX_ENCRYPTED_OPERATION_BITS = 16

ExecutionMode = Literal["INTEGER", "SIMULATION", "REAL"]
IntArray = npt.NDArray[np.int64]


class OPEFHEError(RuntimeError):
    """Base class for failures that must not fall back to clear execution."""


class OPEFHEConformanceError(OPEFHEError):
    """Raised when Concrete disagrees with the frozen integer reference."""


class OPEFHESecurityError(OPEFHEError):
    """Raised when required backend or artifact security evidence is absent."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _spec_digest(spec: OPECircuitSpec) -> str:
    payload = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256(payload)


def _probability(value: Any, field: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise OPEFHESecurityError(f"Concrete did not report {field}") from error
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise OPEFHESecurityError(f"Concrete reported invalid {field}")
    return probability


@dataclass(frozen=True)
class OPEFHEReceipt:
    """Public compilation, probability, domain, cost, and artifact evidence."""

    spec_digest: str
    concrete_python_version: str
    requested_p_error: float | None
    requested_global_p_error: float | None
    compiled_p_error: float
    compiled_global_p_error: float
    security_level: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, int]
    encrypted_output_vectors: int
    integers_per_output_vector: int
    calibration_strategy: str
    quantized_state_domain_points: int
    calibration_rows: int
    calibration_sha256: str
    input_min: tuple[int, ...]
    input_max: tuple[int, ...]
    operations: CircuitOperationCounts
    maximum_integer_bit_width: int
    complexity: float
    compile_ns: int
    mlir_sha256: str
    server_artifact_bytes: int
    server_artifact_sha256: str
    client_specs_bytes: int
    client_specs_sha256: str
    server_secret_key_markers: tuple[str, ...]
    backend: str = "Concrete-Python TFHE"
    mode: str = "COMPILED EXACT OPE CANARY"
    transport: str = "serialized client Value/evaluation keys/server Value"
    output_contract: str = "three encrypted H-vectors; client-only decode and division"
    schema_version: str = "unseen-loop/ope-fhe-receipt-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> OPEFHEReceipt:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("OPE FHE receipt must be a JSON object")
        operations = raw.get("operations")
        if not isinstance(operations, dict):
            raise ValueError("OPE FHE receipt is missing operation counts")
        raw["operations"] = CircuitOperationCounts(**operations)
        for field in (
            "input_shape",
            "output_shape",
            "input_min",
            "input_max",
            "server_secret_key_markers",
        ):
            raw[field] = tuple(raw[field])
        receipt = cls(**raw)
        if receipt.schema_version != "unseen-loop/ope-fhe-receipt-v1":
            raise ValueError("unsupported OPE FHE receipt schema")
        _probability(receipt.compiled_p_error, "compiled p_error")
        _probability(receipt.compiled_global_p_error, "compiled global_p_error")
        if receipt.security_level != 128:
            raise OPEFHESecurityError("receipt does not attest the required 128-bit security level")
        if receipt.server_secret_key_markers:
            raise OPEFHESecurityError("receipt reports secret-key-like server artifact members")
        return receipt


@dataclass(frozen=True)
class SanitizedOPECallEvidence:
    """Transport measurements with no clear inputs, outputs, or estimator values."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, int]
    encrypted_output_vectors: int
    integers_per_output_vector: int
    keygen_ns: int
    encrypt_ns: int
    server_evaluate_ns: int
    decrypt_ns: int
    end_to_end_ns: int
    evaluation_key_bytes: int
    request_bytes: int
    response_bytes: int
    request_sha256: str
    response_sha256: str
    output_matches_integer_reference: bool
    server_secret_key_marker_present: bool
    backend: str = "REAL FHE"
    schema_version: str = "unseen-loop/ope-fhe-call-evidence-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class OPEConformanceResult:
    """Client-visible result for one exact integer, simulation, or REAL run."""

    mode: ExecutionMode
    integer_statistics: IntegerSufficientStatistics
    client_statistics: SufficientStatistics
    integer_receipt: IntegerCircuitReceipt
    call_evidence: SanitizedOPECallEvidence | None = None
    encrypted_output_vectors: int = 3

    def __post_init__(self) -> None:
        if self.mode == "REAL" and self.call_evidence is None:
            raise ValueError("REAL results require sanitized call evidence")
        if self.mode != "REAL" and self.call_evidence is not None:
            raise ValueError("only REAL results carry transport evidence")


@dataclass(frozen=True)
class _InputLayout:
    state: slice
    masks: slice
    rewards: slice
    reciprocals: slice
    width: int


def _input_layout(spec: OPECircuitSpec) -> _InputLayout:
    trajectories = spec.trajectories.trajectories
    horizon = spec.trajectories.horizon
    state_count = trajectories * horizon * spec.trajectories.state_dim
    mask_count = trajectories * horizon * spec.trajectories.action_count
    scalar_count = trajectories * horizon
    state = slice(0, state_count)
    masks = slice(state.stop, state.stop + mask_count)
    rewards = slice(masks.stop, masks.stop + scalar_count)
    reciprocals = slice(rewards.stop, rewards.stop + scalar_count)
    return _InputLayout(state, masks, rewards, reciprocals, reciprocals.stop)


def _require_canary(spec: OPECircuitSpec) -> None:
    shape = spec.trajectories
    actual = (shape.trajectories, shape.horizon, shape.state_dim)
    primary = (CANARY_TRAJECTORIES, CANARY_HORIZON, CANARY_STATE_DIM)
    if actual not in {primary, MINIMAL_CANARY_SHAPE}:
        raise ValueError(
            "exact Concrete OPE is restricted to the declared N,H,state_dim "
            f"canaries {primary} or {MINIMAL_CANARY_SHAPE}"
        )
    numerator_bounds, denominator_bounds, raw_bounds = spec._overflow_bounds()
    maximum = max((*numerator_bounds, *denominator_bounds, *raw_bounds))
    if maximum.bit_length() > MAX_COMPILED_INTEGER_BITS:
        raise ValueError(
            "exact Concrete OPE canary exceeds the supported integer range; "
            "use explicitly smaller fixed-point scales or the approximate CKKS path"
        )


def encode_quantized_inputs(spec: OPECircuitSpec, tensors: QuantizedTrajectoryTensors) -> IntArray:
    """Validate and flatten the four encrypted client tensors in a fixed order."""
    _require_canary(spec)
    spec._validate_tensors(tensors)
    values = np.concatenate(
        (
            np.asarray(tensors.states, dtype=np.int64).reshape(-1),
            np.asarray(tensors.action_masks, dtype=np.int64).reshape(-1),
            np.asarray(tensors.rewards, dtype=np.int64).reshape(-1),
            np.asarray(tensors.behavior_reciprocals, dtype=np.int64).reshape(-1),
        )
    )
    layout = _input_layout(spec)
    if values.shape != (layout.width,):
        raise ValueError("quantized OPE input does not match the frozen encrypted shape")
    return np.asarray(values, dtype=np.int64)


def decode_output(spec: OPECircuitSpec, values: npt.ArrayLike) -> IntegerSufficientStatistics:
    """Decode a decrypted 3H integer tensor without performing estimator division."""
    array = np.asarray(values)
    horizon = spec.trajectories.horizon
    if array.shape == (3, horizon):
        array = array.reshape(-1)
    if array.shape != (3 * horizon,):
        raise ValueError("decrypted OPE output must contain exactly three horizon vectors")
    if not np.issubdtype(array.dtype, np.integer) and (
        np.any(~np.isfinite(array)) or np.any(array != np.floor(array))
    ):
        raise ValueError("decrypted OPE output must contain exact integers")
    integers = np.asarray(array, dtype=np.int64)
    return IntegerSufficientStatistics(
        tuple(int(value) for value in integers[:horizon]),
        tuple(int(value) for value in integers[horizon : 2 * horizon]),
        tuple(int(value) for value in integers[2 * horizon :]),
    )


def _encrypted_arguments(spec: OPECircuitSpec, encoded: IntArray) -> tuple[IntArray, ...]:
    layout = _input_layout(spec)
    trajectories = spec.trajectories
    return (
        encoded[layout.state].reshape(trajectories.state_shape),
        encoded[layout.masks].reshape(
            trajectories.trajectories,
            trajectories.horizon,
            trajectories.action_count,
        ),
        encoded[layout.rewards].reshape(trajectories.batch_shape),
        encoded[layout.reciprocals].reshape(trajectories.batch_shape),
    )


def calibration_inputset(
    spec: OPECircuitSpec, *, max_rows: int = MAX_CALIBRATION_ROWS
) -> tuple[IntArray, int, str]:
    """Enumerate every quantized state and selected action at scalar extrema.

    Every circuit score sees every state point.  Repeating a selected action and
    scalar extremum across all lanes reaches the extrema of cumulative products
    and horizon sums.  The method is rejected rather than sampled when the full
    evidence would exceed ``max_rows``.
    """
    _require_canary(spec)
    if max_rows < 1:
        raise ValueError("calibration row cap must be positive")
    trajectories = spec.trajectories
    state_low = tuple(_quantize(value, spec.scales.state) for value in trajectories.state_min)
    state_high = tuple(_quantize(value, spec.scales.state) for value in trajectories.state_max)
    state_domain_points = int(
        np.prod([high - low + 1 for low, high in zip(state_low, state_high, strict=True)])
    )
    reward_min = trajectories.reward_min
    reward_max = trajectories.reward_max
    assert reward_min is not None and reward_max is not None
    reward_extrema = tuple(
        sorted(
            {_quantize(reward_min, spec.scales.reward), _quantize(reward_max, spec.scales.reward)}
        )
    )
    reciprocal_extrema = tuple(
        sorted(
            {
                spec.scales.reciprocal,
                _quantize_reciprocal(spec.minimum_behavior_propensity, spec.scales.reciprocal),
            }
        )
    )
    rows = (
        state_domain_points
        * trajectories.action_count
        * len(reward_extrema)
        * len(reciprocal_extrema)
    )
    if rows > max_rows:
        raise ValueError(
            f"exact OPE calibration requires {rows:,} rows above the {max_rows:,}-row cap"
        )

    layout = _input_layout(spec)
    result = np.empty((rows, layout.width), dtype=np.int64)
    state_ranges = tuple(
        range(low, high + 1) for low, high in zip(state_low, state_high, strict=True)
    )
    row = 0
    maximum_target = 0
    samples = trajectories.trajectories * trajectories.horizon
    for state in product(*state_ranges):
        repeated_state = np.tile(np.asarray(state, dtype=np.int64), samples)
        for action in range(trajectories.action_count):
            masks = np.zeros((samples, trajectories.action_count), dtype=np.int64)
            masks[:, action] = 1
            action_mask = tuple(
                int(candidate == action) for candidate in range(trajectories.action_count)
            )
            maximum_target = max(
                maximum_target,
                spec._logged_probability_integer(tuple(state), action_mask),
            )
            for reward in reward_extrema:
                for reciprocal in reciprocal_extrema:
                    result[row, layout.state] = repeated_state
                    result[row, layout.masks] = masks.reshape(-1)
                    result[row, layout.rewards] = reward
                    result[row, layout.reciprocals] = reciprocal
                    row += 1
    assert row == rows
    encrypted_domain_values = (
        *state_low,
        *state_high,
        *reward_extrema,
        *reciprocal_extrema,
    )
    maximum_input_bits = max((2 * abs(value) + 1).bit_length() for value in encrypted_domain_values)
    maximum_raw_weight = (maximum_target * max(reciprocal_extrema)) ** trajectories.horizon
    if (
        maximum_input_bits > MAX_ENCRYPTED_OPERATION_BITS
        or maximum_raw_weight.bit_length() > MAX_ENCRYPTED_OPERATION_BITS
    ):
        raise ValueError("exact OPE canary exceeds Concrete's supported encrypted-operation width")
    strategy = (
        "complete quantized state domain x selected action x reward extrema x reciprocal extrema; "
        "each row repeated over all trajectory/horizon lanes"
    )
    return result, state_domain_points, strategy


def _kernel_compiler(spec: OPECircuitSpec) -> Any:
    fhe = _import_fhe()
    trajectories = spec.trajectories.trajectories
    horizon = spec.trajectories.horizon
    state_dim = spec.trajectories.state_dim
    action_count = spec.trajectories.action_count
    degree = spec.target_policy.degree
    coefficients = np.asarray(spec.coefficient_integers, dtype=np.int64)
    gamma_integer = _quantize(spec.gamma, spec.scales.discount)

    def kernel(states: Any, masks: Any, rewards: Any, reciprocals: Any) -> Any:
        numerators: list[Any] = [0 for _ in range(horizon)]
        denominators: list[Any] = [0 for _ in range(horizon)]
        counts: list[Any] = [0 for _ in range(horizon)]

        for trajectory in range(trajectories):
            raw_weight: Any = 1
            for step in range(horizon):
                factor = spec.scales.state if degree == 2 else 1
                features: list[Any] = [spec.scales.state**degree]
                features.extend(
                    states[trajectory, step, dimension] * factor for dimension in range(state_dim)
                )
                if degree == 2:
                    features.extend(
                        states[trajectory, step, left] * states[trajectory, step, right]
                        for left in range(state_dim)
                        for right in range(left, state_dim)
                    )
                scores: list[Any] = []
                for action in range(action_count):
                    score: Any = 0
                    for coefficient, feature in zip(coefficients[action], features, strict=True):
                        score = score + int(coefficient) * feature
                    scores.append(score)
                target: Any = 0
                for action in range(action_count):
                    target = target + masks[trajectory, step, action] * scores[action]
                raw_weight = raw_weight * target * reciprocals[trajectory, step]
                weight_scale = spec.ratio_scale ** (step + 1)
                clip_integer = _quantize(spec.weight_clip, weight_scale)
                # Exact hard clip as one finite-domain Concrete lookup, never an
                # approximate polynomial.
                clipped_weight = fhe.univariate(
                    lambda value, limit=clip_integer: min(value, limit)
                )(raw_weight)
                denominators[step] = denominators[step] + clipped_weight
                numerators[step] = numerators[step] + (
                    clipped_weight * rewards[trajectory, step] * gamma_integer**step
                )
                for action in range(action_count):
                    counts[step] = counts[step] + masks[trajectory, step, action]
        return (
            fhe.array(numerators),
            fhe.array(denominators),
            fhe.array(counts),
        )

    return fhe.compiler(
        {
            "states": "encrypted",
            "masks": "encrypted",
            "rewards": "encrypted",
            "reciprocals": "encrypted",
        }
    )(kernel)


def _requested_errors(
    p_error: float | None, global_p_error: float | None
) -> tuple[float | None, float | None]:
    if p_error is None and global_p_error is None:
        global_p_error = 1e-6
    if p_error is not None and global_p_error is not None:
        raise ValueError("request either p_error or global_p_error, not both")
    requested_p = None if p_error is None else _probability(p_error, "requested p_error")
    requested_global = (
        None if global_p_error is None else _probability(global_p_error, "requested global_p_error")
    )
    if requested_p == 0 or requested_global == 0:
        raise ValueError("requested error probability must be positive")
    if (requested_p is not None and requested_p > 1e-3) or (
        requested_global is not None and requested_global > 1e-3
    ):
        raise ValueError("requested error probability cannot exceed 1e-3")
    return requested_p, requested_global


@dataclass
class CompiledOPECircuit:
    """Architecture-specific canary with client and serialized server APIs."""

    spec: OPECircuitSpec
    server_path: Path
    client_specs_path: Path
    receipt: OPEFHEReceipt
    circuit: Any | None = None

    def _prepared(
        self, batch: TrajectoryBatch
    ) -> tuple[
        QuantizedTrajectoryTensors, IntArray, IntegerSufficientStatistics, IntegerCircuitReceipt
    ]:
        tensors = self.spec.quantize_client_inputs(batch)
        encoded = encode_quantized_inputs(self.spec, tensors)
        expected, integer_receipt = self.spec.integer_reference(batch)
        return tensors, encoded, expected, integer_receipt

    def integer_reference(
        self, batch: TrajectoryBatch, estimator: EstimatorName
    ) -> OPEConformanceResult:
        _, _, expected, integer_receipt = self._prepared(batch)
        return OPEConformanceResult(
            "INTEGER",
            expected,
            self.spec.client_statistics(expected, estimator),
            integer_receipt,
        )

    def simulate(self, batch: TrajectoryBatch, estimator: EstimatorName) -> OPEConformanceResult:
        if self.circuit is None:
            raise FHEUnavailableError(
                "simulation requires the in-process compiled Concrete circuit; no clear fallback"
            )
        _, encoded, expected, integer_receipt = self._prepared(batch)
        observed = decode_output(
            self.spec, self.circuit.simulate(*_encrypted_arguments(self.spec, encoded))
        )
        if observed != expected:
            raise OPEFHEConformanceError("Concrete simulation disagrees with exact integer OPE")
        return OPEConformanceResult(
            "SIMULATION",
            observed,
            self.spec.client_statistics(observed, estimator),
            integer_receipt,
        )

    def _verify_artifacts(self) -> tuple[bytes, bytes]:
        server_payload = self.server_path.read_bytes()
        client_payload = self.client_specs_path.read_bytes()
        if (
            len(server_payload) != self.receipt.server_artifact_bytes
            or _sha256(server_payload) != self.receipt.server_artifact_sha256
        ):
            raise OPEFHESecurityError("server artifact does not match its compilation receipt")
        if (
            len(client_payload) != self.receipt.client_specs_bytes
            or _sha256(client_payload) != self.receipt.client_specs_sha256
        ):
            raise OPEFHESecurityError(
                "client specifications do not match their compilation receipt"
            )
        markers = server_artifact_secret_markers(self.server_path)
        if markers or self.receipt.server_secret_key_markers:
            raise OPEFHESecurityError("server artifact contains secret-key-like member names")
        return server_payload, client_payload

    def real_roundtrip(
        self, batch: TrajectoryBatch, estimator: EstimatorName
    ) -> OPEConformanceResult:
        """Run serialized transport; only the client owns key generation and decryption."""
        fhe = _import_fhe()
        _, encoded, expected, integer_receipt = self._prepared(batch)
        started = time.perf_counter_ns()
        _, client_specs_payload = self._verify_artifacts()
        client_specs = fhe.ClientSpecs.deserialize(client_specs_payload)
        client = fhe.Client(client_specs)

        phase = time.perf_counter_ns()
        client.keys.generate()
        keygen_ns = time.perf_counter_ns() - phase
        evaluation_keys_payload = client.evaluation_keys.serialize()

        phase = time.perf_counter_ns()
        encrypted_inputs = client.encrypt(*_encrypted_arguments(self.spec, encoded))
        if not isinstance(encrypted_inputs, tuple) or len(encrypted_inputs) != 4:
            raise OPEFHEConformanceError("Concrete client did not encrypt four OPE tensors")
        request_payloads = tuple(value.serialize() for value in encrypted_inputs)
        framed_request = b"".join(
            len(payload).to_bytes(8, "big") + payload for payload in request_payloads
        )
        encrypt_ns = time.perf_counter_ns() - phase

        server = fhe.Server.load(str(self.server_path))
        phase = time.perf_counter_ns()
        server_inputs = tuple(fhe.Value.deserialize(payload) for payload in request_payloads)
        server_evaluation_keys = fhe.EvaluationKeys.deserialize(evaluation_keys_payload)
        encrypted_output = server.run(*server_inputs, evaluation_keys=server_evaluation_keys)
        if not isinstance(encrypted_output, tuple) or len(encrypted_output) != 3:
            raise OPEFHEConformanceError("Concrete server did not return three encrypted vectors")
        response_payloads = tuple(value.serialize() for value in encrypted_output)
        framed_response = b"".join(
            len(payload).to_bytes(8, "big") + payload for payload in response_payloads
        )
        server_evaluate_ns = time.perf_counter_ns() - phase

        phase = time.perf_counter_ns()
        client_outputs = tuple(fhe.Value.deserialize(payload) for payload in response_payloads)
        observed = decode_output(self.spec, client.decrypt(*client_outputs))
        decrypt_ns = time.perf_counter_ns() - phase
        if observed != expected:
            raise OPEFHEConformanceError("REAL Concrete execution disagrees with exact integer OPE")
        evidence = SanitizedOPECallEvidence(
            input_shape=(encoded.size,),
            output_shape=(3, self.spec.trajectories.horizon),
            encrypted_output_vectors=3,
            integers_per_output_vector=self.spec.trajectories.horizon,
            keygen_ns=keygen_ns,
            encrypt_ns=encrypt_ns,
            server_evaluate_ns=server_evaluate_ns,
            decrypt_ns=decrypt_ns,
            end_to_end_ns=time.perf_counter_ns() - started,
            evaluation_key_bytes=len(evaluation_keys_payload),
            request_bytes=sum(len(payload) for payload in request_payloads),
            response_bytes=sum(len(payload) for payload in response_payloads),
            request_sha256=_sha256(framed_request),
            response_sha256=_sha256(framed_response),
            output_matches_integer_reference=True,
            server_secret_key_marker_present=False,
        )
        return OPEConformanceResult(
            "REAL",
            observed,
            self.spec.client_statistics(observed, estimator),
            integer_receipt,
            evidence,
        )

    def execute(
        self, batch: TrajectoryBatch, estimator: EstimatorName, mode: ExecutionMode
    ) -> OPEConformanceResult:
        if mode == "INTEGER":
            return self.integer_reference(batch, estimator)
        if mode == "SIMULATION":
            return self.simulate(batch, estimator)
        if mode == "REAL":
            return self.real_roundtrip(batch, estimator)
        raise ValueError(f"unknown OPE execution mode {mode!r}")


def compile_ope_circuit(
    spec: OPECircuitSpec,
    artifact_dir: str | Path,
    *,
    p_error: float | None = None,
    global_p_error: float | None = None,
    security_level: int = 128,
    max_calibration_rows: int = MAX_CALIBRATION_ROWS,
) -> CompiledOPECircuit:
    """Compile and serialize the exact canary, failing closed on missing evidence."""
    _require_canary(spec)
    if security_level != 128:
        raise ValueError("exact OPE requires Concrete's 128-bit security level")
    requested_p, requested_global = _requested_errors(p_error, global_p_error)
    calibration, state_points, calibration_strategy = calibration_inputset(
        spec, max_rows=max_calibration_rows
    )
    fhe = _import_fhe()
    compiler = _kernel_compiler(spec)
    configuration = fhe.Configuration(
        enable_unsafe_features=False,
        use_insecure_key_cache=False,
        show_progress=False,
    )
    configured_security = getattr(configuration, "security_level", None)
    configured_security_bits = getattr(configured_security, "value", configured_security)
    if configured_security_bits != security_level:
        raise OPEFHESecurityError("Concrete configuration did not retain security_level=128")
    error_arguments: dict[str, float] = {}
    if requested_p is not None:
        error_arguments["p_error"] = requested_p
    if requested_global is not None:
        error_arguments["global_p_error"] = requested_global
    started = time.perf_counter_ns()
    compiler_inputset = [_encrypted_arguments(spec, row) for row in calibration]
    circuit = compiler.compile(compiler_inputset, configuration=configuration, **error_arguments)
    compile_ns = time.perf_counter_ns() - started

    server = getattr(circuit, "server", None)
    if server is None:
        raise OPEFHESecurityError("Concrete compilation did not produce a server object")
    compiled_p = _probability(getattr(server, "p_error", None), "server p_error")
    compiled_global = _probability(getattr(server, "global_p_error", None), "server global_p_error")

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    server_path = destination / "ope-server.zip"
    client_specs_path = destination / "ope-client-specs.bin"
    receipt_path = destination / "ope-receipt.json"
    server.save(str(server_path))
    serialized_specs = server.client_specs.serialize()
    specs_payload = (
        serialized_specs.encode() if isinstance(serialized_specs, str) else bytes(serialized_specs)
    )
    client_specs_path.write_bytes(specs_payload)
    server_payload = server_path.read_bytes()
    markers = server_artifact_secret_markers(server_path)
    if markers:
        raise OPEFHESecurityError("serialized server artifact contains secret-key-like members")

    graph = getattr(circuit, "graph", None)
    if graph is None or not callable(getattr(graph, "maximum_integer_bit_width", None)):
        raise OPEFHESecurityError("Concrete did not report maximum integer bit width")
    maximum_bit_width = int(graph.maximum_integer_bit_width())
    complexity_raw = getattr(circuit, "complexity", None)
    if complexity_raw is None or not np.isfinite(float(complexity_raw)):
        raise OPEFHESecurityError("Concrete did not report finite circuit complexity")
    mlir_raw = getattr(circuit, "mlir", None)
    if mlir_raw is None:
        raise OPEFHESecurityError("Concrete did not expose compiled MLIR evidence")
    try:
        concrete_version = importlib.metadata.version("concrete-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise OPEFHESecurityError("Concrete package version evidence is unavailable") from error

    receipt = OPEFHEReceipt(
        spec_digest=_spec_digest(spec),
        concrete_python_version=concrete_version,
        requested_p_error=requested_p,
        requested_global_p_error=requested_global,
        compiled_p_error=compiled_p,
        compiled_global_p_error=compiled_global,
        security_level=security_level,
        input_shape=(_input_layout(spec).width,),
        output_shape=(3, spec.trajectories.horizon),
        encrypted_output_vectors=3,
        integers_per_output_vector=spec.trajectories.horizon,
        calibration_strategy=calibration_strategy,
        quantized_state_domain_points=state_points,
        calibration_rows=int(calibration.shape[0]),
        calibration_sha256=_sha256(calibration.tobytes(order="C")),
        input_min=tuple(int(value) for value in np.min(calibration, axis=0)),
        input_max=tuple(int(value) for value in np.max(calibration, axis=0)),
        operations=spec.operation_counts(),
        maximum_integer_bit_width=maximum_bit_width,
        complexity=float(complexity_raw),
        compile_ns=compile_ns,
        mlir_sha256=_sha256(str(mlir_raw).encode()),
        server_artifact_bytes=len(server_payload),
        server_artifact_sha256=_sha256(server_payload),
        client_specs_bytes=len(specs_payload),
        client_specs_sha256=_sha256(specs_payload),
        server_secret_key_markers=(),
    )
    receipt_path.write_text(receipt.to_json() + "\n")
    return CompiledOPECircuit(spec, server_path, client_specs_path, receipt, circuit)


def load_ope_circuit(spec: OPECircuitSpec, artifact_dir: str | Path) -> CompiledOPECircuit:
    """Load serialized artifacts after receipt, domain, and digest validation."""
    _require_canary(spec)
    destination = Path(artifact_dir)
    receipt = OPEFHEReceipt.from_json((destination / "ope-receipt.json").read_text())
    if receipt.spec_digest != _spec_digest(spec):
        raise OPEFHESecurityError("OPE spec does not match the serialized circuit receipt")
    expected_shape = (_input_layout(spec).width,)
    if receipt.input_shape != expected_shape or receipt.output_shape != (
        3,
        spec.trajectories.horizon,
    ):
        raise OPEFHESecurityError("serialized OPE circuit shape differs from the frozen spec")
    compiled = CompiledOPECircuit(
        spec,
        destination / "ope-server.zip",
        destination / "ope-client-specs.bin",
        receipt,
    )
    compiled._verify_artifacts()
    return compiled
