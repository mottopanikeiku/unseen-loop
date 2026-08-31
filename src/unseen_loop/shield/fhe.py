"""Exact Concrete-Python circuit for the encrypted warehouse safety shield.

The evaluator receives one encrypted signed ``(6,)`` state and public circuit
constants.  It returns one encrypted signed ``(5, 2, 4)`` margin tensor in
``(action, horizon, family)`` order.  Selection is deliberately client-side.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from unseen_loop.shield.certificate import (
    ErrorBuffer,
    HorizonMargins,
    SafetyMargins,
    certify_candidate,
)
from unseen_loop.shield.shield import SelectionResult
from unseen_loop.shield.shield import select_action as select_core_action
from unseen_loop.shield.types import Action, DynamicsConfig, Obstacle, SafetyLimits, ShieldState

QMAX = 2
STATE_SHAPE = (6,)
MARGIN_SHAPE = (5, 2, 4)
OUTPUT_ORDER = ("action", "horizon", "family")
FAMILY_ORDER = ("obstacle", "speed", "tilt", "battery")
DOMAIN_POINTS = (2 * QMAX + 1) ** STATE_SHAPE[0]
SCHEMA_VERSION = "unseen-loop/shield-concrete-v1"


class ShieldFHEUnavailableError(RuntimeError):
    """Raised instead of substituting clear execution for an FHE mode."""


class ShieldFHEMode(StrEnum):
    """Execution labels with intentionally non-overlapping evidence semantics."""

    CLEAR = "CLEAR"
    SIMULATION = "FHE SIMULATED"
    REAL = "REAL FHE"


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("public circuit constants must be finite")
    return Fraction(str(value))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(encoded)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class StateQuantizer:
    """Public affine map from each signed domain coordinate to a physical state."""

    offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
    steps: tuple[float, ...] = (2.0, 2.0, 0.5, 0.5, 0.25, 0.25)
    qmax: int = QMAX

    def __post_init__(self) -> None:
        object.__setattr__(self, "offsets", tuple(float(value) for value in self.offsets))
        object.__setattr__(self, "steps", tuple(float(value) for value in self.steps))
        if len(self.offsets) != STATE_SHAPE[0] or len(self.steps) != STATE_SHAPE[0]:
            raise ValueError("state quantizer offsets and steps must each contain six values")
        if self.qmax != QMAX:
            raise ValueError("the shield conformance protocol fixes qmax=2")
        if any(not math.isfinite(value) for value in (*self.offsets, *self.steps)):
            raise ValueError("state quantizer constants must be finite")
        if any(value <= 0 for value in self.steps):
            raise ValueError("state quantizer steps must be positive")

    def quantize(self, state: ShieldState | Sequence[float]) -> npt.NDArray[np.int64]:
        values = (
            state.as_tuple() if isinstance(state, ShieldState) else tuple(float(x) for x in state)
        )
        if len(values) != STATE_SHAPE[0] or any(not math.isfinite(value) for value in values):
            raise ValueError("state must contain exactly six finite values")
        quantized = np.asarray(
            [
                round((value - offset) / step)
                for value, offset, step in zip(values, self.offsets, self.steps, strict=True)
            ],
            dtype=np.int64,
        )
        _validate_quantized(quantized)
        return quantized

    def dequantize(self, quantized: npt.ArrayLike) -> ShieldState:
        values = _validate_quantized(quantized)
        physical = [
            offset + step * int(value)
            for value, offset, step in zip(values, self.offsets, self.steps, strict=True)
        ]
        return ShieldState.from_array(physical)

    def canonical(self) -> dict[str, object]:
        return {
            "offsets": [_fraction_text(_fraction(value)) for value in self.offsets],
            "steps": [_fraction_text(_fraction(value)) for value in self.steps],
            "qmax": self.qmax,
        }


def _fhe_dynamics_default() -> DynamicsConfig:
    return DynamicsConfig(
        drag=1.0,
        accel=1.0,
        base_drain=0.0,
        motion_drain=0.25,
        tilt_decay=1.0,
        tilt_gain=0.25,
    )


def _fhe_limits_default() -> SafetyLimits:
    return SafetyLimits(
        obstacles=(Obstacle(0.0, 0.0, 1.0),),
        max_speed=2.0,
        max_abs_tilt=0.5,
        min_battery=0.0,
        x_bounds=(-5.0, 5.0),
        y_bounds=(-5.0, 5.0),
        vehicle_radius=0.0,
        obstacle_clearance=0.0,
    )


@dataclass(frozen=True, slots=True)
class ShieldIntegerSpec:
    """Frozen public clear specification compiled into the integer circuit."""

    dynamics: DynamicsConfig = field(default_factory=_fhe_dynamics_default)
    limits: SafetyLimits = field(default_factory=_fhe_limits_default)
    quantizer: StateQuantizer = field(default_factory=StateQuantizer)

    @property
    def spec_digest(self) -> str:
        return _canonical_digest(self.canonical())

    @property
    def digest(self) -> str:
        return self.spec_digest

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dynamics": {
                key: _fraction_text(_fraction(value))
                for key, value in self.dynamics.to_dict().items()
            },
            "limits": {
                "obstacles": [
                    {
                        key: _fraction_text(_fraction(value))
                        for key, value in obstacle.to_dict().items()
                    }
                    for obstacle in self.limits.obstacles
                ],
                "max_speed": _fraction_text(_fraction(self.limits.max_speed)),
                "max_abs_tilt": _fraction_text(_fraction(self.limits.max_abs_tilt)),
                "min_battery": _fraction_text(_fraction(self.limits.min_battery)),
                "x_bounds": [_fraction_text(_fraction(value)) for value in self.limits.x_bounds],
                "y_bounds": [_fraction_text(_fraction(value)) for value in self.limits.y_bounds],
                "vehicle_radius": _fraction_text(_fraction(self.limits.vehicle_radius)),
                "obstacle_clearance": _fraction_text(_fraction(self.limits.obstacle_clearance)),
            },
            "quantizer": self.quantizer.canonical(),
            "input_shape": list(STATE_SHAPE),
            "output_shape": list(MARGIN_SHAPE),
            "output_order": list(OUTPUT_ORDER),
            "family_order": list(FAMILY_ORDER),
        }


# Sparse exact polynomials over the six signed circuit inputs.
_Exponent = tuple[int, int, int, int, int, int]
_ZERO_EXPONENT: _Exponent = (0, 0, 0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class _Polynomial:
    terms: Mapping[_Exponent, Fraction]

    @classmethod
    def constant(cls, value: int | float | Fraction) -> _Polynomial:
        coefficient = _fraction(value)
        return cls({} if coefficient == 0 else {_ZERO_EXPONENT: coefficient})

    @classmethod
    def variable(cls, index: int) -> _Polynomial:
        exponent = cast(_Exponent, tuple(1 if position == index else 0 for position in range(6)))
        return cls({exponent: Fraction(1)})

    def __add__(self, other: _Polynomial | int | float | Fraction) -> _Polynomial:
        right = other if isinstance(other, _Polynomial) else _Polynomial.constant(other)
        result = dict(self.terms)
        for exponent, coefficient in right.terms.items():
            result[exponent] = result.get(exponent, Fraction(0)) + coefficient
            if result[exponent] == 0:
                del result[exponent]
        return _Polynomial(result)

    __radd__ = __add__

    def __neg__(self) -> _Polynomial:
        return _Polynomial({exponent: -coefficient for exponent, coefficient in self.terms.items()})

    def __sub__(self, other: _Polynomial | int | float | Fraction) -> _Polynomial:
        right = other if isinstance(other, _Polynomial) else _Polynomial.constant(other)
        return self + (-right)

    def __rsub__(self, other: int | float | Fraction) -> _Polynomial:
        return _Polynomial.constant(other) - self

    def __mul__(self, other: _Polynomial | int | float | Fraction) -> _Polynomial:
        right = other if isinstance(other, _Polynomial) else _Polynomial.constant(other)
        result: dict[_Exponent, Fraction] = {}
        for left_exp, left_coefficient in self.terms.items():
            for right_exp, right_coefficient in right.terms.items():
                exponent = cast(
                    _Exponent, tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
                )
                if sum(exponent) > 2:
                    raise ValueError(
                        "shield polynomial degree exceeds the exact degree-two protocol"
                    )
                result[exponent] = (
                    result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
                )
        return _Polynomial({key: value for key, value in result.items() if value})

    __rmul__ = __mul__


_MONOMIAL_EXPONENTS: tuple[_Exponent, ...] = (
    _ZERO_EXPONENT,
    *(
        cast(_Exponent, tuple(int(position == index) for position in range(6)))
        for index in range(6)
    ),
    *(
        cast(
            _Exponent,
            tuple(int(position == left) + int(position == right) for position in range(6)),
        )
        for left in range(6)
        for right in range(left, 6)
    ),
)
_MONOMIAL_INDEX = {exponent: index for index, exponent in enumerate(_MONOMIAL_EXPONENTS)}


@dataclass(frozen=True, slots=True)
class IntegerMarginProgram:
    """Exact integer coefficients for all four families and spatial submargins."""

    spatial_coefficients: npt.NDArray[np.int64]
    family_coefficients: npt.NDArray[np.int64]
    margin_scale: int

    @property
    def spatial_constraints(self) -> int:
        return int(self.spatial_coefficients.shape[2])


def _rollout_polynomials(
    spec: ShieldIntegerSpec,
) -> tuple[tuple[tuple[_Polynomial, ...], ...], ...]:
    state = tuple(
        _Polynomial.constant(offset) + _fraction(step) * _Polynomial.variable(index)
        for index, (offset, step) in enumerate(
            zip(spec.quantizer.offsets, spec.quantizer.steps, strict=True)
        )
    )
    dynamics = spec.dynamics
    action_rollouts: list[tuple[tuple[_Polynomial, ...], ...]] = []
    for action in Action:
        ax, ay = action.vector
        current = state
        horizons: list[tuple[_Polynomial, ...]] = []
        for _ in range(2):
            x, y, vx, vy, battery, tilt = current
            norm = ax * ax + ay * ay
            current = (
                x + vx + _fraction(0.5) * _fraction(dynamics.accel) * ax,
                y + vy + _fraction(0.5) * _fraction(dynamics.accel) * ay,
                _fraction(dynamics.drag) * vx + _fraction(dynamics.accel) * ax,
                _fraction(dynamics.drag) * vy + _fraction(dynamics.accel) * ay,
                battery - _fraction(dynamics.base_drain) - _fraction(dynamics.motion_drain) * norm,
                _fraction(dynamics.tilt_decay) * tilt
                + _fraction(dynamics.tilt_gain) * (ax * vy - ay * vx),
            )
            horizons.append(current)
        action_rollouts.append(tuple(horizons))
    return tuple(action_rollouts)


def integer_margin_program(spec: ShieldIntegerSpec) -> IntegerMarginProgram:
    """Expand the frozen clear dynamics exactly, without floating-point arithmetic."""

    limits = spec.limits
    clearance = _fraction(limits.vehicle_radius) + _fraction(limits.obstacle_clearance)
    spatial: list[list[list[_Polynomial]]] = []
    families: list[list[list[_Polynomial]]] = []
    for action_rollout in _rollout_polynomials(spec):
        action_spatial: list[list[_Polynomial]] = []
        action_families: list[list[_Polynomial]] = []
        for x, y, vx, vy, battery, tilt in action_rollout:
            candidates = [
                x - (_fraction(limits.x_bounds[0]) + clearance),
                (_fraction(limits.x_bounds[1]) - clearance) - x,
                y - (_fraction(limits.y_bounds[0]) + clearance),
                (_fraction(limits.y_bounds[1]) - clearance) - y,
            ]
            for obstacle in limits.obstacles:
                required = _fraction(obstacle.radius) + clearance
                dx = x - _fraction(obstacle.x)
                dy = y - _fraction(obstacle.y)
                candidates.append(dx * dx + dy * dy - required * required)
            action_spatial.append(candidates)
            action_families.append(
                [
                    _fraction(limits.max_speed) ** 2 - vx * vx - vy * vy,
                    _fraction(limits.max_abs_tilt) ** 2 - tilt * tilt,
                    battery - _fraction(limits.min_battery),
                ]
            )
        spatial.append(action_spatial)
        families.append(action_families)

    all_polynomials = [
        polynomial for action in spatial for horizon in action for polynomial in horizon
    ] + [polynomial for action in families for horizon in action for polynomial in horizon]
    margin_scale = math.lcm(
        *(
            coefficient.denominator
            for polynomial in all_polynomials
            for coefficient in polynomial.terms.values()
        ),
        1,
    )

    def coefficients(polynomial: _Polynomial) -> npt.NDArray[np.int64]:
        result = np.zeros(len(_MONOMIAL_EXPONENTS), dtype=np.int64)
        for exponent, coefficient in polynomial.terms.items():
            scaled = coefficient * margin_scale
            if scaled.denominator != 1:
                raise AssertionError("internal common denominator is incomplete")
            result[_MONOMIAL_INDEX[exponent]] = scaled.numerator
        return result

    spatial_array = np.asarray(
        [
            [[coefficients(polynomial) for polynomial in horizon] for horizon in action]
            for action in spatial
        ],
        dtype=np.int64,
    )
    family_array = np.asarray(
        [
            [[coefficients(polynomial) for polynomial in horizon] for horizon in action]
            for action in families
        ],
        dtype=np.int64,
    )
    return IntegerMarginProgram(spatial_array, family_array, margin_scale)


def _validate_quantized(quantized: npt.ArrayLike) -> npt.NDArray[np.int64]:
    raw = np.asarray(quantized)
    if raw.shape != STATE_SHAPE:
        raise ValueError("quantized shield state must have shape (6,)")
    if not np.issubdtype(raw.dtype, np.integer) and (
        not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw))
    ):
        raise ValueError("quantized shield state must contain integers")
    values = raw.astype(np.int64)
    if np.any(values < -QMAX) or np.any(values > QMAX):
        raise ValueError("quantized shield state is outside the complete qmax=2 domain")
    return values


def exhaustive_inputset() -> npt.NDArray[np.int64]:
    """Return the complete lexicographic ``5**6 == 15,625`` compilation domain."""

    return np.asarray(tuple(product(range(-QMAX, QMAX + 1), repeat=6)), dtype=np.int64)


def _integer_monomials(quantized: npt.ArrayLike) -> npt.NDArray[np.int64]:
    values = _validate_quantized(quantized)
    result = [1, *(int(value) for value in values)]
    result.extend(
        int(values[left]) * int(values[right]) for left in range(6) for right in range(left, 6)
    )
    return np.asarray(result, dtype=np.int64)


def clear_margin_tensor(
    spec: ShieldIntegerSpec,
    quantized: npt.ArrayLike,
    *,
    program: IntegerMarginProgram | None = None,
) -> npt.NDArray[np.int64]:
    """Evaluate the exact integer oracle returned by the compiled circuit."""

    integer_program = program or integer_margin_program(spec)
    monomials = _integer_monomials(quantized)
    spatial_values = integer_program.spatial_coefficients @ monomials
    spatial_minimum = np.min(spatial_values, axis=2)
    other_families = integer_program.family_coefficients @ monomials
    return np.asarray(
        np.concatenate((spatial_minimum[..., None], other_families), axis=2),
        dtype=np.int64,
    )


def _import_fhe() -> Any:
    try:
        return importlib.import_module("concrete.fhe")
    except ImportError as error:
        raise ShieldFHEUnavailableError(
            "Concrete-Python is unavailable; shield FHE modes do not fall back to clear execution"
        ) from error


def _compiler(spec: ShieldIntegerSpec, program: IntegerMarginProgram) -> Any:
    fhe = _import_fhe()
    spatial_coefficients = program.spatial_coefficients
    family_coefficients = program.family_coefficients

    def kernel(x: Any) -> Any:
        monomials = fhe.array(
            [
                1,
                *(x[index] for index in range(6)),
                *(x[left] * x[right] for left in range(6) for right in range(left, 6)),
            ]
        )
        output = []
        for action in range(5):
            for horizon in range(2):
                candidates = spatial_coefficients[action, horizon] @ monomials
                spatial_minimum = candidates[0]
                for index in range(1, program.spatial_constraints):
                    spatial_minimum = np.minimum(spatial_minimum, candidates[index])
                output.append(spatial_minimum)
                family_values = family_coefficients[action, horizon] @ monomials
                output.extend(family_values[index] for index in range(3))
        return fhe.array(output).reshape(MARGIN_SHAPE)

    return fhe.compiler({"x": "encrypted"})(kernel)


def server_artifact_secret_markers(path: str | Path) -> tuple[str, ...]:
    """Audit suspicious server archive member names; any hit is a hard failure."""

    markers: list[str] = []
    with zipfile.ZipFile(Path(path)) as archive:
        for name in archive.namelist():
            tokens = {
                token
                for token in name.lower()
                .replace("-", "_")
                .replace(".", "_")
                .replace("/", "_")
                .split("_")
                if token
            }
            if (
                "secret" in tokens
                or "private" in tokens
                or "secretkey" in tokens
                or "privatekey" in tokens
                or "clientkey" in tokens
                or {"secret", "key"} <= tokens
                or {"private", "key"} <= tokens
                or {"client", "key"} <= tokens
            ):
                markers.append(name)
    return tuple(sorted(markers))


@dataclass(frozen=True, slots=True)
class ShieldCircuitReceipt:
    spec_digest: str
    output_order: tuple[str, str, str]
    family_order: tuple[str, str, str, str]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    qmax: int
    domain_points: int
    domain_sha256: str
    range_sha256: str
    input_min: tuple[int, ...]
    input_max: tuple[int, ...]
    output_min: tuple[int, ...]
    output_max: tuple[int, ...]
    margin_scale: int
    requested_p_error: float | None
    requested_global_p_error: float | None
    compiled_p_error: float
    compiled_global_p_error: float
    security_level: int
    concrete_python_version: str
    maximum_integer_bit_width: int
    complexity: float
    operation_counts: Mapping[str, int]
    compile_ns: int
    mlir_sha256: str
    server_artifact_bytes: int
    server_artifact_sha256: str
    client_specs_bytes: int
    client_specs_sha256: str
    server_secret_key_markers: tuple[str, ...]
    backend: str = "Concrete-Python TFHE"
    mode: str = "COMPILED"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


@dataclass(frozen=True, slots=True)
class ShieldCallReceipt:
    """Sanitized systems metadata; no input state or decrypted margins are retained."""

    mode: ShieldFHEMode
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    keygen_ns: int
    encrypt_ns: int
    server_evaluate_ns: int
    decrypt_ns: int
    end_to_end_ns: int
    evaluation_key_bytes: int
    request_bytes: int
    response_bytes: int
    evaluation_key_sha256: str
    request_sha256: str
    response_sha256: str
    output_matches_clear: bool
    server_secret_key_marker_present: bool
    schema_version: str = "unseen-loop/shield-call-v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class SimulationConformance:
    mode: ShieldFHEMode
    domain_points: int
    matches: int
    mismatches: int
    domain_sha256: str
    clear_outputs_sha256: str
    simulated_outputs_sha256: str
    output_shape: tuple[int, ...] = MARGIN_SHAPE

    @property
    def exact(self) -> bool:
        return self.mismatches == 0 and self.matches == self.domain_points


@dataclass(frozen=True, slots=True)
class RealCanaryResult:
    selection: SelectionResult
    call: ShieldCallReceipt


class ShieldFHEClient:
    """Key-owning client for quantization, encryption, decryption, and selection."""

    def __init__(self, client_specs: bytes, spec: ShieldIntegerSpec) -> None:
        fhe = _import_fhe()
        self.spec = spec
        self._program = integer_margin_program(spec)
        self._client = fhe.Client(fhe.ClientSpecs.deserialize(client_specs))
        self._keys_generated = False

    @classmethod
    def from_path(cls, path: str | Path, spec: ShieldIntegerSpec) -> ShieldFHEClient:
        return cls(Path(path).read_bytes(), spec)

    def generate_keys(self) -> tuple[int, bytes]:
        started = time.perf_counter_ns()
        self._client.keys.generate()
        elapsed = time.perf_counter_ns() - started
        self._keys_generated = True
        return elapsed, bytes(self._client.evaluation_keys.serialize())

    def quantize(self, state: ShieldState | Sequence[float]) -> npt.NDArray[np.int64]:
        return self.spec.quantizer.quantize(state)

    def encrypt(self, quantized: npt.ArrayLike) -> bytes:
        if not self._keys_generated:
            raise RuntimeError("client keys must be generated before encryption")
        return bytes(self._client.encrypt(_validate_quantized(quantized)).serialize())

    def decrypt_margin_tensor(self, serialized_response: bytes) -> npt.NDArray[np.int64]:
        """Decrypt and validate the concrete ``(5, 2, 4)`` signed tensor."""

        fhe = _import_fhe()
        value = fhe.Value.deserialize(serialized_response)
        margins = np.asarray(self._client.decrypt(value), dtype=np.int64)
        if margins.shape != MARGIN_SHAPE:
            raise ValueError("decrypted shield response does not have shape (5, 2, 4)")
        return margins

    def select_action(
        self,
        margin_tensor: npt.ArrayLike,
        requested_action: Action | int,
        *,
        error_buffer: ErrorBuffer | None = None,
        emergency_action: Action = Action.BRAKE,
    ) -> SelectionResult:
        """Use the shared shield certificate and stable core selection implementation."""

        margins = np.asarray(margin_tensor)
        if margins.shape != MARGIN_SHAPE or not np.issubdtype(margins.dtype, np.integer):
            raise ValueError("margin_tensor must be an integer tensor with shape (5, 2, 4)")
        scale = self._program.margin_scale
        certificates = []
        for action in Action:
            horizons = tuple(
                HorizonMargins(
                    horizon=horizon + 1,
                    margins=SafetyMargins(
                        *(float(value) / scale for value in margins[int(action), horizon])
                    ),
                )
                for horizon in range(2)
            )
            certificates.append(
                certify_candidate(action, horizons, error_buffer=error_buffer or ErrorBuffer())
            )
        return select_core_action(
            certificates,
            Action(requested_action),
            emergency_action=emergency_action,
            enabled=True,
        )


class ShieldFHEServer:
    """Evaluation-only server wrapper; it has no client or secret-key API."""

    def __init__(self, server_path: str | Path) -> None:
        fhe = _import_fhe()
        self.path = Path(server_path)
        markers = server_artifact_secret_markers(self.path)
        if markers:
            raise RuntimeError(f"server artifact failed secret-marker audit: {markers}")
        self._server = fhe.Server.load(str(self.path))

    def evaluate(self, serialized_request: bytes, serialized_evaluation_keys: bytes) -> bytes:
        fhe = _import_fhe()
        request = fhe.Value.deserialize(serialized_request)
        evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
        response = self._server.run(request, evaluation_keys=evaluation_keys)
        return bytes(response.serialize())


@dataclass(slots=True)
class CompiledShield:
    spec: ShieldIntegerSpec
    circuit: Any
    server_path: Path
    client_specs_path: Path
    receipt: ShieldCircuitReceipt
    program: IntegerMarginProgram

    def clear(self, quantized: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return clear_margin_tensor(self.spec, quantized, program=self.program)

    def simulate(self, quantized: npt.ArrayLike) -> npt.NDArray[np.int64]:
        values = _validate_quantized(quantized)
        result = np.asarray(self.circuit.simulate(values), dtype=np.int64)
        if result.shape != MARGIN_SHAPE:
            raise ValueError("simulated shield response does not have shape (5, 2, 4)")
        return result

    def client(self) -> ShieldFHEClient:
        return ShieldFHEClient.from_path(self.client_specs_path, self.spec)

    def server(self) -> ShieldFHEServer:
        return ShieldFHEServer(self.server_path)

    def evaluate(
        self, quantized: npt.ArrayLike, mode: ShieldFHEMode | str
    ) -> npt.NDArray[np.int64]:
        selected_mode = ShieldFHEMode(mode)
        if selected_mode is ShieldFHEMode.CLEAR:
            return self.clear(quantized)
        if selected_mode is ShieldFHEMode.SIMULATION:
            return self.simulate(quantized)
        raise ValueError(
            "REAL FHE requires real_canary so encryption and key separation are explicit"
        )


def _operation_counts(circuit: Any) -> dict[str, int]:
    names = (
        "clear_addition_count",
        "clear_multiplication_count",
        "encrypted_addition_count",
        "encrypted_negation_count",
        "key_switch_count",
        "programmable_bootstrap_count",
    )
    return {name: int(getattr(circuit, name)) for name in names}


def compile_shield(
    spec: ShieldIntegerSpec,
    artifact_dir: str | Path,
    *,
    p_error: float | None = None,
    global_p_error: float | None = 1e-6,
    security_level: int = 128,
) -> CompiledShield:
    """Compile and serialize the exact domain; missing security evidence fails closed."""

    if security_level != 128:
        raise ValueError("the shield release protocol requires 128-bit Concrete security")
    if p_error is not None and not 0 < p_error < 1:
        raise ValueError("p_error must lie in (0, 1)")
    if global_p_error is not None and not 0 < global_p_error <= 1e-3:
        raise ValueError("global_p_error must lie in (0, 1e-3]")
    if p_error is not None and global_p_error is not None:
        raise ValueError("Concrete accepts either p_error or global_p_error, not both")
    if p_error is None and global_p_error is None:
        raise ValueError("an explicit p_error or global_p_error is required")

    fhe = _import_fhe()
    program = integer_margin_program(spec)
    inputset = exhaustive_inputset()
    compiler = _compiler(spec, program)
    configuration = fhe.Configuration(
        enable_unsafe_features=False,
        use_insecure_key_cache=False,
        show_progress=False,
    )
    compile_kwargs = (
        {"p_error": p_error} if p_error is not None else {"global_p_error": global_p_error}
    )
    started = time.perf_counter_ns()
    circuit = compiler.compile(inputset, configuration=configuration, **compile_kwargs)
    compile_ns = time.perf_counter_ns() - started

    compiled_p_error = float(circuit.server.p_error)
    compiled_global_p_error = float(circuit.server.global_p_error)
    if not 0 <= compiled_p_error < 1 or not 0 <= compiled_global_p_error < 1:
        raise RuntimeError("Concrete server did not expose valid compiled error probabilities")
    configured_security = getattr(configuration, "security_level", None)
    configured_security_bits = getattr(configured_security, "value", configured_security)
    if configured_security_bits != security_level:
        raise RuntimeError("Concrete configuration did not retain 128-bit security evidence")

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    server_path = destination / "shield-server.zip"
    client_specs_path = destination / "shield-client-specs.bin"
    circuit.server.save(str(server_path))
    serialized_specs = circuit.server.client_specs.serialize()
    client_specs = (
        serialized_specs.encode() if isinstance(serialized_specs, str) else bytes(serialized_specs)
    )
    client_specs_path.write_bytes(client_specs)
    markers = server_artifact_secret_markers(server_path)
    if markers:
        server_path.unlink(missing_ok=True)
        client_specs_path.unlink(missing_ok=True)
        raise RuntimeError(f"server artifact failed secret-marker audit: {markers}")

    outputs = np.asarray(
        [clear_margin_tensor(spec, row, program=program) for row in inputset], dtype=np.int64
    )
    output_min = tuple(int(value) for value in np.min(outputs, axis=0).reshape(-1))
    output_max = tuple(int(value) for value in np.max(outputs, axis=0).reshape(-1))
    range_payload = {
        "input_min": [-QMAX] * 6,
        "input_max": [QMAX] * 6,
        "output_min": output_min,
        "output_max": output_max,
        "output_shape": MARGIN_SHAPE,
        "margin_scale": program.margin_scale,
    }
    server_bytes = server_path.read_bytes()
    mlir = str(circuit.mlir)
    receipt = ShieldCircuitReceipt(
        spec_digest=spec.spec_digest,
        output_order=OUTPUT_ORDER,
        family_order=FAMILY_ORDER,
        input_shape=STATE_SHAPE,
        output_shape=MARGIN_SHAPE,
        qmax=QMAX,
        domain_points=DOMAIN_POINTS,
        domain_sha256=_sha256(inputset.tobytes(order="C")),
        range_sha256=_canonical_digest(range_payload),
        input_min=(-QMAX,) * 6,
        input_max=(QMAX,) * 6,
        output_min=output_min,
        output_max=output_max,
        margin_scale=program.margin_scale,
        requested_p_error=p_error,
        requested_global_p_error=global_p_error,
        compiled_p_error=compiled_p_error,
        compiled_global_p_error=compiled_global_p_error,
        security_level=security_level,
        concrete_python_version=importlib.metadata.version("concrete-python"),
        maximum_integer_bit_width=int(circuit.graph.maximum_integer_bit_width()),
        complexity=float(circuit.complexity),
        operation_counts=_operation_counts(circuit),
        compile_ns=compile_ns,
        mlir_sha256=_sha256(mlir.encode()),
        server_artifact_bytes=len(server_bytes),
        server_artifact_sha256=_sha256(server_bytes),
        client_specs_bytes=len(client_specs),
        client_specs_sha256=_sha256(client_specs),
        server_secret_key_markers=markers,
    )
    (destination / "shield-receipt.json").write_text(receipt.to_json() + "\n")
    return CompiledShield(spec, circuit, server_path, client_specs_path, receipt, program)


def exhaustive_simulation_conformance(compiled: CompiledShield) -> SimulationConformance:
    """Check all 15,625 points against Concrete simulation, with no sampling."""

    inputset = exhaustive_inputset()
    clear_hash = hashlib.sha256()
    simulation_hash = hashlib.sha256()
    matches = 0
    for row in inputset:
        clear = compiled.clear(row)
        simulated = compiled.simulate(row)
        clear_hash.update(clear.tobytes(order="C"))
        simulation_hash.update(simulated.tobytes(order="C"))
        matches += int(np.array_equal(clear, simulated))
    return SimulationConformance(
        mode=ShieldFHEMode.SIMULATION,
        domain_points=DOMAIN_POINTS,
        matches=matches,
        mismatches=DOMAIN_POINTS - matches,
        domain_sha256=_sha256(inputset.tobytes(order="C")),
        clear_outputs_sha256=clear_hash.hexdigest(),
        simulated_outputs_sha256=simulation_hash.hexdigest(),
    )


def real_fhe_canary(
    compiled: CompiledShield,
    state: ShieldState | Sequence[float],
    requested_action: Action | int,
    *,
    error_buffer: ErrorBuffer | None = None,
) -> RealCanaryResult:
    """Exercise serialized keygen/encrypt/server/decrypt/select without private logs."""

    started = time.perf_counter_ns()
    client = compiled.client()
    server = compiled.server()
    quantized = client.quantize(state)
    keygen_ns, evaluation_keys = client.generate_keys()
    phase = time.perf_counter_ns()
    request = client.encrypt(quantized)
    encrypt_ns = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    response = server.evaluate(request, evaluation_keys)
    server_evaluate_ns = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    margins = client.decrypt_margin_tensor(response)
    decrypt_ns = time.perf_counter_ns() - phase
    selection = client.select_action(margins, requested_action, error_buffer=error_buffer)
    clear = compiled.clear(quantized)
    call = ShieldCallReceipt(
        mode=ShieldFHEMode.REAL,
        input_shape=STATE_SHAPE,
        output_shape=MARGIN_SHAPE,
        keygen_ns=keygen_ns,
        encrypt_ns=encrypt_ns,
        server_evaluate_ns=server_evaluate_ns,
        decrypt_ns=decrypt_ns,
        end_to_end_ns=time.perf_counter_ns() - started,
        evaluation_key_bytes=len(evaluation_keys),
        request_bytes=len(request),
        response_bytes=len(response),
        evaluation_key_sha256=_sha256(evaluation_keys),
        request_sha256=_sha256(request),
        response_sha256=_sha256(response),
        output_matches_clear=bool(np.array_equal(margins, clear)),
        server_secret_key_marker_present=bool(compiled.receipt.server_secret_key_markers),
    )
    return RealCanaryResult(selection=selection, call=call)
