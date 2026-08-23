"""Versioned, JSON-safe specifications shared by every execution backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class QuantizerSpec:
    """Symmetric per-feature observation quantizer with an explicit valid domain."""

    center: tuple[float, ...]
    step: tuple[float, ...]
    qmax: int

    def __post_init__(self) -> None:
        if not self.center or len(self.center) != len(self.step):
            raise ValueError("center and step must have the same non-zero length")
        if self.qmax < 1:
            raise ValueError("qmax must be positive")
        if any(not np.isfinite(value) or value <= 0 for value in self.step):
            raise ValueError("quantizer steps must be finite and positive")
        if any(not np.isfinite(value) for value in self.center):
            raise ValueError("quantizer centers must be finite")

    @property
    def n_features(self) -> int:
        return len(self.center)

    @property
    def input_bits(self) -> int:
        """Signed bit width sufficient for ``[-qmax, qmax]``."""
        return int(np.ceil(np.log2(2 * self.qmax + 1)))

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        center = np.asarray(self.center, dtype=np.float64)
        radius = np.asarray(self.step, dtype=np.float64) * self.qmax
        return center - radius, center + radius

    def quantize(self, observations: npt.ArrayLike, *, reject: bool = True) -> IntArray:
        values = np.asarray(observations, dtype=np.float64)
        if values.shape[-1:] != (self.n_features,):
            raise ValueError(
                f"expected final observation dimension {self.n_features}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must be finite")
        center = np.asarray(self.center, dtype=np.float64)
        step = np.asarray(self.step, dtype=np.float64)
        unbounded = np.rint((values - center) / step)
        if reject and np.any(np.abs(unbounded) > self.qmax):
            raise ValueError("observation is outside the compiled quantization domain")
        return np.clip(unbounded, -self.qmax, self.qmax).astype(np.int64)

    def dequantize(self, quantized: npt.ArrayLike) -> FloatArray:
        values = np.asarray(quantized, dtype=np.int64)
        if values.shape[-1:] != (self.n_features,):
            raise ValueError("quantized observation has the wrong final dimension")
        if np.any(np.abs(values) > self.qmax):
            raise ValueError("quantized observation exceeds qmax")
        return values * np.asarray(self.step) + np.asarray(self.center)

    @classmethod
    def calibrate(
        cls,
        observations: npt.ArrayLike,
        *,
        input_bits: int,
        padding: float = 0.1,
    ) -> QuantizerSpec:
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("calibration observations must have shape (samples, features)")
        if input_bits < 2 or input_bits > 15:
            raise ValueError("input_bits must be between 2 and 15")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        low = np.min(values, axis=0)
        high = np.max(values, axis=0)
        center = (low + high) / 2
        half_range = np.maximum((high - low) / 2, 1e-8) * (1 + padding)
        qmax = (1 << (input_bits - 1)) - 1
        return cls(tuple(center), tuple(half_range / qmax), qmax)


@dataclass(frozen=True)
class PolicySpec:
    """Immutable polynomial score circuit and its frozen quantization contract."""

    name: str
    env_id: str
    degree: int
    actions: int
    quantizer: QuantizerSpec
    float_coefficients: tuple[tuple[float, ...], ...]
    integer_coefficients: tuple[tuple[int, ...], ...]
    coefficient_scale: float
    schema_version: str = "unseen-loop/policy-v1"

    def __post_init__(self) -> None:
        if self.degree not in {1, 2}:
            raise ValueError("only degree-one and degree-two circuits are supported")
        if self.actions < 2:
            raise ValueError("a discrete policy needs at least two actions")
        expected_features = polynomial_feature_count(self.quantizer.n_features, self.degree)
        float_shape = (len(self.float_coefficients), len(self.float_coefficients[0]))
        integer_shape = (len(self.integer_coefficients), len(self.integer_coefficients[0]))
        expected_shape = (self.actions, expected_features)
        if float_shape != expected_shape or integer_shape != expected_shape:
            raise ValueError(
                f"coefficient matrices must have shape {expected_shape}; "
                f"got {float_shape} and {integer_shape}"
            )
        if self.coefficient_scale <= 0 or not np.isfinite(self.coefficient_scale):
            raise ValueError("coefficient_scale must be finite and positive")
        if not self.name or not self.env_id:
            raise ValueError("name and env_id cannot be empty")

    @property
    def feature_count(self) -> int:
        return polynomial_feature_count(self.quantizer.n_features, self.degree)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @property
    def float_array(self) -> FloatArray:
        return np.asarray(self.float_coefficients, dtype=np.float64)

    @property
    def integer_array(self) -> IntArray:
        return np.asarray(self.integer_coefficients, dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicySpec:
        quantizer_raw = raw["quantizer"]
        quantizer = QuantizerSpec(
            center=tuple(float(value) for value in quantizer_raw["center"]),
            step=tuple(float(value) for value in quantizer_raw["step"]),
            qmax=int(quantizer_raw["qmax"]),
        )
        return cls(
            name=str(raw["name"]),
            env_id=str(raw["env_id"]),
            degree=int(raw["degree"]),
            actions=int(raw["actions"]),
            quantizer=quantizer,
            float_coefficients=tuple(
                tuple(float(value) for value in row) for row in raw["float_coefficients"]
            ),
            integer_coefficients=tuple(
                tuple(int(value) for value in row) for row in raw["integer_coefficients"]
            ),
            coefficient_scale=float(raw["coefficient_scale"]),
            schema_version=str(raw.get("schema_version", "unseen-loop/policy-v1")),
        )

    @classmethod
    def from_json(cls, payload: str) -> PolicySpec:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("policy payload must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class CandidateMetrics:
    """Comparable metrics produced by a candidate evaluation."""

    policy_digest: str
    degree: int
    input_bits: int
    coefficient_bits: int
    return_mean: float
    return_std: float
    teacher_agreement: float
    certified_coverage: float
    constraint_cost: float
    estimated_bit_width: int
    encrypted_multiplications: int
    server_p50_ms: float | None = None
    server_p95_ms: float | None = None
    evaluation_key_bytes: int | None = None
    request_bytes: int | None = None
    response_bytes: int | None = None
    backend: str = "QUANTIZED CLEAR"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def polynomial_feature_count(n_features: int, degree: int) -> int:
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if degree == 1:
        return 1 + n_features
    if degree == 2:
        return 1 + n_features + n_features * (n_features + 1) // 2
    raise ValueError("degree must be one or two")
