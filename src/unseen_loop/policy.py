"""FHE-friendly polynomial policy fitting and exact integer execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from unseen_loop.specs import FloatArray, IntArray, PolicySpec, QuantizerSpec


@dataclass(frozen=True)
class FitDiagnostics:
    weighted_mse: float
    coefficient_scale: float
    saturated_coefficients: int
    samples: int


def polynomial_features(quantized: npt.ArrayLike, degree: int) -> IntArray:
    """Map integer observations to ``[1, x, x_i*x_j]`` without hidden casts."""
    values = np.asarray(quantized, dtype=np.int64)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("quantized observations must be one- or two-dimensional")

    columns: list[IntArray] = [np.ones((values.shape[0], 1), dtype=np.int64), values]
    if degree == 2:
        interactions = [
            (values[:, left] * values[:, right])[:, None]
            for left in range(values.shape[1])
            for right in range(left, values.shape[1])
        ]
        columns.extend(interactions)
    elif degree != 1:
        raise ValueError("degree must be one or two")
    result = np.concatenate(columns, axis=1)
    return result[0] if squeeze else result


class PolynomialPolicy:
    """One source of truth for float, integer-clear, simulated, and FHE backends."""

    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec

    def quantize(self, observations: npt.ArrayLike, *, reject: bool = True) -> IntArray:
        return self.spec.quantizer.quantize(observations, reject=reject)

    def features(self, quantized: npt.ArrayLike) -> IntArray:
        return polynomial_features(quantized, self.spec.degree)

    def float_scores_from_quantized(self, quantized: npt.ArrayLike) -> FloatArray:
        features = self.features(quantized).astype(np.float64)
        return np.asarray(features @ self.spec.float_array.T, dtype=np.float64)

    def integer_scores_from_quantized(self, quantized: npt.ArrayLike) -> IntArray:
        features = self.features(quantized)
        return np.asarray(features @ self.spec.integer_array.T, dtype=np.int64)

    def dequantized_integer_scores(self, quantized: npt.ArrayLike) -> FloatArray:
        return self.integer_scores_from_quantized(quantized) / self.spec.coefficient_scale

    def scores(
        self, observations: npt.ArrayLike, *, integer: bool = False
    ) -> FloatArray | IntArray:
        quantized = self.quantize(observations)
        if integer:
            return self.integer_scores_from_quantized(quantized)
        return self.float_scores_from_quantized(quantized)

    def actions_from_quantized(self, quantized: npt.ArrayLike, *, integer: bool = True) -> IntArray:
        scores = (
            self.integer_scores_from_quantized(quantized)
            if integer
            else self.float_scores_from_quantized(quantized)
        )
        return np.asarray(np.argmax(scores, axis=-1), dtype=np.int64)

    def actions(self, observations: npt.ArrayLike, *, integer: bool = True) -> IntArray:
        return self.actions_from_quantized(self.quantize(observations), integer=integer)

    def coefficient_error_bound(self, quantized: npt.ArrayLike) -> FloatArray:
        """Analytical per-action bound from coefficient rounding at fixed integer input."""
        features = np.abs(self.features(quantized).astype(np.float64))
        coefficient_error = np.abs(
            self.spec.float_array - self.spec.integer_array / self.spec.coefficient_scale
        )
        return np.asarray(features @ coefficient_error.T, dtype=np.float64)

    def global_coefficient_error_bound(self) -> FloatArray:
        """Conservative bound over the complete compiled quantizer box."""
        qmax = self.spec.quantizer.qmax
        maxima = np.full(self.spec.feature_count, qmax, dtype=np.float64)
        maxima[0] = 1
        if self.spec.degree == 2:
            maxima[1 + self.spec.quantizer.n_features :] = qmax * qmax
        coefficient_error = np.abs(
            self.spec.float_array - self.spec.integer_array / self.spec.coefficient_scale
        )
        return np.asarray(maxima @ coefficient_error.T, dtype=np.float64)

    def integer_output_bound(self) -> IntArray:
        """Absolute score bound over the quantizer box, used for overflow receipts."""
        qmax = self.spec.quantizer.qmax
        maxima = np.full(self.spec.feature_count, qmax, dtype=np.int64)
        maxima[0] = 1
        if self.spec.degree == 2:
            maxima[1 + self.spec.quantizer.n_features :] = qmax * qmax
        return np.asarray(maxima @ np.abs(self.spec.integer_array).T, dtype=np.int64)

    @property
    def estimated_output_bits(self) -> int:
        maximum = int(np.max(self.integer_output_bound()))
        return max(2, int(np.ceil(np.log2(2 * maximum + 1))))

    @property
    def encrypted_multiplications(self) -> int:
        if self.spec.degree == 1:
            return 0
        n_features = self.spec.quantizer.n_features
        return n_features * (n_features + 1) // 2


def fit_polynomial_policy(
    observations: npt.ArrayLike,
    teacher_scores: npt.ArrayLike,
    *,
    env_id: str,
    name: str,
    degree: int,
    input_bits: int,
    coefficient_bits: int,
    ridge: float = 1e-3,
    sample_weights: npt.ArrayLike | None = None,
    quantizer: QuantizerSpec | None = None,
    calibration_padding: float = 0.15,
) -> tuple[PolynomialPolicy, FitDiagnostics]:
    """Fit a weighted ridge student, then freeze a signed integer circuit."""
    inputs = np.asarray(observations, dtype=np.float64)
    targets = np.asarray(teacher_scores, dtype=np.float64)
    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("observations and scores must be aligned two-dimensional arrays")
    if targets.shape[1] < 2:
        raise ValueError("teacher_scores must contain at least two actions")
    if coefficient_bits < 2 or coefficient_bits > 30:
        raise ValueError("coefficient_bits must be between 2 and 30")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    frozen_quantizer = quantizer or QuantizerSpec.calibrate(
        inputs, input_bits=input_bits, padding=calibration_padding
    )
    quantized = frozen_quantizer.quantize(inputs, reject=False)
    features = polynomial_features(quantized, degree).astype(np.float64)

    if sample_weights is None:
        weights = np.ones(inputs.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (inputs.shape[0],):
            raise ValueError("sample_weights must contain one value per observation")
        if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample_weights must be finite and positive")
    root_weights = np.sqrt(weights / np.mean(weights))[:, None]
    weighted_features = features * root_weights
    weighted_targets = targets * root_weights

    penalty = ridge * np.eye(features.shape[1], dtype=np.float64)
    penalty[0, 0] = 0
    gram = weighted_features.T @ weighted_features + penalty
    rhs = weighted_features.T @ weighted_targets
    coefficients = np.linalg.solve(gram, rhs).T

    coefficient_qmax = (1 << (coefficient_bits - 1)) - 1
    largest = float(np.max(np.abs(coefficients)))
    epsilon = float(np.finfo(np.float64).eps)
    coefficient_scale = coefficient_qmax / max(largest, epsilon)
    quantized_coefficients = np.rint(coefficients * coefficient_scale)
    quantized_coefficients = np.clip(
        quantized_coefficients, -coefficient_qmax, coefficient_qmax
    ).astype(np.int64)

    spec = PolicySpec(
        name=name,
        env_id=env_id,
        degree=degree,
        actions=targets.shape[1],
        quantizer=frozen_quantizer,
        float_coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
        integer_coefficients=tuple(
            tuple(int(value) for value in row) for row in quantized_coefficients
        ),
        coefficient_scale=coefficient_scale,
    )
    policy = PolynomialPolicy(spec)
    residual = policy.float_scores_from_quantized(quantized) - targets
    weighted_mse = float(np.average(np.mean(residual * residual, axis=1), weights=weights))
    saturated = int(np.count_nonzero(np.abs(quantized_coefficients) == coefficient_qmax))
    return policy, FitDiagnostics(
        weighted_mse=weighted_mse,
        coefficient_scale=coefficient_scale,
        saturated_coefficients=saturated,
        samples=inputs.shape[0],
    )
