"""Backend-independent integer circuit specification for horizon-aware OPE.

The client quantizes states, rewards, and reciprocal logging propensities.  The
server-held polynomial is evaluated only after that boundary.  The arithmetic
circuit contains no division: it returns three encrypted integer vectors and
the client alone decodes and divides them.  This module does not import or make
claims about Concrete or any other FHE runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Literal

import numpy as np

from unseen_loop.ope.estimators import cumulative_importance_weights
from unseen_loop.ope.types import (
    FailureRow,
    OPEValidationError,
    PolynomialPolicySpec,
    SufficientStatistics,
    TrajectoryBatch,
    TrajectorySpec,
)

EstimatorName = Literal["clipped_pdis", "clipped_wpdis"]


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round an exact rational half away from zero without a float conversion."""
    if denominator <= 0:
        raise ValueError("fixed-point denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    if 2 * remainder >= denominator:
        quotient += 1
    return sign * quotient


def _quantize(value: float, scale: int) -> int:
    if not isfinite(value):
        raise ValueError("fixed-point values must be finite")
    numerator, denominator = value.as_integer_ratio()
    return _round_ratio(numerator * scale, denominator)


def _quantize_reciprocal(value: float, scale: int) -> int:
    if not isfinite(value) or value <= 0:
        raise ValueError("reciprocal input must be finite and positive")
    numerator, denominator = value.as_integer_ratio()
    return _round_ratio(scale * denominator, numerator)


def _signed_bits(bound: int) -> int:
    if bound < 0:
        raise ValueError("an absolute bound cannot be negative")
    return max(2, (2 * bound + 1).bit_length())


@dataclass(frozen=True)
class FixedPointScales:
    """Positive integer scales at the client/server trust boundary."""

    state: int = 1 << 12
    coefficient: int = 1 << 20
    reciprocal: int = 1 << 16
    reward: int = 1 << 16
    discount: int = 1 << 16

    def __post_init__(self) -> None:
        if min(self.state, self.coefficient, self.reciprocal, self.reward, self.discount) < 2:
            raise ValueError("all fixed-point scales must be integers of at least two")
        if any(not isinstance(value, int) for value in vars(self).values()):
            raise TypeError("fixed-point scales must be integers")


@dataclass(frozen=True)
class QuantizedTrajectoryTensors:
    """Client-produced fixed tensors; reciprocal creation is not server work."""

    states: tuple[tuple[tuple[int, ...], ...], ...]
    action_masks: tuple[tuple[tuple[int, ...], ...], ...]
    rewards: tuple[tuple[int, ...], ...]
    behavior_reciprocals: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class IntegerSufficientStatistics:
    """The exact encrypted payload shape: three vectors, each of length horizon."""

    numerators: tuple[int, ...]
    denominators: tuple[int, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.numerators or not (
            len(self.numerators) == len(self.denominators) == len(self.counts)
        ):
            raise ValueError("integer statistics vectors must have the same non-zero length")
        if any(value < 0 for value in self.denominators) or any(value < 0 for value in self.counts):
            raise ValueError("denominators and counts must be non-negative")


@dataclass(frozen=True)
class CircuitOperationCounts:
    """Backend-neutral DAG counts.

    ``multiplicative_depth`` covers the arithmetic path and excludes the
    backend-specific implementation of the separately counted exact
    comparisons/multiplexers.
    """

    encrypted_multiplications: int
    public_multiplications: int
    additions: int
    comparisons: int
    multiplexers: int
    multiplicative_depth: int
    encrypted_output_integers: int


@dataclass(frozen=True)
class QuantizationErrorBound:
    """Absolute error bounds against the clear polynomial-policy estimators."""

    per_horizon_numerator: tuple[float, ...]
    per_horizon_denominator: tuple[float, ...]
    clipped_pdis: float
    self_normalized_wpdis: float | None
    max_logged_probability_error: float
    max_reciprocal_error: float
    max_reward_error: float
    max_clip_error: float
    discount_error: float


@dataclass(frozen=True)
class CircuitReceipt:
    """Public scale, overflow, cost, and approximation evidence for one run."""

    numerator_scales: tuple[int, ...]
    denominator_scales: tuple[int, ...]
    numerator_abs_bounds: tuple[int, ...]
    denominator_bounds: tuple[int, ...]
    raw_weight_bounds: tuple[int, ...]
    numerator_signed_bits: tuple[int, ...]
    denominator_unsigned_bits: tuple[int, ...]
    raw_weight_unsigned_bits: tuple[int, ...]
    operations: CircuitOperationCounts
    error: QuantizationErrorBound
    invalid_domains: tuple[str, ...]
    schema_version: str = "unseen-loop/ope-circuit-receipt-v1"
    trust_scope: str = (
        "Backend-independent integer reference; client quantization, reciprocal creation, "
        "decryption, and division are outside the server circuit. Not a REAL FHE result."
    )

    @property
    def digest(self) -> str:
        """SHA-256 closure over every scale, bound, count, error, and trust label."""
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class OPECircuitSpec:
    """Frozen integer semantics shared by a future encrypted backend and clear reference."""

    trajectories: TrajectorySpec
    target_policy: PolynomialPolicySpec
    gamma: float = 1.0
    weight_clip: float = 20.0
    minimum_behavior_propensity: float = 1e-3
    scales: FixedPointScales = field(default_factory=FixedPointScales)

    def __post_init__(self) -> None:
        if self.target_policy.state_dim != self.trajectories.state_dim:
            raise ValueError("trajectory and target-policy state dimensions differ")
        if self.target_policy.action_count != self.trajectories.action_count:
            raise ValueError("trajectory and target-policy action counts differ")
        if not self.trajectories.state_min:
            raise ValueError("closed state_min/state_max are required for an overflow receipt")
        if self.trajectories.reward_min is None:
            raise ValueError("closed reward_min/reward_max are required for an overflow receipt")
        if not isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not isfinite(self.weight_clip) or self.weight_clip <= 0:
            raise ValueError("weight_clip must be finite and positive")
        if (
            not isfinite(self.minimum_behavior_propensity)
            or not 0 < self.minimum_behavior_propensity <= 1
        ):
            raise ValueError("minimum_behavior_propensity must be in (0, 1]")
        self._validate_policy_domain()

    @property
    def coefficient_integers(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(_quantize(value, self.scales.coefficient) for value in row)
            for row in self.target_policy.coefficients
        )

    @property
    def target_probability_scale(self) -> int:
        return int(self.scales.coefficient * self.scales.state**self.target_policy.degree)

    @property
    def ratio_scale(self) -> int:
        return self.target_probability_scale * self.scales.reciprocal

    @property
    def numerator_scales(self) -> tuple[int, ...]:
        ratio = self.ratio_scale
        return tuple(
            ratio ** (step + 1) * self.scales.reward * self.scales.discount**step
            for step in range(self.trajectories.horizon)
        )

    @property
    def denominator_scales(self) -> tuple[int, ...]:
        return tuple(self.ratio_scale ** (step + 1) for step in range(self.trajectories.horizon))

    @property
    def invalid_domains(self) -> tuple[str, ...]:
        return (
            "batch shape/spec differs from the frozen TrajectorySpec",
            "state or reward lies outside its closed compiled domain",
            "logged action is outside [0, action_count)",
            "behavior propensity is non-finite, below minimum_behavior_propensity, or above one",
            "the clear or quantized target polynomial leaves [0, 1] on a supplied state",
            "the clear cumulative importance product overflows floating-point reference arithmetic",
            "a WPDIS horizon has zero aggregate target weight",
        )

    def _feature_intervals(self, *, quantized: bool) -> tuple[tuple[int | float, int | float], ...]:
        lows: tuple[int | float, ...]
        highs: tuple[int | float, ...]
        intervals: list[tuple[int | float, int | float]]
        if quantized:
            lows = tuple(
                _quantize(value, self.scales.state) for value in self.trajectories.state_min
            )
            highs = tuple(
                _quantize(value, self.scales.state) for value in self.trajectories.state_max
            )
            one: int | float = self.scales.state**self.target_policy.degree
            linear_factor: int | float = self.scales.state if self.target_policy.degree == 2 else 1
            intervals = [(one, one)]
            intervals.extend(
                (low * linear_factor, high * linear_factor)
                for low, high in zip(lows, highs, strict=True)
            )
        else:
            lows = self.trajectories.state_min
            highs = self.trajectories.state_max
            intervals = [(1.0, 1.0), *zip(lows, highs, strict=True)]
        if self.target_policy.degree == 2:
            for left in range(self.trajectories.state_dim):
                for right in range(left, self.trajectories.state_dim):
                    candidates = (
                        lows[left] * lows[right],
                        lows[left] * highs[right],
                        highs[left] * lows[right],
                        highs[left] * highs[right],
                    )
                    intervals.append((min(candidates), max(candidates)))
        return tuple(intervals)

    def _validate_policy_domain(self) -> None:
        tolerance = self.target_policy.probability_tolerance
        coefficient_sums = np.sum(
            np.asarray(self.target_policy.coefficients, dtype=np.float64), axis=0
        )
        if abs(float(coefficient_sums[0]) - 1.0) > tolerance or np.any(
            np.abs(coefficient_sums[1:]) > tolerance
        ):
            raise ValueError(
                "target polynomial is not proved to sum to one over the closed state box"
            )

        def check(
            *,
            quantized: bool,
            coefficients: Sequence[Sequence[int | float]],
            probability_scale: int | float,
        ) -> None:
            intervals = self._feature_intervals(quantized=quantized)
            for action, row in enumerate(coefficients):
                low: int | float = 0
                high: int | float = 0
                for coefficient, (feature_low, feature_high) in zip(row, intervals, strict=True):
                    endpoints = coefficient * feature_low, coefficient * feature_high
                    low += min(endpoints)
                    high += max(endpoints)
                slack = tolerance if not quantized else 0.0
                if low < -slack or high > probability_scale + slack:
                    kind = "quantized" if quantized else "clear"
                    raise ValueError(
                        f"{kind} action-{action} polynomial is not proved inside [0, 1] "
                        "over the closed state box"
                    )

        check(
            quantized=False,
            coefficients=self.target_policy.coefficients,
            probability_scale=1.0,
        )
        check(
            quantized=True,
            coefficients=self.coefficient_integers,
            probability_scale=self.target_probability_scale,
        )

    def quantize_client_inputs(self, batch: TrajectoryBatch) -> QuantizedTrajectoryTensors:
        """Perform the explicit client-only reciprocal and fixed-point preparation."""
        if batch.spec != self.trajectories:
            raise ValueError("batch spec differs from the circuit's frozen TrajectorySpec")
        failures: list[FailureRow] = []
        behavior = batch.behavior_array
        for trajectory, step in np.argwhere(
            (behavior < self.minimum_behavior_propensity) | (behavior > 1)
        ):
            failures.append(
                FailureRow(
                    code="unsupported_behavior",
                    field="behavior_propensities",
                    message="propensity is outside the compiled reciprocal domain",
                    trajectory=int(trajectory),
                    step=int(step),
                    value=float(behavior[trajectory, step]),
                )
            )
        if failures:
            raise OPEValidationError(tuple(failures))

        states = tuple(
            tuple(
                tuple(_quantize(value, self.scales.state) for value in state)
                for state in trajectory
            )
            for trajectory in batch.states
        )
        masks = tuple(
            tuple(
                tuple(
                    int(candidate == action) for candidate in range(self.trajectories.action_count)
                )
                for action in trajectory
            )
            for trajectory in batch.actions
        )
        rewards = tuple(
            tuple(_quantize(value, self.scales.reward) for value in trajectory)
            for trajectory in batch.rewards
        )
        reciprocals = tuple(
            tuple(_quantize_reciprocal(value, self.scales.reciprocal) for value in trajectory)
            for trajectory in batch.behavior_propensities
        )
        return QuantizedTrajectoryTensors(states, masks, rewards, reciprocals)

    def _integer_features(self, state: tuple[int, ...]) -> tuple[int, ...]:
        degree = self.target_policy.degree
        factor = self.scales.state if degree == 2 else 1
        features = [self.scales.state**degree]
        features.extend(value * factor for value in state)
        if degree == 2:
            features.extend(
                state[left] * state[right]
                for left in range(self.trajectories.state_dim)
                for right in range(left, self.trajectories.state_dim)
            )
        return tuple(features)

    def _logged_probability_integer(
        self, state: tuple[int, ...], action_mask: tuple[int, ...]
    ) -> int:
        features = self._integer_features(state)
        scores = tuple(
            sum(coefficient * feature for coefficient, feature in zip(row, features, strict=True))
            for row in self.coefficient_integers
        )
        probability = sum(mask * score for mask, score in zip(action_mask, scores, strict=True))
        if probability < 0 or probability > self.target_probability_scale:
            raise ValueError("quantized logged-action target probability is outside [0, 1]")
        return probability

    def _validate_tensors(self, tensors: QuantizedTrajectoryTensors) -> None:
        n = self.trajectories.trajectories
        h = self.trajectories.horizon
        d = self.trajectories.state_dim
        a = self.trajectories.action_count
        if not (
            len(tensors.states)
            == len(tensors.action_masks)
            == len(tensors.rewards)
            == len(tensors.behavior_reciprocals)
            == n
        ):
            raise ValueError("quantized tensors do not have the frozen trajectory count")
        state_low = tuple(
            _quantize(value, self.scales.state) for value in self.trajectories.state_min
        )
        state_high = tuple(
            _quantize(value, self.scales.state) for value in self.trajectories.state_max
        )
        reward_min = self.trajectories.reward_min
        reward_max = self.trajectories.reward_max
        assert reward_min is not None and reward_max is not None
        reward_low = _quantize(reward_min, self.scales.reward)
        reward_high = _quantize(reward_max, self.scales.reward)
        reciprocal_high = _quantize_reciprocal(
            self.minimum_behavior_propensity, self.scales.reciprocal
        )
        for trajectory in range(n):
            if not (
                len(tensors.states[trajectory])
                == len(tensors.action_masks[trajectory])
                == len(tensors.rewards[trajectory])
                == len(tensors.behavior_reciprocals[trajectory])
                == h
            ):
                raise ValueError("quantized tensors do not have the frozen horizon")
            for step in range(h):
                state = tensors.states[trajectory][step]
                mask = tensors.action_masks[trajectory][step]
                if len(state) != d:
                    raise ValueError("quantized state does not have the frozen state dimension")
                if any(
                    value < low or value > high
                    for value, low, high in zip(state, state_low, state_high, strict=True)
                ):
                    raise ValueError("quantized state is outside the compiled domain")
                if len(mask) != a or any(value not in {0, 1} for value in mask) or sum(mask) != 1:
                    raise ValueError("logged action mask must be one-hot over action_count")
                reward = tensors.rewards[trajectory][step]
                if reward < reward_low or reward > reward_high:
                    raise ValueError("quantized reward is outside the compiled domain")
                reciprocal = tensors.behavior_reciprocals[trajectory][step]
                if reciprocal < self.scales.reciprocal or reciprocal > reciprocal_high:
                    raise ValueError("quantized reciprocal is outside the compiled domain")

    def evaluate_integer(self, tensors: QuantizedTrajectoryTensors) -> IntegerSufficientStatistics:
        """Evaluate the exact server DAG using additions, products, comparisons, and muxes."""
        self._validate_tensors(tensors)
        horizon = self.trajectories.horizon
        trajectory_count = self.trajectories.trajectories
        numerator = [0] * horizon
        denominator = [0] * horizon
        ratio_scale = self.ratio_scale
        gamma_integer = _quantize(self.gamma, self.scales.discount)

        for trajectory in range(trajectory_count):
            raw_weight = 1
            for step in range(horizon):
                target = self._logged_probability_integer(
                    tensors.states[trajectory][step], tensors.action_masks[trajectory][step]
                )
                ratio_numerator = target * tensors.behavior_reciprocals[trajectory][step]
                raw_weight *= ratio_numerator
                weight_scale = ratio_scale ** (step + 1)
                clip_integer = _quantize(self.weight_clip, weight_scale)
                clipped_weight = min(raw_weight, clip_integer)
                discount_integer = gamma_integer**step
                denominator[step] += clipped_weight
                numerator[step] += (
                    clipped_weight * tensors.rewards[trajectory][step] * discount_integer
                )
        return IntegerSufficientStatistics(
            tuple(numerator), tuple(denominator), (trajectory_count,) * horizon
        )

    def operation_counts(self) -> CircuitOperationCounts:
        n = self.trajectories.trajectories
        h = self.trajectories.horizon
        a = self.trajectories.action_count
        d = self.trajectories.state_dim
        features = self.target_policy.feature_count
        samples = n * h
        quadratics = d * (d + 1) // 2 if self.target_policy.degree == 2 else 0
        encrypted_multiplications = samples * (quadratics + a + 2) + n * (h - 1)
        public_multiplications = samples * a * features + n * (h - 1)
        additions = samples * (a * (features - 1) + (a - 1))
        additions += 2 * max(0, n - 1) * h
        return CircuitOperationCounts(
            encrypted_multiplications=encrypted_multiplications,
            public_multiplications=public_multiplications,
            additions=additions,
            comparisons=samples,
            multiplexers=samples,
            multiplicative_depth=h + self.target_policy.degree + 1,
            encrypted_output_integers=3 * h,
        )

    def clear_statistics(
        self, batch: TrajectoryBatch, estimator: EstimatorName
    ) -> SufficientStatistics:
        """Evaluate the frozen floating-point clipped-PDIS/WPDIS semantics."""
        if batch.spec != self.trajectories:
            raise ValueError("batch spec differs from the circuit's frozen TrajectorySpec")
        behavior = batch.behavior_array
        if np.any(behavior < self.minimum_behavior_propensity):
            raise ValueError("behavior propensity is outside the compiled reciprocal domain")
        target = self.target_policy.logged_action_probabilities(batch)
        weights = cumulative_importance_weights(batch, target, weight_clip=self.weight_clip)
        discounts = self.gamma ** np.arange(self.trajectories.horizon)
        numerators = tuple(
            float(discounts[step] * np.sum(weights[:, step] * batch.reward_array[:, step]))
            for step in range(self.trajectories.horizon)
        )
        if estimator == "clipped_pdis":
            denominators = (float(self.trajectories.trajectories),) * self.trajectories.horizon
        elif estimator == "clipped_wpdis":
            denominators = tuple(
                float(np.sum(weights[:, step])) for step in range(self.trajectories.horizon)
            )
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        failures = (
            tuple(
                FailureRow(
                    "zero_weight_denominator",
                    "denominators",
                    "WPDIS is undefined at a horizon with zero total target weight",
                    step=step,
                    value=0.0,
                )
                for step, value in enumerate(denominators)
                if value == 0
            )
            if estimator == "clipped_wpdis"
            else ()
        )
        return SufficientStatistics(
            estimator=estimator,
            numerators=numerators,
            denominators=denominators,
            counts=(self.trajectories.trajectories,) * self.trajectories.horizon,
            failures=failures,
        )

    def client_statistics(
        self, statistics: IntegerSufficientStatistics, estimator: EstimatorName
    ) -> SufficientStatistics:
        """Decrypt/decode and divide after the server boundary (the only division API)."""
        if len(statistics.numerators) != self.trajectories.horizon:
            raise ValueError("integer statistics do not have the frozen horizon")
        numerators = tuple(
            value / scale
            for value, scale in zip(statistics.numerators, self.numerator_scales, strict=True)
        )
        if estimator == "clipped_pdis":
            denominators = tuple(float(value) for value in statistics.counts)
        elif estimator == "clipped_wpdis":
            denominators = tuple(
                value / scale
                for value, scale in zip(
                    statistics.denominators, self.denominator_scales, strict=True
                )
            )
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        failures = (
            tuple(
                FailureRow(
                    "zero_weight_denominator",
                    "denominators",
                    "WPDIS is undefined at a horizon with zero total target weight",
                    step=step,
                    value=0.0,
                )
                for step, value in enumerate(denominators)
                if value == 0
            )
            if estimator == "clipped_wpdis"
            else ()
        )
        return SufficientStatistics(
            estimator, numerators, denominators, statistics.counts, failures
        )

    def _overflow_bounds(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        n = self.trajectories.trajectories
        reciprocal_max = _quantize_reciprocal(
            self.minimum_behavior_propensity, self.scales.reciprocal
        )
        ratio_numerator_max = self.target_probability_scale * reciprocal_max
        reward_min = self.trajectories.reward_min
        reward_max = self.trajectories.reward_max
        assert reward_min is not None and reward_max is not None
        reward_integer_max = max(
            abs(_quantize(reward_min, self.scales.reward)),
            abs(_quantize(reward_max, self.scales.reward)),
        )
        gamma_integer = _quantize(self.gamma, self.scales.discount)
        raw_bounds = tuple(
            ratio_numerator_max ** (step + 1) for step in range(self.trajectories.horizon)
        )
        weight_bounds = tuple(
            min(raw, _quantize(self.weight_clip, self.ratio_scale ** (step + 1)))
            for step, raw in enumerate(raw_bounds)
        )
        denominator_bounds = tuple(n * weight for weight in weight_bounds)
        numerator_bounds = tuple(
            n * weight * reward_integer_max * gamma_integer**step
            for step, weight in enumerate(weight_bounds)
        )
        return numerator_bounds, denominator_bounds, raw_bounds

    def receipt(
        self,
        batch: TrajectoryBatch,
        tensors: QuantizedTrajectoryTensors,
        statistics: IntegerSufficientStatistics,
    ) -> CircuitReceipt:
        """Derive checked overflow and triangle-inequality quantization bounds."""
        target = self.target_policy.logged_action_probabilities(batch)
        behavior = batch.behavior_array
        rewards = batch.reward_array
        integer_product = [1] * self.trajectories.trajectories
        clear_weight_array = cumulative_importance_weights(
            batch, target, weight_clip=self.weight_clip
        )
        gamma_integer = _quantize(self.gamma, self.scales.discount)
        numerator_errors: list[float] = []
        denominator_errors: list[float] = []
        clear_numerators: list[float] = []
        clear_denominators: list[float] = []
        max_clip_error = 0.0
        max_probability_error = 0.0
        max_reciprocal_error = 0.0
        max_reward_error = 0.0

        for step in range(self.trajectories.horizon):
            integer_terms: list[float] = []
            clear_terms: list[float] = []
            integer_weights: list[float] = []
            clear_weights: list[float] = []
            weight_scale = self.ratio_scale ** (step + 1)
            max_clip_error = max(
                max_clip_error,
                abs(_quantize(self.weight_clip, weight_scale) / weight_scale - self.weight_clip),
            )
            for trajectory in range(self.trajectories.trajectories):
                target_integer = self._logged_probability_integer(
                    tensors.states[trajectory][step], tensors.action_masks[trajectory][step]
                )
                target_fixed = target_integer / self.target_probability_scale
                reciprocal_fixed = (
                    tensors.behavior_reciprocals[trajectory][step] / self.scales.reciprocal
                )
                reward_fixed = tensors.rewards[trajectory][step] / self.scales.reward
                max_probability_error = max(
                    max_probability_error, abs(target_fixed - target[trajectory, step])
                )
                max_reciprocal_error = max(
                    max_reciprocal_error,
                    abs(reciprocal_fixed - 1.0 / behavior[trajectory, step]),
                )
                max_reward_error = max(
                    max_reward_error, abs(reward_fixed - rewards[trajectory, step])
                )
                integer_product[trajectory] *= (
                    target_integer * tensors.behavior_reciprocals[trajectory][step]
                )
                fixed_weight_numerator = min(
                    integer_product[trajectory],
                    _quantize(self.weight_clip, weight_scale),
                )
                integer_weight = fixed_weight_numerator / weight_scale
                clear_weight = float(clear_weight_array[trajectory, step])
                integer_discount = (gamma_integer / self.scales.discount) ** step
                clear_discount = self.gamma**step
                integer_weights.append(integer_weight)
                clear_weights.append(clear_weight)
                integer_terms.append(integer_discount * integer_weight * reward_fixed)
                clear_terms.append(clear_discount * clear_weight * rewards[trajectory, step])
            numerator_errors.append(
                float(
                    sum(
                        abs(left - right)
                        for left, right in zip(integer_terms, clear_terms, strict=True)
                    )
                )
            )
            denominator_errors.append(
                float(
                    sum(
                        abs(left - right)
                        for left, right in zip(integer_weights, clear_weights, strict=True)
                    )
                )
            )
            clear_numerators.append(float(sum(clear_terms)))
            clear_denominators.append(float(sum(clear_weights)))

        pdis_error = float(sum(numerator_errors) / self.trajectories.trajectories)
        wpdis_error_accumulator = 0.0
        wpdis_defined = True
        for step in range(self.trajectories.horizon):
            fixed_denominator = statistics.denominators[step] / self.denominator_scales[step]
            clear_denominator = clear_denominators[step]
            if fixed_denominator == 0 or clear_denominator == 0:
                wpdis_defined = False
                break
            wpdis_error_accumulator += numerator_errors[step] / fixed_denominator
            wpdis_error_accumulator += (
                abs(clear_numerators[step])
                * denominator_errors[step]
                / (fixed_denominator * clear_denominator)
            )
        wpdis_error = wpdis_error_accumulator if wpdis_defined else None

        numerator_bounds, denominator_bounds, raw_bounds = self._overflow_bounds()
        if any(
            abs(value) > bound
            for value, bound in zip(statistics.numerators, numerator_bounds, strict=True)
        ):
            raise AssertionError("observed numerator exceeded its analytical bound")
        if any(
            value > bound
            for value, bound in zip(statistics.denominators, denominator_bounds, strict=True)
        ):
            raise AssertionError("observed denominator exceeded its analytical bound")
        return CircuitReceipt(
            numerator_scales=self.numerator_scales,
            denominator_scales=self.denominator_scales,
            numerator_abs_bounds=numerator_bounds,
            denominator_bounds=denominator_bounds,
            raw_weight_bounds=raw_bounds,
            numerator_signed_bits=tuple(_signed_bits(value) for value in numerator_bounds),
            denominator_unsigned_bits=tuple(
                max(1, value.bit_length()) for value in denominator_bounds
            ),
            raw_weight_unsigned_bits=tuple(max(1, value.bit_length()) for value in raw_bounds),
            operations=self.operation_counts(),
            error=QuantizationErrorBound(
                per_horizon_numerator=tuple(numerator_errors),
                per_horizon_denominator=tuple(denominator_errors),
                clipped_pdis=pdis_error,
                self_normalized_wpdis=wpdis_error,
                max_logged_probability_error=max_probability_error,
                max_clip_error=max_clip_error,
                max_reciprocal_error=max_reciprocal_error,
                max_reward_error=max_reward_error,
                discount_error=abs(gamma_integer / self.scales.discount - self.gamma),
            ),
            invalid_domains=self.invalid_domains,
        )

    def integer_reference(
        self, batch: TrajectoryBatch
    ) -> tuple[IntegerSufficientStatistics, CircuitReceipt]:
        """Run client preparation, exact integer server semantics, and analytical receipt."""
        tensors = self.quantize_client_inputs(batch)
        statistics = self.evaluate_integer(tensors)
        return statistics, self.receipt(batch, tensors, statistics)
