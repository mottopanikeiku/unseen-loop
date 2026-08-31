"""Clear reference estimators for the frozen OPE data contract.

All divisions and bootstrap resampling in this module are ordinary client-side
operations.  Supplying values to these functions does not imply private policy
training or private nuisance-model fitting.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from unseen_loop.ope.types import (
    BootstrapResult,
    FailureRow,
    OPEValidationError,
    SufficientStatistics,
    TrajectoryBatch,
)

FloatArray = npt.NDArray[np.float64]


def _gamma(gamma: float) -> float:
    value = float(gamma)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    return value


def _clip(weight_clip: float | None) -> float | None:
    if weight_clip is None:
        return None
    value = float(weight_clip)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("weight_clip must be finite and positive")
    return value


def _logged_target_array(batch: TrajectoryBatch, target_propensities: npt.ArrayLike) -> FloatArray:
    target = np.asarray(target_propensities, dtype=np.float64)
    if target.shape == (*batch.spec.batch_shape, batch.spec.action_count):
        distribution_rows: list[FailureRow] = []
        for i, t, action in np.argwhere(~np.isfinite(target) | (target < 0) | (target > 1)):
            distribution_rows.append(
                FailureRow(
                    "out_of_range",
                    f"target_propensities[{int(action)}]",
                    "all-action target propensities must be finite and in [0, 1]",
                    int(i),
                    int(t),
                    None if not np.isfinite(target[i, t, action]) else float(target[i, t, action]),
                )
            )
        sums = np.sum(target, axis=-1)
        for i, t in np.argwhere(~np.isclose(sums, 1.0, atol=1e-9, rtol=0)):
            distribution_rows.append(
                FailureRow(
                    "invalid_probability_distribution",
                    "target_propensities",
                    "all-action target propensities must sum to one",
                    int(i),
                    int(t),
                    None if not np.isfinite(sums[i, t]) else float(sums[i, t]),
                )
            )
        if distribution_rows:
            raise OPEValidationError(tuple(distribution_rows))
        target = np.take_along_axis(target, batch.action_array[..., None], axis=-1)[..., 0]
    elif target.shape != batch.spec.batch_shape:
        raise OPEValidationError(
            (
                FailureRow(
                    "shape_mismatch",
                    "target_propensities",
                    "expected logged-action shape "
                    f"{batch.spec.batch_shape} or all-action shape "
                    f"{(*batch.spec.batch_shape, batch.spec.action_count)}, got {target.shape}",
                ),
            )
        )
    failures = validation_failures(batch, target)
    if failures:
        raise OPEValidationError(failures)
    return target


def validation_failures(
    batch: TrajectoryBatch, target_propensities: npt.ArrayLike
) -> tuple[FailureRow, ...]:
    """Return every propensity/support failure without performing a division."""

    target = np.asarray(target_propensities, dtype=np.float64)
    if target.shape != batch.spec.batch_shape:
        return (
            FailureRow(
                "shape_mismatch",
                "target_propensities",
                f"expected {batch.spec.batch_shape}, got {target.shape}",
            ),
        )
    behavior = batch.behavior_array
    rows: list[FailureRow] = []
    for i, t in np.argwhere(~np.isfinite(target)):
        rows.append(
            FailureRow(
                "non_finite",
                "target_propensities",
                "target propensity must be finite",
                int(i),
                int(t),
            )
        )
    for i, t in np.argwhere((target < 0) | (target > 1)):
        rows.append(
            FailureRow(
                "out_of_range",
                "target_propensities",
                "target propensity must be in [0, 1]",
                int(i),
                int(t),
                float(target[i, t]),
            )
        )
    for i, t in np.argwhere(behavior == 0):
        target_value = target[i, t]
        if np.isfinite(target_value) and target_value > 0:
            rows.append(
                FailureRow(
                    "support_violation",
                    "behavior_propensities",
                    "target assigns positive mass where the logged action has zero behavior mass",
                    int(i),
                    int(t),
                    0.0,
                )
            )
        else:
            rows.append(
                FailureRow(
                    "zero_behavior_propensity",
                    "behavior_propensities",
                    "a logged action must have positive behavior propensity",
                    int(i),
                    int(t),
                    0.0,
                )
            )
    return tuple(rows)


def cumulative_importance_weights(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    weight_clip: float | None = None,
) -> FloatArray:
    """Return cumulative products ``rho[i,t]=prod(k<=t) pi/b``.

    When set, ``weight_clip`` clips each cumulative product, not each one-step
    ratio.  This frozen convention is shared by clipped PDIS and WPDIS.
    """

    clip = _clip(weight_clip)
    target = _logged_target_array(batch, target_propensities)
    ratios = target / batch.behavior_array
    weights = np.cumprod(ratios, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        index = np.argwhere(~np.isfinite(weights))[0]
        raise OPEValidationError(
            (
                FailureRow(
                    "weight_overflow",
                    "importance_weights",
                    "cumulative importance weight is not finite",
                    int(index[0]),
                    int(index[1]),
                ),
            )
        )
    if clip is not None:
        weights = np.minimum(weights, clip)
    return np.asarray(weights, dtype=np.float64)


def ordinary_is(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    gamma: float = 1.0,
) -> float:
    """Ordinary trajectory-wise IS: ``mean(rho_H * discounted_return)``."""

    discount = _gamma(gamma)
    weights = cumulative_importance_weights(batch, target_propensities)
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    returns = np.sum(batch.reward_array * powers, axis=1)
    return float(np.mean(weights[:, -1] * returns))


def pdis(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    gamma: float = 1.0,
) -> float:
    """Per-decision IS: ``mean_i sum_t gamma**t rho[i,t] r[i,t]``."""

    discount = _gamma(gamma)
    weights = cumulative_importance_weights(batch, target_propensities)
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    return float(np.mean(np.sum(weights * batch.reward_array * powers, axis=1)))


def clipped_pdis(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    weight_clip: float,
    gamma: float = 1.0,
) -> float:
    """PDIS with every cumulative importance weight capped at ``weight_clip``."""

    discount = _gamma(gamma)
    weights = cumulative_importance_weights(batch, target_propensities, weight_clip=weight_clip)
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    return float(np.mean(np.sum(weights * batch.reward_array * powers, axis=1)))


def wpdis_sufficient_statistics(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    gamma: float = 1.0,
    weight_clip: float | None = None,
) -> SufficientStatistics:
    """Additive per-horizon WPDIS numerators, denominators, and row counts.

    The client estimate is ``sum_t numerator[t] / denominator[t]``.  A horizon
    with no target-policy mass is retained as an observable failure rather than
    silently replacing its zero denominator.
    """

    discount = _gamma(gamma)
    weights = cumulative_importance_weights(batch, target_propensities, weight_clip=weight_clip)
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    numerators = np.sum(weights * batch.reward_array * powers, axis=0)
    denominators = np.sum(weights, axis=0)
    failures = tuple(
        FailureRow(
            "zero_weight_denominator",
            "denominators",
            "WPDIS is undefined at a horizon with zero total target weight",
            step=int(step),
            value=0.0,
        )
        for step in np.flatnonzero(denominators == 0)
    )
    return SufficientStatistics(
        estimator="wpdis" if weight_clip is None else "clipped_wpdis",
        numerators=tuple(float(value) for value in numerators),
        denominators=tuple(float(value) for value in denominators),
        counts=(batch.spec.trajectories,) * batch.spec.horizon,
        failures=failures,
    )


def wpdis(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    gamma: float = 1.0,
    weight_clip: float | None = None,
) -> float:
    """Client-side WPDIS reduction of additive sufficient statistics."""

    statistics = wpdis_sufficient_statistics(
        batch, target_propensities, gamma=gamma, weight_clip=weight_clip
    )
    estimate = statistics.estimate
    if estimate is None:
        raise OPEValidationError(statistics.failures)
    return estimate


def effective_sample_size(weights: npt.ArrayLike) -> float:
    """Kish ESS ``(sum w)^2/sum(w^2)``; all-zero weights have ESS zero."""

    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        raise ValueError("weights cannot be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    square_sum = float(np.sum(np.square(values)))
    if square_sum == 0:
        return 0.0
    return float(np.sum(values) ** 2 / square_sum)


def per_decision_effective_sample_size(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    weight_clip: float | None = None,
) -> tuple[float, ...]:
    """Kish ESS of cumulative trajectory weights at each horizon."""

    weights = cumulative_importance_weights(batch, target_propensities, weight_clip=weight_clip)
    return tuple(effective_sample_size(weights[:, step]) for step in range(weights.shape[1]))


def _all_action_values(
    batch: TrajectoryBatch,
    target_action_propensities: npt.ArrayLike,
    q_values: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    expected = (*batch.spec.batch_shape, batch.spec.action_count)
    target = np.asarray(target_action_propensities, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    rows: list[FailureRow] = []
    if target.shape != expected:
        rows.append(
            FailureRow(
                "shape_mismatch",
                "target_action_propensities",
                f"expected {expected}, got {target.shape}",
            )
        )
    if q.shape != expected:
        rows.append(FailureRow("shape_mismatch", "q_values", f"expected {expected}, got {q.shape}"))
    if rows:
        raise OPEValidationError(tuple(rows))
    if not np.all(np.isfinite(target)) or np.any(target < 0) or np.any(target > 1):
        raise OPEValidationError(
            (
                FailureRow(
                    "invalid_probability_distribution",
                    "target_action_propensities",
                    "all target probabilities must be finite and in [0, 1]",
                ),
            )
        )
    if not np.allclose(np.sum(target, axis=-1), 1.0, atol=1e-9, rtol=0):
        raise OPEValidationError(
            (
                FailureRow(
                    "invalid_probability_distribution",
                    "target_action_propensities",
                    "target probabilities must sum to one at every row",
                ),
            )
        )
    if not np.all(np.isfinite(q)):
        raise OPEValidationError(
            (FailureRow("non_finite", "q_values", "direct-method values must be finite"),)
        )
    return target, q


def direct_method_sufficient_statistics(
    batch: TrajectoryBatch,
    target_action_propensities: npt.ArrayLike,
    q_values: npt.ArrayLike,
    *,
    gamma: float = 1.0,
) -> SufficientStatistics:
    """Per-horizon DM statistics for conditional immediate-reward values.

    Freeze or cross-fit the supplied model independently of evaluation rows.
    No private model training is claimed.
    """

    discount = _gamma(gamma)
    target, q = _all_action_values(batch, target_action_propensities, q_values)
    state_values = np.sum(target * q, axis=-1)
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    numerators = np.sum(state_values * powers, axis=0)
    return SufficientStatistics(
        estimator="direct_method",
        numerators=tuple(float(value) for value in numerators),
        denominators=(float(batch.spec.trajectories),) * batch.spec.horizon,
        counts=(batch.spec.trajectories,) * batch.spec.horizon,
    )


def control_variate_sufficient_statistics(
    batch: TrajectoryBatch,
    target_action_propensities: npt.ArrayLike,
    q_values: npt.ArrayLike,
    *,
    gamma: float = 1.0,
    weight_clip: float | None = None,
) -> SufficientStatistics:
    """Sequential doubly-robust/control-variate additive statistics.

    Here ``Q(s_t,a)`` is an externally supplied model of the immediate reward
    at step ``t``.  The horizon contribution is
    ``gamma**t * (rho_t*(r_t-Q(s_t,a_t)) + rho_(t-1)*V(s_t))``, with
    ``rho_-1=1``.  Freeze or cross-fit Q independently of evaluation rows.  This
    function neither trains Q nor asserts private training.
    """

    discount = _gamma(gamma)
    target, q = _all_action_values(batch, target_action_propensities, q_values)
    logged_target = np.take_along_axis(target, batch.action_array[..., None], axis=-1)[..., 0]
    weights = cumulative_importance_weights(batch, logged_target, weight_clip=weight_clip)
    previous = np.concatenate(
        (np.ones((batch.spec.trajectories, 1), dtype=np.float64), weights[:, :-1]),
        axis=1,
    )
    state_values = np.sum(target * q, axis=-1)
    logged_q = np.take_along_axis(q, batch.action_array[..., None], axis=-1)[..., 0]
    powers = np.power(discount, np.arange(batch.spec.horizon, dtype=np.float64))
    contributions = powers * (weights * (batch.reward_array - logged_q) + previous * state_values)
    numerators = np.sum(contributions, axis=0)
    return SufficientStatistics(
        estimator="control_variate" if weight_clip is None else "clipped_control_variate",
        numerators=tuple(float(value) for value in numerators),
        denominators=(float(batch.spec.trajectories),) * batch.spec.horizon,
        counts=(batch.spec.trajectories,) * batch.spec.horizon,
    )


def bootstrap_ope(
    batch: TrajectoryBatch,
    target_propensities: npt.ArrayLike,
    *,
    estimator: str = "clipped_pdis",
    gamma: float = 1.0,
    weight_clip: float | None = None,
    samples: int = 1_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Deterministic whole-trajectory bootstrap for the named clear estimator.

    This client-side interval requires trajectory-level clear data.  Aggregate
    encrypted sufficient statistics alone do not provide a private bootstrap.
    """

    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(confidence) or confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")
    if estimator == "clipped_pdis" and weight_clip is None:
        raise ValueError("clipped_pdis requires weight_clip")
    target = _logged_target_array(batch, target_propensities)
    if estimator not in {"ordinary_is", "pdis", "clipped_pdis", "wpdis"}:
        raise ValueError(f"unknown estimator {estimator!r}")

    def evaluate(selected: TrajectoryBatch, propensity: FloatArray) -> float:
        if estimator == "ordinary_is":
            return ordinary_is(selected, propensity, gamma=gamma)
        if estimator == "pdis":
            return pdis(selected, propensity, gamma=gamma)
        if estimator == "clipped_pdis":
            if weight_clip is None:  # Guarded above; keeps the type contract explicit.
                raise AssertionError("unreachable missing clip")
            return clipped_pdis(selected, propensity, gamma=gamma, weight_clip=weight_clip)
        return wpdis(selected, propensity, gamma=gamma, weight_clip=weight_clip)

    estimate = evaluate(batch, target)
    rng = np.random.default_rng(int(seed))
    replicates = np.empty(int(samples), dtype=np.float64)
    for sample in range(int(samples)):
        indices = rng.integers(0, batch.spec.trajectories, size=batch.spec.trajectories)
        replicates[sample] = evaluate(batch.take(indices), target[indices])
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, (tail, 1.0 - tail))
    return BootstrapResult(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        samples=int(samples),
        seed=int(seed),
    )


per_decision_is = pdis
clipped_per_decision_is = clipped_pdis
weighted_per_decision_is = wpdis
