"""Frozen queue policy comparison and client-side statistical references.

These functions perform no work at import time.  Queue generation, truth DP,
model fitting and resampling are coordinator-authorized Modal work, not a local
benchmark.  Fitted nuisances never consume the evaluator's true queue kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from importlib import import_module
from typing import Any, Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt

from unseen_loop.ope.estimators import (
    cumulative_importance_weights,
    effective_sample_size,
    ordinary_is,
    pdis,
    wpdis,
    wpdis_sufficient_statistics,
)
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

QUEUE_POLICY_COMPARISON_V1 = "QUEUE_POLICY_COMPARISON_V1"
BASELINE_IDS = (
    "is",
    "pdis",
    "wpdis",
    "clipped_wpdis_2",
    "clipped_wpdis_10",
    "dm",
    "dr",
    "wdr",
    "mis",
)
FloatArray = npt.NDArray[np.float64]


class _FittedBaselineValues(TypedDict):
    dm: float
    dr: float
    wdr: float
    mis: float | None
    mis_failure_code: str | None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _discounts(horizon: int, gamma: float) -> FloatArray:
    horizon = _integer(horizon, "horizon", 1)
    gamma = _finite(gamma, "gamma")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    return np.power(gamma, np.arange(horizon, dtype=np.float64))


def queue_policies() -> tuple[PolynomialPolicySpec, PolynomialPolicySpec]:
    """The immutable, ordered A/B pair; coefficients are never data-fitted."""
    return (
        PolynomialPolicySpec(2, 1, 1, ((0.80, -0.50), (0.20, 0.50))),
        PolynomialPolicySpec(2, 1, 1, ((0.70, -0.50), (0.30, 0.50))),
    )


def queue_behavior_policy(behavior: str) -> PolynomialPolicySpec:
    if behavior == "primary":
        return PolynomialPolicySpec(2, 1, 1, ((0.75, -0.50), (0.25, 0.50)))
    if behavior == "stress":
        return PolynomialPolicySpec(2, 1, 1, ((0.625, -0.25), (0.375, 0.25)))
    raise ValueError("behavior must be primary or stress")


def queue_kernel() -> tuple[FloatArray, FloatArray]:
    """Enumerate P[q,a,q_next] and expected reward R[q,a] exactly as frozen."""
    transition = np.zeros((16, 2, 16), dtype=np.float64)
    rewards = np.zeros((16, 2), dtype=np.float64)
    for q in range(16):
        for action, service in enumerate((0.35, 0.80)):
            departures = ((0, 1.0),) if q == 0 else ((0, 1 - service), (1, service))
            for departure, departure_probability in departures:
                for arrival, arrival_probability in enumerate((0.55, 0.35, 0.10)):
                    probability = departure_probability * arrival_probability
                    u = q - departure + arrival
                    next_q = min(15, u)
                    overflow = max(0, u - 15)
                    transition[q, action, next_q] += probability
                    rewards[q, action] += (
                        probability * -(q / 15 + 0.15 * action + 0.50 * overflow) / 2.15
                    )
    return transition, rewards


def finite_horizon_values(
    transition: npt.ArrayLike,
    rewards: npt.ArrayLike,
    policy_probabilities: npt.ArrayLike,
    *,
    horizon: int,
    gamma: float,
) -> tuple[FloatArray, FloatArray]:
    """Backward DP on the supplied model, with terminal values exactly zero."""
    _discounts(horizon, gamma)
    transition = np.asarray(transition, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    policy = np.asarray(policy_probabilities, dtype=np.float64)
    if rewards.ndim != 2 or transition.shape != (*rewards.shape, rewards.shape[0]):
        raise ValueError("invalid tabular transition/reward shape")
    if policy.shape != rewards.shape:
        raise ValueError("invalid tabular policy shape")
    if any(not np.all(np.isfinite(x)) for x in (transition, rewards, policy)):
        raise ValueError("nonfinite tabular model")
    if np.any(transition < 0) or np.any(policy < 0):
        raise ValueError("negative tabular probability")
    if not np.allclose(transition.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("transition rows must sum to one")
    if not np.allclose(policy.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("policy rows must sum to one")
    values = np.zeros((horizon + 1, rewards.shape[0]), dtype=np.float64)
    q_values = np.empty((horizon, *rewards.shape), dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        q_values[t] = rewards + gamma * (transition @ values[t + 1])
        values[t] = np.sum(policy * q_values[t], axis=-1)
    return q_values, values


def queue_truth(horizon: int, gamma: float, policy: PolynomialPolicySpec) -> float:
    if policy.state_dim != 1 or policy.action_count != 2:
        raise ValueError("queue policies must have one state coordinate and two actions")
    transition, rewards = queue_kernel()
    probabilities = policy.action_probabilities((np.arange(16) / 15)[:, None])
    _, values = finite_horizon_values(
        transition, rewards, probabilities, horizon=horizon, gamma=gamma
    )
    return float(values[0, 8])


def queue_inputs(horizon: int, gamma: float) -> dict[str, Any]:
    """Public, canonical-JSON-compatible inputs and independently bound digests."""
    normalization = float(_discounts(horizon, gamma).sum())
    transition, rewards = queue_kernel()
    left, right = queue_policies()
    kernel = {
        "identifier": QUEUE_POLICY_COMPARISON_V1,
        "queue_capacity": 15,
        "initial_queue": 8,
        "arrival_probabilities": [0.55, 0.35, 0.10],
        "service_probabilities": [0.35, 0.80],
        "effort_cost": 0.15,
        "overflow_cost": 0.50,
        "reward_scale": 2.15,
        "transition_probabilities": transition.tolist(),
        "expected_rewards": rewards.tolist(),
    }
    policies = {
        "identifier": "QUEUE_ORDERED_POLICIES_V1",
        "policy_order": ["A", "B"],
        "A": left.to_dict(),
        "B": right.to_dict(),
        "primary": queue_behavior_policy("primary").to_dict(),
        "stress": queue_behavior_policy("stress").to_dict(),
    }
    truths = {}
    states = (np.arange(16) / 15)[:, None]
    for name, policy in (
        ("A", left),
        ("B", right),
        ("primary", queue_behavior_policy("primary")),
        ("stress", queue_behavior_policy("stress")),
    ):
        _, values = finite_horizon_values(
            transition,
            rewards,
            policy.action_probabilities(states),
            horizon=horizon,
            gamma=gamma,
        )
        truths[name] = float(values[0, 8])
    truth = {
        "horizon": horizon,
        "gamma": gamma,
        "normalization": normalization,
        "values_raw": truths,
        "contrast_normalized": (truths["B"] - truths["A"]) / normalization,
    }
    return {
        "kernel": kernel,
        "policies": policies,
        "truth": truth,
        "kernel_sha256": _digest(kernel),
        "policies_sha256": _digest(policies),
        "truth_sha256": _digest(truth),
    }


def queue_batch(
    seed: int,
    trajectories: int,
    horizon: int,
    behavior: str,
) -> tuple[TrajectoryBatch, npt.NDArray[np.int64]]:
    seed = _integer(seed, "seed")
    trajectories = _integer(trajectories, "trajectories", 1)
    horizon = _integer(horizon, "horizon", 1)
    policy = queue_behavior_policy(behavior)
    rng = np.random.default_rng(seed)
    states = np.empty((trajectories, horizon, 1), dtype=np.float64)
    actions = np.empty((trajectories, horizon), dtype=np.int64)
    rewards = np.empty((trajectories, horizon), dtype=np.float64)
    propensities = np.empty_like(rewards)
    next_states = np.empty_like(actions)
    q = np.full(trajectories, 8, dtype=np.int64)
    for t in range(horizon):
        states[:, t, 0] = q / 15
        surge = policy.action_probabilities(states[:, t])[:, 1]
        action = (rng.random(trajectories) < surge).astype(np.int64)
        departure = (q > 0) & (rng.random(trajectories) < np.where(action == 0, 0.35, 0.80))
        arrival = rng.choice(3, size=trajectories, p=(0.55, 0.35, 0.10))
        u = q - departure + arrival
        overflow = np.maximum(0, u - 15)
        rewards[:, t] = -(q / 15 + 0.15 * action + 0.50 * overflow) / 2.15
        actions[:, t] = action
        propensities[:, t] = np.where(action == 1, surge, 1 - surge)
        q = np.minimum(15, u)
        next_states[:, t] = q
    spec = TrajectorySpec(trajectories, horizon, 1, 2, (0.0,), (1.0,), -1.0, 0.0)
    return TrajectoryBatch(
        spec,
        tuple(tuple(tuple(float(x) for x in state) for state in row) for row in states),
        tuple(tuple(int(x) for x in row) for row in actions),
        tuple(tuple(float(x) for x in row) for row in rewards),
        tuple(tuple(float(x) for x in row) for row in propensities),
    ), next_states


@dataclass(frozen=True)
class PairedWPDISBootstrap:
    normalization: float
    raw_left: float
    raw_right: float
    raw_contrast: float
    raw_left_se: float
    raw_right_se: float
    raw_contrast_se: float
    raw_lower: float
    raw_upper: float
    normalized_left: float
    normalized_right: float
    normalized_contrast: float
    normalized_left_se: float
    normalized_right_se: float
    normalized_contrast_se: float
    normalized_lower: float
    normalized_upper: float
    normalized_width: float

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _finite(getattr(self, field.name), field.name))
        if self.normalization <= 0 or self.raw_lower > self.raw_upper:
            raise ValueError("invalid bootstrap normalization or endpoints")
        for name in (
            "left",
            "right",
            "contrast",
            "left_se",
            "right_se",
            "contrast_se",
            "lower",
            "upper",
        ):
            raw = getattr(self, "raw_" + name)
            normalized = getattr(self, "normalized_" + name)
            if not math.isclose(normalized, raw / self.normalization, abs_tol=1e-12, rel_tol=0):
                raise ValueError("inconsistent bootstrap units")
            if name.endswith("_se") and raw < 0:
                raise ValueError("bootstrap standard error cannot be negative")
        if not math.isclose(
            self.raw_contrast, self.raw_right - self.raw_left, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("bootstrap contrast must be right minus left")
        if not math.isclose(
            self.normalized_width,
            (self.raw_upper - self.raw_lower) / self.normalization,
            abs_tol=1e-12,
            rel_tol=0,
        ):
            raise ValueError("inconsistent bootstrap width")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PairedWPDISBootstrap:
        if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(cls)}:
            raise ValueError("invalid paired bootstrap fields")
        return cls(**raw)


class StatisticalSupportError(ValueError):
    """Undefined support is evidence of failure, never a zero contribution."""

    failure_code = "statistics.invalid_support"


def _wpdis_draws(
    counts: FloatArray, weighted_rewards: FloatArray, weights: FloatArray, discounts: FloatArray
) -> FloatArray:
    denominators = counts @ weights
    if np.any(denominators <= 0) or not np.all(np.isfinite(denominators)):
        raise StatisticalSupportError("bootstrap draw has undefined WPDIS denominator")
    estimates = ((counts @ weighted_rewards) / denominators) @ discounts
    if not np.all(np.isfinite(estimates)):
        raise StatisticalSupportError("bootstrap draw has nonfinite WPDIS estimate")
    return estimates


def paired_wpdis_bootstrap(
    batch: TrajectoryBatch,
    left_policy: PolynomialPolicySpec,
    right_policy: PolynomialPolicySpec,
    *,
    gamma: float,
    repetitions: int,
    seed: int,
) -> PairedWPDISBootstrap:
    """Shared multinomial whole-trajectory draws; no draw is filtered/replaced."""
    repetitions = _integer(repetitions, "repetitions", 2)
    seed = _integer(seed, "seed")
    discounts = _discounts(batch.spec.horizon, gamma)
    normalization = float(discounts.sum())
    weights = tuple(
        cumulative_importance_weights(batch, p.logged_action_probabilities(batch))
        for p in (left_policy, right_policy)
    )
    weighted_rewards = tuple(w * batch.reward_array for w in weights)
    n = batch.spec.trajectories
    original = np.ones((1, n), dtype=np.float64)
    point = [
        _wpdis_draws(original, wr, w, discounts)[0]
        for wr, w in zip(weighted_rewards, weights, strict=True)
    ]
    draws = np.empty((repetitions, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1 / n, dtype=np.float64)
    for start in range(0, repetitions, 64):
        stop = min(start + 64, repetitions)
        counts = rng.multinomial(n, probabilities, size=stop - start).astype(np.float64)
        for policy_index in range(2):
            draws[start:stop, policy_index] = _wpdis_draws(
                counts, weighted_rewards[policy_index], weights[policy_index], discounts
            )
    contrasts = draws[:, 1] - draws[:, 0]
    lower, upper = np.quantile(contrasts, (0.025, 0.975), method="linear")
    raw = dict(
        left=float(point[0]),
        right=float(point[1]),
        contrast=float(point[1] - point[0]),
        left_se=float(np.std(draws[:, 0], ddof=1)),
        right_se=float(np.std(draws[:, 1], ddof=1)),
        contrast_se=float(np.std(contrasts, ddof=1)),
        lower=float(lower),
        upper=float(upper),
    )
    return PairedWPDISBootstrap(
        normalization=normalization,
        **{"raw_" + k: v for k, v in raw.items()},
        **{"normalized_" + k: v / normalization for k, v in raw.items()},
        normalized_width=float((upper - lower) / normalization),
    )


def translate_cipher_interval(
    bootstrap: PairedWPDISBootstrap,
    cipher_contrast_normalized: float,
) -> tuple[float, float, Literal["positive", "negative", "abstain"]]:
    shift = _finite(cipher_contrast_normalized, "cipher contrast") - bootstrap.normalized_contrast
    lower, upper = bootstrap.normalized_lower + shift, bootstrap.normalized_upper + shift
    decision: Literal["positive", "negative", "abstain"] = (
        "positive" if lower > 0 else "negative" if upper < 0 else "abstain"
    )
    return lower, upper, decision


def _sequential_arrays(
    rewards: npt.ArrayLike,
    weights: npt.ArrayLike,
    q_logged: npt.ArrayLike,
    v_current: npt.ArrayLike,
    gamma: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    arrays = (
        np.asarray(rewards, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(q_logged, dtype=np.float64),
        np.asarray(v_current, dtype=np.float64),
    )
    if arrays[0].ndim != 2 or min(arrays[0].shape) < 1:
        raise ValueError("sequential arrays must be nonempty N by H")
    if any(x.shape != arrays[0].shape or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("sequential arrays must have identical finite shapes")
    if np.any(arrays[1] < 0):
        raise ValueError("importance weights cannot be negative")
    return (*arrays, _discounts(arrays[0].shape[1], gamma))


def sequential_dr(
    rewards: npt.ArrayLike,
    weights: npt.ArrayLike,
    q_logged: npt.ArrayLike,
    v_current: npt.ArrayLike,
    v_next: npt.ArrayLike,
    *,
    gamma: float,
) -> float:
    """Full sequential DR with row-specific, excluded-fold finite-horizon nuisances."""
    r, w, q, v, discount = _sequential_arrays(rewards, weights, q_logged, v_current, gamma)
    next_values = np.asarray(v_next, dtype=np.float64)
    if next_values.shape != r.shape or not np.all(np.isfinite(next_values)):
        raise ValueError("invalid next-state values")
    if np.any(next_values[:, -1] != 0):
        raise ValueError("finite-horizon terminal value must be zero")
    rows = v[:, 0] + np.sum(discount * w * (r + gamma * next_values - q), axis=1)
    return float(np.mean(rows))


def globally_normalized_wdr(
    rewards: npt.ArrayLike,
    weights: npt.ArrayLike,
    q_logged: npt.ArrayLike,
    v_current: npt.ArrayLike,
    *,
    gamma: float,
) -> float:
    """Normalize weights across ALL rows, not separately within nuisance folds."""
    r, w, q, v, discount = _sequential_arrays(rewards, weights, q_logged, v_current, gamma)
    sums = w.sum(axis=0)
    if np.any(sums <= 0) or not np.all(np.isfinite(sums)):
        raise StatisticalSupportError("WDR has undefined global denominator")
    normalized = w / sums
    previous = np.empty_like(normalized)
    previous[:, 0] = 1 / len(r)
    previous[:, 1:] = normalized[:, :-1]
    return float(np.sum(discount * np.sum(normalized * (r - q) + previous * v, axis=0)))


def state_action_mis(
    rewards: npt.ArrayLike,
    target_mass: npt.ArrayLike,
    behavior_mass: npt.ArrayLike,
    *,
    gamma: float,
) -> float:
    """Average held-out occupancy-ratio row contributions over the global N."""
    r, target, behavior = (
        np.asarray(x, dtype=np.float64) for x in (rewards, target_mass, behavior_mass)
    )
    if r.ndim != 2 or min(r.shape) < 1 or target.shape != r.shape or behavior.shape != r.shape:
        raise ValueError("MIS arrays must have identical nonempty N by H shapes")
    if any(not np.all(np.isfinite(x)) for x in (r, target, behavior)) or np.any(target < 0):
        raise StatisticalSupportError("MIS has invalid occupancy mass")
    if np.any(behavior <= 0):
        raise StatisticalSupportError("MIS has undefined held-out behavior support")
    result = float(np.mean(np.sum(_discounts(r.shape[1], gamma) * target / behavior * r, axis=1)))
    if not math.isfinite(result):
        raise StatisticalSupportError("MIS has nonfinite estimate")
    return result


def _queue_observations(batch: TrajectoryBatch, next_states: npt.ArrayLike) -> tuple[Any, Any, str]:
    if batch.spec.state_dim != 1 or batch.spec.action_count != 2:
        raise ValueError("invalid queue batch dimensions")
    scaled = batch.state_array[..., 0] * 15
    q = np.rint(scaled).astype(np.int64)
    raw_next = np.asarray(next_states)
    if raw_next.shape != batch.spec.batch_shape or raw_next.dtype.kind not in "iu":
        raise ValueError("next queue states must be an integer N by H array")
    nxt = raw_next.astype(np.int64, copy=False)
    if np.any(q < 0) or np.any(q > 15) or not np.allclose(scaled, q, atol=1e-12, rtol=0):
        raise ValueError("states must be q/15 for integer q in [0,15]")
    if np.any(nxt < 0) or np.any(nxt > 15) or np.any(q[:, 0] != 8):
        raise ValueError("invalid initial or next queue states")
    if not np.array_equal(q[:, 1:], nxt[:, :-1]):
        raise ValueError("queue transitions must preserve whole trajectories")
    if np.any(nxt < np.maximum(0, q - 1)) or np.any(nxt > np.minimum(15, q + 2)):
        raise ValueError("queue transition is outside the fixed kernel support")
    if np.any(batch.reward_array < -1) or np.any(batch.reward_array > 0):
        raise ValueError("queue rewards must be in [-1,0]")
    for behavior in ("primary", "stress"):
        logged = queue_behavior_policy(behavior).logged_action_probabilities(batch)
        if np.allclose(batch.behavior_array, logged, atol=1e-12, rtol=0):
            return q, nxt, behavior
    raise ValueError("batch does not use a frozen queue behavior")


def _occupancies(transition: FloatArray, policy: FloatArray, horizon: int) -> FloatArray:
    occupancy = np.zeros((horizon, 16), dtype=np.float64)
    occupancy[0, 8] = 1
    controlled = np.einsum("sa,sak->sk", policy, transition)
    for t in range(1, horizon):
        occupancy[t] = occupancy[t - 1] @ controlled
    return occupancy


def cross_fitted_queue_baselines(
    batch: TrajectoryBatch,
    next_states: npt.ArrayLike,
    *,
    gamma: float,
) -> tuple[_FittedBaselineValues, _FittedBaselineValues]:
    """Stationary pooled training model, four fixed index-mod-four held-out folds."""
    q, nxt, behavior = _queue_observations(batch, next_states)
    n, horizon = q.shape
    actions, rewards = batch.action_array, batch.reward_array
    folds = np.arange(n) % 4
    states = (np.arange(16) / 15)[:, None]
    mu = queue_behavior_policy(behavior).action_probabilities(states)
    policies = queue_policies()
    nuisances = [
        dict(
            q=np.empty_like(rewards),
            v=np.empty_like(rewards),
            next=np.empty_like(rewards),
            target=np.empty_like(rewards),
            behavior=np.empty_like(rewards),
        )
        for _ in policies
    ]
    for fold in range(4):
        held = np.flatnonzero(folds == fold)
        if not len(held):
            continue
        train = folds != fold
        cells = (q[train] * 2 + actions[train]).ravel()
        counts = np.bincount(cells, minlength=32).reshape(16, 2)
        reward_sum = np.bincount(cells, weights=rewards[train].ravel(), minlength=32).reshape(16, 2)
        fitted_rewards = (reward_sum - 0.25) / (counts + 0.5)
        transition_counts = np.bincount(cells * 16 + nxt[train].ravel(), minlength=512).reshape(
            16, 2, 16
        )
        transition = (transition_counts + 0.5) / (counts[..., None] + 8)
        behavior_occupancy = _occupancies(transition, mu, horizon)
        step = np.arange(horizon)[None, :]
        held_q, held_a, held_next = q[held], actions[held], nxt[held]
        for policy, nuisance in zip(policies, nuisances, strict=True):
            probabilities = policy.action_probabilities(states)
            qhat, vhat = finite_horizon_values(
                transition, fitted_rewards, probabilities, horizon=horizon, gamma=gamma
            )
            target_occupancy = _occupancies(transition, probabilities, horizon)
            nuisance["q"][held] = qhat[step, held_q, held_a]
            nuisance["v"][held] = vhat[step, held_q]
            nuisance["next"][held] = vhat[step + 1, held_next]
            nuisance["target"][held] = (
                target_occupancy[step, held_q] * probabilities[held_q, held_a]
            )
            nuisance["behavior"][held] = behavior_occupancy[step, held_q] * mu[held_q, held_a]
    results = []
    for policy, nuisance in zip(policies, nuisances, strict=True):
        weights = cumulative_importance_weights(batch, policy.logged_action_probabilities(batch))
        result: _FittedBaselineValues = {
            "dm": float(np.mean(nuisance["v"][:, 0])),
            "dr": sequential_dr(
                rewards, weights, nuisance["q"], nuisance["v"], nuisance["next"], gamma=gamma
            ),
            "wdr": globally_normalized_wdr(
                rewards, weights, nuisance["q"], nuisance["v"], gamma=gamma
            ),
            "mis": None,
            "mis_failure_code": None,
        }
        try:
            result["mis"] = state_action_mis(
                rewards, nuisance["target"], nuisance["behavior"], gamma=gamma
            )
        except StatisticalSupportError:
            result["mis_failure_code"] = "statistics.invalid_support"
        results.append(result)
    return results[0], results[1]


def evaluate_queue_batch(
    batch: TrajectoryBatch,
    next_states: npt.ArrayLike,
    *,
    gamma: float,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return only released paired aggregate evidence, never trajectory rows."""
    _queue_observations(batch, next_states)
    discounts = _discounts(batch.spec.horizon, gamma)
    normalization = float(discounts.sum())
    inputs = queue_inputs(batch.spec.horizon, gamma)
    policies = queue_policies()
    bootstrap = paired_wpdis_bootstrap(
        batch, *policies, gamma=gamma, repetitions=repetitions, seed=bootstrap_seed
    )
    fitted = cross_fitted_queue_baselines(batch, next_states, gamma=gamma)
    rows = []
    baseline_values: list[dict[str, float | None]] = []
    for index, (name, policy) in enumerate(zip(("A", "B"), policies, strict=True)):
        logged = policy.logged_action_probabilities(batch)
        weights = cumulative_importance_weights(batch, logged)
        statistics = wpdis_sufficient_statistics(batch, logged, gamma=gamma)
        raw_value = statistics.estimate
        if raw_value is None:
            raise StatisticalSupportError("undefined primary WPDIS denominator")
        rows.append(
            {
                "id": name,
                "policy_sha256": _digest(policy.to_dict()),
                "mean_weighted_rewards": np.mean(weights * batch.reward_array, axis=0).tolist(),
                "mean_weights": np.mean(weights, axis=0).tolist(),
                "raw_ess": [
                    effective_sample_size(weights[:, t]) for t in range(batch.spec.horizon)
                ],
                "counts": [batch.spec.trajectories] * batch.spec.horizon,
                "raw_value": raw_value,
                "normalized_value": raw_value / normalization,
                "minimum_logged_propensity": float(np.min(batch.behavior_array)),
                "maximum_logged_ratio": float(np.max(logged / batch.behavior_array)),
                "maximum_cumulative_weight": float(np.max(weights)),
                "support_failures": 0,
            }
        )
        baseline_values.append(
            {
                "is": ordinary_is(batch, logged, gamma=gamma),
                "pdis": pdis(batch, logged, gamma=gamma),
                "wpdis": raw_value,
                "clipped_wpdis_2": wpdis(batch, logged, gamma=gamma, weight_clip=2),
                "clipped_wpdis_10": wpdis(batch, logged, gamma=gamma, weight_clip=10),
                "dm": fitted[index]["dm"],
                "dr": fitted[index]["dr"],
                "wdr": fitted[index]["wdr"],
                "mis": fitted[index]["mis"],
            }
        )
    baseline_rows = []
    for estimator in BASELINE_IDS:
        left, right = (values[estimator] for values in baseline_values)
        failure = "statistics.invalid_support" if left is None or right is None else None
        baseline_rows.append(
            {
                "estimator_id": estimator,
                "left_raw": left,
                "right_raw": right,
                "contrast_normalized": (
                    None if left is None or right is None else (right - left) / normalization
                ),
                "failure_code": failure,
            }
        )
    truth = inputs["truth"]
    return {
        "kind": "batch_reference",
        "batch_sha256": _digest(batch.to_dict()),
        "kernel_sha256": inputs["kernel_sha256"],
        "policies_sha256": inputs["policies_sha256"],
        "normalization": normalization,
        "truth_left_raw": truth["values_raw"]["A"],
        "truth_right_raw": truth["values_raw"]["B"],
        "truth_contrast_normalized": truth["contrast_normalized"],
        "policy_rows": rows,
        "bootstrap": bootstrap.to_dict(),
        "baseline_rows": baseline_rows,
    }


def _sample(values: npt.ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("summary requires a nonempty finite one-dimensional sample")
    return result


def student_t_bias_interval(errors: npt.ArrayLike) -> tuple[float, float | None, float | None]:
    t_ppf = cast(Callable[[float, int], float], import_module("scipy.stats").t.ppf)

    values = _sample(errors)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    radius = float(t_ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - radius, mean + radius


def clopper_pearson_lower(successes: int, trials: int) -> float:
    beta_ppf = cast(Callable[[float, int, int], float], import_module("scipy.stats").beta.ppf)

    successes, trials = _integer(successes, "successes"), _integer(trials, "trials", 1)
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    return 0.0 if successes == 0 else float(beta_ppf(0.05, successes, trials - successes + 1))


def batch_rmse_interval(
    errors: npt.ArrayLike,
    *,
    seed: int,
    repetitions: int = 10000,
) -> tuple[float, float | None, float | None]:
    values = _sample(errors)
    seed, repetitions = _integer(seed, "seed"), _integer(repetitions, "repetitions", 2)
    squared = np.square(values)
    rmse = float(np.sqrt(np.mean(squared)))
    if len(values) < 2:
        return rmse, None, None
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 64):
        stop = min(start + 64, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = np.sqrt(np.mean(squared[indices], axis=1))
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return rmse, float(lower), float(upper)


def timing_speedup_interval(
    speedups: npt.ArrayLike,
    *,
    seed: int,
    repetitions: int = 10000,
) -> tuple[float, float]:
    values = _sample(speedups)
    seed, repetitions = _integer(seed, "seed"), _integer(repetitions, "repetitions", 2)
    if np.any(values <= 0):
        raise ValueError("timing speedups must be positive; invalid pairs cannot be filtered")
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 64):
        stop = min(start + 64, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        medians[start:stop] = np.median(values[indices], axis=1)
    return float(np.median(values)), float(np.quantile(medians, 0.05, method="linear"))
