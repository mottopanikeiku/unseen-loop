from __future__ import annotations

import numpy as np
import pytest

from unseen_loop.ope.study import (
    PairedWPDISBootstrap,
    StatisticalSupportError,
    cross_fitted_queue_baselines,
    globally_normalized_wdr,
    paired_wpdis_bootstrap,
    queue_policies,
    sequential_dr,
    state_action_mis,
    translate_cipher_interval,
)
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec


def _tiny_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        TrajectorySpec(3, 2, 1, 2),
        (((0.0,), (0.0,)), ((0.0,), (0.0,)), ((0.0,), (0.0,))),
        ((0, 1), (1, 0), (1, 1)),
        ((-1.0, 0.0), (0.0, -1.0), (-0.5, -0.25)),
        ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
    )


def test_paired_bootstrap_matches_shared_whole_trajectory_draws_and_units() -> None:
    batch = _tiny_batch()
    left = PolynomialPolicySpec(2, 1, 1, ((0.6, 0.0), (0.4, 0.0)))
    right = PolynomialPolicySpec(2, 1, 1, ((0.4, 0.0), (0.6, 0.0)))
    result = paired_wpdis_bootstrap(batch, left, right, gamma=0.5, repetitions=67, seed=39)
    # Independent literal implementation crosses the frozen 64-draw block boundary.
    rng = np.random.default_rng(39)
    counts = np.concatenate(
        (rng.multinomial(3, [1 / 3] * 3, size=64), rng.multinomial(3, [1 / 3] * 3, size=3))
    )
    draws = []
    points = []
    for policy in (left, right):
        ratios = policy.logged_action_probabilities(batch) / 0.5
        weights = np.column_stack((ratios[:, 0], ratios[:, 0] * ratios[:, 1]))
        weighted = weights * batch.reward_array
        draws.append(np.sum((counts @ weighted) / (counts @ weights) * [1.0, 0.5], axis=1))
        points.append(float(np.sum(weighted.sum(axis=0) / weights.sum(axis=0) * [1.0, 0.5])))
    contrast = draws[1] - draws[0]
    lower, upper = np.quantile(contrast, [0.025, 0.975], method="linear")
    assert result.raw_left == pytest.approx(points[0])
    assert result.raw_right == pytest.approx(points[1])
    assert result.raw_contrast == pytest.approx(points[1] - points[0])
    assert result.raw_lower == pytest.approx(lower)
    assert result.raw_upper == pytest.approx(upper)
    for name, sample in (("left", draws[0]), ("right", draws[1]), ("contrast", contrast)):
        assert getattr(result, f"raw_{name}_se") == pytest.approx(np.std(sample, ddof=1))
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
        assert getattr(result, "normalized_" + name) == pytest.approx(
            getattr(result, "raw_" + name) / 1.5
        )
    assert result.normalized_width == pytest.approx((upper - lower) / 1.5)
    assert PairedWPDISBootstrap.from_dict(result.to_dict()) == result
    invalid = result.to_dict() | {"normalized_left_se": result.normalized_left_se + 1}
    with pytest.raises(ValueError):
        PairedWPDISBootstrap.from_dict(invalid)
    with pytest.raises(ValueError):
        PairedWPDISBootstrap.from_dict(result.to_dict() | {"extra": 0})
    with pytest.raises(ValueError):
        PairedWPDISBootstrap.from_dict(result.to_dict() | {"normalization": True})


def test_bootstrap_retains_undefined_resample_as_support_failure() -> None:
    batch = _tiny_batch()
    batch = TrajectoryBatch(
        batch.spec,
        batch.states,
        ((0, 0), (1, 0), (1, 1)),
        batch.rewards,
        batch.behavior_propensities,
    )
    deterministic = PolynomialPolicySpec(2, 1, 1, ((1.0, 0.0), (0.0, 0.0)))
    with pytest.raises(StatisticalSupportError):
        paired_wpdis_bootstrap(
            batch, deterministic, deterministic, gamma=1.0, repetitions=67, seed=39
        )


def test_sequential_dr_exact_model_telescopes_with_off_policy_weights() -> None:
    # A deterministic two-step model with rewards -1 then -2. Exact Q/V
    # eliminate the entire TD correction even for non-unit cumulative weights.
    rewards = np.asarray([[-1.0, -2.0], [-1.0, -2.0]])
    values = np.asarray([[-2.0, -2.0], [-2.0, -2.0]])
    next_values = np.asarray([[-2.0, 0.0], [-2.0, 0.0]])
    weights = np.asarray([[0.5, 0.25], [1.5, 2.25]])
    assert sequential_dr(rewards, weights, values, values, next_values, gamma=0.5) == -2.0
    # On-policy telescoping must hold even when the nuisance is wrong and
    # differs between excluded folds, provided adjacent V values are coherent.
    wrong = np.asarray([[9.0, 3.0], [-4.0, 7.0]])
    wrong_next = np.column_stack((wrong[:, 1], np.zeros(2)))
    assert sequential_dr(
        rewards, np.ones_like(weights), wrong, wrong, wrong_next, gamma=0.5
    ) == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        sequential_dr(rewards, weights, values, values, np.ones_like(values), gamma=0.5)


def test_wdr_uses_global_current_and_previous_weights_across_fold_nuisances() -> None:
    rewards = np.asarray([[1.0, 2.0], [3.0, 5.0], [-2.0, 4.0], [8.0, -1.0]])
    weights = np.asarray([[1.0, 2.0], [2.0, 1.0], [4.0, 8.0], [8.0, 4.0]])
    qhat = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    vhat = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    expected = (
        np.dot(weights[:, 0] / 15, rewards[:, 0] - qhat[:, 0])
        + np.mean(vhat[:, 0])
        + 0.5
        * (
            np.dot(weights[:, 1] / 15, rewards[:, 1] - qhat[:, 1])
            + np.dot(weights[:, 0] / 15, vhat[:, 1])
        )
    )
    actual = globally_normalized_wdr(rewards, weights, qhat, vhat, gamma=0.5)
    assert actual == pytest.approx(expected)
    fold_normalized = np.mean(
        [
            globally_normalized_wdr(
                rewards[i : i + 1], weights[i : i + 1], qhat[i : i + 1], vhat[i : i + 1], gamma=0.5
            )
            for i in range(4)
        ]
    )
    assert abs(actual - fold_normalized) > 0.1
    with pytest.raises(StatisticalSupportError):
        globally_normalized_wdr(rewards, np.zeros_like(weights), qhat, vhat, gamma=0.5)


def test_cross_fitted_queue_wdr_pools_training_steps_but_not_weight_normalizers() -> None:
    # Fixed inputs, not a generated queue study. Each held-out fold has a
    # different training model; derive the H2 nuisance without calling its DP.
    q = np.asarray([[8, 7], [8, 8], [8, 9], [8, 10], [8, 8], [8, 9], [8, 10], [8, 7]])
    actions = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
    next_states = np.column_stack((q[:, 1], [8, 9, 10, 11, 8, 8, 9, 7]))
    rewards = -(q / 15 + 0.15 * actions) / 2.15
    surge = 0.25 + 0.5 * q / 15
    mu_logged = np.where(actions == 1, surge, 1 - surge)
    batch = TrajectoryBatch(
        TrajectorySpec(8, 2, 1, 2), (q / 15)[..., None], actions, rewards, mu_logged
    )
    observed = cross_fitted_queue_baselines(batch, next_states, gamma=0.5)
    for policy_index, policy in enumerate(queue_policies()):
        probabilities = policy.action_probabilities((np.arange(16) / 15)[:, None])
        q_logged, v_current = np.empty((8, 2)), np.empty((8, 2))
        for fold in range(4):
            count = np.zeros((16, 2))
            reward_sum = np.zeros((16, 2))
            transitions = np.zeros((16, 2, 16))
            for i in range(8):
                if i % 4 == fold:
                    continue
                for t in range(2):
                    state, action = q[i, t], actions[i, t]
                    count[state, action] += 1
                    reward_sum[state, action] += rewards[i, t]
                    transitions[state, action, next_states[i, t]] += 1
            rhat = (reward_sum - 0.25) / (count + 0.5)
            phat = (transitions + 0.5) / (count[..., None] + 8)
            v1 = (probabilities * rhat).sum(axis=1)
            q0 = rhat + 0.5 * (phat @ v1)
            v0 = (probabilities * q0).sum(axis=1)
            for i in range(fold, 8, 4):
                q_logged[i] = [q0[q[i, 0], actions[i, 0]], rhat[q[i, 1], actions[i, 1]]]
                v_current[i] = [v0[q[i, 0]], v1[q[i, 1]]]
        ratios = policy.logged_action_probabilities(batch) / mu_logged
        weights = np.cumprod(ratios, axis=1)
        current = weights / weights.sum(axis=0)
        previous = np.column_stack((np.full(8, 1 / 8), current[:, 0]))
        expected = np.sum(
            [1.0, 0.5] * np.sum(current * (rewards - q_logged) + previous * v_current, axis=0)
        )
        assert observed[policy_index]["wdr"] == pytest.approx(expected)
        assert observed[policy_index]["dm"] == pytest.approx(np.mean(v_current[:, 0]))


def test_mis_uses_global_row_average_and_retains_undefined_support() -> None:
    assert state_action_mis(
        [[1.0, 2.0], [3.0, 4.0]], [[0.2, 0.1], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.2]], gamma=0.5
    ) == pytest.approx(4.75)
    for target in (0.0, 1.0):
        with pytest.raises(StatisticalSupportError):
            state_action_mis([[1.0]], [[target]], [[0.0]], gamma=1.0)


def _interval() -> PairedWPDISBootstrap:
    return PairedWPDISBootstrap(
        normalization=2.0,
        raw_left=1.0,
        raw_right=1.2,
        raw_contrast=0.2,
        raw_left_se=0.1,
        raw_right_se=0.1,
        raw_contrast_se=0.1,
        raw_lower=-0.2,
        raw_upper=0.6,
        normalized_left=0.5,
        normalized_right=0.6,
        normalized_contrast=0.1,
        normalized_left_se=0.05,
        normalized_right_se=0.05,
        normalized_contrast_se=0.05,
        normalized_lower=-0.1,
        normalized_upper=0.3,
        normalized_width=0.4,
    )


def test_ciphertext_interval_translates_in_normalized_units_and_includes_zero() -> None:
    bootstrap = _interval()
    lower, upper, decision = translate_cipher_interval(bootstrap, 0.4)
    assert (lower, upper) == pytest.approx((0.2, 0.6))
    assert decision == "positive"
    assert translate_cipher_interval(bootstrap, -0.4)[2] == "negative"
    assert translate_cipher_interval(bootstrap, 0.1)[2] == "abstain"
    assert translate_cipher_interval(bootstrap, 0.2)[2] == "abstain"
    with pytest.raises(ValueError):
        translate_cipher_interval(bootstrap, float("nan"))
