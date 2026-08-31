from __future__ import annotations

import math

import pytest

from unseen_loop.flagship.integration import (
    ExperimentPlan,
    FrozenRequestedPolicy,
    Outcome,
    ShieldVariant,
    TrajectoryKind,
    behavior_probabilities,
    bootstrap_paired_effect,
    build_ope_batch,
    run_paired_online_truth,
    run_trajectory,
)
from unseen_loop.shield.types import Action, SafetyLimits, ScenarioSpec, ShieldState


def _policy() -> FrozenRequestedPolicy:
    return FrozenRequestedPolicy.constant((0.05, 0.30, 0.25, 0.20, 0.20))


def _all_unsafe_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        initial_state=ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        goal=(8.0, 8.0),
        safety=SafetyLimits(min_battery=10.0),
        max_steps=64,
    )


def test_production_plan_has_frozen_counts_and_ope_shapes() -> None:
    plan = ExperimentPlan()

    assert len(plan.scenario_ids) == 12
    assert plan.shields == (ShieldVariant.OFF, ShieldVariant.H1, ShieldVariant.H2)
    assert plan.behavior_trajectory_count == 12 * 3 * 4_096
    assert plan.direct_trajectory_count == 12 * 3 * 2_048
    assert plan.ope_call_count == 1_152
    assert len(plan.trajectory_plans(TrajectoryKind.BEHAVIOR)) == 12 * 3
    assert len(plan.trajectory_plans(TrajectoryKind.DIRECT)) == 12 * 3
    assert len(plan.ope_calls()) == 1_152
    assert {call.shape for call in plan.ope_calls()} == {(256, 64)}
    assert {call.outcome for call in plan.ope_calls()} == set(Outcome)


def test_behavior_is_exact_half_target_half_uniform() -> None:
    target = _policy().probabilities((0.0,) * 6)
    behavior = behavior_probabilities(target)

    assert sum(behavior) == pytest.approx(1.0)
    assert behavior == pytest.approx(tuple(0.5 * value + 0.1 for value in target))
    assert min(behavior) >= 0.1


def test_many_requested_actions_map_to_one_execution_without_propensity_coalescing() -> None:
    policy = _policy()
    scenario = _all_unsafe_scenario()
    logs = tuple(
        run_trajectory(
            "unsafe",
            scenario,
            policy,
            ShieldVariant.H2,
            TrajectoryKind.BEHAVIOR,
            trajectory_index=index,
            seed=index,
            horizon=1,
        )
        for index in range(32)
    )
    requested = {log.steps[0].requested_action for log in logs}

    assert len(requested - {Action.BRAKE}) >= 2
    assert {log.steps[0].executed_action for log in logs} == {Action.BRAKE}
    for log in logs:
        step = log.steps[0]
        pi = policy.probabilities(step.state)
        assert step.mu_propensity == pytest.approx(0.5 * pi[int(step.requested_action)] + 0.1)

    prepared = build_ope_batch(logs, policy, Outcome.RETURN)
    # OPE uses requested actions and their mu propensities, not the many-to-one
    # executed action or an invented propensity for it.
    assert set(prepared.trajectories.action_array[:, 0]) == {int(action) for action in requested}
    assert set(prepared.trajectories.action_array[:, 0]) != {int(Action.BRAKE)}
    assert prepared.trajectories.behavior_array[:, 0] == pytest.approx(
        [log.steps[0].mu_propensity for log in logs]
    )
    assert [row[0] for row in prepared.target_propensities] == pytest.approx(
        [
            policy.probabilities(log.steps[0].state)[int(log.steps[0].requested_action)]
            for log in logs
        ]
    )


def test_return_and_unsafe_batches_share_requested_action_semantics() -> None:
    policy = _policy()
    scenario = _all_unsafe_scenario()
    logs = tuple(
        run_trajectory(
            "unsafe",
            scenario,
            policy,
            ShieldVariant.H2,
            TrajectoryKind.BEHAVIOR,
            trajectory_index=index,
            seed=index,
            horizon=2,
        )
        for index in range(4)
    )

    return_batch = build_ope_batch(logs, policy, Outcome.RETURN)
    unsafe_batch = build_ope_batch(logs, policy, Outcome.UNSAFE_STEPS)

    assert return_batch.trajectories.actions == unsafe_batch.trajectories.actions
    assert (
        return_batch.trajectories.behavior_propensities
        == unsafe_batch.trajectories.behavior_propensities
    )
    assert unsafe_batch.trajectories.rewards == ((1.0, 1.0),) * 4


def test_paired_online_truth_and_effect_bootstrap_keep_common_replicates() -> None:
    policy = _policy()
    scenario = _all_unsafe_scenario()
    truth = tuple(
        run_paired_online_truth(
            "unsafe",
            scenario,
            policy,
            replicate=index,
            seed=100 + index,
            horizon=2,
        )
        for index in range(4)
    )

    for pair in truth:
        assert tuple(outcome.shield for outcome in pair.outcomes) == tuple(ShieldVariant)
        assert {outcome.seed for outcome in pair.outcomes} == {pair.seed}
        assert {outcome.replicate for outcome in pair.outcomes} == {pair.replicate}

    effect = bootstrap_paired_effect(
        truth,
        ShieldVariant.H2,
        samples=32,
        seed=7,
        confidence=0.8,
    )
    expected_returns = [
        pair.for_shield(ShieldVariant.H2).total_return
        - pair.for_shield(ShieldVariant.OFF).total_return
        for pair in truth
    ]
    expected_unsafe = [
        pair.for_shield(ShieldVariant.H2).unsafe_steps
        - pair.for_shield(ShieldVariant.OFF).unsafe_steps
        for pair in truth
    ]
    assert effect.return_effect.estimate == pytest.approx(sum(expected_returns) / len(truth))
    assert effect.unsafe_step_effect.estimate == pytest.approx(sum(expected_unsafe) / len(truth))
    assert math.isfinite(effect.return_effect.lower)
    assert math.isfinite(effect.return_effect.upper)
