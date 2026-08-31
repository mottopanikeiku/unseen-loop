from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from unseen_loop.ope import (
    OPEValidationError,
    PolynomialPolicySpec,
    TrajectoryBatch,
    TrajectorySpec,
    bootstrap_ope,
    clipped_pdis,
    control_variate_sufficient_statistics,
    cumulative_importance_weights,
    direct_method_sufficient_statistics,
    effective_sample_size,
    ordinary_is,
    pdis,
    per_decision_effective_sample_size,
    validation_failures,
    wpdis,
    wpdis_sufficient_statistics,
)


def batch() -> TrajectoryBatch:
    spec = TrajectorySpec(
        trajectories=2,
        horizon=2,
        state_dim=1,
        action_count=2,
        state_min=(-2.0,),
        state_max=(2.0,),
        reward_min=-10.0,
        reward_max=10.0,
    )
    return TrajectoryBatch(
        spec=spec,
        states=(((-1.0,), (0.0,)), ((0.5,), (1.0,))),
        actions=((0, 1), (1, 0)),
        rewards=((1.0, 2.0), (3.0, 4.0)),
        behavior_propensities=((0.5, 0.5), (0.5, 0.25)),
    )


def target_logged() -> np.ndarray:
    return np.asarray(((1.0, 0.5), (0.25, 0.5)), dtype=np.float64)


def target_all_actions() -> np.ndarray:
    return np.asarray(
        (
            ((1.0, 0.0), (0.5, 0.5)),
            ((0.75, 0.25), (0.5, 0.5)),
        ),
        dtype=np.float64,
    )


def test_is_pdis_clipping_and_wpdis_match_closed_form_definitions() -> None:
    logged = target_logged()
    np.testing.assert_allclose(
        cumulative_importance_weights(batch(), logged), ((2.0, 2.0), (0.5, 1.0))
    )
    assert ordinary_is(batch(), logged) == pytest.approx(6.5)
    assert pdis(batch(), logged) == pytest.approx(5.75)
    assert clipped_pdis(batch(), logged, weight_clip=1.0) == pytest.approx(4.25)
    assert pdis(batch(), target_all_actions()) == pytest.approx(5.75)

    statistics = wpdis_sufficient_statistics(batch(), logged)
    assert statistics.numerators == pytest.approx((3.5, 8.0))
    assert statistics.denominators == pytest.approx((2.5, 3.0))
    assert statistics.counts == (2, 2)
    assert statistics.estimate == pytest.approx(3.5 / 2.5 + 8.0 / 3.0)
    assert wpdis(batch(), logged) == pytest.approx(statistics.estimate)
    assert type(statistics).from_json(statistics.to_json()) == statistics


def test_reduction_identities_for_on_policy_and_one_step_data() -> None:
    logged = batch().behavior_array
    empirical_return = np.mean(np.sum(batch().reward_array, axis=1))
    assert ordinary_is(batch(), logged) == pytest.approx(empirical_return)
    assert pdis(batch(), logged) == pytest.approx(empirical_return)
    assert clipped_pdis(batch(), logged, weight_clip=1.0) == pytest.approx(empirical_return)
    assert wpdis(batch(), logged) == pytest.approx(empirical_return)

    one_step = TrajectoryBatch(
        TrajectorySpec(2, 1, 1, 2),
        (((0.0,),), ((1.0,),)),
        ((0,), (1,)),
        ((2.0,), (4.0,)),
        ((0.5,), (0.5,)),
    )
    target = ((0.25,), (1.0,))
    assert ordinary_is(one_step, target) == pytest.approx(pdis(one_step, target))


def test_clipping_caps_cumulative_not_one_step_weights() -> None:
    logged = np.asarray(((1.0, 1.0), (0.5, 0.5)))
    clipped = cumulative_importance_weights(batch(), logged, weight_clip=1.5)
    # First trajectory has ratios (2, 2): cumulative (2, 4), then each product is capped.
    np.testing.assert_allclose(clipped[0], (1.5, 1.5))
    with pytest.raises(ValueError, match="positive"):
        clipped_pdis(batch(), logged, weight_clip=0.0)


def test_zero_support_and_zero_weight_denominator_are_observable() -> None:
    zero_behavior = TrajectoryBatch(
        TrajectorySpec(1, 2, 1, 2),
        (((0.0,), (0.0,)),),
        ((0, 0),),
        ((1.0, 1.0),),
        ((0.0, 0.5),),
    )
    rows = validation_failures(zero_behavior, ((0.25, 0.5),))
    assert [row.code for row in rows] == ["support_violation"]
    assert rows[0].trajectory == 0 and rows[0].step == 0
    with pytest.raises(OPEValidationError) as caught:
        pdis(zero_behavior, ((0.25, 0.5),))
    assert caught.value.rows == rows

    zero_target = ((1.0, 0.0), (1.0, 0.0))
    statistics = wpdis_sufficient_statistics(batch(), zero_target)
    assert statistics.estimate is None
    assert [row.code for row in statistics.failures] == ["zero_weight_denominator"]
    with pytest.raises(OPEValidationError, match="zero_weight_denominator"):
        wpdis(batch(), zero_target)


def test_shapes_domains_and_propensity_ranges_return_failure_rows() -> None:
    with pytest.raises(OPEValidationError) as caught:
        TrajectoryBatch(
            TrajectorySpec(1, 1, 1, 2, reward_min=-1.0, reward_max=1.0),
            (((0.0,),),),
            ((2,),),
            ((2.0,),),
            ((1.25,),),
        )
    assert {row.field for row in caught.value.rows} == {
        "actions",
        "rewards",
        "behavior_propensities",
    }

    with pytest.raises(OPEValidationError) as fractional:
        TrajectoryBatch(
            TrajectorySpec(1, 1, 1, 2),
            (((0.0,),),),
            ((0.5,),),  # type: ignore[arg-type]
            ((0.0,),),
            ((0.5,),),
        )
    assert [row.code for row in fractional.value.rows] == ["invalid_integer"]

    with pytest.raises(OPEValidationError) as target_error:
        pdis(batch(), ((-0.1, 0.5), (0.25, 1.1)))
    assert {row.code for row in target_error.value.rows} == {"out_of_range"}
    invalid_distribution = target_all_actions()
    invalid_distribution[0, 0] = (0.8, 0.3)
    with pytest.raises(OPEValidationError) as distribution_error:
        pdis(batch(), invalid_distribution)
    assert {row.code for row in distribution_error.value.rows} == {
        "invalid_probability_distribution"
    }
    with pytest.raises(OPEValidationError, match="shape_mismatch"):
        pdis(batch(), ((0.5,),))


def test_policy_and_trajectory_contracts_are_immutable_and_json_round_trip() -> None:
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.5, 0.1), (0.5, -0.1)),
    )
    np.testing.assert_allclose(
        policy.action_probabilities(batch().state_array)[0], ((0.4, 0.6), (0.5, 0.5))
    )
    assert PolynomialPolicySpec.from_json(policy.to_json()) == policy
    restored = TrajectoryBatch.from_json(batch().to_json())
    assert restored == batch()
    assert json.loads(restored.to_json())["spec"]["horizon"] == 2
    with pytest.raises(FrozenInstanceError):
        policy.degree = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        restored.rewards[0][0] = 99.0  # type: ignore[index]


def test_effective_sample_size_and_optional_nuisance_statistics() -> None:
    assert effective_sample_size((1.0, 1.0, 0.0)) == pytest.approx(2.0)
    assert effective_sample_size((0.0, 0.0)) == 0.0
    assert per_decision_effective_sample_size(batch(), target_logged()) == pytest.approx(
        (25.0 / 17.0, 9.0 / 5.0)
    )

    q_values = np.zeros((2, 2, 2), dtype=np.float64)
    control = control_variate_sufficient_statistics(batch(), target_all_actions(), q_values)
    assert control.estimate == pytest.approx(pdis(batch(), target_logged()))

    constant_q = np.full((2, 2, 2), 2.0)
    direct = direct_method_sufficient_statistics(batch(), target_all_actions(), constant_q)
    assert direct.estimate == pytest.approx(4.0)
    assert direct.counts == (2, 2)


def test_bootstrap_is_seeded_deterministic_and_serializable() -> None:
    first = bootstrap_ope(
        batch(),
        target_logged(),
        estimator="clipped_pdis",
        weight_clip=1.0,
        samples=40,
        seed=17,
        confidence=0.8,
    )
    second = bootstrap_ope(
        batch(),
        target_logged(),
        estimator="clipped_pdis",
        weight_clip=1.0,
        samples=40,
        seed=17,
        confidence=0.8,
    )
    assert first == second
    assert first.estimate == pytest.approx(4.25)
    assert first.lower <= first.estimate <= first.upper
    assert json.loads(first.to_json())["seed"] == 17
    assert type(first).from_json(first.to_json()) == first
