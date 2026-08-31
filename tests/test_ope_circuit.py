from __future__ import annotations

import math

import pytest

from unseen_loop.ope.circuit import (
    FixedPointScales,
    OPECircuitSpec,
    QuantizedTrajectoryTensors,
)
from unseen_loop.ope.estimators import clipped_pdis, wpdis
from unseen_loop.ope.types import (
    OPEValidationError,
    PolynomialPolicySpec,
    TrajectoryBatch,
    TrajectorySpec,
)


def _fixture() -> tuple[OPECircuitSpec, TrajectoryBatch]:
    trajectory_spec = TrajectorySpec(
        trajectories=2,
        horizon=3,
        state_dim=1,
        action_count=2,
        state_min=(-1.0,),
        state_max=(1.0,),
        reward_min=-2.0,
        reward_max=2.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.6, 0.1), (0.4, -0.1)),
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states=(((-1.0,), (0.25,), (0.75,)), ((1.0,), (-0.5,), (0.0,))),
        actions=((0, 1, 0), (1, 0, 1)),
        rewards=((1.25, -0.5, 2.0), (0.5, 1.0, -1.25)),
        behavior_propensities=((0.5, 0.4, 0.7), (0.3, 0.6, 0.5)),
    )
    circuit = OPECircuitSpec(
        trajectories=trajectory_spec,
        target_policy=policy,
        gamma=0.9,
        weight_clip=2.5,
        minimum_behavior_propensity=0.25,
        scales=FixedPointScales(
            state=1 << 10,
            coefficient=1 << 16,
            reciprocal=1 << 14,
            reward=1 << 14,
            discount=1 << 14,
        ),
    )
    return circuit, batch


def test_integer_dag_matches_its_fixed_point_definition_exactly() -> None:
    circuit, batch = _fixture()
    tensors = circuit.quantize_client_inputs(batch)
    statistics = circuit.evaluate_integer(tensors)

    expected_numerators = [0, 0, 0]
    expected_denominators = [0, 0, 0]
    gamma_integer = int(circuit.gamma * circuit.scales.discount + 0.5)
    for trajectory in range(batch.spec.trajectories):
        raw_weight = 1
        for step in range(batch.spec.horizon):
            state = tensors.states[trajectory][step][0]
            aligned_features = (circuit.scales.state, state)
            scores = tuple(
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(row, aligned_features, strict=True)
                )
                for row in circuit.coefficient_integers
            )
            target = sum(
                mask * score
                for mask, score in zip(tensors.action_masks[trajectory][step], scores, strict=True)
            )
            raw_weight *= target * tensors.behavior_reciprocals[trajectory][step]
            weight_scale = circuit.ratio_scale ** (step + 1)
            clipped = min(raw_weight, int(circuit.weight_clip * weight_scale + 0.5))
            expected_denominators[step] += clipped
            expected_numerators[step] += (
                clipped * tensors.rewards[trajectory][step] * gamma_integer**step
            )

    assert statistics.numerators == tuple(expected_numerators)
    assert statistics.denominators == tuple(expected_denominators)
    assert statistics.counts == (2, 2, 2)


def test_client_estimates_match_clear_estimators_within_receipt_bounds() -> None:
    circuit, batch = _fixture()
    integer_statistics, receipt = circuit.integer_reference(batch)

    integer_pdis = circuit.client_statistics(integer_statistics, "clipped_pdis")
    integer_wpdis = circuit.client_statistics(integer_statistics, "clipped_wpdis")
    clear_pdis = circuit.clear_statistics(batch, "clipped_pdis")
    clear_wpdis = circuit.clear_statistics(batch, "clipped_wpdis")
    target = circuit.target_policy.logged_action_probabilities(batch)
    assert clear_pdis.estimate == pytest.approx(
        clipped_pdis(
            batch,
            target,
            gamma=circuit.gamma,
            weight_clip=circuit.weight_clip,
        )
    )
    assert clear_wpdis.estimate == pytest.approx(
        wpdis(
            batch,
            target,
            gamma=circuit.gamma,
            weight_clip=circuit.weight_clip,
        )
    )

    assert integer_pdis.estimate is not None
    assert integer_wpdis.estimate is not None
    assert clear_pdis.estimate is not None
    assert clear_wpdis.estimate is not None
    assert abs(integer_pdis.estimate - clear_pdis.estimate) <= (receipt.error.clipped_pdis + 1e-15)
    assert receipt.error.self_normalized_wpdis is not None
    assert abs(integer_wpdis.estimate - clear_wpdis.estimate) <= (
        receipt.error.self_normalized_wpdis + 1e-15
    )
    assert receipt.error.max_logged_probability_error > 0
    assert receipt.error.max_reciprocal_error > 0


def test_receipt_closes_scales_overflow_depth_and_payload_shape() -> None:
    circuit, batch = _fixture()
    statistics, receipt = circuit.integer_reference(batch)

    assert len(receipt.numerator_scales) == batch.spec.horizon
    assert len(receipt.denominator_scales) == batch.spec.horizon
    assert receipt.denominator_scales == tuple(
        circuit.ratio_scale ** (step + 1) for step in range(batch.spec.horizon)
    )
    assert all(
        abs(value) <= bound
        for value, bound in zip(statistics.numerators, receipt.numerator_abs_bounds, strict=True)
    )
    assert all(
        value <= bound
        for value, bound in zip(statistics.denominators, receipt.denominator_bounds, strict=True)
    )
    assert receipt.operations.multiplicative_depth == (
        batch.spec.horizon + circuit.target_policy.degree + 1
    )
    assert receipt.operations.encrypted_output_integers == 3 * batch.spec.horizon
    assert receipt.operations.comparisons == batch.spec.trajectories * batch.spec.horizon
    assert len(receipt.invalid_domains) == 7
    assert "Not a REAL FHE result" in receipt.trust_scope
    assert len(receipt.digest) == 64
    assert (
        receipt.digest
        == circuit.receipt(
            batch,
            circuit.quantize_client_inputs(batch),
            statistics,
        ).digest
    )


def test_behavior_support_domain_is_rejected_before_reciprocal() -> None:
    circuit, batch = _fixture()
    unsupported = TrajectoryBatch(
        batch.spec,
        batch.states,
        batch.actions,
        batch.rewards,
        ((0.5, 0.0, 0.7), (0.3, 0.6, 0.5)),
    )

    with pytest.raises(OPEValidationError) as caught:
        circuit.quantize_client_inputs(unsupported)

    assert caught.value.rows[0].code == "unsupported_behavior"
    assert caught.value.rows[0].trajectory == 0
    assert caught.value.rows[0].step == 1


def test_policy_must_be_proved_on_entire_closed_state_box() -> None:
    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=1,
        state_dim=1,
        action_count=2,
        state_min=(-1.0,),
        state_max=(1.0,),
        reward_min=0.0,
        reward_max=1.0,
    )
    unsafe = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.5, 0.75), (0.5, -0.75)),
    )

    with pytest.raises(ValueError, match="not proved inside"):
        OPECircuitSpec(trajectory_spec, unsafe)


def test_wpdis_zero_weight_is_an_explicit_invalid_domain() -> None:
    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=1,
        state_dim=1,
        action_count=2,
        state_min=(0.0,),
        state_max=(1.0,),
        reward_min=-1.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.0, 0.0), (1.0, 0.0)),
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states=(((0.5,),),),
        actions=((0,),),
        rewards=((1.0,),),
        behavior_propensities=((0.5,),),
    )
    circuit = OPECircuitSpec(trajectory_spec, policy, minimum_behavior_propensity=0.1)

    statistics, receipt = circuit.integer_reference(batch)
    wpdis = circuit.client_statistics(statistics, "clipped_wpdis")

    assert statistics.denominators == (0,)
    assert wpdis.estimate is None
    assert receipt.error.self_normalized_wpdis is None
    assert any("zero aggregate target weight" in reason for reason in receipt.invalid_domains)


def test_exact_clip_scaling_does_not_convert_large_horizon_scales_to_float() -> None:
    horizon = 24
    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=horizon,
        state_dim=1,
        action_count=2,
        state_min=(0.0,),
        state_max=(1.0,),
        reward_min=0.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.5, 0.0), (0.5, 0.0)),
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states=(tuple((0.5,) for _ in range(horizon)),),
        actions=(tuple(0 for _ in range(horizon)),),
        rewards=(tuple(1.0 for _ in range(horizon)),),
        behavior_propensities=(tuple(0.5 for _ in range(horizon)),),
    )
    circuit = OPECircuitSpec(
        trajectory_spec,
        policy,
        gamma=1.0,
        weight_clip=20.25,
        minimum_behavior_propensity=0.5,
    )

    statistics, receipt = circuit.integer_reference(batch)

    assert statistics.denominators[-1] == receipt.denominator_scales[-1]
    assert receipt.raw_weight_unsigned_bits[-1] > 1_000


def test_fixed_tensor_shape_and_one_hot_actions_are_part_of_the_contract() -> None:
    circuit, batch = _fixture()
    tensors = circuit.quantize_client_inputs(batch)

    assert len(tensors.states) == batch.spec.trajectories
    assert all(len(row) == batch.spec.horizon for row in tensors.states)
    assert all(sum(mask) == 1 for trajectory in tensors.action_masks for mask in trajectory)
    assert all(
        math.isfinite(value / circuit.scales.reciprocal)
        for trajectory in tensors.behavior_reciprocals
        for value in trajectory
    )

    # The server API accepts integer tensors only; a malformed action mask cannot
    # silently select two target actions.
    malformed = QuantizedTrajectoryTensors(
        tensors.states,
        (((1, 1), *tensors.action_masks[0][1:]), tensors.action_masks[1]),
        tensors.rewards,
        tensors.behavior_reciprocals,
    )
    with pytest.raises(ValueError, match="one-hot"):
        circuit.evaluate_integer(malformed)
