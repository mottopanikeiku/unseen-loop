from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unseen_loop.crypto.ckks import CKKSParameters
from unseen_loop.ope.ckks import (
    POLYNOMIAL_APPROX_OPE_V1,
    SOFT_CLIP_COEFFICIENTS,
    EncryptedOPERequest,
    OPECKKSClient,
    OPECKKSServer,
    PolynomialApproxOPESpec,
    clear_oracle,
    executable_ckks_parameters,
    generate_ope_contexts,
    plan_chunks,
)
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec


def test_executable_parameters_cover_h8_and_reject_unsupported_depth() -> None:
    parameters = executable_ckks_parameters(64, 8)

    assert parameters.poly_modulus_degree == 16_384
    assert len(parameters.coeff_mod_bit_sizes) == 16
    assert sum(parameters.coeff_mod_bit_sizes) == 416
    assert parameters.slot_capacity == 8_192
    with pytest.raises(ValueError, match="tc128 frontier"):
        executable_ckks_parameters(256, 32)


def _fixture(
    *, trajectories: int = 3, horizon: int = 2
) -> tuple[PolynomialApproxOPESpec, TrajectoryBatch]:
    trajectory_spec = TrajectorySpec(
        trajectories=trajectories,
        horizon=horizon,
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
    base_states = (-1.0, 0.0, 1.0)
    states = tuple(
        tuple(
            ((base_states[index % 3] + 0.25 * step) / (1.0 + 0.25 * step),)
            for step in range(horizon)
        )
        for index in range(trajectories)
    )
    actions = tuple(
        tuple((index + step) % 2 for step in range(horizon)) for index in range(trajectories)
    )
    rewards = tuple(
        tuple(((-1.0) ** (index + step)) * (0.5 + 0.1 * step) for step in range(horizon))
        for index in range(trajectories)
    )
    behavior = tuple((0.5,) * horizon for _ in range(trajectories))
    batch = TrajectoryBatch(trajectory_spec, states, actions, rewards, behavior)
    # min behavior=.5 gives a closed raw bound 2**H; this clip places it in [0, 2C].
    spec = PolynomialApproxOPESpec(
        trajectory_spec,
        policy,
        gamma=0.9,
        weight_clip=float(2 ** (horizon - 1)),
        minimum_behavior_propensity=0.5,
    )
    return spec, batch


def test_clear_oracle_uses_the_frozen_soft_polynomial_not_hard_clip() -> None:
    spec, batch = _fixture()

    wpdis = clear_oracle(spec, batch, "clipped_wpdis")
    pdis = clear_oracle(spec, batch, "clipped_pdis")

    target = spec.target_policy.logged_action_probabilities(batch)
    raw = np.cumprod(target / batch.behavior_array, axis=1)
    normalized = raw / spec.weight_clip
    soft = spec.weight_clip * (
        SOFT_CLIP_COEFFICIENTS[1] * normalized + SOFT_CLIP_COEFFICIENTS[2] * normalized**2
    )
    expected_numerators = tuple(
        float(spec.gamma**step * np.sum(soft[:, step] * batch.reward_array[:, step]))
        for step in range(batch.spec.horizon)
    )
    expected_denominators = tuple(
        float(np.sum(soft[:, step])) for step in range(batch.spec.horizon)
    )

    assert wpdis.numerators == pytest.approx(expected_numerators)
    assert wpdis.denominators == pytest.approx(expected_denominators)
    assert pdis.numerators == pytest.approx(expected_numerators)
    assert pdis.denominators == (3.0, 3.0)
    assert wpdis.counts == pdis.counts == (3, 3)
    assert PolynomialApproxOPESpec.soft_clip(1.0, 1.0) == pytest.approx(0.75)
    assert PolynomialApproxOPESpec.soft_clip(1.0, 1.0) != min(1.0, 1.0)


def test_receipt_closes_scale_depth_modulus_range_error_and_three_h_output() -> None:
    spec, _ = _fixture()
    parameters = CKKSParameters(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=(40, 20, 20, 20, 20, 20, 20, 20, 20, 40),
        global_scale=float(2**20),
    )

    receipt = spec.receipt(parameters)

    assert receipt.identifier == POLYNOMIAL_APPROX_OPE_V1
    assert receipt.scale_bits == 20
    assert receipt.required_multiplicative_depth == 8
    assert receipt.available_multiplicative_depth == 8
    assert receipt.depth_supported
    assert receipt.modulus_supported
    assert receipt.configured_modulus_bits == 240
    assert receipt.raw_weight_bounds == (2.0, 4.0)
    assert receipt.normalized_soft_clip_domain == (0.0, 2.0)
    assert receipt.soft_clip_coefficients == (0.0, 1.0, -0.25)
    assert receipt.soft_clip_absolute_error_bound == spec.weight_clip / 4
    assert receipt.output_ciphertexts == 3 * spec.trajectories.horizon
    assert receipt.required_security_level == "tc128"
    assert len(receipt.digest) == 64
    persisted = dataclasses.asdict(receipt)
    assert not {"plaintext", "states", "actions", "rewards", "secret_key"} & persisted.keys()


def test_depth_check_fails_closed_before_context_generation() -> None:
    spec, _ = _fixture()
    shallow = CKKSParameters()

    assert not spec.receipt(shallow).depth_supported
    with pytest.raises(ValueError, match="modulus chain provides"):
        spec.receipt(shallow).require_executable()


def test_lane_plan_accepts_4096_and_chunks_above_4096_without_oversized_vectors() -> None:
    parameters = CKKSParameters()

    exact = plan_chunks(4096, parameters)
    chunked = plan_chunks(4097, parameters)
    much_larger = plan_chunks(10_000, parameters)

    assert tuple((chunk.start, chunk.stop, chunk.slots) for chunk in exact.chunks) == (
        (0, 4096, 4096),
    )
    assert tuple((chunk.start, chunk.stop, chunk.slots) for chunk in chunked.chunks) == (
        (0, 4096, 4096),
        (4096, 4097, 1),
    )
    assert [chunk.slots for chunk in much_larger.chunks] == [4096, 4096, 1808]
    assert max(chunk.slots for chunk in much_larger.chunks) <= parameters.slot_capacity
    assert chunked.is_chunked


def test_request_contract_has_no_target_model_or_clear_private_log_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(EncryptedOPERequest)}

    assert field_names == {"chunks", "chunk_plan", "identifier"}
    assert not {"target_policy", "states", "actions", "rewards", "plaintext"} & field_names


def test_closed_weight_range_rejects_a_polynomial_outside_its_approximation_domain() -> None:
    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=3,
        state_dim=1,
        action_count=2,
        state_min=(-1.0,),
        state_max=(1.0,),
        reward_min=-1.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(2, 1, 1, ((0.5, 0.0), (0.5, 0.0)))

    with pytest.raises(ValueError, match="soft-clip domain"):
        PolynomialApproxOPESpec(
            trajectory_spec,
            policy,
            weight_clip=1.0,
            minimum_behavior_propensity=0.5,
        )


@pytest.mark.fhe
@pytest.mark.slow
def test_real_ckks_precision_matches_the_same_polynomial_clear_oracle() -> None:
    pytest.importorskip("tenseal")
    spec, batch = _fixture(horizon=1)
    parameters = CKKSParameters(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=(39, 20, 20, 20, 20, 20, 20, 20, 39),
        global_scale=float(2**20),
    )
    contexts = generate_ope_contexts(spec, parameters)
    client = OPECKKSClient.from_serialized(
        contexts.ckks.client_context, parameters=parameters, spec=spec
    )
    server = OPECKKSServer.from_serialized(
        contexts.ckks.server_context, parameters=parameters, spec=spec
    )

    request, encryption_receipt = client.encrypt_batch(batch)
    response, evaluation_receipt = server.evaluate(request)
    actual_wpdis, decryption_receipt = client.decrypt_statistics(response, "clipped_wpdis")
    actual_pdis, _ = client.decrypt_statistics(response, "clipped_pdis")
    expected_wpdis = spec.clear_oracle(batch, "clipped_wpdis")
    expected_pdis = spec.clear_oracle(batch, "clipped_pdis")

    assert actual_wpdis.numerators == pytest.approx(expected_wpdis.numerators, abs=5e-2)
    assert actual_wpdis.denominators == pytest.approx(expected_wpdis.denominators, abs=5e-2)
    assert actual_wpdis.estimate == pytest.approx(expected_wpdis.estimate, abs=5e-2)
    assert actual_pdis.estimate == pytest.approx(expected_pdis.estimate, abs=5e-2)
    assert actual_wpdis.counts == expected_wpdis.counts
    assert evaluation_receipt.output_ciphertexts == 3 * spec.trajectories.horizon
    assert encryption_receipt.input_sha256 is None
    assert decryption_receipt.output_sha256 is None
    assert contexts.ckks.receipt.effective_security_level == "tc128"
    assert not contexts.ckks.receipt.server_context_is_private
