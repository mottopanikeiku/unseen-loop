from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from unseen_loop.ope.circuit import FixedPointScales, OPECircuitSpec, QuantizedTrajectoryTensors
from unseen_loop.ope.fhe import (
    CompiledOPECircuit,
    OPEFHEConformanceError,
    SanitizedOPECallEvidence,
    _requested_errors,
    calibration_inputset,
    compile_ope_circuit,
    decode_output,
    encode_quantized_inputs,
)
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec


def _fixture() -> tuple[OPECircuitSpec, TrajectoryBatch]:
    trajectory_spec = TrajectorySpec(
        trajectories=4,
        horizon=4,
        state_dim=6,
        action_count=2,
        state_min=(0.0,) * 6,
        state_max=(0.0,) * 6,
        reward_min=-1.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=6,
        degree=1,
        coefficients=((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * 2,
    )
    spec = OPECircuitSpec(
        trajectory_spec,
        policy,
        gamma=1.0,
        weight_clip=2.0,
        minimum_behavior_propensity=1.0,
        scales=FixedPointScales(state=2, coefficient=2, reciprocal=2, reward=2, discount=2),
    )
    actions = tuple(tuple((trajectory + step) % 2 for step in range(4)) for trajectory in range(4))
    batch = TrajectoryBatch(
        trajectory_spec,
        states=tuple(tuple((0.0,) * 6 for _ in range(4)) for _ in range(4)),
        actions=actions,
        rewards=(
            (1.0, 0.5, -0.5, -1.0),
            (0.5, 1.0, -1.0, -0.5),
            (-0.5, -1.0, 1.0, 0.5),
            (-1.0, -0.5, 0.5, 1.0),
        ),
        behavior_propensities=tuple((1.0, 1.0, 1.0, 1.0) for _ in range(4)),
    )
    return spec, batch


def _uncompiled(spec: OPECircuitSpec, circuit: Any | None = None) -> CompiledOPECircuit:
    return CompiledOPECircuit(
        spec,
        Path("unused-server.zip"),
        Path("unused-client-specs.bin"),
        cast(Any, None),
        circuit,
    )


def test_integer_api_returns_exact_counts_and_client_only_division() -> None:
    spec, batch = _fixture()
    backend = _uncompiled(spec)

    pdis = backend.integer_reference(batch, "clipped_pdis")
    wpdis = backend.execute(batch, "clipped_wpdis", "INTEGER")

    assert pdis.mode == "INTEGER"
    assert pdis.integer_statistics.counts == (4, 4, 4, 4)
    assert len(pdis.integer_statistics.numerators) == 4
    assert len(pdis.integer_statistics.denominators) == 4
    assert pdis.client_statistics.denominators == (4.0, 4.0, 4.0, 4.0)
    assert wpdis.client_statistics.denominators != tuple(
        float(value) for value in wpdis.integer_statistics.denominators
    )
    assert pdis.integer_receipt.operations.encrypted_output_integers == 12
    assert pdis.integer_receipt.operations == spec.operation_counts()
    assert spec.operation_counts().comparisons == 16
    assert spec.operation_counts().multiplexers == 16


def test_fixed_transport_encoding_and_three_vector_decode() -> None:
    spec, batch = _fixture()
    tensors = spec.quantize_client_inputs(batch)
    encoded = encode_quantized_inputs(spec, tensors)
    expected = spec.evaluate_integer(tensors)
    output = np.asarray(
        [*expected.numerators, *expected.denominators, *expected.counts], dtype=np.int64
    )

    assert encoded.shape == (4 * 4 * (6 + 2 + 2),)
    assert decode_output(spec, output) == expected
    assert decode_output(spec, output.reshape(3, 4)) == expected
    with pytest.raises(ValueError, match="exactly three"):
        decode_output(spec, output[:-1])
    with pytest.raises(ValueError, match="exact integers"):
        decode_output(spec, output.astype(np.float64) + 0.25)


def test_simulation_must_conform_to_exact_integer_reference() -> None:
    spec, batch = _fixture()
    expected, _ = spec.integer_reference(batch)
    output = np.asarray(
        [*expected.numerators, *expected.denominators, *expected.counts], dtype=np.int64
    )

    class MatchingCircuit:
        def simulate(self, *arguments: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            assert tuple(argument.shape for argument in arguments) == (
                (4, 4, 6),
                (4, 4, 2),
                (4, 4),
                (4, 4),
            )
            return output

    result = _uncompiled(spec, MatchingCircuit()).simulate(batch, "clipped_wpdis")
    assert result.mode == "SIMULATION"
    assert result.integer_statistics == expected

    class WrongCircuit:
        def simulate(self, *arguments: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            assert len(arguments) == 4
            wrong = output.copy()
            wrong[0] += 1
            return wrong

    with pytest.raises(OPEFHEConformanceError, match="disagrees"):
        _uncompiled(spec, WrongCircuit()).simulate(batch, "clipped_pdis")


def test_calibration_is_complete_and_not_sampled() -> None:
    spec, _ = _fixture()
    calibration, state_points, strategy = calibration_inputset(spec)

    # One quantized state, two selected actions, two reward extrema, and the
    # single reciprocal endpoint are represented over every trajectory/time lane.
    assert state_points == 1
    assert calibration.shape == (4, 160)
    assert "complete quantized state domain" in strategy
    assert set(calibration[:, 96:128].sum(axis=1)) == {16}
    assert set(calibration[:, 128:144].reshape(-1)) == {-2, 2}
    assert set(calibration[:, 144:160].reshape(-1)) == {2}
    with pytest.raises(ValueError, match="above the 3-row cap"):
        calibration_inputset(spec, max_rows=3)


def test_canary_and_quantized_domains_are_rejected_before_backend_use() -> None:
    spec, batch = _fixture()
    tensors = spec.quantize_client_inputs(batch)
    malformed = QuantizedTrajectoryTensors(
        (
            ((1, *tensors.states[0][0][1:]), *tensors.states[0][1:]),
            *tensors.states[1:],
        ),
        tensors.action_masks,
        tensors.rewards,
        tensors.behavior_reciprocals,
    )
    with pytest.raises(ValueError, match="outside the compiled domain"):
        encode_quantized_inputs(spec, malformed)

    wrong_shape = TrajectorySpec(
        trajectories=1,
        horizon=4,
        state_dim=6,
        action_count=2,
        state_min=(0.0,) * 6,
        state_max=(0.0,) * 6,
        reward_min=-1.0,
        reward_max=1.0,
    )
    wrong_spec = OPECircuitSpec(
        wrong_shape,
        spec.target_policy,
        scales=spec.scales,
        minimum_behavior_propensity=0.5,
    )
    with pytest.raises(ValueError, match="restricted"):
        calibration_inputset(wrong_spec)


def test_requested_error_contract_is_unambiguous_and_complete() -> None:
    assert _requested_errors(None, None) == (None, 1e-6)
    assert _requested_errors(1e-5, None) == (1e-5, None)
    assert _requested_errors(None, 1e-7) == (None, 1e-7)
    with pytest.raises(ValueError, match="either"):
        _requested_errors(1e-5, 1e-6)
    with pytest.raises(ValueError, match="positive"):
        _requested_errors(0.0, None)


def test_call_evidence_json_cannot_persist_private_vectors() -> None:
    evidence = SanitizedOPECallEvidence(
        input_shape=(160,),
        output_shape=(3, 4),
        encrypted_output_vectors=3,
        integers_per_output_vector=4,
        keygen_ns=1,
        encrypt_ns=2,
        server_evaluate_ns=3,
        decrypt_ns=4,
        end_to_end_ns=10,
        evaluation_key_bytes=100,
        request_bytes=200,
        response_bytes=300,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        output_matches_integer_reference=True,
        server_secret_key_marker_present=False,
    )
    payload = evidence.to_json()

    assert "states" not in payload
    assert "rewards" not in payload
    assert "numerators" not in payload
    assert "denominators" not in payload
    assert "behavior_propensities" not in payload


@pytest.mark.fhe
@pytest.mark.slow
def test_real_serialized_roundtrip_conforms_when_concrete_is_installed(tmp_path: Path) -> None:
    if os.environ.get("UNSEEN_LOOP_RUN_OPE_FHE") != "1":
        pytest.skip("set UNSEEN_LOOP_RUN_OPE_FHE=1 for the expensive Concrete canary")
    pytest.importorskip("concrete.fhe")
    spec, batch = _fixture()

    compiled = compile_ope_circuit(spec, tmp_path, global_p_error=1e-3)
    simulated = compiled.simulate(batch, "clipped_wpdis")
    real = compiled.real_roundtrip(batch, "clipped_wpdis")

    assert simulated.integer_statistics == real.integer_statistics
    assert real.call_evidence is not None
    assert real.call_evidence.output_matches_integer_reference
    assert not real.call_evidence.server_secret_key_marker_present
    assert compiled.receipt.requested_global_p_error == 1e-3
    assert 0 <= compiled.receipt.compiled_p_error <= 1
    assert 0 <= compiled.receipt.compiled_global_p_error <= 1
    assert compiled.receipt.security_level == 128
    assert compiled.receipt.encrypted_output_vectors == 3
    assert compiled.receipt.integers_per_output_vector == 4
