from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unseen_loop.shield.environment import candidate_rollouts, safety_report
from unseen_loop.shield.fhe import (
    DOMAIN_POINTS,
    FAMILY_ORDER,
    MARGIN_SHAPE,
    OUTPUT_ORDER,
    CompiledShield,
    ShieldFHEClient,
    ShieldFHEMode,
    ShieldIntegerSpec,
    StateQuantizer,
    clear_margin_tensor,
    compile_shield,
    exhaustive_inputset,
    exhaustive_simulation_conformance,
    integer_margin_program,
    real_fhe_canary,
)
from unseen_loop.shield.types import Action, Obstacle, SafetyLimits, ShieldState


def test_exhaustive_inputset_is_complete_qmax2_domain() -> None:
    domain = exhaustive_inputset()

    assert domain.shape == (15_625, 6)
    assert DOMAIN_POINTS == 15_625
    assert np.array_equal(domain[0], np.full(6, -2))
    assert np.array_equal(domain[-1], np.full(6, 2))
    assert len({tuple(row) for row in domain}) == DOMAIN_POINTS
    assert np.array_equal(np.min(domain, axis=0), np.full(6, -2))
    assert np.array_equal(np.max(domain, axis=0), np.full(6, 2))


def test_exact_integer_oracle_matches_frozen_saturated_margin_spec() -> None:
    spec = ShieldIntegerSpec(
        limits=SafetyLimits(
            obstacles=(Obstacle(x=1.0, y=-2.0, radius=0.75),),
            max_speed=3.0,
            max_abs_tilt=0.75,
            min_battery=-0.25,
            x_bounds=(-12.0, 12.0),
            y_bounds=(-11.0, 11.0),
            vehicle_radius=0.25,
            obstacle_clearance=0.125,
        )
    )
    program = integer_margin_program(spec)
    quantized = np.asarray([1, -1, 1, 0, 1, -1], dtype=np.int64)
    integer_margins = clear_margin_tensor(spec, quantized, program=program)
    state = spec.quantizer.dequantize(quantized)

    expected = np.empty(MARGIN_SHAPE, dtype=np.float64)
    for action, rollout in zip(Action, candidate_rollouts(state, spec.dynamics), strict=True):
        for horizon, future in enumerate(rollout):
            report = safety_report(future, spec.limits)
            spatial = min((*report.boundary_margins, *report.obstacle_margins))
            expected[int(action), horizon] = (
                spatial,
                report.speed_margin,
                report.tilt_margin,
                report.battery_margin,
            )
    expected[..., 0] = np.clip(
        expected[..., 0],
        -program.spatial_margin_clip / program.margin_scale,
        program.spatial_margin_clip / program.margin_scale,
    )

    assert integer_margins.shape == MARGIN_SHAPE
    assert np.allclose(integer_margins / program.margin_scale, expected, rtol=0, atol=1e-12)


def test_output_encoding_places_all_margins_in_one_unsigned_bit_width() -> None:
    spec = ShieldIntegerSpec()
    program = integer_margin_program(spec)
    encoded = np.asarray(
        [
            clear_margin_tensor(spec, quantized, program=program)
            + program.output_encoding_offset
            for quantized in exhaustive_inputset()
        ],
        dtype=np.int64,
    )

    assert int(encoded.min()) > 0
    bit_widths = {int(value).bit_length() for value in encoded.reshape(-1)}
    assert len(bit_widths) == 1


def test_default_spatial_pruning_proves_one_active_obstacle_per_output() -> None:
    program = integer_margin_program(ShieldIntegerSpec())

    assert program.spatial_constraints == 1
    assert program.spatial_active_indices == ((4,),) * 10


def test_spec_digest_binds_public_spec_and_output_order() -> None:
    original = ShieldIntegerSpec()
    changed = replace(
        original, quantizer=replace(original.quantizer, steps=(4.0, 5.0, 1.25, 1.25, 0.25, 0.25))
    )

    assert original.spec_digest != changed.spec_digest
    assert original.canonical()["output_order"] == list(OUTPUT_ORDER)
    assert original.canonical()["family_order"] == list(FAMILY_ORDER)
    assert "state" not in original.canonical()


def test_quantizer_rejects_non_protocol_domain_without_clipping() -> None:
    quantizer = StateQuantizer()

    with pytest.raises(ValueError, match="outside"):
        quantizer.quantize(ShieldState(20.0, 0.0, 0.0, 0.0, 0.5, 0.0))
    with pytest.raises(ValueError, match="qmax=2"):
        StateQuantizer(qmax=3)


def _selection_client(spec: ShieldIntegerSpec) -> ShieldFHEClient:
    # Selection is a clear client responsibility and does not require constructing
    # Concrete client keys.  This deliberately bypasses only the crypto constructor.
    client = object.__new__(ShieldFHEClient)
    client.spec = spec
    client._program = integer_margin_program(spec)
    return client


def test_client_selection_uses_requested_then_stable_core_order() -> None:
    spec = ShieldIntegerSpec()
    client = _selection_client(spec)
    margins = np.ones(MARGIN_SHAPE, dtype=np.int64)
    margins[int(Action.EAST), 0, 0] = 0

    requested_safe = client.select_action(margins, Action.NORTH)
    assert requested_safe.action is Action.NORTH
    assert requested_safe.reason == "requested_certified"

    # WEST and NORTH tie as safest certified alternatives.  The shared core
    # selector resolves the tie by frozen Action enum order, selecting WEST.
    margins[:] = 1
    margins[int(Action.EAST), 0, 0] = 0
    margins[int(Action.WEST)] = 7
    margins[int(Action.NORTH)] = 7
    alternative = client.select_action(margins, Action.EAST)
    assert alternative.action is Action.WEST
    assert alternative.reason == "safest_certified_alternative"


def test_simulation_conformance_checks_every_domain_point() -> None:
    class ExactSimulation:
        spec = ShieldIntegerSpec()
        program = integer_margin_program(spec)

        def clear(self, quantized: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
            return clear_margin_tensor(self.spec, quantized, program=self.program)

        def simulate(self, quantized: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
            return self.clear(quantized)

    result = exhaustive_simulation_conformance(ExactSimulation(), workers=4)  # type: ignore[arg-type]

    assert result.mode is ShieldFHEMode.SIMULATION
    assert result.domain_points == 15_625
    assert result.matches == 15_625
    assert result.mismatches == 0
    assert result.exact
    assert result.clear_outputs_sha256 == result.simulated_outputs_sha256


class _FakeClient:
    def __init__(
        self, margins: np.ndarray[Any, np.dtype[np.int64]], spec: ShieldIntegerSpec
    ) -> None:
        self._margins = margins
        self._selector = _selection_client(spec)

    def quantize(self, state: ShieldState) -> np.ndarray[Any, np.dtype[np.int64]]:
        del state
        return np.zeros(6, dtype=np.int64)

    def generate_keys(self) -> tuple[int, bytes]:
        return 11, b"public-evaluation-material"

    def encrypt(self, quantized: Any) -> bytes:
        del quantized
        return b"ciphertext-request"

    def decrypt_margin_tensor(self, response: bytes) -> np.ndarray[Any, np.dtype[np.int64]]:
        assert response == b"ciphertext-response"
        return self._margins

    def select_action(self, margins: Any, requested_action: Action, *, error_buffer: Any) -> Any:
        return self._selector.select_action(margins, requested_action, error_buffer=error_buffer)


class _FakeServer:
    def evaluate(self, request: bytes, evaluation_keys: bytes) -> bytes:
        assert request == b"ciphertext-request"
        assert evaluation_keys == b"public-evaluation-material"
        return b"ciphertext-response"


def test_real_canary_keeps_private_state_and_margins_out_of_receipt() -> None:
    spec = ShieldIntegerSpec()
    margins = clear_margin_tensor(spec, np.zeros(6, dtype=np.int64))

    class FakeCompiled:
        receipt = type("Receipt", (), {"server_secret_key_markers": ()})()

        def client(self) -> _FakeClient:
            return _FakeClient(margins, spec)

        def server(self) -> _FakeServer:
            return _FakeServer()

        def clear(self, quantized: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
            return clear_margin_tensor(spec, quantized)

    private = ShieldState(0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
    result = real_fhe_canary(FakeCompiled(), private, Action.BRAKE)  # type: ignore[arg-type]
    payload = json.loads(result.call.to_json())

    assert result.call.mode is ShieldFHEMode.REAL
    assert result.call.output_matches_clear
    assert result.call.request_bytes == len(b"ciphertext-request")
    assert result.call.response_bytes == len(b"ciphertext-response")
    assert result.call.evaluation_key_bytes == len(b"public-evaluation-material")
    assert result.call.server_secret_key_marker_present is False
    assert "state" not in payload
    assert "margin" not in payload
    assert "requested_action" not in payload
    assert "selected_action" not in payload
    assert "ciphertext-request" not in result.call.to_json()


def test_real_mode_never_aliases_clear_or_simulation() -> None:
    fake = object.__new__(CompiledShield)
    fake.spec = ShieldIntegerSpec()
    fake.program = integer_margin_program(fake.spec)
    fake.circuit = None

    with pytest.raises(ValueError, match="real_canary"):
        fake.evaluate(np.zeros(6, dtype=np.int64), ShieldFHEMode.REAL)


@pytest.mark.fhe
@pytest.mark.slow
def test_concrete_complete_domain_and_serialized_real_canary(tmp_path: Path) -> None:
    if os.environ.get("UNSEEN_LOOP_RUN_SHIELD_FHE") != "1":
        pytest.skip("set UNSEEN_LOOP_RUN_SHIELD_FHE=1 for the expensive Concrete canary")
    pytest.importorskip("concrete.fhe")

    compiled = compile_shield(ShieldIntegerSpec(), tmp_path)
    conformance = exhaustive_simulation_conformance(compiled)
    canary = real_fhe_canary(
        compiled,
        ShieldState(0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        Action.BRAKE,
    )

    assert conformance.exact
    assert compiled.receipt.spec_digest == compiled.spec.spec_digest
    assert compiled.receipt.output_order == OUTPUT_ORDER
    assert compiled.receipt.compiled_p_error == compiled.circuit.server.p_error
    assert compiled.receipt.compiled_global_p_error == compiled.circuit.server.global_p_error
    assert compiled.receipt.server_secret_key_markers == ()
    assert canary.call.output_matches_clear
