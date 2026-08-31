from __future__ import annotations

from dataclasses import replace

import pytest

from unseen_loop.shield.certificate import (
    CandidateCertificate,
    ErrorBuffer,
    HorizonMargins,
    MarginFamily,
    SafetyMargins,
    certify_candidate,
)
from unseen_loop.shield.shield import (
    ShieldConfig,
    ShieldMode,
    compute_safety_margins,
    compute_shield_metrics,
    rollout_candidates,
    select_action,
    shield_step,
    state_safety_margins,
)
from unseen_loop.shield.types import Action, DynamicsConfig, Obstacle, SafetyLimits, ShieldState


def _horizon(value: float, *, horizon: int = 1) -> HorizonMargins:
    return HorizonMargins(
        horizon=horizon,
        margins=SafetyMargins(obstacle=value, speed=value, tilt=value, battery=value),
    )


def _certificate(action: Action, value: float) -> CandidateCertificate:
    return certify_candidate(action, (_horizon(value),))


def _certificates(values: dict[Action, float]) -> tuple[CandidateCertificate, ...]:
    return tuple(_certificate(action, values.get(action, -1.0)) for action in Action)


def _safe_limits(**overrides: object) -> SafetyLimits:
    values: dict[str, object] = {
        "max_speed": 10.0,
        "max_abs_tilt": 2.0,
        "min_battery": 0.0,
        "x_bounds": (-20.0, 20.0),
        "y_bounds": (-20.0, 20.0),
        "vehicle_radius": 0.0,
    }
    values.update(overrides)
    return SafetyLimits(**values)  # type: ignore[arg-type]


def test_counterfactual_rollout_is_two_steps_in_public_action_order() -> None:
    state = ShieldState(x=0.0, y=0.0, vx=0.0, vy=0.0, battery=1.0, tilt=0.0)
    dynamics = DynamicsConfig()

    rollouts = rollout_candidates(state, dynamics)

    assert tuple(item.action for item in rollouts) == tuple(Action)
    assert all(len(item.states) == 2 for item in rollouts)
    east = rollouts[Action.EAST]
    assert east.states[0].as_tuple() == pytest.approx((0.5, 0.0, 1.0, 0.0, 0.95, 0.0))
    assert east.states[1].as_tuple() == pytest.approx((2.0, 0.0, 1.9, 0.0, 0.9, 0.0))


def test_four_margin_families_use_strict_polynomial_safety_signs() -> None:
    limits = _safe_limits(
        obstacles=(Obstacle(x=0.0, y=0.0, radius=0.5),),
        max_speed=2.0,
        max_abs_tilt=0.5,
        min_battery=0.2,
    )
    state = ShieldState(x=1.0, y=0.0, vx=1.0, vy=1.0, battery=0.7, tilt=0.3)

    margins = state_safety_margins(state, limits)

    assert margins.obstacle == pytest.approx(0.75)
    assert margins.speed == pytest.approx(2.0)
    assert margins.tilt == pytest.approx(0.16)
    assert margins.battery == pytest.approx(0.5)


def test_certified_safe_requires_every_buffered_margin_strictly_positive() -> None:
    raw = HorizonMargins(
        horizon=1,
        margins=SafetyMargins(obstacle=0.3, speed=0.4, tilt=0.2, battery=0.1),
    )
    certificate = certify_candidate(
        Action.BRAKE,
        (raw,),
        error_buffer=ErrorBuffer(obstacle=0.1, speed=0.1, tilt=0.1, battery=0.1),
    )

    assert not certificate.certified
    assert certificate.failed_obligations == ((1, MarginFamily.BATTERY),)
    assert all(
        step.buffered.for_family(family) > 0
        for step in certificate.steps
        for family in certificate.active_families
        if certificate.certified
    )

    safe = certify_candidate(
        Action.BRAKE,
        (raw,),
        error_buffer=ErrorBuffer(obstacle=0.1, speed=0.1, tilt=0.1, battery=0.09),
    )
    assert safe.certified
    assert all(
        step.buffered.for_family(family) > 0
        for step in safe.steps
        for family in safe.active_families
    )


def test_selection_preserves_safe_request_and_breaks_alternative_ties_by_enum_order() -> None:
    certificates = _certificates(
        {
            Action.BRAKE: 0.5,
            Action.EAST: 0.5,
            Action.NORTH: 0.1,
        }
    )

    preserved = select_action(certificates, Action.NORTH)
    tied = select_action(certificates, Action.SOUTH)

    assert preserved.action is Action.NORTH
    assert preserved.reason == "requested_certified"
    assert tied.action is Action.BRAKE
    assert tied.reason == "safest_certified_alternative"
    assert tied.selected_certified


def test_selection_uses_explicit_uncertified_emergency_fallback() -> None:
    result = select_action(_certificates({}), Action.EAST, emergency_action=Action.BRAKE)

    assert result.action is Action.BRAKE
    assert result.emergency_fallback
    assert not result.selected_certified
    assert result.reason == "uncertified_emergency_fallback"


def test_baseline_and_ablation_modes_have_distinct_observable_semantics() -> None:
    state = ShieldState(x=0.0, y=0.0, vx=1.0, vy=0.0, battery=1.0, tilt=0.0)
    dynamics = DynamicsConfig()
    short_workspace = _safe_limits(x_bounds=(-10.0, 2.0))
    robust = shield_step(
        state,
        Action.EAST,
        step=0,
        dynamics=dynamics,
        limits=short_workspace,
        config=ShieldConfig(mode=ShieldMode.ROBUST),
    )
    one_step = shield_step(
        state,
        Action.EAST,
        step=0,
        dynamics=dynamics,
        limits=short_workspace,
        config=ShieldConfig(mode=ShieldMode.ONE_STEP_ABLATION),
    )
    east_robust = robust.candidates[Action.EAST]
    east_one_step = one_step.candidates[Action.EAST]
    assert len(east_robust.steps) == 2
    assert not east_robust.certified
    assert len(east_one_step.steps) == 1
    assert east_one_step.certified
    assert one_step.selected_action is Action.EAST

    large_buffer = ErrorBuffer(obstacle=100.0, speed=100.0, tilt=100.0, battery=100.0)
    robust_buffered = shield_step(
        ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        Action.NORTH,
        step=1,
        dynamics=dynamics,
        limits=_safe_limits(),
        config=ShieldConfig(mode=ShieldMode.ROBUST, error_buffer=large_buffer),
    )
    clear = shield_step(
        ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        Action.NORTH,
        step=1,
        dynamics=dynamics,
        limits=_safe_limits(),
        config=ShieldConfig(mode=ShieldMode.CLEAR_BASELINE, error_buffer=large_buffer),
    )
    assert robust_buffered.emergency_fallback
    assert clear.selected_action is Action.NORTH
    assert clear.selected_certified

    no_shield = shield_step(
        ShieldState(0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        Action.SOUTH,
        step=2,
        dynamics=dynamics,
        limits=_safe_limits(),
        config=ShieldConfig(mode=ShieldMode.NO_SHIELD_ABLATION),
    )
    assert no_shield.selected_action is Action.SOUTH
    assert no_shield.reason == "shield_disabled"
    assert not no_shield.selected_certified


def test_receipt_replays_exactly_and_never_serializes_private_state() -> None:
    kwargs = {
        "state": ShieldState(0.1, -0.2, 0.0, 0.0, 1.0, 0.0),
        "requested_action": Action.BRAKE,
        "step": 7,
        "dynamics": DynamicsConfig(),
        "limits": _safe_limits(),
    }
    first = shield_step(**kwargs)
    second = shield_step(**kwargs)

    first.verify()
    assert first == second
    assert first.receipt_digest == second.receipt_digest
    payload = first.to_dict()
    assert "state" not in payload
    assert "SafetyMargins" not in repr(first)
    candidate_payloads = payload["candidates"]
    assert isinstance(candidate_payloads, list)
    for candidate in candidate_payloads:
        assert isinstance(candidate, dict)
        private_fields = {"state", "raw", "buffer", "buffered", "minimum_buffered_margin"}
        assert not private_fields & candidate.keys()
        assert "selection_rank" in candidate

    tampered = replace(first, selected_action=Action.EAST)
    with pytest.raises(ValueError, match="does not replay"):
        tampered.verify()


def test_metrics_detect_false_safe_and_report_coverage_without_states() -> None:
    safe_receipt = shield_step(
        ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        Action.BRAKE,
        step=0,
        dynamics=DynamicsConfig(),
        limits=_safe_limits(),
    )
    fallback_receipt = shield_step(
        ShieldState(0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        Action.BRAKE,
        step=1,
        dynamics=DynamicsConfig(),
        limits=_safe_limits(),
    )

    metrics = compute_shield_metrics(
        (safe_receipt, fallback_receipt),
        realized_safe=(False, False),
    )

    assert metrics.decisions == 2
    assert metrics.covered_decisions == 1
    assert metrics.coverage == 0.5
    assert metrics.false_safes == 1
    assert metrics.false_safe_rate == 1.0
    assert metrics.emergency_fallbacks == 1
    assert 0.0 <= metrics.candidate_coverage <= 1.0


def test_margin_matrix_covers_every_candidate_and_horizon() -> None:
    state = ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    rollouts = rollout_candidates(state, DynamicsConfig())

    matrix = compute_safety_margins(rollouts, _safe_limits())

    assert tuple(matrix) == tuple(Action)
    assert all(tuple(item.horizon for item in rows) == (1, 2) for rows in matrix.values())
