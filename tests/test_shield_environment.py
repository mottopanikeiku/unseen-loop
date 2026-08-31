from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from unseen_loop.shield import (
    STATE_FEATURES,
    Action,
    DynamicsConfig,
    Obstacle,
    SafetyLimits,
    ScenarioSpec,
    ShieldState,
    WarehouseEnvironment,
    candidate_rollouts,
    polynomial_step,
    rollout_states,
    safety_report,
    simulate_rollout,
)


def scenario(**overrides: object) -> ScenarioSpec:
    values: dict[str, object] = {
        "initial_state": ShieldState(0.0, 0.0, 0.2, -0.1, 1.0, 0.0),
        "goal": (8.0, 8.0),
        "safety": SafetyLimits(
            obstacles=(Obstacle(3.0, 3.0, 0.5),),
            max_speed=3.0,
            max_abs_tilt=0.5,
            min_battery=0.1,
            x_bounds=(-5.0, 5.0),
            y_bounds=(-5.0, 5.0),
            vehicle_radius=0.25,
        ),
        "reset_jitter": (0.5, 0.5, 0.1, 0.1, 0.05, 0.02),
        "max_steps": 5,
    }
    values.update(overrides)
    return ScenarioSpec(**values)  # type: ignore[arg-type]


def test_protocol_state_and_action_order_is_frozen() -> None:
    assert STATE_FEATURES == ("x", "y", "vx", "vy", "battery", "tilt")
    assert [(action.value, action.vector) for action in Action] == [
        (0, (0, 0)),
        (1, (1, 0)),
        (2, (-1, 0)),
        (3, (0, 1)),
        (4, (0, -1)),
    ]
    with pytest.raises(ValueError):
        Action(5)


def test_specs_are_immutable_and_reject_invalid_values() -> None:
    state = ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        state.x = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        ShieldState(float("nan"), 0.0, 0.0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="radius"):
        Obstacle(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="two-step"):
        scenario(horizon=3)


def test_polynomial_step_matches_the_frozen_clear_equations() -> None:
    state = ShieldState(1.0, 2.0, 0.4, -0.2, 0.8, 0.1)
    following = polynomial_step(state, Action.NORTH, DynamicsConfig())
    assert following.as_tuple() == pytest.approx((1.4, 2.3, 0.36, 0.82, 0.75, 0.06))

    brake = polynomial_step(state, Action.BRAKE, DynamicsConfig())
    assert brake.as_tuple() == pytest.approx((1.4, 1.8, 0.36, -0.18, 0.78, 0.08))


def test_two_step_candidate_rollouts_repeat_each_public_action() -> None:
    initial = ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    config = DynamicsConfig()
    east = rollout_states(initial, Action.EAST, config)
    assert len(east) == 2
    assert east[0].as_tuple() == pytest.approx((0.5, 0.0, 1.0, 0.0, 0.95, 0.0))
    assert east[1].as_tuple() == pytest.approx((2.0, 0.0, 1.9, 0.0, 0.9, 0.0))

    candidates = candidate_rollouts(initial, config)
    assert len(candidates) == 5
    assert all(len(path) == 2 for path in candidates)
    assert candidates[Action.WEST][1].x == pytest.approx(-2.0)
    assert candidates[Action.NORTH][1].y == pytest.approx(2.0)
    assert candidates[Action.SOUTH][1].y == pytest.approx(-2.0)


def test_polynomial_safety_margins_require_strictly_positive_boundaries() -> None:
    limits = SafetyLimits(
        obstacles=(Obstacle(0.0, 0.0, 1.0),),
        max_speed=2.5,
        max_abs_tilt=0.5,
        min_battery=0.1,
        x_bounds=(-5.0, 5.0),
        y_bounds=(-5.0, 5.0),
        vehicle_radius=0.25,
    )
    touching = ShieldState(1.25, 0.0, 2.5, 0.0, 0.1, -0.5)
    report = safety_report(touching, limits)
    assert report.obstacle_margins == pytest.approx((0.0,))
    assert report.speed_margin == pytest.approx(0.0)
    assert report.tilt_margin == pytest.approx(0.0)
    assert report.battery_margin == pytest.approx(0.0)
    assert report.unsafe
    assert report.obstacle_unsafe
    assert report.speed_unsafe
    assert report.tilt_unsafe
    assert report.battery_unsafe

    collision = safety_report(ShieldState(1.249, 0.0, 0.0, 0.0, 1.0, 0.0), limits)
    assert collision.obstacle_unsafe
    outside = safety_report(ShieldState(-4.751, 2.0, 0.0, 0.0, 1.0, 0.0), limits)
    assert outside.boundary_unsafe
    assert safety_report(ShieldState(-4.75, 2.0, 0.0, 0.0, 1.0, 0.0), limits).boundary_unsafe
    strictly_inside = ShieldState(-4.749, 2.0, 0.0, 0.0, 1.0, 0.0)
    assert not safety_report(strictly_inside, limits).unsafe


def test_seeded_reset_reproduces_trajectory_without_global_rng_state() -> None:
    spec = scenario()
    first = WarehouseEnvironment(spec)
    second = WarehouseEnvironment(spec)
    state_a = first.reset(seed=2026)
    state_b = second.reset(seed=2026)
    assert state_a == state_b
    assert (
        first.rollout([Action.EAST, Action.NORTH]).states
        == second.rollout([Action.EAST, Action.NORTH]).states
    )

    different = WarehouseEnvironment(spec).reset(seed=2027)
    assert different != state_a
    for sampled, center, radius in zip(
        state_a.as_tuple(), spec.initial_state.as_tuple(), spec.reset_jitter, strict=True
    ):
        assert center - radius <= sampled <= center + radius


def test_unsafe_events_count_categories_once_per_post_step_state() -> None:
    limits = SafetyLimits(
        obstacles=(Obstacle(0.5, 0.0, 0.3),),
        max_speed=0.1,
        max_abs_tilt=0.5,
        min_battery=0.99,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        vehicle_radius=0.1,
    )
    spec = scenario(
        initial_state=ShieldState(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        safety=limits,
        reset_jitter=(0.0,) * 6,
        terminate_on_unsafe=True,
    )
    environment = WarehouseEnvironment(spec)
    environment.reset(seed=1)
    result = environment.step(Action.EAST)
    assert result.terminated
    assert result.safety.obstacle_unsafe
    assert result.safety.speed_unsafe
    assert result.safety.battery_unsafe
    assert not result.safety.boundary_unsafe
    assert result.unsafe_events.elapsed_steps == 1
    assert result.unsafe_events.unsafe_steps == 1
    assert result.unsafe_events.total_events == 3
    assert result.unsafe_events.unsafe_rate == 1.0
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(Action.BRAKE)


def test_max_steps_truncates_and_rollout_stops() -> None:
    spec = scenario(max_steps=2, reset_jitter=(0.0,) * 6)
    environment = WarehouseEnvironment(spec)
    environment.reset(seed=10)
    rollout = environment.rollout([Action.BRAKE, Action.BRAKE, Action.BRAKE])
    assert len(rollout.actions) == 2
    assert len(rollout.states) == 3
    assert environment.done
    assert environment.elapsed_steps == 2


def test_scenario_and_rollout_serialization_are_stable() -> None:
    spec = scenario()
    encoded = spec.to_json()
    assert ScenarioSpec.from_json(encoded) == spec
    assert ScenarioSpec.from_json(encoded).to_json() == encoded

    rollout = simulate_rollout(spec, [Action.BRAKE, Action.EAST])
    assert rollout.states[0] == spec.initial_state
    assert rollout.states[1] == polynomial_step(spec.initial_state, Action.BRAKE, spec.dynamics)
    payload = rollout.to_json()
    assert '"actions":["BRAKE","EAST"]' in payload
    assert '"unsafe_events"' in payload
