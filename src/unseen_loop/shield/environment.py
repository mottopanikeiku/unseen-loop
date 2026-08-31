"""Deterministic clear warehouse dynamics shared with the safety circuit.

This module deliberately has no Gym dependency.  In particular,
:func:`polynomial_step` is the single clear definition of the transition that
encrypted candidate evaluation must reproduce.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .types import Action, DynamicsConfig, SafetyLimits, ScenarioSpec, ShieldState


@dataclass(frozen=True, slots=True)
class SafetyReport:
    """Polynomial margins and constraint events for one post-transition state."""

    obstacle_margins: tuple[float, ...]
    boundary_margins: tuple[float, float, float, float]
    speed_margin: float
    tilt_margin: float
    battery_margin: float
    obstacle_unsafe: bool
    boundary_unsafe: bool
    speed_unsafe: bool
    tilt_unsafe: bool
    battery_unsafe: bool

    @property
    def unsafe(self) -> bool:
        return any(
            (
                self.obstacle_unsafe,
                self.boundary_unsafe,
                self.speed_unsafe,
                self.tilt_unsafe,
                self.battery_unsafe,
            )
        )

    @property
    def violation_count(self) -> int:
        return sum(
            (
                self.obstacle_unsafe,
                self.boundary_unsafe,
                self.speed_unsafe,
                self.tilt_unsafe,
                self.battery_unsafe,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_margins": list(self.obstacle_margins),
            "boundary_margins": list(self.boundary_margins),
            "speed_margin": self.speed_margin,
            "tilt_margin": self.tilt_margin,
            "battery_margin": self.battery_margin,
            "obstacle_unsafe": self.obstacle_unsafe,
            "boundary_unsafe": self.boundary_unsafe,
            "speed_unsafe": self.speed_unsafe,
            "tilt_unsafe": self.tilt_unsafe,
            "battery_unsafe": self.battery_unsafe,
            "unsafe": self.unsafe,
        }


@dataclass(frozen=True, slots=True)
class UnsafeEventCounts:
    """Cumulative post-step events, with each category counted once per step."""

    elapsed_steps: int = 0
    unsafe_steps: int = 0
    obstacle_events: int = 0
    boundary_events: int = 0
    speed_events: int = 0
    tilt_events: int = 0
    battery_events: int = 0

    @property
    def total_events(self) -> int:
        return (
            self.obstacle_events
            + self.boundary_events
            + self.speed_events
            + self.tilt_events
            + self.battery_events
        )

    @property
    def unsafe_rate(self) -> float:
        return self.unsafe_steps / self.elapsed_steps if self.elapsed_steps else 0.0

    def add(self, report: SafetyReport) -> UnsafeEventCounts:
        return UnsafeEventCounts(
            elapsed_steps=self.elapsed_steps + 1,
            unsafe_steps=self.unsafe_steps + int(report.unsafe),
            obstacle_events=self.obstacle_events + int(report.obstacle_unsafe),
            boundary_events=self.boundary_events + int(report.boundary_unsafe),
            speed_events=self.speed_events + int(report.speed_unsafe),
            tilt_events=self.tilt_events + int(report.tilt_unsafe),
            battery_events=self.battery_events + int(report.battery_unsafe),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "elapsed_steps": self.elapsed_steps,
            "unsafe_steps": self.unsafe_steps,
            "obstacle_events": self.obstacle_events,
            "boundary_events": self.boundary_events,
            "speed_events": self.speed_events,
            "tilt_events": self.tilt_events,
            "battery_events": self.battery_events,
            "total_events": self.total_events,
            "unsafe_rate": self.unsafe_rate,
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    state: ShieldState
    reward: float
    terminated: bool
    truncated: bool
    safety: SafetyReport
    unsafe_events: UnsafeEventCounts

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "safety": self.safety.to_dict(),
            "unsafe_events": self.unsafe_events.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """A complete deterministic rollout; states include the initial state."""

    states: tuple[ShieldState, ...]
    actions: tuple[Action, ...]
    rewards: tuple[float, ...]
    safety: tuple[SafetyReport, ...]
    unsafe_events: UnsafeEventCounts

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [state.to_dict() for state in self.states],
            "actions": [action.name for action in self.actions],
            "rewards": list(self.rewards),
            "safety": [report.to_dict() for report in self.safety],
            "unsafe_events": self.unsafe_events.to_dict(),
            "total_reward": self.total_reward,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def polynomial_step(
    state: ShieldState,
    action: Action | int,
    config: DynamicsConfig,
) -> ShieldState:
    """Apply the exact public polynomial dynamics used by the shield circuit.

    No clipping or collision response occurs here: either would make the clear
    transition diverge from encrypted evaluation.  Constraint violations are
    measured separately by :func:`safety_report`.
    """

    candidate = Action(action)
    ax, ay = candidate.vector
    action_norm_squared = ax * ax + ay * ay
    return ShieldState(
        x=state.x + state.vx + 0.5 * config.accel * ax,
        y=state.y + state.vy + 0.5 * config.accel * ay,
        vx=config.drag * state.vx + config.accel * ax,
        vy=config.drag * state.vy + config.accel * ay,
        battery=state.battery - config.base_drain - config.motion_drain * action_norm_squared,
        tilt=config.tilt_decay * state.tilt + config.tilt_gain * (ax * state.vy - ay * state.vx),
    )


def rollout_states(
    state: ShieldState,
    action: Action | int,
    config: DynamicsConfig,
    *,
    horizon: int = 2,
) -> tuple[ShieldState, ...]:
    """Roll a public candidate action forward repeatedly for ``horizon`` steps."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    states: list[ShieldState] = []
    current = state
    for _ in range(horizon):
        current = polynomial_step(current, action, config)
        states.append(current)
    return tuple(states)


def candidate_rollouts(
    state: ShieldState,
    config: DynamicsConfig,
    *,
    horizon: int = 2,
) -> tuple[tuple[ShieldState, ...], ...]:
    """Return one rollout per action, indexed by the frozen action enum value."""

    return tuple(rollout_states(state, action, config, horizon=horizon) for action in Action)


def safety_report(state: ShieldState, limits: SafetyLimits) -> SafetyReport:
    """Evaluate polynomial safety margins; only strictly positive margins are safe."""

    obstacle_margins = tuple(
        (state.x - obstacle.x) ** 2
        + (state.y - obstacle.y) ** 2
        - (obstacle.radius + limits.vehicle_radius + limits.obstacle_clearance) ** 2
        for obstacle in limits.obstacles
    )
    x_min, x_max = limits.x_bounds
    y_min, y_max = limits.y_bounds
    required_clearance = limits.vehicle_radius + limits.obstacle_clearance
    boundary_margins = (
        state.x - (x_min + required_clearance),
        (x_max - required_clearance) - state.x,
        state.y - (y_min + required_clearance),
        (y_max - required_clearance) - state.y,
    )
    speed_margin = limits.max_speed**2 - state.vx**2 - state.vy**2
    tilt_margin = limits.max_abs_tilt**2 - state.tilt**2
    battery_margin = state.battery - limits.min_battery
    return SafetyReport(
        obstacle_margins=obstacle_margins,
        boundary_margins=boundary_margins,
        speed_margin=speed_margin,
        tilt_margin=tilt_margin,
        battery_margin=battery_margin,
        obstacle_unsafe=any(margin <= 0 for margin in obstacle_margins),
        boundary_unsafe=any(margin <= 0 for margin in boundary_margins),
        speed_unsafe=speed_margin <= 0,
        tilt_unsafe=tilt_margin <= 0,
        battery_unsafe=battery_margin <= 0,
    )


def _goal_reached(state: ShieldState, scenario: ScenarioSpec) -> bool:
    dx = state.x - scenario.goal[0]
    dy = state.y - scenario.goal[1]
    return dx * dx + dy * dy <= scenario.reward.goal_radius**2


def transition_reward(
    before: ShieldState,
    after: ShieldState,
    action: Action | int,
    report: SafetyReport,
    scenario: ScenarioSpec,
) -> float:
    """Compute the clear task reward without affecting shield decisions."""

    before_dx = before.x - scenario.goal[0]
    before_dy = before.y - scenario.goal[1]
    after_dx = after.x - scenario.goal[0]
    after_dy = after.y - scenario.goal[1]
    squared_progress = before_dx**2 + before_dy**2 - after_dx**2 - after_dy**2
    ax, ay = Action(action).vector
    reward = (
        scenario.reward.progress_weight * squared_progress
        - scenario.reward.action_cost * (ax * ax + ay * ay)
        - scenario.reward.unsafe_cost * report.violation_count
    )
    if _goal_reached(after, scenario):
        reward += scenario.reward.goal_bonus
    return reward


def simulate_rollout(
    scenario: ScenarioSpec,
    actions: Iterable[Action | int],
    *,
    initial_state: ShieldState | None = None,
) -> RolloutResult:
    """Pure deterministic rollout that does not mutate an environment instance."""

    current = scenario.initial_state if initial_state is None else initial_state
    states = [current]
    normalized_actions: list[Action] = []
    rewards: list[float] = []
    reports: list[SafetyReport] = []
    counts = UnsafeEventCounts()
    for raw_action in actions:
        action = Action(raw_action)
        following = polynomial_step(current, action, scenario.dynamics)
        report = safety_report(following, scenario.safety)
        normalized_actions.append(action)
        rewards.append(transition_reward(current, following, action, report, scenario))
        reports.append(report)
        counts = counts.add(report)
        states.append(following)
        current = following
    return RolloutResult(
        states=tuple(states),
        actions=tuple(normalized_actions),
        rewards=tuple(rewards),
        safety=tuple(reports),
        unsafe_events=counts,
    )


class WarehouseEnvironment:
    """Small deterministic environment around the shared polynomial transition."""

    def __init__(self, scenario: ScenarioSpec, *, seed: int | None = None) -> None:
        self.scenario = scenario
        self._random = random.Random(seed)
        self._state = scenario.initial_state
        self._unsafe_events = UnsafeEventCounts()
        self._done = False

    @property
    def state(self) -> ShieldState:
        return self._state

    @property
    def elapsed_steps(self) -> int:
        return self._unsafe_events.elapsed_steps

    @property
    def unsafe_events(self) -> UnsafeEventCounts:
        return self._unsafe_events

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, *, seed: int | None = None) -> ShieldState:
        """Reset state, optionally reseeding the independent environment RNG."""

        if seed is not None:
            self._random.seed(seed)
        values = tuple(
            base + self._random.uniform(-radius, radius) if radius else base
            for base, radius in zip(
                self.scenario.initial_state.as_tuple(),
                self.scenario.reset_jitter,
                strict=True,
            )
        )
        self._state = ShieldState.from_array(values)
        self._unsafe_events = UnsafeEventCounts()
        self._done = False
        return self._state

    def step(self, action: Action | int) -> StepResult:
        if self._done:
            raise RuntimeError("environment is done; call reset before stepping again")
        candidate = Action(action)
        following = polynomial_step(self._state, candidate, self.scenario.dynamics)
        report = safety_report(following, self.scenario.safety)
        reward = transition_reward(self._state, following, candidate, report, self.scenario)
        events = self._unsafe_events.add(report)
        terminated = _goal_reached(following, self.scenario) or (
            self.scenario.terminate_on_unsafe and report.unsafe
        )
        truncated = not terminated and events.elapsed_steps >= self.scenario.max_steps
        self._state = following
        self._unsafe_events = events
        self._done = terminated or truncated
        return StepResult(
            state=following,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            safety=report,
            unsafe_events=events,
        )

    def rollout(self, actions: Iterable[Action | int]) -> RolloutResult:
        """Advance through actions, stopping at termination or truncation."""

        initial = self._state
        states = [initial]
        normalized_actions: list[Action] = []
        rewards: list[float] = []
        reports: list[SafetyReport] = []
        for raw_action in actions:
            action = Action(raw_action)
            result = self.step(action)
            normalized_actions.append(action)
            states.append(result.state)
            rewards.append(result.reward)
            reports.append(result.safety)
            if result.done:
                break
        return RolloutResult(
            states=tuple(states),
            actions=tuple(normalized_actions),
            rewards=tuple(rewards),
            safety=tuple(reports),
            unsafe_events=self._unsafe_events,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe episode state for experiment artifacts."""

        return {
            "scenario": self.scenario.to_dict(),
            "state": self._state.to_dict(),
            "unsafe_events": self._unsafe_events.to_dict(),
            "done": self._done,
        }
