"""Immutable data contracts for the warehouse safety shield.

The state ordering and action ordering in this module are part of the encrypted
circuit protocol.  Change them only as a versioned protocol migration.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Any

STATE_FEATURES = ("x", "y", "vx", "vy", "battery", "tilt")
STATE_DIM = len(STATE_FEATURES)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _pair(raw: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(raw[0]), float(raw[1]))
    _require_finite(f"{name}[0]", result[0])
    _require_finite(f"{name}[1]", result[1])
    return result


def _state_values(
    raw: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float, float, float, float]:
    if len(raw) != STATE_DIM:
        raise ValueError(f"{name} must have exactly {STATE_DIM} values")
    return (
        float(raw[0]),
        float(raw[1]),
        float(raw[2]),
        float(raw[3]),
        float(raw[4]),
        float(raw[5]),
    )


class Action(IntEnum):
    """The five public candidate actions in protocol order."""

    BRAKE = 0
    EAST = 1
    WEST = 2
    NORTH = 3
    SOUTH = 4

    @property
    def vector(self) -> tuple[int, int]:
        return _ACTION_VECTORS[self]


_ACTION_VECTORS: dict[Action, tuple[int, int]] = {
    Action.BRAKE: (0, 0),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
    Action.NORTH: (0, 1),
    Action.SOUTH: (0, -1),
}


@dataclass(frozen=True, slots=True)
class ShieldState:
    """Six signed features encrypted by the client, in frozen protocol order."""

    x: float
    y: float
    vx: float
    vy: float
    battery: float
    tilt: float

    def __post_init__(self) -> None:
        for name, value in zip(STATE_FEATURES, self.as_tuple(), strict=True):
            _require_finite(name, value)

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.vx, self.vy, self.battery, self.tilt)

    def as_array(self) -> tuple[float, float, float, float, float, float]:
        """Return an immutable array-shaped representation in protocol order."""
        return self.as_tuple()

    def to_dict(self) -> dict[str, float]:
        return dict(zip(STATE_FEATURES, self.as_tuple(), strict=True))

    @classmethod
    def from_array(cls, values: Sequence[float]) -> ShieldState:
        return cls(*_state_values(values, name="state"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShieldState:
        return cls.from_array([float(raw[name]) for name in STATE_FEATURES])


@dataclass(frozen=True, slots=True)
class Obstacle:
    """A closed circular obstacle in warehouse coordinates."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        _require_finite("obstacle.x", self.x)
        _require_finite("obstacle.y", self.y)
        _require_finite("obstacle.radius", self.radius)
        if self.radius <= 0:
            raise ValueError("obstacle radius must be positive")

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "radius": self.radius}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Obstacle:
        return cls(x=float(raw["x"]), y=float(raw["y"]), radius=float(raw["radius"]))


# Descriptive alias retained in the public model vocabulary.
CircularObstacle = Obstacle


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    """Public coefficients for the polynomial warehouse transition."""

    drag: float = 0.9
    accel: float = 1.0
    base_drain: float = 0.02
    motion_drain: float = 0.03
    tilt_decay: float = 0.8
    tilt_gain: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "drag",
            "accel",
            "base_drain",
            "motion_drain",
            "tilt_decay",
            "tilt_gain",
        ):
            _require_finite(name, getattr(self, name))
        if not 0 <= self.drag <= 1:
            raise ValueError("drag must be in [0, 1]")
        if self.accel <= 0:
            raise ValueError("accel must be positive")
        if self.base_drain < 0 or self.motion_drain < 0:
            raise ValueError("battery drains must be non-negative")
        if not 0 <= self.tilt_decay <= 1:
            raise ValueError("tilt_decay must be in [0, 1]")
        if self.tilt_gain < 0:
            raise ValueError("tilt_gain must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "drag": self.drag,
            "accel": self.accel,
            "base_drain": self.base_drain,
            "motion_drain": self.motion_drain,
            "tilt_decay": self.tilt_decay,
            "tilt_gain": self.tilt_gain,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DynamicsConfig:
        return cls(**{name: float(raw[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Public safety constraints used by clear and encrypted evaluation."""

    obstacles: tuple[Obstacle, ...] = ()
    max_speed: float = 2.5
    max_abs_tilt: float = 0.5
    min_battery: float = 0.0
    x_bounds: tuple[float, float] = (-10.0, 10.0)
    y_bounds: tuple[float, float] = (-10.0, 10.0)
    vehicle_radius: float = 0.25
    obstacle_clearance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        if any(not isinstance(obstacle, Obstacle) for obstacle in self.obstacles):
            raise TypeError("obstacles must contain only Obstacle instances")
        object.__setattr__(self, "x_bounds", _pair(self.x_bounds, name="x_bounds"))
        object.__setattr__(self, "y_bounds", _pair(self.y_bounds, name="y_bounds"))
        if self.x_bounds[0] >= self.x_bounds[1] or self.y_bounds[0] >= self.y_bounds[1]:
            raise ValueError("workspace bounds must be strictly increasing")
        for name in (
            "max_speed",
            "max_abs_tilt",
            "min_battery",
            "vehicle_radius",
            "obstacle_clearance",
        ):
            _require_finite(name, getattr(self, name))
        if self.max_speed <= 0 or self.max_abs_tilt <= 0:
            raise ValueError("speed and tilt limits must be positive")
        if self.vehicle_radius < 0 or self.obstacle_clearance < 0:
            raise ValueError("clearances must be non-negative")
        required_clearance = self.vehicle_radius + self.obstacle_clearance
        if self.x_bounds[1] - self.x_bounds[0] < 2 * required_clearance:
            raise ValueError("x_bounds are too narrow for the required clearance")
        if self.y_bounds[1] - self.y_bounds[0] < 2 * required_clearance:
            raise ValueError("y_bounds are too narrow for the required clearance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "max_speed": self.max_speed,
            "max_abs_tilt": self.max_abs_tilt,
            "min_battery": self.min_battery,
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "vehicle_radius": self.vehicle_radius,
            "obstacle_clearance": self.obstacle_clearance,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SafetyLimits:
        return cls(
            obstacles=tuple(Obstacle.from_dict(item) for item in raw.get("obstacles", ())),
            max_speed=float(raw["max_speed"]),
            max_abs_tilt=float(raw["max_abs_tilt"]),
            min_battery=float(raw["min_battery"]),
            x_bounds=_pair(raw.get("x_bounds", (-10.0, 10.0)), name="x_bounds"),
            y_bounds=_pair(raw.get("y_bounds", (-10.0, 10.0)), name="y_bounds"),
            vehicle_radius=float(raw.get("vehicle_radius", 0.25)),
            obstacle_clearance=float(raw.get("obstacle_clearance", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Clear-environment reward coefficients; not part of shield selection."""

    progress_weight: float = 1.0
    action_cost: float = 0.02
    unsafe_cost: float = 5.0
    goal_bonus: float = 10.0
    goal_radius: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_finite(name, getattr(self, name))
        if (
            min(
                self.progress_weight,
                self.action_cost,
                self.unsafe_cost,
                self.goal_bonus,
            )
            < 0
        ):
            raise ValueError("reward weights must be non-negative")
        if self.goal_radius <= 0:
            raise ValueError("goal_radius must be positive")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RewardSpec:
        return cls(**{name: float(raw[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A complete, immutable and JSON-serializable warehouse scenario."""

    initial_state: ShieldState
    goal: tuple[float, float]
    safety: SafetyLimits
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    reward: RewardSpec = field(default_factory=RewardSpec)
    reset_jitter: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    horizon: int = 2
    max_steps: int = 100
    terminate_on_unsafe: bool = False
    schema_version: str = "unseen-loop/shield-scenario-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal", _pair(self.goal, name="goal"))
        jitter = _state_values(self.reset_jitter, name="reset_jitter")
        if any(not isfinite(value) or value < 0 for value in jitter):
            raise ValueError("reset_jitter values must be finite and non-negative")
        object.__setattr__(self, "reset_jitter", jitter)
        if self.horizon != 2:
            raise ValueError("the shield protocol uses an exact two-step horizon")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "goal": list(self.goal),
            "safety": self.safety.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "reward": self.reward.to_dict(),
            "reset_jitter": list(self.reset_jitter),
            "horizon": self.horizon,
            "max_steps": self.max_steps,
            "terminate_on_unsafe": self.terminate_on_unsafe,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScenarioSpec:
        return cls(
            initial_state=ShieldState.from_dict(raw["initial_state"]),
            goal=_pair(raw["goal"], name="goal"),
            safety=SafetyLimits.from_dict(raw["safety"]),
            dynamics=DynamicsConfig.from_dict(raw.get("dynamics", DynamicsConfig().to_dict())),
            reward=RewardSpec.from_dict(raw.get("reward", RewardSpec().to_dict())),
            reset_jitter=_state_values(
                raw.get("reset_jitter", (0.0,) * STATE_DIM),
                name="reset_jitter",
            ),
            horizon=int(raw.get("horizon", 2)),
            max_steps=int(raw.get("max_steps", 100)),
            terminate_on_unsafe=bool(raw.get("terminate_on_unsafe", False)),
            schema_version=str(raw.get("schema_version", "unseen-loop/shield-scenario-v1")),
        )

    @classmethod
    def from_json(cls, payload: str) -> ScenarioSpec:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("scenario payload must be a JSON object")
        return cls.from_dict(raw)
