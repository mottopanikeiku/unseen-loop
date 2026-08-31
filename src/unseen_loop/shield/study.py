"""Clear, paired CipherShield study runner.

This module is deliberately independent of every FHE backend.  Persisted rows
contain actions, public outcomes, counts, and exact denominators, but never
states, candidate margins, or other plaintext client inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

from .certificate import MARGIN_FAMILIES, ErrorBuffer, MarginFamily
from .environment import WarehouseEnvironment, polynomial_step
from .shield import compute_safety_margins, rollout_candidates
from .types import (
    Action,
    DynamicsConfig,
    Obstacle,
    RewardSpec,
    SafetyLimits,
    ScenarioSpec,
    ShieldState,
)

PLAN_SCHEMA_VERSION = "unseen-loop/shield-study-plan-v1"
STEP_SCHEMA_VERSION = "unseen-loop/shield-study-step-v1"
EPISODE_SCHEMA_VERSION = "unseen-loop/shield-study-episode-v1"
SUMMARY_SCHEMA_VERSION = "unseen-loop/shield-study-summary-v1"

SCENARIO_CATEGORIES = (
    "benign_open_floor",
    "static_obstacle",
    "crossing_obstacle",
    "narrow_aisle",
    "workspace_boundary",
    "overspeed",
    "tilt_stress",
    "low_battery",
    "obstacle_and_speed",
    "boundary_and_tilt",
    "battery_and_obstacle",
    "compound_stress",
)
BENIGN_SCENARIO_CATEGORIES = frozenset(("benign_open_floor",))
CONTROLLER_CELLS = (
    "disabled",
    "always_brake",
    "h1",
    "h2",
    "leave_out_obstacle",
    "leave_out_speed",
    "leave_out_tilt",
    "leave_out_battery",
    "zero_buffer",
    "double_buffer",
    "model_minus_10pct",
    "model_nominal",
    "model_plus_10pct",
)
LEAVE_OUT_CONTROLLERS = {
    "leave_out_obstacle": MarginFamily.OBSTACLE,
    "leave_out_speed": MarginFamily.SPEED,
    "leave_out_tilt": MarginFamily.TILT,
    "leave_out_battery": MarginFamily.BATTERY,
}
MODEL_MISMATCH_CONTROLLERS = (
    "model_minus_10pct",
    "model_nominal",
    "model_plus_10pct",
)
DEFAULT_ERROR_BUFFER = ErrorBuffer(obstacle=0.05, speed=0.05, tilt=0.005, battery=0.005)


class JobStatus(StrEnum):
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def content_seed(*parts: object) -> int:
    """Produce a portable 63-bit seed from canonical content only."""

    return int(_digest(parts)[:16], 16) & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class ShieldJob:
    job_id: str
    pair_id: str
    scenario_category: str
    controller: str
    replicate: int
    seed: int
    status: JobStatus = JobStatus.PLANNED
    schema_version: str = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", JobStatus(self.status))
        if self.scenario_category not in SCENARIO_CATEGORIES:
            raise ValueError(f"unknown scenario category: {self.scenario_category}")
        if self.controller not in CONTROLLER_CELLS:
            raise ValueError(f"unknown controller cell: {self.controller}")
        if self.replicate < 0 or self.seed < 0:
            raise ValueError("replicate and seed must be non-negative")
        if len(self.job_id) != 64 or len(self.pair_id) != 64:
            raise ValueError("job_id and pair_id must be full SHA-256 digests")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "pair_id": self.pair_id,
            "scenario_category": self.scenario_category,
            "controller": self.controller,
            "replicate": self.replicate,
            "seed": self.seed,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class ShieldStudyPlan:
    seed_root: str
    scenario_categories: tuple[str, ...]
    controller_cells: tuple[str, ...]
    seeds_per_controller_cell: int
    jobs: tuple[ShieldJob, ...]
    schema_version: str = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_categories", tuple(self.scenario_categories))
        object.__setattr__(self, "controller_cells", tuple(self.controller_cells))
        object.__setattr__(self, "jobs", tuple(self.jobs))
        if not self.seed_root or self.seeds_per_controller_cell < 1:
            raise ValueError("plan requires a seed root and positive seed count")
        if len(set(self.scenario_categories)) != len(self.scenario_categories):
            raise ValueError("scenario categories must be unique")
        if len(set(self.controller_cells)) != len(self.controller_cells):
            raise ValueError("controller cells must be unique")
        expected = (
            len(self.scenario_categories)
            * len(self.controller_cells)
            * self.seeds_per_controller_cell
        )
        if len(self.jobs) != expected:
            raise ValueError(f"plan has {len(self.jobs)} jobs; expected {expected}")
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("plan job IDs must be unique")

    @property
    def episode_count(self) -> int:
        return len(self.jobs)

    @property
    def pair_count(self) -> int:
        return len({job.pair_id for job in self.jobs})

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed_root": self.seed_root,
            "scenario_categories": list(self.scenario_categories),
            "controller_cells": list(self.controller_cells),
            "seeds_per_controller_cell": self.seeds_per_controller_cell,
            "episode_count": self.episode_count,
            "pair_count": self.pair_count,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def _manifest_values(
    manifest: str | Path | Mapping[str, Any] | object,
) -> tuple[str, Mapping[str, Any]]:
    if isinstance(manifest, (str, Path)):
        with Path(manifest).open("rb") as handle:
            raw = tomllib.load(handle)
        return str(raw["seed_root"]), raw["shield"]
    if isinstance(manifest, Mapping):
        shield = manifest["shield"]
        if not isinstance(shield, Mapping):
            shield = {
                name: getattr(shield, name)
                for name in ("scenarios", "seeds_per_controller_cell", "controller_cells")
            }
        return str(manifest["seed_root"]), shield
    try:
        manifest_object = cast(Any, manifest)
        shield_object = manifest_object.shield
        shield = {
            name: getattr(shield_object, name)
            for name in ("scenarios", "seeds_per_controller_cell", "controller_cells")
        }
        return str(manifest_object.seed_root), shield
    except AttributeError as error:
        raise ValueError("manifest must expose seed_root and shield study fields") from error


def build_shield_plan(
    manifest: str | Path | Mapping[str, Any] | object,
    *,
    scenario_categories: Sequence[str] = SCENARIO_CATEGORIES,
    controller_cells: Sequence[str] | None = None,
    seeds_per_controller_cell: int | None = None,
) -> ShieldStudyPlan:
    """Build the exact paired matrix described by a flagship manifest."""

    seed_root, shield = _manifest_values(manifest)
    categories = tuple(scenario_categories)
    controllers = tuple(
        controller_cells if controller_cells is not None else shield["controller_cells"]
    )
    seed_count = int(
        seeds_per_controller_cell
        if seeds_per_controller_cell is not None
        else shield["seeds_per_controller_cell"]
    )
    unknown_categories = set(categories) - set(SCENARIO_CATEGORIES)
    unknown_controllers = set(controllers) - set(CONTROLLER_CELLS)
    if unknown_categories or unknown_controllers:
        raise ValueError(
            "unknown categories/controllers: "
            f"{sorted(unknown_categories)}, {sorted(unknown_controllers)}"
        )
    if (
        "scenarios" in shield
        and scenario_categories == SCENARIO_CATEGORIES
        and int(shield["scenarios"]) != len(categories)
    ):
        raise ValueError("manifest scenario count disagrees with registered factories")
    if seed_count < 1:
        raise ValueError("seeds_per_controller_cell must be positive")
    jobs: list[ShieldJob] = []
    for category in categories:
        for replicate in range(seed_count):
            pair_content = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "seed_root": seed_root,
                "scenario_category": category,
                "replicate": replicate,
            }
            pair_id = _digest(pair_content)
            seed = content_seed(pair_content)
            for controller in controllers:
                jobs.append(
                    ShieldJob(
                        job_id=_digest({"pair_id": pair_id, "controller": controller}),
                        pair_id=pair_id,
                        scenario_category=category,
                        controller=controller,
                        replicate=replicate,
                        seed=seed,
                    )
                )
    return ShieldStudyPlan(
        seed_root=seed_root,
        scenario_categories=categories,
        controller_cells=controllers,
        seeds_per_controller_cell=seed_count,
        jobs=tuple(jobs),
    )


def _scenario(
    state: ShieldState,
    goal: tuple[float, float],
    *,
    obstacles: tuple[Obstacle, ...] = (),
    max_speed: float = 2.5,
    max_tilt: float = 0.5,
    min_battery: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-10.0, 10.0), (-10.0, 10.0)),
) -> ScenarioSpec:
    return ScenarioSpec(
        initial_state=state,
        goal=goal,
        safety=SafetyLimits(
            obstacles=obstacles,
            max_speed=max_speed,
            max_abs_tilt=max_tilt,
            min_battery=min_battery,
            x_bounds=bounds[0],
            y_bounds=bounds[1],
            vehicle_radius=0.25,
            obstacle_clearance=0.1,
        ),
        dynamics=DynamicsConfig(),
        reward=RewardSpec(),
        reset_jitter=(0.15, 0.15, 0.08, 0.08, 0.01, 0.01),
        max_steps=32,
    )


def _benign() -> ScenarioSpec:
    return _scenario(ShieldState(-2, 0, 0, 0, 1, 0), (2, 0), max_speed=5)


def _static_obstacle() -> ScenarioSpec:
    return _scenario(ShieldState(-5, 0, 0.4, 0, 1, 0), (5, 0), obstacles=(Obstacle(0, 0, 1.2),))


def _crossing() -> ScenarioSpec:
    return _scenario(
        ShieldState(-4, -3, 0.5, 0.5, 1, 0.05),
        (4, 3),
        obstacles=(Obstacle(0, 0, 1), Obstacle(1.8, 1, 0.7)),
    )


def _aisle() -> ScenarioSpec:
    obstacles = tuple(Obstacle(x, y, 0.65) for x in (-1.5, 1.5) for y in (-2, 0, 2))
    return _scenario(ShieldState(0, -5, 0, 0.4, 1, 0), (0, 5), obstacles=obstacles)


def _boundary() -> ScenarioSpec:
    return _scenario(ShieldState(4.2, 0, 0.9, 0, 1, 0), (-4, 0), bounds=((-5, 5), (-5, 5)))


def _speed() -> ScenarioSpec:
    return _scenario(ShieldState(-4, 0, 1.6, 0, 1, 0), (4, 0), max_speed=1.9)


def _tilt() -> ScenarioSpec:
    return _scenario(ShieldState(-3, -3, 0.7, 0.5, 1, 0.24), (3, 3), max_tilt=0.27)


def _battery() -> ScenarioSpec:
    return _scenario(ShieldState(-2, 0, 0.1, 0, 0.16, 0), (2, 0), min_battery=0.08)


def _obstacle_speed() -> ScenarioSpec:
    return _scenario(
        ShieldState(-4, 0, 1.3, 0, 1, 0),
        (4, 0),
        obstacles=(Obstacle(0, 0, 1),),
        max_speed=1.8,
    )


def _boundary_tilt() -> ScenarioSpec:
    return _scenario(
        ShieldState(4, 3.5, 0.8, 0.5, 1, 0.25),
        (-4, -3.5),
        max_tilt=0.3,
        bounds=((-5, 5), (-5, 5)),
    )


def _battery_obstacle() -> ScenarioSpec:
    return _scenario(
        ShieldState(-3, 0, 0.3, 0, 0.2, 0),
        (3, 0),
        obstacles=(Obstacle(0, 0, 0.9),),
        min_battery=0.08,
    )


def _compound() -> ScenarioSpec:
    return _scenario(
        ShieldState(-3.8, -3.5, 1, 0.7, 0.22, 0.22),
        (3.8, 3.5),
        obstacles=(Obstacle(-0.5, -0.2, 1), Obstacle(1.7, 1.5, 0.8)),
        max_speed=1.8,
        max_tilt=0.3,
        min_battery=0.08,
        bounds=((-5, 5), (-5, 5)),
    )


SCENARIO_FACTORIES: Mapping[str, Callable[[], ScenarioSpec]] = {
    "benign_open_floor": _benign,
    "static_obstacle": _static_obstacle,
    "crossing_obstacle": _crossing,
    "narrow_aisle": _aisle,
    "workspace_boundary": _boundary,
    "overspeed": _speed,
    "tilt_stress": _tilt,
    "low_battery": _battery,
    "obstacle_and_speed": _obstacle_speed,
    "boundary_and_tilt": _boundary_tilt,
    "battery_and_obstacle": _battery_obstacle,
    "compound_stress": _compound,
}


def make_scenario(category: str) -> ScenarioSpec:
    try:
        return SCENARIO_FACTORIES[category]()
    except KeyError as error:
        raise ValueError(f"unknown scenario category: {category}") from error


def nominal_requested_policy(state: ShieldState, scenario: ScenarioSpec) -> Action:
    """Stable one-step greedy goal policy which does not inspect safety limits."""

    gx, gy = scenario.goal
    candidates = tuple(
        (action, polynomial_step(state, action, scenario.dynamics)) for action in Action
    )
    return min(
        candidates,
        key=lambda item: ((item[1].x - gx) ** 2 + (item[1].y - gy) ** 2, int(item[0])),
    )[0]


@dataclass(frozen=True, slots=True)
class _Controller:
    enabled: bool = True
    always_brake: bool = False
    horizons: int = 2
    buffer_scale: float = 1.0
    omitted: MarginFamily | None = None
    model_scale: float = 1.0


def _controller(name: str) -> _Controller:
    if name == "disabled":
        return _Controller(enabled=False)
    if name == "always_brake":
        return _Controller(enabled=False, always_brake=True)
    if name == "h1":
        return _Controller(horizons=1)
    if name in ("h2", "model_nominal"):
        return _Controller()
    if name in LEAVE_OUT_CONTROLLERS:
        return _Controller(omitted=LEAVE_OUT_CONTROLLERS[name])
    if name == "zero_buffer":
        return _Controller(buffer_scale=0)
    if name == "double_buffer":
        return _Controller(buffer_scale=2)
    if name == "model_minus_10pct":
        return _Controller(model_scale=0.9)
    if name == "model_plus_10pct":
        return _Controller(model_scale=1.1)
    raise ValueError(f"unknown controller: {name}")


def _select(
    state: ShieldState,
    requested: Action,
    scenario: ScenarioSpec,
    controller: _Controller,
) -> tuple[Action, bool, bool, bool, int, int]:
    if controller.always_brake:
        return Action.BRAKE, False, False, False, 0, 0
    if not controller.enabled:
        return requested, False, False, False, 0, 0
    dynamics = replace(scenario.dynamics, accel=scenario.dynamics.accel * controller.model_scale)
    margins = compute_safety_margins(rollout_candidates(state, dynamics), scenario.safety)
    active = tuple(family for family in MARGIN_FAMILIES if family != controller.omitted)
    base = DEFAULT_ERROR_BUFFER.as_margins().as_tuple()
    buffer = ErrorBuffer(*(controller.buffer_scale * value for value in base))
    certified: dict[Action, bool] = {}
    minima: dict[Action, float] = {}
    for action in Action:
        obligations = tuple(
            horizon.margins.for_family(family) - buffer.for_family(family)
            for horizon in margins[action][: controller.horizons]
            for family in active
        )
        certified[action] = all(value > 0 for value in obligations)
        minima[action] = min(obligations)
    requested_certified = certified[requested]
    if requested_certified:
        selected, fallback = requested, False
    else:
        safe = tuple(action for action in Action if certified[action])
        if safe:
            selected = max(safe, key=lambda action: (minima[action], -int(action)))
            fallback = False
        else:
            selected, fallback = Action.BRAKE, True
    return (
        selected,
        fallback,
        requested_certified,
        certified[selected],
        sum(certified.values()),
        len(Action),
    )


@dataclass(frozen=True, slots=True)
class ShieldStepRow:
    job_id: str
    pair_id: str
    scenario_category: str
    controller: str
    replicate: int
    seed: int
    step: int
    step_denominator: int
    requested_action: str
    executed_action: str
    reward: float
    unsafe: bool
    obstacle_event: bool
    boundary_event: bool
    speed_event: bool
    tilt_event: bool
    battery_event: bool
    intervention: int
    intervention_denominator: int
    benign_intervention: int
    benign_intervention_denominator: int
    fallback: int
    fallback_denominator: int
    requested_certified: int
    requested_preserved: int
    selected_certified: int
    selected_denominator: int
    certified_candidates: int
    candidate_evaluations: int
    terminated: bool
    truncated: bool
    goal_reached: bool
    job_status: JobStatus = JobStatus.COMPLETED
    schema_version: str = STEP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_status", JobStatus(self.job_status))
        if self.step < 0 or self.step_denominator != 1 or not math.isfinite(self.reward):
            raise ValueError("step must be non-negative, unit-denominated, and reward finite")
        for numerator, denominator in (
            (self.intervention, self.intervention_denominator),
            (self.benign_intervention, self.benign_intervention_denominator),
            (self.fallback, self.fallback_denominator),
            (self.requested_preserved, self.requested_certified),
            (self.selected_certified, self.selected_denominator),
            (self.certified_candidates, self.candidate_evaluations),
        ):
            if numerator < 0 or denominator < numerator:
                raise ValueError("step numerator must lie within its denominator")

    def to_dict(self) -> dict[str, object]:
        return {
            field: (value.value if isinstance(value, StrEnum) else value)
            for field, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


@dataclass(frozen=True, slots=True)
class ShieldEpisodeRow:
    job_id: str
    pair_id: str
    scenario_category: str
    controller: str
    replicate: int
    seed: int
    status: JobStatus
    failure_code: str | None
    episode_denominator: int
    elapsed_steps: int
    step_denominator: int
    total_return: float
    goal_reached: bool
    terminated: bool
    truncated: bool
    unsafe_steps: int
    unsafe_episode: int
    obstacle_events: int
    boundary_events: int
    speed_events: int
    tilt_events: int
    battery_events: int
    interventions: int
    intervention_denominator: int
    benign_interventions: int
    benign_intervention_denominator: int
    fallbacks: int
    fallback_denominator: int
    requested_preserved: int
    requested_certified_denominator: int
    selected_certified: int
    selected_denominator: int
    certified_candidates: int
    candidate_evaluations: int
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", JobStatus(self.status))
        if self.episode_denominator != 1:
            raise ValueError("every attempted job contributes one episode denominator")
        if self.elapsed_steps != self.step_denominator or not math.isfinite(self.total_return):
            raise ValueError(
                "step denominator must be actual elapsed steps and return must be finite"
            )
        pairs = (
            (self.unsafe_steps, self.step_denominator),
            (self.unsafe_episode, self.episode_denominator),
            (self.interventions, self.intervention_denominator),
            (self.benign_interventions, self.benign_intervention_denominator),
            (self.fallbacks, self.fallback_denominator),
            (self.requested_preserved, self.requested_certified_denominator),
            (self.selected_certified, self.selected_denominator),
            (self.certified_candidates, self.candidate_evaluations),
        )
        if any(numerator < 0 or denominator < numerator for numerator, denominator in pairs):
            raise ValueError("episode numerator must lie within its denominator")
        if self.status is JobStatus.COMPLETED and self.failure_code is not None:
            raise ValueError("completed jobs cannot have a failure code")
        if self.status is JobStatus.FAILED and not self.failure_code:
            raise ValueError("failed jobs require a non-sensitive failure code")

    def to_dict(self) -> dict[str, object]:
        return {
            field: (value.value if isinstance(value, StrEnum) else value)
            for field, value in ((name, getattr(self, name)) for name in self.__dataclass_fields__)
        }


def _aggregate_episode(job: ShieldJob, steps: Sequence[ShieldStepRow]) -> ShieldEpisodeRow:
    final = steps[-1]
    return ShieldEpisodeRow(
        job_id=job.job_id,
        pair_id=job.pair_id,
        scenario_category=job.scenario_category,
        controller=job.controller,
        replicate=job.replicate,
        seed=job.seed,
        status=JobStatus.COMPLETED,
        failure_code=None,
        episode_denominator=1,
        elapsed_steps=len(steps),
        step_denominator=len(steps),
        total_return=sum(row.reward for row in steps),
        terminated=final.terminated,
        goal_reached=final.goal_reached,
        truncated=final.truncated,
        unsafe_steps=sum(row.unsafe for row in steps),
        unsafe_episode=int(any(row.unsafe for row in steps)),
        obstacle_events=sum(row.obstacle_event for row in steps),
        boundary_events=sum(row.boundary_event for row in steps),
        speed_events=sum(row.speed_event for row in steps),
        tilt_events=sum(row.tilt_event for row in steps),
        battery_events=sum(row.battery_event for row in steps),
        interventions=sum(row.intervention for row in steps),
        intervention_denominator=sum(row.intervention_denominator for row in steps),
        benign_interventions=sum(row.benign_intervention for row in steps),
        benign_intervention_denominator=sum(row.benign_intervention_denominator for row in steps),
        fallbacks=sum(row.fallback for row in steps),
        fallback_denominator=sum(row.fallback_denominator for row in steps),
        requested_preserved=sum(row.requested_preserved for row in steps),
        requested_certified_denominator=sum(row.requested_certified for row in steps),
        selected_certified=sum(row.selected_certified for row in steps),
        selected_denominator=sum(row.selected_denominator for row in steps),
        certified_candidates=sum(row.certified_candidates for row in steps),
        candidate_evaluations=sum(row.candidate_evaluations for row in steps),
    )


def _failed_episode(job: ShieldJob, error: Exception) -> ShieldEpisodeRow:
    return ShieldEpisodeRow(
        job_id=job.job_id,
        pair_id=job.pair_id,
        scenario_category=job.scenario_category,
        controller=job.controller,
        replicate=job.replicate,
        seed=job.seed,
        status=JobStatus.FAILED,
        failure_code=type(error).__name__,
        episode_denominator=1,
        elapsed_steps=0,
        step_denominator=0,
        total_return=0.0,
        goal_reached=False,
        terminated=False,
        truncated=False,
        unsafe_steps=0,
        unsafe_episode=0,
        obstacle_events=0,
        boundary_events=0,
        speed_events=0,
        tilt_events=0,
        battery_events=0,
        interventions=0,
        intervention_denominator=0,
        benign_interventions=0,
        benign_intervention_denominator=0,
        fallbacks=0,
        fallback_denominator=0,
        requested_preserved=0,
        requested_certified_denominator=0,
        selected_certified=0,
        selected_denominator=0,
        certified_candidates=0,
        candidate_evaluations=0,
    )


def run_shield_job(
    job: ShieldJob,
    *,
    scenario_factory: Callable[[], ScenarioSpec] | None = None,
    requested_policy: Callable[[ShieldState, ScenarioSpec], Action] = nominal_requested_policy,
) -> tuple[tuple[ShieldStepRow, ...], ShieldEpisodeRow]:
    """Run one clear job, retaining a failed attempt in its episode denominator."""

    if job.status is not JobStatus.PLANNED:
        raise ValueError("only planned jobs can run")
    try:
        scenario = (scenario_factory or SCENARIO_FACTORIES[job.scenario_category])()
        environment = WarehouseEnvironment(scenario, seed=job.seed)
        environment.reset(seed=job.seed)
        controller = _controller(job.controller)
        rows: list[ShieldStepRow] = []
        while not environment.done:
            requested = Action(requested_policy(environment.state, scenario))
            selected, fallback, requested_certified, selection_certified, certified, evaluated = (
                _select(environment.state, requested, scenario, controller)
            )
            result = environment.step(selected)
            benign = job.scenario_category in BENIGN_SCENARIO_CATEGORIES
            rows.append(
                ShieldStepRow(
                    job_id=job.job_id,
                    pair_id=job.pair_id,
                    step_denominator=1,
                    scenario_category=job.scenario_category,
                    controller=job.controller,
                    replicate=job.replicate,
                    seed=job.seed,
                    step=len(rows),
                    requested_action=requested.name,
                    executed_action=selected.name,
                    reward=result.reward,
                    unsafe=result.safety.unsafe,
                    obstacle_event=result.safety.obstacle_unsafe,
                    boundary_event=result.safety.boundary_unsafe,
                    speed_event=result.safety.speed_unsafe,
                    tilt_event=result.safety.tilt_unsafe,
                    battery_event=result.safety.battery_unsafe,
                    intervention=int(selected != requested),
                    intervention_denominator=1,
                    benign_intervention=int(benign and selected != requested),
                    benign_intervention_denominator=int(benign),
                    fallback=int(fallback),
                    fallback_denominator=1,
                    requested_certified=int(requested_certified),
                    requested_preserved=int(requested_certified and selected == requested),
                    goal_reached=(
                        (result.state.x - scenario.goal[0]) ** 2
                        + (result.state.y - scenario.goal[1]) ** 2
                        <= scenario.reward.goal_radius**2
                    ),
                    selected_certified=int(
                        controller.enabled and not fallback and selection_certified
                    ),
                    selected_denominator=int(controller.enabled and not fallback),
                    certified_candidates=certified,
                    candidate_evaluations=evaluated,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            )
        return tuple(rows), _aggregate_episode(job, rows)
    except Exception as error:
        return (), _failed_episode(job, error)


@dataclass(frozen=True, slots=True)
class WilsonBounds:
    lower: float
    upper: float
    numerator: int
    denominator: int
    confidence: float = 0.95

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def wilson_bounds(numerator: int, denominator: int, confidence: float = 0.95) -> WilsonBounds:
    if denominator < 0 or not 0 <= numerator <= denominator:
        raise ValueError("numerator must lie within denominator")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if denominator == 0:
        return WilsonBounds(0.0, 1.0, numerator, denominator, confidence)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = numerator / denominator
    divisor = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / divisor
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / divisor
    lower = 0.0 if numerator == 0 else max(0.0, centre - radius)
    upper = 1.0 if numerator == denominator else min(1.0, centre + radius)
    return WilsonBounds(lower, upper, numerator, denominator, confidence)


@dataclass(frozen=True, slots=True)
class BootstrapBounds:
    estimate: float
    lower: float
    upper: float
    repetitions: int
    pair_denominator: int
    category_denominator: int
    confidence: float = 0.95

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def paired_hierarchical_bootstrap(
    differences_by_category: Mapping[str, Sequence[float]],
    *,
    repetitions: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapBounds:
    """Resample categories, then paired episode differences within each category."""

    categories = tuple(sorted(differences_by_category))
    if (
        repetitions < 1
        or not categories
        or any(not differences_by_category[key] for key in categories)
    ):
        raise ValueError("bootstrap needs repetitions and non-empty categories")
    values = [value for key in categories for value in differences_by_category[key]]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    generator = random.Random(seed)
    category_means = [
        sum(differences_by_category[key]) / len(differences_by_category[key]) for key in categories
    ]
    draws: list[float] = []
    for _ in range(repetitions):
        sampled_category_means: list[float] = []
        for _category in categories:
            selected_category = generator.choice(categories)
            candidates = differences_by_category[selected_category]
            inner = [generator.choice(candidates) for _ in range(len(candidates))]
            sampled_category_means.append(sum(inner) / len(inner))
        draws.append(sum(sampled_category_means) / len(sampled_category_means))
    tail = (1 - confidence) / 2
    return BootstrapBounds(
        estimate=sum(category_means) / len(category_means),
        lower=_quantile(draws, tail),
        upper=_quantile(draws, 1 - tail),
        repetitions=repetitions,
        pair_denominator=len(values),
        category_denominator=len(categories),
        confidence=confidence,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(rows: Sequence[ShieldEpisodeRow]) -> dict[str, object]:
    denominator_names = (
        "episode_denominator",
        "step_denominator",
        "intervention_denominator",
        "benign_intervention_denominator",
        "fallback_denominator",
        "requested_certified_denominator",
        "selected_denominator",
        "candidate_evaluations",
    )
    count_names = (
        "unsafe_steps",
        "unsafe_episode",
        "obstacle_events",
        "boundary_events",
        "speed_events",
        "tilt_events",
        "battery_events",
        "interventions",
        "benign_interventions",
        "fallbacks",
        "requested_preserved",
        "selected_certified",
        "certified_candidates",
    )
    denominators = {name: sum(getattr(row, name) for row in rows) for name in denominator_names}
    counts = {name: sum(getattr(row, name) for row in rows) for name in count_names}
    goals = sum(row.goal_reached for row in rows)
    completed = sum(row.status is JobStatus.COMPLETED for row in rows)
    counts.update(
        {
            "goals": goals,
            "completed_jobs": completed,
            "failed_jobs": len(rows) - completed,
        }
    )
    rates = {
        "unsafe_step": _ratio(counts["unsafe_steps"], denominators["step_denominator"]),
        "unsafe_episode": _ratio(counts["unsafe_episode"], denominators["episode_denominator"]),
        "goal": _ratio(goals, denominators["episode_denominator"]),
        "intervention": _ratio(counts["interventions"], denominators["intervention_denominator"]),
        "benign_intervention": _ratio(
            counts["benign_interventions"], denominators["benign_intervention_denominator"]
        ),
        "fallback": _ratio(counts["fallbacks"], denominators["fallback_denominator"]),
        "requested_preservation": _ratio(
            counts["requested_preserved"], denominators["requested_certified_denominator"]
        ),
        "certified_candidate": _ratio(
            counts["selected_certified"], denominators["selected_denominator"]
        ),
        "all_candidate_certification": _ratio(
            counts["certified_candidates"], denominators["candidate_evaluations"]
        ),
    }
    return {
        "denominators": denominators,
        "counts": counts,
        "rates": rates,
        "wilson": {
            "unsafe_step": wilson_bounds(
                counts["unsafe_steps"], denominators["step_denominator"]
            ).to_dict(),
            "unsafe_episode": wilson_bounds(
                counts["unsafe_episode"], denominators["episode_denominator"]
            ).to_dict(),
            "goal": wilson_bounds(goals, denominators["episode_denominator"]).to_dict(),
            "benign_intervention": wilson_bounds(
                counts["benign_interventions"],
                denominators["benign_intervention_denominator"],
            ).to_dict(),
            "fallback": wilson_bounds(
                counts["fallbacks"], denominators["fallback_denominator"]
            ).to_dict(),
        },
        "return": {
            "sum": sum(row.total_return for row in rows),
            "mean": sum(row.total_return for row in rows) / completed if completed else 0.0,
        },
    }


@dataclass(frozen=True, slots=True)
class ShieldGateResult:
    name: str
    value: float | None
    threshold: float
    comparison: str
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ShieldStudySummary:
    complete: bool
    episode_denominator: int
    pair_denominator: int
    step_denominator: int
    category_episode_denominators: Mapping[str, int]
    category_step_denominators: Mapping[str, int]
    controllers: Mapping[str, object]
    categories: Mapping[str, object]
    paired_comparisons: Mapping[str, object]
    model_mismatch: Mapping[str, object]
    leave_one_constraint_out: Mapping[str, object]
    gates: tuple[ShieldGateResult, ...]
    all_gates_passed: bool
    schema_version: str = SUMMARY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "complete": self.complete,
            "episode_denominator": self.episode_denominator,
            "pair_denominator": self.pair_denominator,
            "step_denominator": self.step_denominator,
            "category_episode_denominators": dict(self.category_episode_denominators),
            "category_step_denominators": dict(self.category_step_denominators),
            "controllers": dict(self.controllers),
            "categories": dict(self.categories),
            "paired_comparisons": dict(self.paired_comparisons),
            "model_mismatch": dict(self.model_mismatch),
            "leave_one_constraint_out": dict(self.leave_one_constraint_out),
            "gates": [gate.to_dict() for gate in self.gates],
            "all_gates_passed": self.all_gates_passed,
        }


DEFAULT_GATES: Mapping[str, float] = {
    "minimum_unsafe_reduction_vs_disabled": 0.50,
    "minimum_lookahead_reduction_vs_h1": 0.25,
    "maximum_benign_intervention_rate": 0.01,
    "maximum_benign_intervention_wilson_upper": 0.015,
    "maximum_fallback_rate": 0.01,
    "maximum_fallback_wilson_upper": 0.02,
    "minimum_certified_requested_action_preservation": 1.0,
    "minimum_certified_candidate_rate": 1.0,
    "normalized_return_delta_lower_bound": -0.05,
    "goal_rate_delta_lower_bound": -0.02,
    "model_mismatch_max_unsafe_multiplier": 2.0,
}


def _gate(name: str, value: float | None, threshold: float, comparison: str) -> ShieldGateResult:
    if value is None or not math.isfinite(value):
        return ShieldGateResult(name, value, threshold, comparison, False)
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return ShieldGateResult(name, value, threshold, comparison, passed)


def _reduction(target: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if target == 0 else 0.0
    return 1 - target / baseline


def _paired(
    rows: Sequence[ShieldEpisodeRow],
    controller: str,
    baseline: str,
    function: Callable[[ShieldEpisodeRow, ShieldEpisodeRow], float],
) -> dict[str, list[float]]:
    lookup = {(row.pair_id, row.controller): row for row in rows}
    result: dict[str, list[float]] = defaultdict(list)
    for row in sorted(
        (item for item in rows if item.controller == controller),
        key=lambda item: (item.scenario_category, item.replicate, item.pair_id),
    ):
        other = lookup.get((row.pair_id, baseline))
        if other is None:
            raise ValueError(f"missing paired {baseline} row for {row.job_id}")
        if row.status is not JobStatus.COMPLETED or other.status is not JobStatus.COMPLETED:
            continue
        result[row.scenario_category].append(function(row, other))
    if not result:
        raise ValueError(f"no completed pairs for {controller} versus {baseline}")
    return result


def summarize_shield_study(
    episodes: Iterable[ShieldEpisodeRow],
    *,
    plan: ShieldStudyPlan | None = None,
    gates: Mapping[str, float] = DEFAULT_GATES,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int | None = None,
) -> ShieldStudySummary:
    """Summarize exact denominators, paired uncertainty, ablations, and all gates."""

    rows = tuple(episodes)
    if not rows or len({row.job_id for row in rows}) != len(rows):
        raise ValueError("summary needs non-empty, unique episode rows")
    observed = {row.job_id for row in rows}
    complete = all(row.status is JobStatus.COMPLETED for row in rows)
    if plan is not None:
        expected = {job.job_id for job in plan.jobs}
        if observed != expected:
            raise ValueError(
                "plan completeness mismatch: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
    required = {"disabled", "h1", "h2", *MODEL_MISMATCH_CONTROLLERS}
    missing = required - {row.controller for row in rows}
    if missing:
        raise ValueError(f"missing required controller cells: {sorted(missing)}")
    controllers = {
        controller: _aggregate(tuple(row for row in rows if row.controller == controller))
        for controller in sorted({row.controller for row in rows})
    }
    categories = {
        category: {
            controller: _aggregate(
                tuple(
                    row
                    for row in rows
                    if row.scenario_category == category and row.controller == controller
                )
            )
            for controller in sorted(
                {row.controller for row in rows if row.scenario_category == category}
            )
        }
        for category in sorted({row.scenario_category for row in rows})
    }
    seed = (
        bootstrap_seed
        if bootstrap_seed is not None
        else content_seed(SUMMARY_SCHEMA_VERSION, sorted(observed))
    )
    return_delta = paired_hierarchical_bootstrap(
        _paired(
            rows,
            "h2",
            "disabled",
            lambda target, baseline: (
                (target.total_return - baseline.total_return) / max(1.0, abs(baseline.total_return))
            ),
        ),
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    goal_delta = paired_hierarchical_bootstrap(
        _paired(
            rows,
            "h2",
            "disabled",
            lambda target, baseline: float(target.goal_reached) - float(baseline.goal_reached),
        ),
        repetitions=bootstrap_repetitions,
        seed=content_seed(seed, "goal"),
    )
    h2 = cast(dict[str, Any], controllers["h2"])
    h1 = cast(dict[str, Any], controllers["h1"])
    disabled = cast(dict[str, Any], controllers["disabled"])
    h2_rates = cast(dict[str, float], h2["rates"])
    h1_rates = cast(dict[str, float], h1["rates"])
    disabled_rates = cast(dict[str, float], disabled["rates"])
    mismatch_rates = {
        name: float(cast(dict[str, Any], controllers[name])["rates"]["unsafe_step"])
        for name in MODEL_MISMATCH_CONTROLLERS
    }
    nominal = mismatch_rates["model_nominal"]
    mismatch_multipliers: dict[str, float | None] = {
        name: (rate / nominal if nominal else (1.0 if rate == 0 else None))
        for name, rate in mismatch_rates.items()
    }
    finite_multipliers = [value for value in mismatch_multipliers.values() if value is not None]
    mismatch_max = (
        max(finite_multipliers) if len(finite_multipliers) == len(mismatch_multipliers) else None
    )
    wilson = cast(dict[str, dict[str, float]], h2["wilson"])
    gate_results = (
        _gate(
            "minimum_unsafe_reduction_vs_disabled",
            _reduction(h2_rates["unsafe_step"], disabled_rates["unsafe_step"]),
            gates["minimum_unsafe_reduction_vs_disabled"],
            ">=",
        ),
        _gate(
            "minimum_lookahead_reduction_vs_h1",
            _reduction(h2_rates["unsafe_step"], h1_rates["unsafe_step"]),
            gates["minimum_lookahead_reduction_vs_h1"],
            ">=",
        ),
        _gate(
            "maximum_benign_intervention_rate",
            h2_rates["benign_intervention"],
            gates["maximum_benign_intervention_rate"],
            "<=",
        ),
        _gate(
            "maximum_benign_intervention_wilson_upper",
            wilson["benign_intervention"]["upper"],
            gates["maximum_benign_intervention_wilson_upper"],
            "<=",
        ),
        _gate(
            "maximum_fallback_rate",
            h2_rates["fallback"],
            gates["maximum_fallback_rate"],
            "<=",
        ),
        _gate(
            "maximum_fallback_wilson_upper",
            wilson["fallback"]["upper"],
            gates["maximum_fallback_wilson_upper"],
            "<=",
        ),
        _gate(
            "minimum_certified_requested_action_preservation",
            h2_rates["requested_preservation"],
            gates["minimum_certified_requested_action_preservation"],
            ">=",
        ),
        _gate(
            "minimum_certified_candidate_rate",
            h2_rates["certified_candidate"],
            gates["minimum_certified_candidate_rate"],
            ">=",
        ),
        _gate(
            "normalized_return_delta_lower_bound",
            return_delta.lower,
            gates["normalized_return_delta_lower_bound"],
            ">=",
        ),
        _gate(
            "goal_rate_delta_lower_bound",
            goal_delta.lower,
            gates["goal_rate_delta_lower_bound"],
            ">=",
        ),
        _gate(
            "model_mismatch_max_unsafe_multiplier",
            mismatch_max,
            gates["model_mismatch_max_unsafe_multiplier"],
            "<=",
        ),
    )
    leave_out: dict[str, object] = {}
    for controller, family in LEAVE_OUT_CONTROLLERS.items():
        if controller not in controllers:
            continue
        aggregate = controllers[controller]
        event_name = f"{family.value}_events"
        omitted_events = aggregate["counts"][event_name]  # type: ignore[index]
        if family is MarginFamily.OBSTACLE:
            omitted_events += aggregate["counts"]["boundary_events"]  # type: ignore[index]
        leave_out[family.value] = {
            "controller": controller,
            "unsafe_step_rate": aggregate["rates"]["unsafe_step"],  # type: ignore[index]
            "unsafe_step_rate_delta_vs_h2": (
                aggregate["rates"]["unsafe_step"] - h2_rates["unsafe_step"]  # type: ignore[index]
            ),
            "omitted_family_events": omitted_events,
            "event_step_denominator": aggregate["denominators"][  # type: ignore[index]
                "step_denominator"
            ],
        }
    category_names = sorted({row.scenario_category for row in rows})
    return ShieldStudySummary(
        complete=complete,
        episode_denominator=sum(row.episode_denominator for row in rows),
        pair_denominator=len({row.pair_id for row in rows}),
        step_denominator=sum(row.step_denominator for row in rows),
        category_episode_denominators={
            category: sum(
                row.episode_denominator for row in rows if row.scenario_category == category
            )
            for category in category_names
        },
        category_step_denominators={
            category: sum(row.step_denominator for row in rows if row.scenario_category == category)
            for category in category_names
        },
        controllers=controllers,
        categories=categories,
        paired_comparisons={
            "h2_vs_disabled_normalized_return_delta": return_delta.to_dict(),
            "h2_vs_disabled_goal_rate_delta": goal_delta.to_dict(),
        },
        model_mismatch={
            "unsafe_step_rates": mismatch_rates,
            "unsafe_multipliers_vs_nominal": mismatch_multipliers,
            "maximum_unsafe_multiplier": mismatch_max,
        },
        leave_one_constraint_out=leave_out,
        gates=gate_results,
        all_gates_passed=complete and all(gate.passed for gate in gate_results),
    )
