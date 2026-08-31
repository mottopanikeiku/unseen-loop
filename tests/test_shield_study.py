from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from unseen_loop.shield.study import (
    CONTROLLER_CELLS,
    EPISODE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    SCENARIO_CATEGORIES,
    STEP_SCHEMA_VERSION,
    JobStatus,
    ShieldEpisodeRow,
    build_shield_plan,
    paired_hierarchical_bootstrap,
    run_shield_job,
    summarize_shield_study,
    wilson_bounds,
)
from unseen_loop.shield.types import (
    DynamicsConfig,
    RewardSpec,
    SafetyLimits,
    ScenarioSpec,
    ShieldState,
)

ROOT = Path(__file__).parents[1]


def _manifest(*, seeds: int = 1, controllers: tuple[str, ...] = CONTROLLER_CELLS) -> dict:
    return {
        "seed_root": "tiny-content-root",
        "shield": {
            "scenarios": 12,
            "seeds_per_controller_cell": seeds,
            "controller_cells": list(controllers),
        },
    }


def _tiny_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        initial_state=ShieldState(0, 0, 0, 0, 1, 0),
        goal=(8, 0),
        safety=SafetyLimits(max_speed=4, max_abs_tilt=1),
        dynamics=DynamicsConfig(),
        reward=RewardSpec(),
        reset_jitter=(0, 0, 0, 0, 0, 0),
        max_steps=3,
    )


def test_flagship_plan_has_exact_paired_matrix_and_content_ids() -> None:
    first = build_shield_plan(ROOT / "experiments" / "flagship.toml")
    second = build_shield_plan(ROOT / "experiments" / "flagship.toml")

    assert first.schema_version == PLAN_SCHEMA_VERSION
    assert first.episode_count == 79_872
    assert first.pair_count == 12 * 512
    assert first.jobs == second.jobs
    assert len({job.job_id for job in first.jobs}) == 79_872

    pair = [
        job
        for job in first.jobs
        if job.scenario_category == SCENARIO_CATEGORIES[0] and job.replicate == 0
    ]
    assert len(pair) == 13
    assert len({job.pair_id for job in pair}) == 1
    assert len({job.seed for job in pair}) == 1
    assert {job.controller for job in pair} == set(CONTROLLER_CELLS)
    assert all(job.status is JobStatus.PLANNED for job in pair)
    assert json.loads(json.dumps(first.jobs[0].to_dict()))["status"] == "planned"


def test_tiny_plan_override_retains_category_controller_pairing() -> None:
    plan = build_shield_plan(
        _manifest(seeds=2),
        scenario_categories=("benign_open_floor", "overspeed"),
        controller_cells=("disabled", "h2"),
        seeds_per_controller_cell=2,
    )
    assert plan.episode_count == 8
    assert plan.pair_count == 4
    grouped: dict[str, list] = {}
    for job in plan.jobs:
        grouped.setdefault(job.pair_id, []).append(job)
    assert all({job.controller for job in jobs} == {"disabled", "h2"} for jobs in grouped.values())
    assert all(len({job.seed for job in jobs}) == 1 for jobs in grouped.values())

    @dataclass(frozen=True, slots=True)
    class ShieldSection:
        scenarios: int
        seeds_per_controller_cell: int
        controller_cells: tuple[str, ...]

    @dataclass(frozen=True, slots=True)
    class Manifest:
        seed_root: str
        shield: ShieldSection
        digest: str

    object_plan = build_shield_plan(
        Manifest(
            seed_root="object-root",
            shield=ShieldSection(12, 1, (CONTROLLER_CELLS[0],)),
            digest="unused-by-seed-contract",
        ),
        scenario_categories=(SCENARIO_CATEGORIES[0],),
    )
    assert object_plan.episode_count == 1


def test_clear_job_persists_exact_elapsed_denominators_without_private_state() -> None:
    plan = build_shield_plan(
        _manifest(),
        scenario_categories=("benign_open_floor",),
        controller_cells=("h2",),
    )
    steps, episode = run_shield_job(plan.jobs[0], scenario_factory=_tiny_scenario)

    assert episode.status is JobStatus.COMPLETED
    assert episode.schema_version == EPISODE_SCHEMA_VERSION
    assert len(steps) == episode.elapsed_steps == episode.step_denominator == 3
    assert episode.episode_denominator == 1
    assert episode.intervention_denominator == 3
    assert episode.benign_intervention_denominator == 3
    assert episode.fallback_denominator == 3
    assert all(step.schema_version == STEP_SCHEMA_VERSION for step in steps)
    assert [step.step for step in steps] == list(range(len(steps)))
    assert all(step.intervention_denominator == 1 for step in steps)
    assert sum(step.reward for step in steps) == episode.total_return
    serialized = json.dumps([step.to_dict() for step in steps] + [episode.to_dict()])
    assert '"state"' not in serialized
    assert "margin" not in serialized
    assert all(step.step_denominator == 1 for step in steps)


def test_failed_job_is_retained_in_episode_denominator() -> None:
    job = build_shield_plan(
        _manifest(),
        scenario_categories=("benign_open_floor",),
        controller_cells=("h2",),
    ).jobs[0]

    def fail() -> ScenarioSpec:
        raise RuntimeError("private detail is not persisted")

    steps, episode = run_shield_job(job, scenario_factory=fail)
    assert steps == ()
    assert episode.status is JobStatus.FAILED
    assert episode.failure_code == "RuntimeError"
    assert episode.episode_denominator == 1
    assert episode.step_denominator == 0
    assert "private detail" not in json.dumps(episode.to_dict())


def test_wilson_and_paired_hierarchical_bootstrap_are_deterministic() -> None:
    interval = wilson_bounds(0, 10)
    assert interval.lower == 0
    assert 0 < interval.upper < 0.5
    assert wilson_bounds(0, 0).upper == 1

    differences = {"a": (1.0, 2.0), "b": (-1.0, 0.0)}
    first = paired_hierarchical_bootstrap(differences, repetitions=100, seed=17)
    second = paired_hierarchical_bootstrap(differences, repetitions=100, seed=17)
    assert first == second
    assert first.estimate == pytest.approx(0.5)
    assert first.pair_denominator == 4
    assert first.category_denominator == 2


def test_summary_has_category_denominators_ablations_variants_and_all_gates() -> None:
    plan = build_shield_plan(
        _manifest(),
        scenario_categories=("benign_open_floor",),
        controller_cells=CONTROLLER_CELLS,
    )
    episodes: list[ShieldEpisodeRow] = []
    for job in plan.jobs:
        _, episode = run_shield_job(job, scenario_factory=_tiny_scenario)
        episodes.append(episode)

    summary = summarize_shield_study(
        episodes,
        plan=plan,
        bootstrap_repetitions=40,
        bootstrap_seed=4,
    )
    payload = summary.to_dict()
    assert summary.complete
    assert summary.episode_denominator == 13
    assert summary.pair_denominator == 1
    assert summary.step_denominator == sum(episode.elapsed_steps for episode in episodes)
    assert summary.category_episode_denominators == {"benign_open_floor": 13}
    assert summary.category_step_denominators["benign_open_floor"] == summary.step_denominator
    assert set(summary.leave_one_constraint_out) == {"obstacle", "speed", "tilt", "battery"}
    assert set(summary.model_mismatch["unsafe_step_rates"]) == {
        "model_minus_10pct",
        "model_nominal",
        "model_plus_10pct",
    }
    assert {gate.name for gate in summary.gates} == {
        "minimum_unsafe_reduction_vs_disabled",
        "minimum_lookahead_reduction_vs_h1",
        "maximum_benign_intervention_rate",
        "maximum_benign_intervention_wilson_upper",
        "maximum_fallback_rate",
        "maximum_fallback_wilson_upper",
        "minimum_certified_requested_action_preservation",
        "minimum_certified_candidate_rate",
        "normalized_return_delta_lower_bound",
        "goal_rate_delta_lower_bound",
        "model_mismatch_max_unsafe_multiplier",
    }
    assert payload["schema_version"] == "unseen-loop/shield-study-summary-v1"
    json.dumps(payload, allow_nan=False)


def test_summary_rejects_missing_planned_attempt() -> None:
    plan = build_shield_plan(
        _manifest(),
        scenario_categories=("benign_open_floor",),
        controller_cells=CONTROLLER_CELLS,
    )
    episodes = [run_shield_job(job, scenario_factory=_tiny_scenario)[1] for job in plan.jobs]
    with pytest.raises(ValueError, match="plan completeness mismatch"):
        summarize_shield_study(episodes[:-1], plan=plan, bootstrap_repetitions=5)
