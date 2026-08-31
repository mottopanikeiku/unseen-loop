"""Execute one clear CipherShield flagship job and retain canonical public evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from unseen_loop.shield.study import (
    PLAN_SCHEMA_VERSION,
    SCENARIO_CATEGORIES,
    ShieldJob,
    make_scenario,
    run_shield_job,
)
from unseen_loop.shield.types import ScenarioSpec

from .manifest import canonical_json, content_digest

ARTIFACT_SCHEMA_VERSION = "unseen-loop/flagship-clear-shield-artifact-v1"
_STAGE = "clear_shield_matrix"
_JOB_ID = re.compile(r"job-clear_shield_matrix-[0-9a-f]{24}\Z")
_JOB_KEYS = frozenset(("job_id", "stage", "seed", "coordinates"))
_COORDINATE_KEYS = frozenset(("scenario", "controller", "repetition"))


def _rejected(reason_code: str) -> dict[str, str | None]:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason_code,
    }


def _valid_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _map_job(
    manifest: Mapping[str, Any], job: Mapping[str, Any]
) -> tuple[ShieldJob, str, str, str, ScenarioSpec] | str:
    if set(job) != _JOB_KEYS:
        return "clear-shield.invalid-job"
    flagship_job_id = job.get("job_id")
    if not isinstance(flagship_job_id, str) or _JOB_ID.fullmatch(flagship_job_id) is None:
        return "clear-shield.invalid-job-id"
    if job.get("stage") != _STAGE:
        return "clear-shield.invalid-stage"
    seed = job.get("seed")
    if not _valid_integer(seed):
        return "clear-shield.invalid-seed"
    coordinates = job.get("coordinates")
    if not isinstance(coordinates, Mapping) or set(coordinates) != _COORDINATE_KEYS:
        return "clear-shield.unknown-coordinates"
    scenario_index = coordinates.get("scenario")
    repetition = coordinates.get("repetition")
    controller = coordinates.get("controller")
    if not _valid_integer(scenario_index) or not _valid_integer(repetition):
        return "clear-shield.unknown-coordinates"
    if not isinstance(controller, str):
        return "clear-shield.unknown-coordinates"
    scenario_index = cast(int, scenario_index)
    repetition = cast(int, repetition)
    seed = cast(int, seed)

    shield = manifest.get("shield")
    seed_root = manifest.get("seed_root")
    if not isinstance(shield, Mapping) or not isinstance(seed_root, str) or not seed_root:
        return "clear-shield.invalid-manifest"
    scenario_count = shield.get("scenarios")
    repetition_count = shield.get("seeds_per_controller_cell")
    controllers = shield.get("controller_cells")
    if (
        type(scenario_count) is not int
        or scenario_count != len(SCENARIO_CATEGORIES)
        or type(repetition_count) is not int
        or repetition_count <= 0
        or not isinstance(controllers, (list, tuple))
        or not all(isinstance(item, str) for item in controllers)
    ):
        return "clear-shield.invalid-manifest"
    if (
        scenario_index >= scenario_count
        or repetition >= repetition_count
        or controller not in controllers
    ):
        return "clear-shield.unknown-coordinates"

    scenario_category = SCENARIO_CATEGORIES[scenario_index]
    manifest_digest = content_digest(manifest)
    normalized_job = {
        "job_id": flagship_job_id,
        "stage": _STAGE,
        "seed": seed,
        "coordinates": {
            "scenario": scenario_index,
            "controller": controller,
            "repetition": repetition,
        },
    }
    job_digest = content_digest(normalized_job)
    pair_id = content_digest(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "seed_root": seed_root,
            "scenario_category": scenario_category,
            "replicate": repetition,
        }
    )
    shield_job = ShieldJob(
        job_id=content_digest({"pair_id": pair_id, "controller": controller}),
        pair_id=pair_id,
        scenario_category=scenario_category,
        controller=controller,
        replicate=repetition,
        seed=seed,
    )
    scenario = make_scenario(scenario_category)
    spec_digest = content_digest(scenario.to_dict())
    return shield_job, manifest_digest, job_digest, spec_digest, scenario


def execute_flagship_job(
    manifest: object, job: object, evidence_root: str | Path
) -> dict[str, str | None]:
    """Run exactly one clear episode, retaining completed and failed attempts alike."""

    if not isinstance(manifest, Mapping) or not isinstance(job, Mapping):
        return _rejected("clear-shield.invalid-input")
    mapped = _map_job(
        cast(Mapping[str, Any], manifest),
        cast(Mapping[str, Any], job),
    )
    if isinstance(mapped, str):
        return _rejected(mapped)
    shield_job, manifest_digest, job_digest, spec_digest, scenario = mapped
    steps, episode = run_shield_job(shield_job, scenario_factory=lambda: scenario)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "manifest_digest": manifest_digest,
        "job_digest": job_digest,
        "spec_digest": spec_digest,
        "status": episode.status.value,
        "stage": _STAGE,
        "job_id": job["job_id"],
        "scenario_category": shield_job.scenario_category,
        "controller": shield_job.controller,
        "step_denominator": episode.step_denominator,
        "category_denominators": {
            shield_job.scenario_category: episode.episode_denominator,
        },
        "episode": episode.to_dict(),
        "steps": [step.to_dict() for step in steps],
    }
    payload = canonical_json(artifact) + b"\n"
    relative_path = Path(_STAGE) / f"{job['job_id']}.json"
    destination = Path(evidence_root) / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    return {
        "status": "succeeded",
        "artifact_path": relative_path.as_posix(),
        "artifact_digest": hashlib.sha256(payload).hexdigest(),
        "reason_code": None,
    }
