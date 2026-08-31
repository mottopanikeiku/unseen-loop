from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from unseen_loop.flagship.executor_analysis import (
    AnalysisError,
    _expected_jobs,
    _verify_evidence,
    execute_flagship_job,
)
from unseen_loop.flagship.manifest import PlannedJob, canonical_json, content_digest
from unseen_loop.flagship.registry import (
    AppendOnlyRegistry,
    JobStatus,
    Provenance,
    Transition,
)


def _tiny_manifest(*, timing_jobs: int = 1) -> dict[str, object]:
    return {
        "schema_version": "unseen-loop/flagship-study-v1",
        "name": "tiny-closed-evidence",
        "seed_root": "tiny-seed-root",
        "claims": {"novel_conjunction": "test", "forbidden": []},
        "shield": {
            "scenarios": 0,
            "seeds_per_controller_cell": 0,
            "controller_cells": [],
            "fhe_challenge": {
                "occupancy_states": 0,
                "extrema_states": 0,
                "threshold_states": 0,
                "tie_states": 0,
                "canary_states": 0,
                "canary_encryptions_per_state": 0,
                "invalid_domain_rejections": 0,
            },
        },
        "ope": {
            "reference": {"analytic_fixtures": 0, "random_fixed_point_cases": 0},
            "clip_values": [],
            "include_unclipped": False,
            "horizons": [],
            "trajectory_counts": [],
            "overlap_lambdas": [],
            "estimators": [],
            "independent_batches": 0,
            "fhe_challenge": {
                "occupancy_batches": 0,
                "extrema_batches": 0,
                "terminal_padding_batches": 0,
                "rounding_boundary_batches": 0,
                "invalid_batch_rejections": 0,
            },
        },
        "integration": {
            "behavior_trajectories_per_cell": 0,
            "ope_batch_trajectories": 1,
            "direct_target_trajectories_per_cell": 0,
            "scenarios": 0,
            "shield_modes": [],
            "outcomes": [],
        },
        "systems": {
            "shield_timing_containers": timing_jobs,
            "ope_timing_containers": 0,
            "scale_trajectory_counts": [],
            "scale_horizons": [],
            "scale_containers_per_cell": 0,
            "concurrent_clients": 0,
        },
    }


def _planned_jobs(manifest: dict[str, object]) -> tuple[PlannedJob, ...]:
    return tuple(
        PlannedJob(item.job_id, item.stage, item.seed, tuple(item.coordinates.items()))
        for item in _expected_jobs(manifest)
    )


def _registry(
    root: Path,
    manifest: dict[str, object],
    *,
    jobs: tuple[PlannedJob, ...] | None = None,
) -> AppendOnlyRegistry:
    provenance = Provenance.from_mapping(
        source_digest="1" * 64,
        config_digest=content_digest(manifest),
        image_digests={"core": "2" * 64},
    )
    return AppendOnlyRegistry.create(
        root / "registry.jsonl",
        jobs=_planned_jobs(manifest) if jobs is None else jobs,
        provenance=provenance,
    )


def _close_timing_and_start_analysis(
    root: Path, manifest: dict[str, object], registry: AppendOnlyRegistry
) -> tuple[dict[str, object], Path]:
    expected = _expected_jobs(manifest)
    timing = next(item for item in expected if item.stage == "timing")
    analysis = next(item for item in expected if item.stage == "analysis")
    relative = f"timing/{timing.job_id}.json"
    artifact = root / relative
    artifact.parent.mkdir()
    raw = (
        canonical_json(
            {
                "job_id": timing.job_id,
                "stage": timing.stage,
                "coordinates": timing.coordinates,
                "timing": {"server_p95_seconds": 0.1, "client_p95_seconds": 0.2},
            }
        )
        + b"\n"
    )
    artifact.write_bytes(raw)
    registry.apply(
        (
            Transition(timing.job_id, JobStatus.STARTED),
            Transition(
                timing.job_id,
                JobStatus.SUCCEEDED,
                artifact_path=relative,
                artifact_digest=hashlib.sha256(raw).hexdigest(),
            ),
            Transition(analysis.job_id, JobStatus.STARTED),
        )
    )
    return {
        "job_id": analysis.job_id,
        "stage": analysis.stage,
        "seed": analysis.seed,
        "coordinates": analysis.coordinates,
    }, artifact


def test_closed_fixture_accepts_exact_denominator_and_hashes(tmp_path: Path) -> None:
    manifest = _tiny_manifest()
    registry = _registry(tmp_path, manifest)
    job, _artifact = _close_timing_and_start_analysis(tmp_path, manifest, registry)

    artifacts, closure = _verify_evidence(manifest, job, tmp_path)

    assert len(artifacts["timing"]) == 1
    assert closure["planned_jobs"] == 3
    assert closure["stage_artifact_counts"]["timing"] == 1
    assert len(closure["stage_artifact_set_sha256"]["timing"]) == 64


def test_missing_registry_denominator_is_rejected(tmp_path: Path) -> None:
    manifest = _tiny_manifest()
    complete = _planned_jobs(manifest)
    incomplete = tuple(job for job in complete if job.stage != "timing")
    registry = _registry(tmp_path, manifest, jobs=incomplete)
    analysis = next(job for job in incomplete if job.stage == "analysis")
    registry.started(analysis.job_id)
    payload = {
        "job_id": analysis.job_id,
        "stage": analysis.stage,
        "seed": analysis.seed,
        "coordinates": analysis.coordinate_dict(),
    }

    with pytest.raises(AnalysisError, match="plan has a missing or extra"):
        _verify_evidence(manifest, payload, tmp_path)


def test_registered_artifact_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = _tiny_manifest()
    registry = _registry(tmp_path, manifest)
    job, artifact = _close_timing_and_start_analysis(tmp_path, manifest, registry)
    artifact.write_bytes(
        canonical_json(
            {
                "job_id": "tampered",
                "stage": "timing",
                "timing": {"server_p95_seconds": 0.1, "client_p95_seconds": 0.2},
            }
        )
        + b"\n"
    )

    with pytest.raises(AnalysisError, match="digest does not match"):
        _verify_evidence(manifest, job, tmp_path)


def test_unregistered_extra_file_is_rejected(tmp_path: Path) -> None:
    manifest = _tiny_manifest()
    registry = _registry(tmp_path, manifest)
    job, _artifact = _close_timing_and_start_analysis(tmp_path, manifest, registry)
    (tmp_path / "private-values.json").write_text("{}\n")

    with pytest.raises(AnalysisError, match="missing or extra files"):
        _verify_evidence(manifest, job, tmp_path)


def test_invalid_analysis_job_rejects_without_an_artifact(tmp_path: Path) -> None:
    result = execute_flagship_job(
        _tiny_manifest(),
        {
            "job_id": "job-analysis-not-planned",
            "stage": "analysis",
            "seed": 0,
            "coordinates": {"kind": "singleton"},
        },
        tmp_path,
    )

    assert result == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "analysis.unverifiable-evidence",
    }
    assert not (tmp_path / "analysis").exists()
