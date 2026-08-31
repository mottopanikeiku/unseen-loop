from __future__ import annotations

import dataclasses
import hashlib
import json
from itertools import islice
from pathlib import Path

import pytest

from unseen_loop.flagship.manifest import (
    ManifestError,
    PlannedJob,
    assert_disjoint_job_seeds,
    iter_stage_jobs,
    load_manifest,
    parse_manifest_bytes,
    planned_job_ids,
    stage_dag,
)
from unseen_loop.flagship.registry import (
    AppendOnlyRegistry,
    JobStatus,
    Provenance,
    RegistryError,
    finalize_evidence,
)

FLAGSHIP = Path(__file__).parents[1] / "experiments" / "flagship.toml"
SMOKE = Path(__file__).parents[1] / "experiments" / "flagship-smoke.toml"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _provenance(config_digest: str = DIGEST_B) -> Provenance:
    return Provenance.from_mapping(
        source_digest=DIGEST_A,
        config_digest=config_digest,
        image_digests={"core": DIGEST_C},
    )


def _job(job_id: str, *, invalid: bool = False) -> PlannedJob:
    kind = "invalid" if invalid else "unit"
    return PlannedJob(
        job_id,
        "unit_stage",
        int(hashlib.sha256(job_id.encode()).hexdigest(), 16),
        (("kind", kind),),
    )


def test_flagship_manifest_is_strict_typed_and_immutable() -> None:
    manifest = load_manifest(FLAGSHIP)

    assert manifest.schema_version == "unseen-loop/flagship-study-v1"
    assert manifest.shield.output_shape == (5, 2, 4)
    assert manifest.ope.horizons == (8, 32, 64)
    assert manifest.digest == hashlib.sha256(FLAGSHIP.read_bytes()).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.name = "replacement"  # type: ignore[misc]

    extra = FLAGSHIP.read_text().replace(
        'name = "ciphershield-private-ope-flagship-2026-08"',
        'name = "ciphershield-private-ope-flagship-2026-08"\nunknown = 1',
    )
    with pytest.raises(ManifestError, match="extra"):
        parse_manifest_bytes(extra.encode())

    inconsistent = FLAGSHIP.read_text().replace("valid_calls = 448", "valid_calls = 447")
    with pytest.raises(ManifestError, match="denominator"):
        parse_manifest_bytes(inconsistent.encode())


def test_stage_dag_is_topological_bounded_and_content_derived() -> None:
    manifest = load_manifest(FLAGSHIP)
    stages = stage_dag(manifest)
    seen: set[str] = set()
    for stage in stages:
        assert set(stage.dependencies) <= seen
        assert 1 <= stage.max_parallel <= 64
        assert manifest.digest[:8] not in stage.stage_id
        seen.add(stage.name)

    modified = parse_manifest_bytes(
        FLAGSHIP.read_text()
        .replace(
            'seed_root = "unseen-loop-flagship-2026-08"',
            'seed_root = "unseen-loop-flagship-2026-08-replay"',
        )
        .encode()
    )
    assert [stage.stage_id for stage in stage_dag(modified)] != [stage.stage_id for stage in stages]


def test_every_configured_denominator_expands_to_explicit_stable_jobs() -> None:
    manifest = load_manifest(FLAGSHIP)
    expected = {
        "clear_shield_matrix": 12 * 13 * 512,
        "shield_fhe_challenge": 448 + 64,
        "ope_validation": 32 + 4096 + 3 * 3 * 3 * 4 * 2 * 100 + 96 + 16,
        "integration": 147456 + 73728 + 1152,
        "timing": 8 + 8 + 3 * 3 * 4 + 8,
        "analysis": 1,
        "evidence_finalizer": 1,
    }
    for stage, count in expected.items():
        assert sum(1 for _ in iter_stage_jobs(manifest, stage)) == count

    first = tuple(islice(iter_stage_jobs(manifest, "clear_shield_matrix"), 256))
    again = tuple(islice(iter_stage_jobs(manifest, "clear_shield_matrix"), 256))
    assert first == again
    assert len({job.job_id for job in first}) == len(first)
    assert_disjoint_job_seeds(first)
    assert planned_job_ids(manifest, "analysis") == (
        next(iter_stage_jobs(manifest, "analysis")).job_id,
    )


def test_integration_commits_all_trajectory_jobs_before_ope_consumers() -> None:
    manifest = load_manifest(SMOKE)
    seen_ope = False
    trajectory_jobs = 0
    ope_jobs = 0
    for job in iter_stage_jobs(manifest, "integration"):
        kind = job.coordinate_dict()["kind"]
        if kind == "real_fhe_ope":
            seen_ope = True
            ope_jobs += 1
        else:
            assert not seen_ope
            trajectory_jobs += 1

    assert trajectory_jobs == (
        manifest.integration.expected_behavior_trajectories
        + manifest.integration.expected_direct_trajectories
    )
    assert ope_jobs == manifest.integration.expected_real_fhe_calls


def test_registry_is_append_only_and_does_not_replace_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text('{"ok":true}\n')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    succeeded = _job("job-unit-success")
    failed = _job("job-unit-failed")
    registry = AppendOnlyRegistry.create(
        tmp_path / "registry.jsonl",
        jobs=(succeeded, failed),
        provenance=_provenance(),
    )

    registry.started(succeeded.job_id)
    registry.succeeded(succeeded.job_id, artifact_path=artifact.name, artifact_digest=digest)
    registry.started(failed.job_id)
    registry.failed(failed.job_id, reason_code="worker.deliberate-failure")

    snapshot = registry.snapshot()
    statuses = {record.job_id: record.status for record in snapshot.records}
    assert statuses == {
        succeeded.job_id: JobStatus.SUCCEEDED,
        failed.job_id: JobStatus.FAILED,
    }
    with pytest.raises(RegistryError, match="retries cannot replace"):
        registry.started(failed.job_id)
    with pytest.raises(RegistryError, match="requires exactly one started"):
        registry.succeeded(failed.job_id, artifact_path=artifact.name, artifact_digest=digest)
    with pytest.raises(RegistryError, match="failed, timed-out"):
        finalize_evidence(registry, evidence_root=tmp_path)


def test_finalization_rejects_incomplete_extra_and_missing_artifacts(tmp_path: Path) -> None:
    job = _job("job-unit-incomplete")
    registry = AppendOnlyRegistry.create(
        tmp_path / "registry.jsonl", jobs=(job,), provenance=_provenance()
    )
    registry.started(job.job_id)
    with pytest.raises(RegistryError, match="incomplete"):
        finalize_evidence(registry, evidence_root=tmp_path)
    with pytest.raises(RegistryError, match="immutable plan"):
        registry.started("job-extra")

    other_root = tmp_path / "missing"
    other_root.mkdir()
    other_registry = AppendOnlyRegistry.create(
        other_root / "registry.jsonl",
        jobs=(_job("job-missing-artifact"),),
        provenance=_provenance(),
    )
    other_registry.started("job-missing-artifact")
    other_registry.succeeded(
        "job-missing-artifact", artifact_path="absent.json", artifact_digest=DIGEST_A
    )
    with pytest.raises(RegistryError, match="missing regular artifact"):
        finalize_evidence(other_registry, evidence_root=other_root)


def test_finalizer_closes_one_root_index_and_validates_rejections(tmp_path: Path) -> None:
    artifact = tmp_path / "result.bin"
    artifact.write_bytes(b"encrypted-output")
    succeeded = _job("job-unit-evidence")
    rejected = _job("job-unit-invalid", invalid=True)
    registry = AppendOnlyRegistry.create(
        tmp_path / "registry.jsonl",
        jobs=(succeeded, rejected),
        provenance=_provenance(),
    )
    registry.started(succeeded.job_id)
    registry.succeeded(
        succeeded.job_id,
        artifact_path=artifact.name,
        artifact_digest=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    registry.started(rejected.job_id)
    registry.rejected(rejected.job_id, reason_code="input.out-of-domain")

    index = finalize_evidence(registry, evidence_root=tmp_path)
    payload = json.loads(index.read_text())
    assert index.name == "evidence-index.json"
    assert payload["planned_job_ids"] == sorted((succeeded.job_id, rejected.job_id))
    assert payload["status_counts"] == {"rejected": 1, "succeeded": 1}
    assert payload["provenance"]["source_digest"] == DIGEST_A
    with pytest.raises(RegistryError, match="replace"):
        finalize_evidence(registry, evidence_root=tmp_path, reject_extra_files=False)


def test_registry_detects_hash_chain_tampering(tmp_path: Path) -> None:
    job = _job("job-unit-tamper")
    path = tmp_path / "registry.jsonl"
    registry = AppendOnlyRegistry.create(path, jobs=(job,), provenance=_provenance())
    registry.started(job.job_id)
    lines = path.read_text().splitlines()
    event = json.loads(lines[1])
    event["job_id"] = "job-unit-substituted"
    lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(RegistryError, match="hash chain"):
        registry.snapshot()
