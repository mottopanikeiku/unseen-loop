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
    supporting = tmp_path / "shared" / "server.zip"
    supporting.parent.mkdir()
    supporting.write_bytes(b"public-server-artifact")
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

    index = finalize_evidence(
        registry,
        evidence_root=tmp_path,
        supporting_paths=("shared/server.zip",),
    )
    payload = json.loads(index.read_text())
    assert index.name == "evidence-index.json"
    assert payload["planned_job_ids"] == sorted((succeeded.job_id, rejected.job_id))
    assert payload["status_counts"] == {"rejected": 1, "succeeded": 1}
    assert payload["provenance"]["source_digest"] == DIGEST_A
    assert payload["supporting_artifacts"] == {
        "shared/server.zip": hashlib.sha256(supporting.read_bytes()).hexdigest()
    }
    closed_bytes = index.read_bytes()
    closed_mtime = index.stat().st_mtime_ns
    assert (
        finalize_evidence(
            registry,
            evidence_root=tmp_path,
            supporting_paths=("shared/server.zip",),
        )
        == index
    )
    assert index.read_bytes() == closed_bytes
    assert index.stat().st_mtime_ns == closed_mtime
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


def _closed_registry(root: Path) -> tuple[AppendOnlyRegistry, Path]:
    artifact = root / "attempt.json"
    # Scientific failure is valid successful evidence, unlike a failed registry transition.
    artifact.write_bytes(b'{"completed":false,"failure_code":"runtime.timeout"}\n')
    job = _job("job-retained-failure")
    registry = AppendOnlyRegistry.create(
        root / "registry.jsonl", jobs=(job,), provenance=_provenance()
    )
    registry.started(job.job_id)
    registry.succeeded(
        job.job_id,
        artifact_path=artifact.name,
        artifact_digest=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    return registry, artifact


def test_duplicate_registry_creation_cannot_reset_attempts(tmp_path: Path) -> None:
    registry, _ = _closed_registry(tmp_path)
    original = registry.path.read_bytes()
    with pytest.raises(RegistryError):
        AppendOnlyRegistry.create(
            registry.path,
            jobs=(_job("job-retained-failure"),),
            provenance=_provenance(),
        )
    assert registry.path.read_bytes() == original
    assert registry.snapshot().records[0].status == JobStatus.SUCCEEDED


@pytest.mark.parametrize("mutation", ["artifact", "index", "extra", "registry"])
def test_repeat_closure_revalidates_all_evidence(tmp_path: Path, mutation: str) -> None:
    registry, artifact = _closed_registry(tmp_path)
    index = finalize_evidence(registry, evidence_root=tmp_path)
    if mutation == "artifact":
        artifact.write_bytes(b"changed result")
    elif mutation == "index":
        # Even semantically equivalent but noncanonical marker bytes are not replaceable.
        index.write_bytes(index.read_bytes() + b"\n")
    elif mutation == "extra":
        (tmp_path / "late-worker-result.json").write_bytes(b"orphan result")
    else:
        with registry.path.open("ab") as handle:
            handle.write(b'{"invalid":"event"}\n')
    preserved = index.read_bytes()
    with pytest.raises(RegistryError):
        finalize_evidence(registry, evidence_root=tmp_path)
    assert index.read_bytes() == preserved


def test_checksum_closure_is_acyclic_complete_and_repeatable(tmp_path: Path) -> None:
    registry, artifact = _closed_registry(tmp_path)
    ledger = tmp_path / "checksums.sha256"
    ledger.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (artifact, registry.path)
        )
    )
    index = finalize_evidence(
        registry,
        evidence_root=tmp_path,
        supporting_paths=(ledger.name,),
    )
    payload = json.loads(index.read_bytes())
    assert (
        payload["supporting_artifacts"][ledger.name]
        == hashlib.sha256(ledger.read_bytes()).hexdigest()
    )
    assert (
        finalize_evidence(
            registry,
            evidence_root=tmp_path,
            supporting_paths=(ledger.name,),
        )
        == index
    )


@pytest.mark.parametrize("invalid", ["self", "index", "missing", "duplicate", "wrong_digest"])
def test_checksum_ledger_cannot_omit_files_or_create_cycles(tmp_path: Path, invalid: str) -> None:
    registry, artifact = _closed_registry(tmp_path)
    rows = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in (artifact, registry.path)
    ]
    if invalid == "self":
        rows.append(f"{DIGEST_A}  checksums.sha256\n")
    elif invalid == "index":
        rows.append(f"{DIGEST_A}  evidence-index.json\n")
    elif invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(rows[0])
    else:
        rows[0] = f"{DIGEST_A}  {artifact.name}\n"
    ledger = tmp_path / "checksums.sha256"
    ledger.write_text("".join(rows))
    with pytest.raises(RegistryError):
        finalize_evidence(registry, evidence_root=tmp_path, supporting_paths=(ledger.name,))
    assert not (tmp_path / "evidence-index.json").exists()


def test_atomic_closure_does_not_leave_a_marker_on_failed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unseen_loop.flagship.registry as registry_module

    registry, _ = _closed_registry(tmp_path)
    real_rename = registry_module.os.rename

    def interrupted_rename(*args, **kwargs):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(registry_module.os, "rename", interrupted_rename)
    with pytest.raises(OSError):
        finalize_evidence(registry, evidence_root=tmp_path)
    assert not (tmp_path / "evidence-index.json").exists()
    monkeypatch.setattr(registry_module.os, "rename", real_rename)
    index = finalize_evidence(registry, evidence_root=tmp_path)
    assert json.loads(index.read_bytes())["planned_job_ids"] == ["job-retained-failure"]
