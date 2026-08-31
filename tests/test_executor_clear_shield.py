from __future__ import annotations

import hashlib
import json
from pathlib import Path

from unseen_loop.flagship import executor_clear_shield as executor
from unseen_loop.flagship.manifest import iter_stage_jobs, load_manifest
from unseen_loop.shield.study import run_shield_job


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_manifest(Path(__file__).parents[1] / "experiments" / "flagship-smoke.toml")
    planned = next(iter_stage_jobs(manifest, "clear_shield_matrix"))
    job = {
        "job_id": planned.job_id,
        "stage": planned.stage,
        "seed": planned.seed,
        "coordinates": planned.coordinate_dict(),
    }
    return manifest.canonical_payload(), job


def test_clear_executor_runs_one_episode_and_writes_closed_public_evidence(tmp_path) -> None:
    manifest, job = _payloads()

    result = executor.execute_flagship_job(manifest, job, tmp_path)

    assert set(result) == {"status", "artifact_path", "artifact_digest", "reason_code"}
    assert result["status"] == "succeeded"
    assert result["reason_code"] is None
    artifact_path = tmp_path / str(result["artifact_path"])
    payload = artifact_path.read_bytes()
    assert result["artifact_digest"] == hashlib.sha256(payload).hexdigest()
    artifact = json.loads(payload)
    assert artifact["status"] == "completed"
    assert artifact["episode"]["episode_denominator"] == 1
    assert artifact["step_denominator"] == len(artifact["steps"])
    assert artifact["category_denominators"] == {
        artifact["scenario_category"]: 1,
    }
    assert len(artifact["manifest_digest"]) == 64
    assert len(artifact["job_digest"]) == 64
    assert len(artifact["spec_digest"]) == 64
    assert '"state"' not in payload.decode()
    assert "margin" not in payload.decode()


def test_clear_executor_retains_failed_episode_as_successful_attempt(tmp_path, monkeypatch) -> None:
    manifest, job = _payloads()

    def failed_runner(shield_job, *, scenario_factory):
        def fail():
            raise RuntimeError("private details must not escape")

        return run_shield_job(shield_job, scenario_factory=fail)

    monkeypatch.setattr(executor, "run_shield_job", failed_runner)
    result = executor.execute_flagship_job(manifest, job, tmp_path)
    artifact = json.loads((tmp_path / str(result["artifact_path"])).read_bytes())

    assert result["status"] == "succeeded"
    assert artifact["status"] == "failed"
    assert artifact["episode"]["failure_code"] == "RuntimeError"
    assert artifact["episode"]["episode_denominator"] == 1
    assert artifact["step_denominator"] == 0
    assert artifact["steps"] == []
    assert "private details" not in json.dumps(artifact)


def test_clear_executor_rejects_unknown_coordinates_without_artifact(tmp_path) -> None:
    manifest, job = _payloads()
    job["coordinates"] = {**job["coordinates"], "scenario": 999}

    result = executor.execute_flagship_job(manifest, job, tmp_path)

    assert result == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "clear-shield.unknown-coordinates",
    }
    assert not any(tmp_path.iterdir())
