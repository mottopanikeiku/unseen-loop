from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from unseen_loop.flagship.executor_integration import execute_flagship_job
from unseen_loop.flagship.manifest import (
    PLAN_SCHEMA_VERSION,
    content_digest,
    derive_seed,
    load_manifest,
)
from unseen_loop.ope.fhe import calibration_inputset
from unseen_loop.ope.types import SufficientStatistics


def _manifest(*, horizon: int = 8, scenarios: int = 12) -> dict[str, Any]:
    payload = load_manifest("experiments/flagship-smoke.toml").canonical_payload()
    integration = payload["integration"]
    integration["scenarios"] = scenarios
    integration["behavior_trajectories_per_cell"] = 64
    integration["direct_target_trajectories_per_cell"] = 1
    integration["horizon"] = horizon
    integration["ope_batch_trajectories"] = 64
    payload["ope"]["bootstrap_repetitions"] = 16
    return payload


def _root(tmp_path: Path, config_digest: str = "a" * 64) -> Path:
    header = {"type": "header", "provenance": {"config_digest": config_digest}}
    (tmp_path / "registry.jsonl").write_text(
        json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return tmp_path


def _job(
    manifest: dict[str, Any],
    coordinates: dict[str, str | int | float],
    config_digest: str = "a" * 64,
) -> dict[str, object]:
    digest = content_digest(
        {
            "schema": PLAN_SCHEMA_VERSION,
            "manifest": config_digest,
            "stage": "integration",
            "coordinates": tuple(sorted(coordinates.items())),
        }
    )
    job_id = f"job-integration-{digest[:24]}"
    return {
        "job_id": job_id,
        "stage": "integration",
        "seed": derive_seed(manifest["seed_root"], job_id),
        "coordinates": coordinates,
    }


def _execute_trajectory(
    root: Path,
    manifest: dict[str, Any],
    *,
    kind: str,
    scenario: int,
    shield_mode: str,
    trajectory: int,
) -> dict[str, object]:
    coordinates: dict[str, str | int | float] = {
        "kind": kind,
        "scenario": scenario,
        "shield_mode": shield_mode,
        "trajectory": trajectory,
    }
    job = _job(manifest, coordinates)
    result = execute_flagship_job(manifest, job, root)
    assert result["status"] == "succeeded"
    artifact = root / str(result["artifact_path"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == result["artifact_digest"]
    return json.loads(artifact.read_text())


def test_integration_accepts_a_registered_scenario_prefix(tmp_path: Path) -> None:
    manifest = _manifest(scenarios=2)
    root = _root(tmp_path)

    included = _execute_trajectory(
        root,
        manifest,
        kind="behavior_trajectory",
        scenario=1,
        shield_mode="off",
        trajectory=0,
    )
    excluded_job = _job(
        manifest,
        {
            "kind": "behavior_trajectory",
            "scenario": 2,
            "shield_mode": "off",
            "trajectory": 0,
        },
    )
    excluded = execute_flagship_job(manifest, excluded_job, root)

    assert included["coordinates"]["scenario"] == 1
    assert excluded["status"] == "rejected"


def test_trajectory_release_preserves_many_to_one_requested_semantics_without_plain_logs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(horizon=32)
    root = _root(tmp_path)
    artifact = _execute_trajectory(
        root,
        manifest,
        kind="behavior_trajectory",
        scenario=11,
        shield_mode="h2",
        trajectory=0,
    )

    assert artifact["evidence_type"] == "client_released_trajectory_aggregate"
    released = artifact["trajectory"]
    assert sum(released["requested_action_counts"]) == 32
    assert sum(released["executed_action_counts"]) == 32
    assert sum(map(sum, released["requested_to_executed_counts"])) == 32
    assert any(
        sum(row[column] > 0 for row in released["requested_to_executed_counts"]) >= 2
        for column in range(5)
    )
    observed_mu = {value for values in released["mu_by_requested_action"] for value in values}
    assert list(observed_mu) == pytest.approx([0.2])

    encoded = json.dumps(artifact, sort_keys=True)
    assert '"steps"' not in encoded
    assert '"state"' not in encoded
    assert '"rewards"' not in encoded
    assert "executed_action_propensity" not in encoded
    assert "per-step" in artifact["release_scope"]

    invalid = _job(
        manifest,
        {
            "kind": "behavior_trajectory",
            "scenario": 11,
            "shield_mode": "h2",
            "trajectory": 1,
        },
    )
    invalid["seed"] = int(invalid["seed"]) + 1
    rejected = execute_flagship_job(manifest, invalid, root)
    assert rejected == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "integration.invalid-job",
    }
    assert not (root / "integration" / f"{invalid['job_id']}.json").exists()


def test_ope_loads_exact_deterministic_64_batch_and_uses_fake_ckks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(horizon=4)
    root = _root(tmp_path)
    for trajectory in range(64):
        _execute_trajectory(
            root,
            manifest,
            kind="behavior_trajectory",
            scenario=0,
            shield_mode="h2",
            trajectory=trajectory,
        )
    _execute_trajectory(
        root,
        manifest,
        kind="direct_trajectory",
        scenario=0,
        shield_mode="h2",
        trajectory=0,
    )

    class ExecutableReceipt:
        parameters = object()

        def require_executable(self) -> None:
            return None

    captured: dict[str, object] = {}

    def fake_ckks(spec: object, batch: object, receipt: object):
        captured["shape"] = batch.spec.batch_shape
        captured["actions"] = batch.actions
        captured["mu"] = batch.behavior_propensities
        return (
            SufficientStatistics(
                "clipped_wpdis",
                numerators=(64.0, 32.0, 16.0, 8.0),
                denominators=(64.0,) * 4,
                counts=(64,) * 4,
            ),
            [{"backend": "FAKE REAL CKKS", "plaintext": False}],
        )

    monkeypatch.setattr(
        "unseen_loop.flagship.executor_integration._ckks_receipt",
        lambda spec: ExecutableReceipt(),
    )
    monkeypatch.setattr("unseen_loop.flagship.executor_integration._run_ckks", fake_ckks)

    coordinates: dict[str, str | int | float] = {
        "kind": "real_fhe_ope",
        "scenario": 0,
        "shield_mode": "h2",
        "outcome": "return",
        "batch": 0,
    }
    result = execute_flagship_job(manifest, _job(manifest, coordinates), root)

    assert result["status"] == "succeeded"
    artifact = json.loads((root / str(result["artifact_path"])).read_text())
    assert captured["shape"] == (64, 4)
    assert len(captured["actions"]) == 64
    assert list({value for row in captured["mu"] for value in row}) == pytest.approx([0.2])
    assert artifact["batch"] == {
        "horizon": 4,
        "source": "deterministically reconstructed from 64 bound client releases",
        "trajectory_count": 64,
        "trajectory_start": 0,
    }
    assert artifact["backend"]["label"] == "CKKS_POLYNOMIAL_APPROX_OPE"
    assert artifact["backend"]["real_fhe"] is True
    assert artifact["backend"]["ckks_failure_label"] is None
    assert artifact["statistics"]["shape"] == [3, 4]
    assert artifact["statistics"]["denominators"] == [64.0] * 4
    assert artifact["effect_channel"]["fhe_estimate"] == pytest.approx(1.875)
    assert "truth_excludes_zero" in artifact["effect_channel"]
    assert "sign_error" in artifact["effect_channel"]
    serialized = json.dumps(artifact, sort_keys=True)
    assert '"states"' not in serialized
    assert '"steps"' not in serialized
    assert '"ciphertext"' not in serialized


def test_unexecutable_ckks_runs_exact_fake_concrete_canary_with_distinct_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(horizon=4)
    root = _root(tmp_path)
    for trajectory in range(64):
        _execute_trajectory(
            root,
            manifest,
            kind="behavior_trajectory",
            scenario=1,
            shield_mode="h1",
            trajectory=trajectory,
        )
    _execute_trajectory(
        root,
        manifest,
        kind="direct_trajectory",
        scenario=1,
        shield_mode="h1",
        trajectory=0,
    )

    class UnsupportedReceipt:
        def require_executable(self) -> None:
            raise ValueError("provides 2 multiplication levels but requires 69")

    captured: dict[str, object] = {}

    def fake_concrete(spec: object, batch: object):
        calibration, _, _ = calibration_inputset(spec)
        captured["calibration_rows"] = len(calibration)
        captured["shape"] = batch.spec.batch_shape
        return (
            SufficientStatistics(
                "clipped_wpdis",
                numerators=(1.0, 0.5),
                denominators=(1.0, 1.0),
                counts=(1, 1),
            ),
            [{"backend": "FAKE REAL CONCRETE", "output_shape": [3, 2]}],
        )

    monkeypatch.setattr(
        "unseen_loop.flagship.executor_integration._ckks_receipt",
        lambda spec: UnsupportedReceipt(),
    )
    monkeypatch.setattr("unseen_loop.flagship.executor_integration._run_concrete", fake_concrete)
    monkeypatch.setattr(
        "unseen_loop.flagship.executor_integration._run_ckks",
        lambda *args: pytest.fail("clear or CKKS execution must not replace the Concrete fallback"),
    )

    coordinates: dict[str, str | int | float] = {
        "kind": "real_fhe_ope",
        "scenario": 1,
        "shield_mode": "h1",
        "outcome": "unsafe_step_cost",
        "batch": 0,
    }
    result = execute_flagship_job(manifest, _job(manifest, coordinates), root)

    assert result["status"] == "succeeded"
    artifact = json.loads((root / str(result["artifact_path"])).read_text())
    assert captured["shape"] == (1, 2)
    assert int(captured["calibration_rows"]) > 0
    assert artifact["backend"] == {
        "ckks_failure_label": "ckks.insufficient-multiplicative-depth",
        "context_scope": "one compiled Concrete context for this bounded canary call",
        "label": "CONCRETE_EXACT_SMALL_CANARY",
        "real_fhe": True,
        "statistics_scope": "1 trajectory x 2 horizons canary; not a clear substitute",
        "trust_scope": "colocated-client-server",
        "trust_scope_detail": (
            "Concrete client and server execute in one Modal worker; REAL FHE attests backend "
            "execution but does not claim input privacy from the colocated worker"
        ),
    }
    assert artifact["statistics"]["shape"] == [3, 2]
    assert artifact["effect_channel"]["scope"].endswith("not a clear substitute")
