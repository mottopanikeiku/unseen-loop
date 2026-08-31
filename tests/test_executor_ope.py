from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from unseen_loop.flagship import executor_ope
from unseen_loop.flagship.manifest import load_manifest
from unseen_loop.ope.fhe import OPEConformanceResult, SanitizedOPECallEvidence


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    parsed = load_manifest(Path(__file__).parents[1] / "experiments" / "flagship-smoke.toml")
    return parsed.canonical_payload()


def _job(kind: str, seed: int = 1729, **coordinates: object) -> dict[str, object]:
    return {
        "job_id": "job-ope-validation-test",
        "stage": "ope_validation",
        "seed": seed,
        "coordinates": {"kind": kind, **coordinates},
    }


def _execute(
    manifest: dict[str, Any], job: dict[str, object], root: Path
) -> tuple[dict[str, object], dict[str, Any]]:
    result = executor_ope.execute_flagship_job(manifest, job, root)
    assert result["status"] == "succeeded"
    assert result["reason_code"] == "completed"
    relative = result["artifact_path"]
    assert isinstance(relative, str)
    payload = (root / relative).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == result["artifact_digest"]
    assert (
        payload
        == json.dumps(
            json.loads(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )
    return result, json.loads(payload)


def test_analytic_fixture_is_deterministic_and_contains_exact_truth(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    job = _job("analytic_fixture", case=3)
    first_result, first = _execute(manifest, job, tmp_path / "first")
    second_result, second = _execute(manifest, job, tmp_path / "second")

    assert first == second
    assert first_result["artifact_digest"] == second_result["artifact_digest"]
    assert first["evidence_class"] == "CLEAR ANALYTIC REFERENCE"
    assert first["truth"]["value"] == pytest.approx(
        sum(first["truth"]["per_horizon_contributions"])
    )
    assert first["policy"]["behavior"]["support_valid"] is True
    assert first["diagnostics"]["positive_horizon_denominators"] is True
    assert first["diagnostics"]["logged_batch_persisted"] is False
    assert "behavior_propensities" not in json.dumps(first)


def test_empirical_persists_denominators_errors_ess_and_closed_ci_inputs(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    _, artifact = _execute(
        manifest,
        _job(
            "empirical",
            horizon=8,
            trajectories=64,
            overlap=0.2,
            clip=5.0,
            estimator="clipped_wpdis",
            batch=0,
        ),
        tmp_path,
    )

    assert artifact["evidence_class"] == "CLEAR STATISTICAL EVIDENCE"
    assert artifact["fhe"] is None
    assert artifact["fixed_point"] is None
    assert len(artifact["estimates"]["pdis"]["denominators"]) == 8
    assert len(artifact["estimates"]["wpdis"]["denominators"]) == 8
    assert artifact["estimates"]["pdis"]["absolute_error"] >= 0
    assert artifact["estimates"]["wpdis"]["squared_error"] >= 0
    assert len(artifact["diagnostics"]["per_horizon_ess"]) == 8
    assert artifact["diagnostics"]["minimum_ess_fraction"] > 0
    assert artifact["ci"]["method"] == "deterministic_multinomial_whole_trajectory_percentile"
    assert artifact["ci"]["repetitions"] == manifest["ope"]["bootstrap_repetitions"]
    assert artifact["ci"]["estimator"] == "clipped_wpdis"
    assert artifact["ci"]["lower"] <= artifact["ci"]["upper"]
    assert isinstance(artifact["ci"]["replicate_sum"], float)
    assert isinstance(artifact["ci"]["replicate_sum_squares"], float)


def test_unclipped_empirical_uses_valid_target_behavior_support(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    _, artifact = _execute(
        manifest,
        _job(
            "empirical",
            horizon=8,
            trajectories=64,
            overlap=0.8,
            clip="unclipped",
            estimator="clipped_pdis",
            batch=1,
        ),
        tmp_path,
    )

    assert artifact["policy"]["behavior"]["support_valid"] is True
    assert artifact["policy"]["behavior"]["minimum_logged_probability"] > 0
    assert all(value == 64.0 for value in artifact["estimates"]["pdis"]["denominators"])


def test_fixed_point_reference_closes_integer_denominators_and_errors(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    _, artifact = _execute(manifest, _job("fixed_point_reference", case=7), tmp_path)

    fixed = artifact["fixed_point"]
    assert artifact["evidence_class"] == "CLEAR EXACT FIXED-POINT REFERENCE"
    assert len(fixed["integer_statistics"]["denominators"]) == 4
    assert fixed["integer_statistics"]["counts"] == [4, 4, 4, 4]
    assert fixed["decoded"]["pdis"]["denominators"] == [4.0, 4.0, 4.0, 4.0]
    assert fixed["error"]["absolute_pdis"] >= 0
    assert fixed["error"]["absolute_wpdis"] >= 0
    assert fixed["receipt"]["error"]["clipped_pdis"] == pytest.approx(
        fixed["error"]["absolute_pdis"], rel=1e-10, abs=1e-15
    )
    assert artifact["diagnostics"]["logged_batch_persisted"] is False


def test_fhe_valid_uses_real_concrete_boundary_and_colocated_scope(
    manifest: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int, int]] = []

    def fake_concrete(
        spec: Any, batch: Any, security_level: int
    ) -> tuple[OPEConformanceResult, object]:
        calls.append(
            (
                spec.trajectories.trajectories,
                spec.trajectories.horizon,
                security_level,
            )
        )
        integers, receipt = spec.integer_reference(batch)
        evidence = SanitizedOPECallEvidence(
            input_shape=(160,),
            output_shape=(3, 4),
            encrypted_output_vectors=3,
            integers_per_output_vector=4,
            keygen_ns=11,
            encrypt_ns=12,
            server_evaluate_ns=13,
            decrypt_ns=14,
            end_to_end_ns=50,
            evaluation_key_bytes=101,
            request_bytes=102,
            response_bytes=103,
            request_sha256="a" * 64,
            response_sha256="b" * 64,
            output_matches_integer_reference=True,
            server_secret_key_marker_present=False,
        )
        return (
            OPEConformanceResult(
                "REAL",
                integers,
                spec.client_statistics(integers, "clipped_wpdis"),
                receipt,
                evidence,
            ),
            {"security_level": security_level, "compiled_global_p_error": 1e-6},
        )

    monkeypatch.setattr(executor_ope, "_run_concrete_canary", fake_concrete)
    _, artifact = _execute(manifest, _job("fhe_valid", category="occupancy", batch=0), tmp_path)

    assert calls == [(4, 4, 128)]
    assert artifact["evidence_class"] == "REAL COLOCATED FHE CANARY"
    assert artifact["configured_challenge_shape"] == {"horizon": 64, "trajectories": 256}
    assert artifact["policy"]["behavior"]["support_valid"] is True
    assert artifact["truth"]["value"] == pytest.approx(
        sum(artifact["truth"]["per_horizon_contributions"])
    )
    assert artifact["estimates"]["pdis"]["denominators"] == [4.0, 4.0, 4.0, 4.0]
    assert artifact["estimates"]["wpdis"]["denominators"] == [4.0, 4.0, 4.0, 4.0]
    assert artifact["diagnostics"]["per_horizon_ess"] == [4.0, 4.0, 4.0, 4.0]
    assert artifact["diagnostics"]["positive_horizon_denominators"] is True
    assert artifact["fhe"]["mode"] == "REAL"
    assert artifact["fhe"]["canary_shape"] == {
        "horizon": 4,
        "state_dim": 6,
        "trajectories": 4,
    }
    assert artifact["fhe"]["conforms_to_integer_reference"] is True
    assert artifact["fhe"]["integer_statistics"]["counts"] == [4, 4, 4, 4]
    assert artifact["fhe"]["call_evidence"]["server_secret_key_marker_present"] is False
    assert artifact["fhe"]["compilation_receipt"]["security_level"] == 128
    assert "colocated" in artifact["fhe"]["trust_scope"]


def test_fhe_invalid_rejects_before_concrete_and_writes_no_artifact(
    manifest: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid jobs must reject before Concrete")

    monkeypatch.setattr(executor_ope, "_run_concrete_canary", forbidden)
    result = executor_ope.execute_flagship_job(manifest, _job("fhe_invalid", batch=0), tmp_path)

    assert result == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "declared_invalid_ope_batch",
    }
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        (_job("unknown"), "unknown_ope_job_kind"),
        (_job("analytic_fixture", case=999), "case_out_of_range"),
        (_job("empirical", horizon=7), "unexpected_coordinates"),
        (
            {
                "job_id": "../escape",
                "stage": "ope_validation",
                "seed": 1,
                "coordinates": {"kind": "analytic_fixture", "case": 0},
            },
            "invalid_job_id",
        ),
    ],
)
def test_malformed_jobs_reject_without_artifacts(
    manifest: dict[str, Any], tmp_path: Path, job: dict[str, object], reason: str
) -> None:
    result = executor_ope.execute_flagship_job(manifest, job, tmp_path)

    assert result["status"] == "rejected"
    assert result["reason_code"] == reason
    assert result["artifact_path"] is None
    assert result["artifact_digest"] is None
    assert not any(tmp_path.iterdir())
