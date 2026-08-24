from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from unseen_loop.artifacts import ArtifactLedger, dataclass_dict
from unseen_loop.cli import main
from unseen_loop.experiment import ExperimentSummary, ResearchPreset
from unseen_loop.suite import (
    _hierarchical_interval,
    load_release_config,
    run_release_suite,
)


def release_config() -> str:
    return """\
schema_version = "unseen-loop/release-suite-v1"
name = "test-release"
seed_root = "test-root"
fhe_runtime = "concrete-python-2.10.0"
security_level = 128
global_p_error = 1e-6
stable_argmax = "lowest-index"

[workloads]
environments = ["CartPole-v1", "MountainCar-v0"]
checkpoints_per_environment = 2
selection_episodes = 2
evaluation_episodes = 3

[search]
degrees = [1]
input_bits = [3]
coefficient_bits = [8]
ridge = [1e-3]
refinement_rounds = 1
calibration_padding = 0.5
certificate_weighting = false
student_occupancy_refinement = true

[training.gpu]
population = 4
iterations = 1
episodes_per_candidate = 1
hidden_size = 2

[gates]
minimum_certified_occupancy = 0.99
maximum_certified_mismatches = 0

"""


def test_checked_in_release_config_has_the_documented_matrix() -> None:
    path = Path("experiments/release.toml")
    config, digest = load_release_config(path)

    assert config.environments == ("CartPole-v1", "MountainCar-v0", "Acrobot-v1")
    assert config.checkpoints_per_environment == 5
    assert config.selection_episodes == 100
    assert config.evaluation_episodes == 100
    assert config.expected_runs == 15
    assert config.expected_paired_episodes == 1_500
    assert config.expected_episode_rows == 3_000
    assert config.candidates_per_run == 120
    assert config.expected_candidate_rows == 1_800
    assert config.expected_selection_episodes == 180_000
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_expanded_configs_have_matched_budgets_and_all_variants() -> None:
    expanded, _ = load_release_config("experiments/expanded-multitask.toml")
    assert expanded.environments == (
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
    )
    assert expanded.checkpoints_per_environment == 5
    assert expanded.selection_episodes == 50
    assert expanded.evaluation_episodes == 100
    assert expanded.expected_runs == 15
    assert expanded.candidates_per_run == 8
    assert expanded.expected_selection_episodes == 6_000
    assert expanded.expected_paired_episodes == 1_500
    assert (
        expanded.teacher_population,
        expanded.teacher_iterations,
        expanded.episodes_per_candidate,
        expanded.hidden_size,
        expanded.calibration_padding,
    ) == (64, 12, 2, 16, 2.0)
    assert expanded.certificate_weighting is True
    assert expanded.student_occupancy_refinement is True

    paths = sorted(Path("experiments").glob("ablation-cartpole-*.toml"))
    assert len(paths) == 4
    configs = [load_release_config(path)[0] for path in paths]
    assert {
        (config.certificate_weighting, config.student_occupancy_refinement) for config in configs
    } == {(False, False), (False, True), (True, False), (True, True)}
    assert {
        (
            config.seed_root,
            config.environments,
            config.checkpoints_per_environment,
            config.selection_episodes,
            config.evaluation_episodes,
            config.degrees,
            config.input_bits,
            config.coefficient_bits,
            config.ridge_values,
            config.refinement_rounds,
            config.calibration_padding,
            config.teacher_population,
            config.teacher_iterations,
            config.episodes_per_candidate,
            config.hidden_size,
            config.minimum_certified_occupancy,
            config.maximum_certified_mismatches,
        )
        for config in configs
    } == {
        (
            "expanded-cartpole-ablation-2026-08",
            ("CartPole-v1",),
            5,
            50,
            100,
            (1, 2),
            (5, 6),
            (8, 10),
            (0.001,),
            3,
            2.0,
            64,
            12,
            2,
            16,
            0.99,
            0,
        )
    }
    assert all(config.candidates_per_run == 8 for config in configs)
    assert all(config.expected_selection_episodes == 2_000 for config in configs)
    assert sum(config.expected_selection_episodes for config in configs) == 8_000


def test_release_config_variant_switches_default_true_and_require_booleans(tmp_path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        release_config()
        .replace("certificate_weighting = false\n", "")
        .replace("student_occupancy_refinement = true\n", "")
    )
    config, _ = load_release_config(legacy)
    assert config.certificate_weighting is True
    assert config.student_occupancy_refinement is True

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(
        release_config().replace(
            "certificate_weighting = false",
            'certificate_weighting = "false"',
        )
    )
    with pytest.raises(ValueError, match=r"certificate_weighting.*boolean"):
        load_release_config(malformed)


def test_release_config_rejects_ambiguous_or_nonpositive_dimensions(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.toml"
    duplicate.write_text(
        release_config().replace(
            '["CartPole-v1", "MountainCar-v0"]', '["CartPole-v1", "CartPole-v1"]'
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_release_config(duplicate)

    zero = tmp_path / "zero.toml"
    zero.write_text(release_config().replace("evaluation_episodes = 3", "evaluation_episodes = 0"))
    with pytest.raises(ValueError, match="evaluation_episodes"):
        load_release_config(zero)


def test_hierarchical_paired_interval_is_deterministic_and_task_balanced() -> None:
    deltas = {
        "task-a": ((1.0, 1.0), (3.0, 3.0)),
        "task-b": ((9.0, 9.0),),
    }

    first = _hierarchical_interval(deltas, seed=1234, repetitions=500)
    second = _hierarchical_interval(deltas, seed=1234, repetitions=500)

    assert first == second
    assert first[0] == 5.5
    assert first[1] <= 2.0
    assert first[2] == 9.0


def _write_fake_run(
    *,
    env_id: str,
    output: str | Path,
    preset: ResearchPreset,
    run_id: str,
    range_valid: bool,
    selection_episode_seeds: tuple[int, ...] = (30, 31),
    invalid_selection_counters: bool = False,
    selection_agreement_count: int = 10,
) -> ExperimentSummary:
    assert preset.selection_episodes == 2
    assert preset.evaluation_episodes == 3
    assert preset.search.certificate_weighting is False
    assert preset.search.student_occupancy_refinement is True
    destination = Path(output)
    checkpoint_index = int(run_id.rsplit("-", 1)[1])
    environment_offset = 1.0 if env_id == "CartPole-v1" else 3.0
    delta = environment_offset + checkpoint_index
    summary = ExperimentSummary(
        run_id=run_id,
        env_id=env_id,
        backend="clear",
        teacher_digest=f"teacher-{run_id}",
        teacher_return_mean=51.0,
        candidates=1,
        frontier_candidates=1,
        champion_policy_digest=f"policy-{run_id}",
        champion_name="test-policy",
        champion_return_mean=51.0 + delta,
        champion_return_delta=delta,
        teacher_agreement=1.0,
        certified_coverage=0.995,
        constraint_cost=0.0,
        estimated_output_bits=8,
        encrypted_multiplications=0,
        box_certificate_coverage=1.0,
        box_certificate_points=1,
        simulated_matches_integer=None,
        real_fhe_calls=0,
        real_fhe_all_match=None,
        label="QUANTIZED CLEAR",
        privacy_evidence=False,
    )
    ledger = ArtifactLedger(destination)
    ledger.write_json(
        "seeds.json",
        {
            "training": 1,
            "distillation": [10],
            "refinement": [20],
            "selection": [30, 31],
            "evaluation": [40, 41, 42],
            "real_fhe": [50],
            "namespace": f"test:{run_id}",
        },
    )
    ledger.write_jsonl(
        "evaluation/episodes.jsonl",
        (
            {
                "seed": seed,
                "total_return": (10.0 + seed + (delta if mode == "QUANTIZED CLEAR" else 0.0)),
                "constraint_cost": 0.0,
                "length": 10,
                "terminated": True,
                "truncated": False,
                "action_digest": f"{mode}-{seed}",
                "mode": mode,
                "policy_digest": "test",
            }
            for mode in ("FLOAT TEACHER", "QUANTIZED CLEAR")
            for seed in (40, 41, 42)
        ),
    )
    ledger.write_jsonl(
        "search/candidates.jsonl",
        (
            {
                "metrics": {
                    "policy_digest": summary.champion_policy_digest,
                    "range_valid": range_valid,
                    "teacher_agreement": 1.0,
                    "certified_coverage": 1.0,
                }
            },
        ),
    )
    ledger.write_jsonl(
        "search/selection-episodes.jsonl",
        (
            {
                "candidate_digest": summary.champion_policy_digest,
                "seed": seed,
                "total_return": 50.0,
                "constraint_cost": 0.0,
                "range_valid": range_valid,
                "action_digest": f"selection-{seed}",
                "steps": 10,
                "teacher_agreement_count": selection_agreement_count,
                "certified_count": 11 if invalid_selection_counters else 10,
                "certified_mismatch_count": 0,
                "saturation_count": 0 if range_valid else 1,
                "mode": "QUANTIZED CLEAR SELECTION",
            }
            for seed in selection_episode_seeds
        ),
    )
    ledger.write_json(
        "certificates/heldout.json",
        {"coverage": 0.995, "certified_mismatches": 0},
    )
    ledger.write_json("summary.json", dataclass_dict(summary))
    ledger.finalize()
    return summary


def test_release_suite_materializes_matrix_and_retains_paired_rows(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())

    def fake_run_experiment(**kwargs: Any) -> ExperimentSummary:
        return _write_fake_run(
            env_id=kwargs["env_id"],
            output=kwargs["output"],
            preset=kwargs["preset"],
            run_id=kwargs["run_id"],
            range_valid=True,
        )

    monkeypatch.setattr("unseen_loop.suite.run_experiment", fake_run_experiment)
    output = tmp_path / "suite"
    summary = run_release_suite(config_path=config_path, output=output)

    assert summary["expected_runs"] == summary["completed_runs"] == 4
    assert summary["retained_paired_episodes"] == 12
    assert summary["retained_episode_rows"] == 24
    assert summary["paired_return_delta"]["mean"] == 2.5
    assert summary["candidates_per_run"] == 1
    assert summary["expected_candidate_rows"] == 4
    assert summary["expected_selection_episodes"] == 8
    assert summary["retained_selection_episode_rows"] == 8
    assert summary["selection_episodes_per_candidate"] == 2
    assert summary["selection_evidence_scope"].startswith("complete candidate-by-seed")
    assert summary["certificate_weighting"] is False
    assert summary["student_occupancy_refinement"] is True
    assert summary["all_runs_complete"] is True
    assert summary["all_suite_gates_passed"] is True
    run_rows = [json.loads(line) for line in (output / "suite-runs.jsonl").read_text().splitlines()]
    paired_rows = [
        json.loads(line) for line in (output / "suite-episodes.jsonl").read_text().splitlines()
    ]
    assert len(run_rows) == 4
    assert len(paired_rows) == 12
    assert all(row["retained_episode_rows"] == 6 for row in run_rows)
    assert all(row["retained_selection_episode_rows"] == 2 for row in run_rows)
    assert all(row["certificate_weighting"] is False for row in run_rows)
    assert all(row["student_occupancy_refinement"] is True for row in run_rows)
    assert not any("observation" in row for row in paired_rows)
    assert ArtifactLedger(output).verify() == (True, ())
    child_summary = output / "runs" / "CartPole-v1--checkpoint-00" / "summary.json"
    child_summary.write_text("{}\n")
    valid, failures = ArtifactLedger(output).verify()
    assert not valid
    assert any("runs/CartPole-v1--checkpoint-00/summary.json" in row for row in failures)


def test_release_suite_rejects_nonempty_output_before_launch(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())
    output = tmp_path / "suite"
    output.mkdir()
    (output / "stale.json").write_text("{}\n")

    def unexpected_run(**kwargs: Any) -> ExperimentSummary:
        raise AssertionError(f"experiment should not launch: {kwargs}")

    monkeypatch.setattr("unseen_loop.suite.run_experiment", unexpected_run)
    with pytest.raises(ValueError, match="absent or empty"):
        run_release_suite(config_path=config_path, output=output)


def test_release_suite_rejects_a_range_invalid_champion(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())

    def fake_run_experiment(**kwargs: Any) -> ExperimentSummary:
        return _write_fake_run(
            env_id=kwargs["env_id"],
            output=kwargs["output"],
            preset=kwargs["preset"],
            run_id=kwargs["run_id"],
            range_valid=False,
        )

    monkeypatch.setattr("unseen_loop.suite.run_experiment", fake_run_experiment)
    with pytest.raises(RuntimeError, match="range-invalid"):
        run_release_suite(config_path=config_path, output=tmp_path / "suite")


def test_release_suite_rejects_incomplete_selection_episode_pairs(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())

    def fake_run_experiment(**kwargs: Any) -> ExperimentSummary:
        return _write_fake_run(
            env_id=kwargs["env_id"],
            output=kwargs["output"],
            preset=kwargs["preset"],
            run_id=kwargs["run_id"],
            range_valid=True,
            selection_episode_seeds=(30, 30),
        )

    monkeypatch.setattr("unseen_loop.suite.run_experiment", fake_run_experiment)
    with pytest.raises(RuntimeError, match=r"selection episode.*duplicated"):
        run_release_suite(config_path=config_path, output=tmp_path / "suite")


def test_release_suite_rejects_invalid_selection_counters(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())

    def fake_run_experiment(**kwargs: Any) -> ExperimentSummary:
        return _write_fake_run(
            env_id=kwargs["env_id"],
            output=kwargs["output"],
            preset=kwargs["preset"],
            run_id=kwargs["run_id"],
            range_valid=True,
            invalid_selection_counters=True,
        )

    monkeypatch.setattr("unseen_loop.suite.run_experiment", fake_run_experiment)
    with pytest.raises(RuntimeError, match="counters exceed episode steps"):
        run_release_suite(config_path=config_path, output=tmp_path / "suite")


def test_release_suite_recomputes_candidate_selection_metrics(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "release.toml"
    config_path.write_text(release_config())

    def fake_run_experiment(**kwargs: Any) -> ExperimentSummary:
        return _write_fake_run(
            env_id=kwargs["env_id"],
            output=kwargs["output"],
            preset=kwargs["preset"],
            run_id=kwargs["run_id"],
            range_valid=True,
            selection_agreement_count=9,
        )

    monkeypatch.setattr("unseen_loop.suite.run_experiment", fake_run_experiment)
    with pytest.raises(RuntimeError, match="agreement disagrees"):
        run_release_suite(config_path=config_path, output=tmp_path / "suite")


def test_suite_cli_forwards_toml_backend_and_provenance(tmp_path, monkeypatch, capsys) -> None:
    captured: dict[str, Any] = {}

    def fake_suite(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema_version": "unseen-loop/release-suite-v1", "completed_runs": 15}

    monkeypatch.setattr("unseen_loop.cli._git_state", lambda: ("abc123", False))
    monkeypatch.setattr("unseen_loop.cli.run_release_suite", fake_suite)
    config = tmp_path / "release.toml"
    output = tmp_path / "output"

    assert (
        main(
            [
                "suite",
                "--config",
                str(config),
                "--backend",
                "simulate",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured == {
        "config_path": config,
        "output": output,
        "backend": "simulate",
        "git_commit": "abc123",
        "git_dirty": False,
    }
    assert json.loads(capsys.readouterr().out)["completed_runs"] == 15
