from __future__ import annotations

import json

import numpy as np
import pytest

from unseen_loop.cli import main
from unseen_loop.experiment import ResearchPreset, load_summary, run_experiment, verify_artifact
from unseen_loop.search import SearchConfig
from unseen_loop.teacher import TeacherCheckpoint


def checkpoint() -> TeacherCheckpoint:
    w1 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, -5.0], [1.0, -1.0]], dtype=np.float64)
    b1 = np.zeros(2)
    w2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    b2 = np.zeros(2)
    parameters = np.concatenate((w1.ravel(), b1, w2.ravel(), b2))
    return TeacherCheckpoint(
        env_id="CartPole-v1",
        observation_size=4,
        actions=2,
        hidden_size=2,
        parameters=tuple(parameters),
        training_seed=1,
        iterations=0,
        population=0,
        elite_fraction=0.15,
    )


def tiny_preset() -> ResearchPreset:
    return ResearchPreset(
        full=False,
        teacher_iterations=1,
        teacher_population=4,
        episodes_per_candidate=1,
        selection_episodes=2,
        evaluation_episodes=2,
        hidden_size=2,
        search=SearchConfig(
            degrees=(1,),
            input_bits=(3,),
            coefficient_bits=(8,),
            ridge_values=(1e-3,),
            refinement_rounds=1,
            calibration_padding=5.0,
        ),
    )


def test_clear_experiment_writes_self_verifying_artifact(tmp_path) -> None:
    output = tmp_path / "run"
    summary = run_experiment(
        env_id="CartPole-v1",
        output=output,
        backend="clear",
        preset=tiny_preset(),
        teacher_checkpoint=checkpoint(),
        seed_root="test",
    )

    assert summary.label == "QUANTIZED CLEAR"
    assert not summary.privacy_evidence
    assert summary.candidates == 1
    assert load_summary(output) == summary
    assert verify_artifact(output) == (True, ())
    claims = json.loads((output / "claims.json").read_text())
    assert claims["privacy_evidence"] is False
    assert "malicious-server computation integrity" in claims["not_supported"]


def test_experiment_rejects_wrong_environment_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="environment"):
        run_experiment(
            env_id="MountainCar-v0",
            output=tmp_path,
            backend="clear",
            preset=tiny_preset(),
            teacher_checkpoint=checkpoint(),
        )


def test_report_command_validates_and_publishes_evidence(tmp_path, capsys) -> None:
    evidence = {
        "schema_version": "unseen-loop/modal-evidence-v1",
        "run_id": "test-run",
        "circuit_receipt": {},
        "real_fhe_trials": [],
    }
    source = tmp_path / "source.json"
    output = tmp_path / "site" / "evidence.json"
    source.write_text(json.dumps(evidence))

    assert main(["report", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == evidence
    assert "test-run" in capsys.readouterr().out


def test_report_command_rejects_wrong_schema(tmp_path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="schema"):
        main(["report", str(source), "--output", str(tmp_path / "out")])
