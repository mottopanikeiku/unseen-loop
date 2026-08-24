from __future__ import annotations

import json

import numpy as np
import pytest

import unseen_loop.experiment as experiment_module
from unseen_loop.experiment import ResearchPreset, run_experiment
from unseen_loop.search import IntegerStudent, SearchConfig
from unseen_loop.teacher import EpisodeResult, ScorePolicy, TeacherCheckpoint, TrajectoryBatch


def checkpoint() -> TeacherCheckpoint:
    w1 = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, -5.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )
    b1 = np.zeros(2)
    w2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    b2 = np.zeros(2)
    return TeacherCheckpoint(
        env_id="CartPole-v1",
        observation_size=4,
        actions=2,
        hidden_size=2,
        parameters=tuple(np.concatenate((w1.ravel(), b1, w2.ravel(), b2))),
        training_seed=1,
        iterations=0,
        population=0,
        elite_fraction=0.15,
    )


def evaluation_preset() -> ResearchPreset:
    return ResearchPreset(
        full=False,
        teacher_iterations=1,
        teacher_population=4,
        episodes_per_candidate=1,
        selection_episodes=2,
        evaluation_episodes=3,
        hidden_size=2,
        search=SearchConfig(
            degrees=(1,),
            input_bits=(7,),
            coefficient_bits=(10,),
            ridge_values=(1e-3,),
            refinement_rounds=0,
            calibration_padding=10.0,
        ),
    )


def test_post_selection_metrics_and_rows_use_paired_student_occupancy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, TrajectoryBatch] = {}

    def heldout_batch(env_id: str, policy: ScorePolicy, seeds: tuple[int, ...]) -> TrajectoryBatch:
        assert env_id == "CartPole-v1"
        assert len(seeds) == 3
        is_student = isinstance(policy, IntegerStudent)
        lengths = (1, 1, 1) if is_student else (2, 2, 2)
        returns = tuple(float(index + (10 if is_student else 100)) for index in range(3))
        costs = tuple(float(index + (1 if is_student else 20)) for index in range(3))
        observations = np.concatenate(
            [np.full((length, 4), index / 100.0) for index, length in enumerate(lengths)]
        )
        scores = np.asarray(policy.score(observations), dtype=np.float64)
        actions = np.asarray(np.argmax(scores, axis=1), dtype=np.int64)
        episodes = tuple(
            EpisodeResult(
                seed=seed,
                total_return=returns[index],
                constraint_cost=costs[index],
                length=lengths[index],
                terminated=True,
                truncated=False,
                action_digest=f"{'student' if is_student else 'teacher'}-{seed}",
            )
            for index, seed in enumerate(seeds)
        )
        batch = TrajectoryBatch(
            observations=observations,
            scores=scores,
            actions=actions,
            episode_ids=np.concatenate(
                [np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)]
            ),
            steps=np.concatenate([np.arange(length, dtype=np.int64) for length in lengths]),
            returns=returns,
            constraint_costs=costs,
            episodes=episodes,
        )
        observed["student" if is_student else "teacher"] = batch
        return batch

    monkeypatch.setattr(experiment_module, "collect_trajectories", heldout_batch)
    output = tmp_path / "evaluation"
    summary = run_experiment(
        env_id="CartPole-v1",
        output=output,
        backend="clear",
        preset=evaluation_preset(),
        seed_root="heldout-occupancy-test",
        teacher_checkpoint=checkpoint(),
    )

    student = observed["student"]
    teacher = observed["teacher"]
    assert summary.champion_return_mean == pytest.approx(np.mean(student.returns))
    assert summary.teacher_return_mean == pytest.approx(np.mean(teacher.returns))
    assert summary.constraint_cost == pytest.approx(np.mean(student.constraint_costs))

    certificate = json.loads((output / "certificates" / "heldout.json").read_text())
    assert certificate["observations"] == student.observations.shape[0]
    assert certificate["observations"] != teacher.observations.shape[0]
    assert summary.certified_coverage == certificate["coverage"]

    rows = [
        json.loads(line)
        for line in (output / "evaluation" / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 * len(student.episodes)
    for index in range(0, len(rows), 2):
        teacher_row, student_row = rows[index : index + 2]
        assert teacher_row["mode"] == "FLOAT TEACHER"
        assert student_row["mode"] == "QUANTIZED CLEAR"
        assert teacher_row["seed"] == student_row["seed"]

    seed_plan = json.loads((output / "seeds.json").read_text())
    assert len(seed_plan["selection"]) == evaluation_preset().selection_episodes
    assert len(seed_plan["evaluation"]) == evaluation_preset().evaluation_episodes
    assert set(seed_plan["selection"]).isdisjoint(seed_plan["evaluation"])

    selection_rows = [
        json.loads(line)
        for line in (output / "search" / "selection-episodes.jsonl").read_text().splitlines()
    ]
    assert len(selection_rows) == summary.candidates * evaluation_preset().selection_episodes
    assert all(
        set(row)
        == {
            "candidate_digest",
            "seed",
            "total_return",
            "constraint_cost",
            "range_valid",
            "steps",
            "teacher_agreement_count",
            "certified_count",
            "certified_mismatch_count",
            "saturation_count",
            "action_digest",
            "mode",
        }
        for row in selection_rows
    )
    assert all(row["mode"] == "QUANTIZED CLEAR SELECTION" for row in selection_rows)
    selection_keys = {(row["candidate_digest"], row["seed"]) for row in selection_rows}
    assert len(selection_keys) == len(selection_rows)
    candidate_digests = {row["candidate_digest"] for row in selection_rows}
    candidate_rows = [
        json.loads(line)
        for line in (output / "search" / "candidates.jsonl").read_text().splitlines()
    ]
    candidate_metrics = {row["metrics"]["policy_digest"]: row["metrics"] for row in candidate_rows}
    for candidate_digest in candidate_digests:
        candidate_seeds = {
            row["seed"] for row in selection_rows if row["candidate_digest"] == candidate_digest
        }
        assert candidate_seeds == set(seed_plan["selection"])
        rows_for_candidate = [
            row for row in selection_rows if row["candidate_digest"] == candidate_digest
        ]
        steps = sum(row["steps"] for row in rows_for_candidate)
        metrics = candidate_metrics[candidate_digest]
        assert metrics["teacher_agreement"] == (
            sum(row["teacher_agreement_count"] for row in rows_for_candidate) / steps
        )
        assert metrics["certified_coverage"] == (
            sum(row["certified_count"] for row in rows_for_candidate) / steps
        )
        saturation_count = sum(row["saturation_count"] for row in rows_for_candidate)
        assert metrics["range_valid"] == (saturation_count == 0)
        assert all(
            row["range_valid"] == (row["saturation_count"] == 0) for row in rows_for_candidate
        )
        assert sum(row["certified_mismatch_count"] for row in rows_for_candidate) <= sum(
            row["certified_count"] for row in rows_for_candidate
        )
        assert metrics["return_mean"] == pytest.approx(
            np.mean([row["total_return"] for row in rows_for_candidate])
        )
        assert metrics["constraint_cost"] == pytest.approx(
            np.mean([row["constraint_cost"] for row in rows_for_candidate])
        )
