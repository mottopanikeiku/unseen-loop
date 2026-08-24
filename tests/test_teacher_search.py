from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

import unseen_loop.search as search_module
from unseen_loop.experiment import ResearchPreset, SeedPlan
from unseen_loop.search import SearchConfig, pareto_front, search_policies
from unseen_loop.teacher import MLPTeacher, TeacherCheckpoint, rollout, train_cem_teacher


def cartpole_teacher() -> MLPTeacher:
    w1 = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, -5.0],
            [1.0, -1.0],
        ]
    )
    b1 = np.zeros(2)
    w2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    b2 = np.zeros(2)
    parameters = np.concatenate((w1.ravel(), b1, w2.ravel(), b2))
    return MLPTeacher(
        TeacherCheckpoint(
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
    )


def test_teacher_rollout_is_seed_deterministic() -> None:
    teacher = cartpole_teacher()
    first, _ = rollout("CartPole-v1", teacher, seed=42, max_steps=50)
    second, _ = rollout("CartPole-v1", teacher, seed=42, max_steps=50)
    assert first == second
    assert first.length > 0


def test_small_search_returns_a_valid_pareto_candidate() -> None:
    records = search_policies(
        cartpole_teacher(),
        distillation_seeds=(1, 2),
        refinement_seeds=(3,),
        selection_seeds=(4, 5),
        config=SearchConfig(
            degrees=(1,),
            input_bits=(4,),
            coefficient_bits=(6,),
            ridge_values=(1e-3,),
            refinement_rounds=1,
            calibration_padding=5.0,
        ),
    )
    assert len(records) == 1
    assert pareto_front(records) == records
    assert 0 <= records[0].metrics.teacher_agreement <= 1
    assert 0 <= records[0].metrics.certified_coverage <= 1
    assert tuple(episode.seed for episode in records[0].selection_episodes) == (4, 5)
    selection = records[0].selection_episodes
    selection_steps = sum(episode.steps for episode in selection)
    assert records[0].metrics.teacher_agreement == (
        sum(episode.teacher_agreement_count for episode in selection) / selection_steps
    )
    assert records[0].metrics.certified_coverage == (
        sum(episode.certified_count for episode in selection) / selection_steps
    )
    assert records[0].metrics.range_valid == (
        sum(episode.saturation_count for episode in selection) == 0
    )


def test_seed_plan_splits_are_disjoint_and_respect_episode_counts() -> None:
    seeds = SeedPlan.derive(
        "split-test",
        "CartPole-v1",
        full=False,
        selection_episodes=7,
        evaluation_episodes=11,
    )

    assert len(seeds.selection) == 7
    assert len(seeds.evaluation) == 11
    splits = (
        seeds.distillation,
        seeds.refinement,
        seeds.selection,
        seeds.evaluation,
    )
    assert all(not (set(left) & set(right)) for left, right in combinations(splits, 2))
    quick = ResearchPreset.quick()
    release = ResearchPreset.release()
    assert (quick.selection_episodes, quick.evaluation_episodes) == (8, 8)
    assert (release.selection_episodes, release.evaluation_episodes) == (100, 100)


def test_search_rejects_overlapping_split_seeds() -> None:
    with pytest.raises(ValueError, match="selection and refinement"):
        search_policies(
            cartpole_teacher(),
            distillation_seeds=(1,),
            selection_seeds=(2,),
            refinement_seeds=(2,),
            config=SearchConfig(
                degrees=(1,),
                input_bits=(4,),
                coefficient_bits=(6,),
                ridge_values=(1e-3,),
                refinement_rounds=0,
            ),
        )


def test_range_invalid_candidate_is_excluded_from_pareto_front() -> None:
    (valid,) = search_policies(
        cartpole_teacher(),
        distillation_seeds=(101,),
        refinement_seeds=(102,),
        selection_seeds=(103,),
        config=SearchConfig(
            degrees=(1,),
            input_bits=(4,),
            coefficient_bits=(6,),
            ridge_values=(1e-3,),
            refinement_rounds=0,
            calibration_padding=5.0,
        ),
    )
    valid = replace(valid, metrics=replace(valid.metrics, range_valid=True))
    invalid = replace(
        valid,
        metrics=replace(
            valid.metrics,
            range_valid=False,
            return_mean=valid.metrics.return_mean + 1_000.0,
            certified_coverage=1.0,
        ),
    )

    assert pareto_front((invalid, valid)) == (valid,)
    assert pareto_front((invalid,)) == ()


def test_certificate_and_occupancy_ablation_switches_form_full_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weighting_calls = 0
    original_weighting = search_module.certificate_guided_weights

    def count_weighting(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal weighting_calls
        weighting_calls += 1
        return original_weighting(*args, **kwargs)

    monkeypatch.setattr(search_module, "certificate_guided_weights", count_weighting)
    outcomes: dict[tuple[bool, bool], tuple[int, int]] = {}
    for certificate_weighting, student_occupancy in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        calls_before = weighting_calls
        (record,) = search_policies(
            cartpole_teacher(),
            distillation_seeds=(201,),
            refinement_seeds=(202,),
            selection_seeds=(203,),
            config=SearchConfig(
                degrees=(1,),
                input_bits=(4,),
                coefficient_bits=(6,),
                ridge_values=(1e-3,),
                refinement_rounds=1,
                calibration_padding=5.0,
                certificate_weighting=certificate_weighting,
                student_occupancy_refinement=student_occupancy,
            ),
        )
        outcomes[(certificate_weighting, student_occupancy)] = (
            weighting_calls - calls_before,
            record.train_samples,
        )

    assert outcomes[(False, False)][0] == 0
    assert outcomes[(False, True)][0] == 0
    assert outcomes[(True, False)][0] > 0
    assert outcomes[(True, True)][0] > outcomes[(True, False)][0]
    base_samples = outcomes[(False, False)][1]
    assert outcomes[(True, False)][1] == base_samples
    assert outcomes[(False, True)][1] > base_samples
    assert outcomes[(True, True)][1] > base_samples


def test_cem_training_executes_real_environment_steps() -> None:
    teacher, history = train_cem_teacher(
        "CartPole-v1",
        seed=3,
        hidden_size=2,
        iterations=1,
        population=4,
        episodes_per_candidate=1,
        max_steps=5,
    )
    assert len(history) == 1
    assert teacher.checkpoint.iterations == 1
    assert np.isfinite(history[0].best_return)
