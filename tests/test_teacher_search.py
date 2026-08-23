from __future__ import annotations

import numpy as np

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
        evaluation_seeds=(4, 5),
        config=SearchConfig(
            degrees=(1,),
            input_bits=(4,),
            coefficient_bits=(6,),
            ridge_values=(1e-3,),
            refinement_rounds=0,
        ),
    )
    assert len(records) == 1
    assert pareto_front(records) == records
    assert 0 <= records[0].metrics.teacher_agreement <= 1
    assert 0 <= records[0].metrics.certified_coverage <= 1


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
