from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from unseen_loop.gpu_teacher import _cartpole_returns, train_cartpole_gpu
from unseen_loop.teacher import parameter_count

torch = pytest.importorskip("torch")


def gym_constant_left_return(initial_state: np.ndarray) -> float:
    env = gym.make("CartPole-v1")
    try:
        env.reset(seed=0)
        env.unwrapped.state = initial_state.copy()
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(0)
            total += float(reward)
        return total
    finally:
        env.close()


def test_vectorized_cartpole_matches_gym_for_constant_policy() -> None:
    initial = np.asarray([[0.01, -0.02, 0.03, -0.01], [-0.04, 0.01, -0.02, 0.02]], dtype=np.float32)
    hidden_size = 2
    parameters = torch.zeros((1, parameter_count(4, hidden_size, 2)), dtype=torch.float32)
    vectorized = _cartpole_returns(
        parameters,
        torch.from_numpy(initial),
        hidden_size=hidden_size,
    )
    expected = np.mean([gym_constant_left_return(row) for row in initial])
    assert float(vectorized[0]) == pytest.approx(expected)


def test_gpu_trainer_runs_on_cpu_for_conformance() -> None:
    result = train_cartpole_gpu(
        seed=9,
        hidden_size=2,
        iterations=1,
        population=4,
        episodes_per_candidate=2,
        device="cpu",
    )
    assert result.device == "cpu"
    assert result.checkpoint.population == 4
    assert 0 < result.best_vectorized_return <= 500
