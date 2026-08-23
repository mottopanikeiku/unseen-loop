"""Deterministic NumPy CEM teachers and student-occupancy trajectory collection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from unseen_loop.specs import FloatArray, IntArray


class ScorePolicy(Protocol):
    def score(self, observation: npt.ArrayLike) -> FloatArray: ...


@dataclass(frozen=True)
class TeacherCheckpoint:
    env_id: str
    observation_size: int
    actions: int
    hidden_size: int
    parameters: tuple[float, ...]
    training_seed: int
    iterations: int
    population: int
    elite_fraction: float
    schema_version: str = "unseen-loop/teacher-v1"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TeacherCheckpoint:
        return cls(
            env_id=str(raw["env_id"]),
            observation_size=int(raw["observation_size"]),
            actions=int(raw["actions"]),
            hidden_size=int(raw["hidden_size"]),
            parameters=tuple(float(value) for value in raw["parameters"]),
            training_seed=int(raw["training_seed"]),
            iterations=int(raw["iterations"]),
            population=int(raw["population"]),
            elite_fraction=float(raw["elite_fraction"]),
            schema_version=str(raw.get("schema_version", "unseen-loop/teacher-v1")),
        )

    @classmethod
    def from_json(cls, payload: str) -> TeacherCheckpoint:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("teacher checkpoint must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    total_return: float
    constraint_cost: float
    length: int
    terminated: bool
    truncated: bool
    action_digest: str


@dataclass(frozen=True)
class TrajectoryBatch:
    observations: FloatArray
    scores: FloatArray
    actions: IntArray
    episode_ids: IntArray
    steps: IntArray
    returns: tuple[float, ...]
    constraint_costs: tuple[float, ...]
    episodes: tuple[EpisodeResult, ...]


@dataclass(frozen=True)
class CEMIteration:
    iteration: int
    elite_mean_return: float
    population_mean_return: float
    best_return: float
    sigma_mean: float


class MLPTeacher:
    """Small deterministic tanh teacher used as a clear RL utility ceiling."""

    def __init__(self, checkpoint: TeacherCheckpoint) -> None:
        self.checkpoint = checkpoint
        expected = parameter_count(
            checkpoint.observation_size, checkpoint.hidden_size, checkpoint.actions
        )
        if len(checkpoint.parameters) != expected:
            raise ValueError(f"expected {expected} teacher parameters")
        self._parameters = np.asarray(checkpoint.parameters, dtype=np.float64)
        self._w1, self._b1, self._w2, self._b2 = unpack_parameters(
            self._parameters,
            checkpoint.observation_size,
            checkpoint.hidden_size,
            checkpoint.actions,
        )

    def score(self, observation: npt.ArrayLike) -> FloatArray:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape[-1:] != (self.checkpoint.observation_size,):
            raise ValueError("observation has the wrong shape")
        hidden = np.tanh(values @ self._w1 + self._b1)
        return np.asarray(hidden @ self._w2 + self._b2, dtype=np.float64)

    def action(self, observation: npt.ArrayLike) -> int:
        return int(np.argmax(self.score(observation)))


def parameter_count(observations: int, hidden: int, actions: int) -> int:
    if min(observations, hidden, actions) < 1:
        raise ValueError("network dimensions must be positive")
    return observations * hidden + hidden + hidden * actions + actions


def unpack_parameters(
    parameters: FloatArray,
    observations: int,
    hidden: int,
    actions: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    offset = 0
    w1_size = observations * hidden
    w1 = parameters[offset : offset + w1_size].reshape(observations, hidden)
    offset += w1_size
    b1 = parameters[offset : offset + hidden]
    offset += hidden
    w2_size = hidden * actions
    w2 = parameters[offset : offset + w2_size].reshape(hidden, actions)
    offset += w2_size
    b2 = parameters[offset : offset + actions]
    return w1, b1, w2, b2


def observation_constraint_cost(env_id: str, observation: npt.ArrayLike) -> float:
    """Public benchmark cost: proximity to known classic-control failure boundaries."""
    values = np.asarray(observation, dtype=np.float64)
    if env_id == "CartPole-v1":
        position_cost = max(0.0, abs(float(values[0])) / 2.4 - 0.75)
        angle_limit = 12 * np.pi / 180
        angle_cost = max(0.0, abs(float(values[2])) / angle_limit - 0.75)
        return position_cost + angle_cost
    if env_id == "MountainCar-v0":
        return max(0.0, -float(values[0]) - 1.1)
    if env_id == "Acrobot-v1":
        angular_velocity = abs(float(values[4])) + abs(float(values[5]))
        return max(0.0, angular_velocity - 8.0) / 8.0
    return 0.0


def rollout(
    env_id: str,
    policy: ScorePolicy,
    *,
    seed: int,
    max_steps: int | None = None,
    collect: bool = False,
) -> tuple[EpisodeResult, tuple[list[FloatArray], list[FloatArray], list[int]] | None]:
    env = gym.make(env_id)
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_return = 0.0
    constraint_cost = 0.0
    action_hasher = hashlib.sha256()
    observations: list[FloatArray] = []
    scores_list: list[FloatArray] = []
    actions: list[int] = []
    terminated = False
    truncated = False
    step = 0
    configured_limit = env.spec.max_episode_steps if env.spec is not None else None
    limit = max_steps if max_steps is not None else int(configured_limit or 1_000)
    try:
        while step < limit and not (terminated or truncated):
            scores = np.asarray(policy.score(observation), dtype=np.float64)
            action = int(np.argmax(scores))
            if collect:
                observations.append(np.asarray(observation, dtype=np.float64).copy())
                scores_list.append(scores.copy())
                actions.append(action)
            action_hasher.update(action.to_bytes(4, "little", signed=False))
            constraint_cost += observation_constraint_cost(env_id, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            step += 1
    finally:
        env.close()
    result = EpisodeResult(
        seed=seed,
        total_return=total_return,
        constraint_cost=constraint_cost,
        length=step,
        terminated=terminated,
        truncated=truncated,
        action_digest=action_hasher.hexdigest(),
    )
    traces = (observations, scores_list, actions) if collect else None
    return result, traces


def collect_trajectories(
    env_id: str,
    policy: ScorePolicy,
    seeds: tuple[int, ...],
    *,
    max_steps: int | None = None,
) -> TrajectoryBatch:
    observations: list[FloatArray] = []
    scores: list[FloatArray] = []
    actions: list[int] = []
    episode_ids: list[int] = []
    steps: list[int] = []
    returns: list[float] = []
    costs: list[float] = []
    episode_results: list[EpisodeResult] = []
    for episode_id, seed in enumerate(seeds):
        result, traces = rollout(env_id, policy, seed=seed, max_steps=max_steps, collect=True)
        assert traces is not None
        obs_rows, score_rows, action_rows = traces
        observations.extend(obs_rows)
        scores.extend(score_rows)
        actions.extend(action_rows)
        episode_ids.extend([episode_id] * len(obs_rows))
        steps.extend(range(len(obs_rows)))
        returns.append(result.total_return)
        costs.append(result.constraint_cost)
        episode_results.append(result)
    return TrajectoryBatch(
        observations=np.asarray(observations, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        returns=tuple(returns),
        constraint_costs=tuple(costs),
        episodes=tuple(episode_results),
    )


class _CandidatePolicy:
    def __init__(
        self, parameters: FloatArray, observation_size: int, hidden_size: int, actions: int
    ) -> None:
        self._w1, self._b1, self._w2, self._b2 = unpack_parameters(
            parameters, observation_size, hidden_size, actions
        )

    def score(self, observation: npt.ArrayLike) -> FloatArray:
        values = np.asarray(observation, dtype=np.float64)
        return np.asarray(np.tanh(values @ self._w1 + self._b1) @ self._w2 + self._b2)


def train_cem_teacher(
    env_id: str,
    *,
    seed: int,
    hidden_size: int = 16,
    iterations: int = 24,
    population: int = 64,
    elite_fraction: float = 0.15,
    episodes_per_candidate: int = 2,
    max_steps: int | None = None,
) -> tuple[MLPTeacher, tuple[CEMIteration, ...]]:
    """Train a policy by reward-ranked parameter search; no supervised shortcut."""
    if iterations < 1 or population < 4 or episodes_per_candidate < 1:
        raise ValueError("iterations, population, and episodes_per_candidate must be positive")
    if not 0 < elite_fraction < 1:
        raise ValueError("elite_fraction must lie in (0, 1)")
    probe = gym.make(env_id)
    try:
        if not isinstance(probe.observation_space, gym.spaces.Box):
            raise ValueError("CEM teacher requires a vector Box observation space")
        if probe.observation_space.shape is None:
            raise ValueError("CEM teacher requires a fixed observation shape")
        observation_size = int(np.prod(probe.observation_space.shape))
        if not isinstance(probe.action_space, gym.spaces.Discrete):
            raise ValueError("CEM teacher currently supports discrete action spaces only")
        actions = int(probe.action_space.n)
    finally:
        probe.close()

    rng = np.random.default_rng(seed)
    count = parameter_count(observation_size, hidden_size, actions)
    mean = np.zeros(count, dtype=np.float64)
    sigma = np.ones(count, dtype=np.float64)
    elite_count = max(2, int(np.ceil(population * elite_fraction)))
    history: list[CEMIteration] = []
    best_parameters = mean.copy()
    best_return = -np.inf

    for iteration in range(iterations):
        candidates = rng.normal(mean, sigma, size=(population, count))
        returns = np.empty(population, dtype=np.float64)
        episode_seeds = tuple(
            seed * 100_000 + iteration * 1_000 + index for index in range(episodes_per_candidate)
        )
        for candidate_index, parameters in enumerate(candidates):
            policy = _CandidatePolicy(parameters, observation_size, hidden_size, actions)
            candidate_returns = [
                rollout(env_id, policy, seed=episode_seed, max_steps=max_steps)[0].total_return
                for episode_seed in episode_seeds
            ]
            returns[candidate_index] = float(np.mean(candidate_returns))
        elite_indices = np.argpartition(returns, -elite_count)[-elite_count:]
        elites = candidates[elite_indices]
        mean = np.mean(elites, axis=0)
        sigma = np.maximum(np.std(elites, axis=0), 0.03)
        iteration_best = int(np.argmax(returns))
        if returns[iteration_best] > best_return:
            best_return = float(returns[iteration_best])
            best_parameters = candidates[iteration_best].copy()
        history.append(
            CEMIteration(
                iteration=iteration,
                elite_mean_return=float(np.mean(returns[elite_indices])),
                population_mean_return=float(np.mean(returns)),
                best_return=best_return,
                sigma_mean=float(np.mean(sigma)),
            )
        )

    checkpoint = TeacherCheckpoint(
        env_id=env_id,
        observation_size=observation_size,
        actions=actions,
        hidden_size=hidden_size,
        parameters=tuple(float(value) for value in best_parameters),
        training_seed=seed,
        iterations=iterations,
        population=population,
        elite_fraction=elite_fraction,
    )
    return MLPTeacher(checkpoint), tuple(history)
