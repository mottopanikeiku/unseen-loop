"""GPU-vectorized CartPole CEM training for Modal research sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unseen_loop.teacher import TeacherCheckpoint, parameter_count


@dataclass(frozen=True)
class GPUTrainingResult:
    checkpoint: TeacherCheckpoint
    device: str
    iterations: int
    population: int
    episodes_per_candidate: int
    best_vectorized_return: float
    elapsed_ns: int
    torch_version: str
    cuda_version: str | None
    device_name: str
    schema_version: str = "unseen-loop/gpu-training-v1"


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GPU training requires the optional torch dependency") from error
    return torch


def _cartpole_returns(
    parameters: Any,
    initial_states: Any,
    *,
    hidden_size: int,
    max_steps: int = 500,
) -> Any:
    """Evaluate an entire policy population under Gymnasium's CartPole equations."""
    torch = _torch()
    population = parameters.shape[0]
    episodes = initial_states.shape[0]
    observations = 4
    actions = 2
    offset = 0
    w1_size = observations * hidden_size
    w1 = parameters[:, offset : offset + w1_size].reshape(population, observations, hidden_size)
    offset += w1_size
    b1 = parameters[:, offset : offset + hidden_size]
    offset += hidden_size
    w2_size = hidden_size * actions
    w2 = parameters[:, offset : offset + w2_size].reshape(population, hidden_size, actions)
    offset += w2_size
    b2 = parameters[:, offset : offset + actions]

    state = initial_states.unsqueeze(0).expand(population, episodes, observations).clone()
    alive = torch.ones((population, episodes), dtype=torch.bool, device=parameters.device)
    returns = torch.zeros((population, episodes), dtype=torch.float32, device=parameters.device)
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    for _ in range(max_steps):
        hidden = torch.tanh(torch.einsum("peo,poh->peh", state, w1) + b1[:, None, :])
        scores = torch.einsum("peh,pha->pea", hidden, w2) + b2[:, None, :]
        action = torch.argmax(scores, dim=-1)
        x, x_dot, theta, theta_dot = state.unbind(dim=-1)
        force = torch.where(action == 1, force_mag, -force_mag)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        temporary = (force + polemass_length * theta_dot.square() * sin_theta) / total_mass
        theta_acceleration = (gravity * sin_theta - cos_theta * temporary) / (
            length * (4.0 / 3.0 - masspole * cos_theta.square() / total_mass)
        )
        x_acceleration = temporary - polemass_length * theta_acceleration * cos_theta / total_mass
        next_state = torch.stack(
            (
                x + tau * x_dot,
                x_dot + tau * x_acceleration,
                theta + tau * theta_dot,
                theta_dot + tau * theta_acceleration,
            ),
            dim=-1,
        )
        returns += alive
        state = torch.where(alive[..., None], next_state, state)
        alive &= (
            (state[..., 0] >= -x_threshold)
            & (state[..., 0] <= x_threshold)
            & (state[..., 2] >= -theta_threshold)
            & (state[..., 2] <= theta_threshold)
        )
        if not bool(torch.any(alive)):
            break
    return torch.mean(returns, dim=1)


def train_cartpole_gpu(
    *,
    seed: int,
    hidden_size: int = 16,
    iterations: int = 30,
    population: int = 4_096,
    elite_fraction: float = 0.05,
    episodes_per_candidate: int = 16,
    device: str = "cuda",
) -> GPUTrainingResult:
    """Train thousands of closed-loop policies concurrently on one GPU."""
    torch = _torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if iterations < 1 or population < 4 or episodes_per_candidate < 1:
        raise ValueError("training dimensions must be positive")
    if not 0 < elite_fraction < 1:
        raise ValueError("elite_fraction must lie in (0, 1)")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    count = parameter_count(4, hidden_size, 2)
    mean = torch.zeros(count, dtype=torch.float32, device=device)
    sigma = torch.ones(count, dtype=torch.float32, device=device)
    elite_count = max(2, int(np.ceil(population * elite_fraction)))
    best_return = -np.inf
    best_parameters = mean.clone()
    torch.cuda.synchronize() if device.startswith("cuda") else None
    import time

    wall_started = time.perf_counter_ns()
    for _ in range(iterations):
        candidates = mean + sigma * torch.randn(
            (population, count), generator=generator, device=device
        )
        initial_states = torch.empty(
            (episodes_per_candidate, 4), dtype=torch.float32, device=device
        ).uniform_(-0.05, 0.05, generator=generator)
        returns = _cartpole_returns(
            candidates, initial_states, hidden_size=hidden_size, max_steps=500
        )
        elite_indices = torch.topk(returns, elite_count, largest=True).indices
        elites = candidates[elite_indices]
        mean = torch.mean(elites, dim=0)
        sigma = torch.clamp(torch.std(elites, dim=0, unbiased=False), min=0.02)
        iteration_value, iteration_index = torch.max(returns, dim=0)
        if float(iteration_value) > best_return:
            best_return = float(iteration_value)
            best_parameters = candidates[int(iteration_index)].clone()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - wall_started

    checkpoint = TeacherCheckpoint(
        env_id="CartPole-v1",
        observation_size=4,
        actions=2,
        hidden_size=hidden_size,
        parameters=tuple(float(value) for value in best_parameters.detach().cpu().numpy()),
        training_seed=seed,
        iterations=iterations,
        population=population,
        elite_fraction=elite_fraction,
    )
    device_name = torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU"
    return GPUTrainingResult(
        checkpoint=checkpoint,
        device=device,
        iterations=iterations,
        population=population,
        episodes_per_candidate=episodes_per_candidate,
        best_vectorized_return=best_return,
        elapsed_ns=elapsed_ns,
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
        device_name=device_name,
    )
