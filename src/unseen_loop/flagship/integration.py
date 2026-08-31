"""Integrated CipherShield and private-OPE experiment semantics.

The shield changes the executed-action transition kernel, while OPE remains over
requested actions.  In particular, a behavior row always stores
``mu(requested_action | state)`` even when several requested actions are mapped
to the same executed action by the shield.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from unseen_loop.ope.types import TrajectoryBatch, TrajectorySpec
from unseen_loop.shield.environment import WarehouseEnvironment
from unseen_loop.shield.shield import ShieldConfig, ShieldMode, shield_step
from unseen_loop.shield.types import STATE_DIM, Action, ScenarioSpec, ShieldState

ACTION_COUNT = len(Action)
BEHAVIOR_POLICY_MIX = 0.5
DEFAULT_SCENARIO_IDS = tuple(f"scenario-{index:02d}" for index in range(1, 13))


class ShieldVariant(StrEnum):
    """The three shield-induced MDPs in the flagship study."""

    OFF = "off"
    H1 = "h1"
    H2 = "h2"

    @property
    def config(self) -> ShieldConfig:
        if self is ShieldVariant.OFF:
            return ShieldConfig(mode=ShieldMode.NO_SHIELD_ABLATION)
        if self is ShieldVariant.H1:
            return ShieldConfig(mode=ShieldMode.ONE_STEP_ABLATION)
        return ShieldConfig(mode=ShieldMode.ROBUST)


class TrajectoryKind(StrEnum):
    BEHAVIOR = "behavior"
    DIRECT = "direct"


class Outcome(StrEnum):
    RETURN = "return"
    UNSAFE_STEPS = "unsafe_steps"


@dataclass(frozen=True, slots=True)
class FrozenRequestedPolicy:
    """Frozen stochastic requested-action policy with affine softmax logits."""

    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    temperature: float = 1.0
    schema_version: str = "unseen-loop/frozen-requested-policy-v1"

    def __post_init__(self) -> None:
        weights = tuple(tuple(float(value) for value in row) for row in self.weights)
        bias = tuple(float(value) for value in self.bias)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        if len(weights) != ACTION_COUNT or any(len(row) != STATE_DIM for row in weights):
            raise ValueError(f"weights must have shape ({ACTION_COUNT}, {STATE_DIM})")
        if len(bias) != ACTION_COUNT:
            raise ValueError(f"bias must have {ACTION_COUNT} entries")
        flat = (*bias, *(value for row in weights for value in row), float(self.temperature))
        if not all(math.isfinite(value) for value in flat):
            raise ValueError("policy parameters must be finite")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")

    @classmethod
    def constant(cls, probabilities: Sequence[float]) -> FrozenRequestedPolicy:
        """Build a state-independent policy; every action retains stochastic support."""

        values = tuple(float(value) for value in probabilities)
        if len(values) != ACTION_COUNT:
            raise ValueError(f"probabilities must have {ACTION_COUNT} entries")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("constant policy probabilities must be finite and positive")
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("constant policy probabilities must sum to one")
        return cls(
            weights=tuple((0.0,) * STATE_DIM for _ in Action),
            bias=tuple(math.log(value) for value in values),
        )

    def probabilities(self, state: ShieldState | Sequence[float]) -> tuple[float, ...]:
        values = (
            state.as_tuple() if isinstance(state, ShieldState) else tuple(float(x) for x in state)
        )
        if len(values) != STATE_DIM or not all(math.isfinite(value) for value in values):
            raise ValueError(f"state must contain {STATE_DIM} finite values")
        logits = tuple(
            (sum(weight * value for weight, value in zip(row, values, strict=True)) + bias)
            / self.temperature
            for row, bias in zip(self.weights, self.bias, strict=True)
        )
        maximum = max(logits)
        exponentials = tuple(math.exp(value - maximum) for value in logits)
        denominator = sum(exponentials)
        return tuple(value / denominator for value in exponentials)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "weights": self.weights,
                "bias": self.bias,
                "temperature": self.temperature,
                "schema_version": self.schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def behavior_probabilities(target_probabilities: Sequence[float]) -> tuple[float, ...]:
    """Return the frozen behavior policy ``mu = .5*pi + .5*Uniform``."""

    target = tuple(float(value) for value in target_probabilities)
    if len(target) != ACTION_COUNT:
        raise ValueError(f"target probabilities must have {ACTION_COUNT} entries")
    if any(not math.isfinite(value) or value < 0 for value in target):
        raise ValueError("target probabilities must be finite and non-negative")
    if not math.isclose(sum(target), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("target probabilities must sum to one")
    uniform = 1.0 / ACTION_COUNT
    return tuple(BEHAVIOR_POLICY_MIX * value + BEHAVIOR_POLICY_MIX * uniform for value in target)


def _sample(probabilities: Sequence[float], rng: random.Random) -> Action:
    draw = rng.random()
    cumulative = 0.0
    for action, probability in zip(Action, probabilities, strict=True):
        cumulative += probability
        if draw < cumulative:
            return action
    return tuple(Action)[-1]


@dataclass(frozen=True, slots=True)
class LoggedStep:
    """Client-local transition log with requested and executed actions separated."""

    step: int
    state: tuple[float, ...]
    requested_action: Action
    requested_action_probability: float
    mu_propensity: float
    executed_action: Action
    reward: float
    unsafe_step: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", tuple(float(value) for value in self.state))
        object.__setattr__(self, "requested_action", Action(self.requested_action))
        object.__setattr__(self, "executed_action", Action(self.executed_action))
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if len(self.state) != STATE_DIM or not all(math.isfinite(value) for value in self.state):
            raise ValueError(f"state must contain {STATE_DIM} finite values")
        for name in ("requested_action_probability", "mu_propensity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0 or value > 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "requested_action": int(self.requested_action),
            "requested_action_probability": self.requested_action_probability,
            "mu_propensity": self.mu_propensity,
            "executed_action": int(self.executed_action),
            "reward": self.reward,
            "unsafe_step": self.unsafe_step,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryLog:
    scenario_id: str
    shield: ShieldVariant
    kind: TrajectoryKind
    trajectory_index: int
    seed: int
    policy_digest: str
    steps: tuple[LoggedStep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        object.__setattr__(self, "kind", TrajectoryKind(self.kind))
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.trajectory_index < 0 or self.seed < 0:
            raise ValueError("trajectory_index and seed must be non-negative")
        if not self.steps:
            raise ValueError("trajectory must contain at least one step")
        if tuple(step.step for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("trajectory step indices must be consecutive from zero")
        if len(self.policy_digest) != 64:
            raise ValueError("policy_digest must be a SHA-256 digest")

    @property
    def total_return(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def unsafe_steps(self) -> int:
        return sum(step.unsafe_step for step in self.steps)


@dataclass(frozen=True, slots=True)
class TrajectoryPlan:
    scenario_id: str
    shield: ShieldVariant
    kind: TrajectoryKind
    trajectory_count: int
    horizon: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        object.__setattr__(self, "kind", TrajectoryKind(self.kind))
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.trajectory_count < 1 or self.horizon < 1:
            raise ValueError("trajectory_count and horizon must be positive")


@dataclass(frozen=True, slots=True)
class OPECallPlan:
    scenario_id: str
    shield: ShieldVariant
    outcome: Outcome
    batch_index: int
    trajectory_start: int
    trajectories: int
    horizon: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.batch_index < 0 or self.trajectory_start < 0:
            raise ValueError("batch_index and trajectory_start must be non-negative")
        if self.trajectories < 1 or self.horizon < 1:
            raise ValueError("trajectories and horizon must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.trajectories, self.horizon


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Executable production schedule; two outcomes cause two OPE calls per batch."""

    scenario_ids: tuple[str, ...] = DEFAULT_SCENARIO_IDS
    shields: tuple[ShieldVariant, ...] = tuple(ShieldVariant)
    behavior_trajectories: int = 4_096
    direct_trajectories: int = 2_048
    batch_size: int = 256
    horizon: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ids", tuple(self.scenario_ids))
        object.__setattr__(self, "shields", tuple(ShieldVariant(value) for value in self.shields))
        if not self.scenario_ids or any(not value for value in self.scenario_ids):
            raise ValueError("scenario_ids must be non-empty strings")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("scenario_ids must be unique")
        if self.shields != tuple(ShieldVariant):
            raise ValueError("the flagship plan requires off, h1, and h2 MDPs in that order")
        counts = (
            self.behavior_trajectories,
            self.direct_trajectories,
            self.batch_size,
            self.horizon,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts
        ):
            raise ValueError("trajectory counts, batch_size, and horizon must be positive integers")
        if self.behavior_trajectories % self.batch_size:
            raise ValueError("behavior_trajectories must be divisible by batch_size")

    @property
    def behavior_trajectory_count(self) -> int:
        return len(self.scenario_ids) * len(self.shields) * self.behavior_trajectories

    @property
    def direct_trajectory_count(self) -> int:
        return len(self.scenario_ids) * len(self.shields) * self.direct_trajectories

    @property
    def ope_call_count(self) -> int:
        return (
            len(self.scenario_ids)
            * len(self.shields)
            * (self.behavior_trajectories // self.batch_size)
            * len(Outcome)
        )

    def trajectory_plans(self, kind: TrajectoryKind | None = None) -> tuple[TrajectoryPlan, ...]:
        kinds = tuple(TrajectoryKind) if kind is None else (TrajectoryKind(kind),)
        return tuple(
            TrajectoryPlan(
                scenario_id=scenario_id,
                shield=shield,
                kind=trajectory_kind,
                trajectory_count=(
                    self.behavior_trajectories
                    if trajectory_kind is TrajectoryKind.BEHAVIOR
                    else self.direct_trajectories
                ),
                horizon=self.horizon,
            )
            for scenario_id in self.scenario_ids
            for shield in self.shields
            for trajectory_kind in kinds
        )

    def ope_calls(self) -> tuple[OPECallPlan, ...]:
        batch_count = self.behavior_trajectories // self.batch_size
        return tuple(
            OPECallPlan(
                scenario_id=scenario_id,
                shield=shield,
                outcome=outcome,
                batch_index=batch_index,
                trajectory_start=batch_index * self.batch_size,
                trajectories=self.batch_size,
                horizon=self.horizon,
            )
            for scenario_id in self.scenario_ids
            for shield in self.shields
            for batch_index in range(batch_count)
            for outcome in Outcome
        )


def run_trajectory(
    scenario_id: str,
    scenario: ScenarioSpec,
    policy: FrozenRequestedPolicy,
    shield: ShieldVariant,
    kind: TrajectoryKind,
    *,
    trajectory_index: int,
    seed: int,
    horizon: int = 64,
) -> TrajectoryLog:
    """Run one fixed-horizon requested-action MDP trajectory.

    Terminal states are continued as absorbing states with zero reward and zero
    unsafe cost so the encrypted OPE tensor remains rectangular.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    variant = ShieldVariant(shield)
    source = TrajectoryKind(kind)
    environment = WarehouseEnvironment(scenario, seed=seed)
    state = environment.reset(seed=seed)
    rng = random.Random(seed)
    done = False
    rows: list[LoggedStep] = []
    for step_index in range(horizon):
        decision_state = state.as_tuple()
        target = policy.probabilities(state)
        behavior = behavior_probabilities(target)
        sampling = behavior if source is TrajectoryKind.BEHAVIOR else target
        requested = _sample(sampling, rng)
        decision = shield_step(
            state,
            requested,
            step=step_index,
            dynamics=scenario.dynamics,
            limits=scenario.safety,
            config=variant.config,
        )
        executed = decision.selected_action
        if done:
            reward = 0.0
            unsafe = False
        else:
            result = environment.step(executed)
            reward = result.reward
            unsafe = result.safety.unsafe
            state = result.state
            done = result.done
        rows.append(
            LoggedStep(
                step=step_index,
                state=decision_state,
                requested_action=requested,
                requested_action_probability=sampling[int(requested)],
                mu_propensity=behavior[int(requested)],
                executed_action=executed,
                reward=reward,
                unsafe_step=unsafe,
            )
        )
    return TrajectoryLog(
        scenario_id=scenario_id,
        shield=variant,
        kind=source,
        trajectory_index=trajectory_index,
        seed=seed,
        policy_digest=policy.digest,
        steps=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class PreparedOPEBatch:
    """Requested-action OPE tensors plus clear target propensities for conformance."""

    scenario_id: str
    shield: ShieldVariant
    outcome: Outcome
    trajectories: TrajectoryBatch
    target_propensities: tuple[tuple[float, ...], ...]


def build_ope_batch(
    logs: Sequence[TrajectoryLog],
    policy: FrozenRequestedPolicy,
    outcome: Outcome,
) -> PreparedOPEBatch:
    """Construct one OPE call without ever deriving an executed-action propensity."""

    rows = tuple(logs)
    if not rows:
        raise ValueError("logs cannot be empty")
    first = rows[0]
    horizon = len(first.steps)
    expected = (first.scenario_id, first.shield, TrajectoryKind.BEHAVIOR, policy.digest, horizon)
    for row in rows:
        observed = (row.scenario_id, row.shield, row.kind, row.policy_digest, len(row.steps))
        if observed != expected:
            raise ValueError("OPE batches require homogeneous behavior trajectories and policy")
        for step in row.steps:
            target_probabilities = policy.probabilities(step.state)
            expected_mu = behavior_probabilities(target_probabilities)[int(step.requested_action)]
            if not math.isclose(step.mu_propensity, expected_mu, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("mu_propensity does not match the requested action")
            if not math.isclose(
                step.requested_action_probability,
                step.mu_propensity,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("behavior rows must be sampled with their logged mu propensity")
    selected_outcome = Outcome(outcome)
    rewards = tuple(
        tuple(
            step.reward if selected_outcome is Outcome.RETURN else float(step.unsafe_step)
            for step in row.steps
        )
        for row in rows
    )
    spec = TrajectorySpec(
        trajectories=len(rows),
        horizon=horizon,
        state_dim=STATE_DIM,
        action_count=ACTION_COUNT,
    )
    batch = TrajectoryBatch(
        spec=spec,
        states=tuple(tuple(step.state for step in row.steps) for row in rows),
        actions=tuple(tuple(int(step.requested_action) for step in row.steps) for row in rows),
        rewards=rewards,
        behavior_propensities=tuple(
            tuple(step.mu_propensity for step in row.steps) for row in rows
        ),
    )
    logged_target = tuple(
        tuple(policy.probabilities(step.state)[int(step.requested_action)] for step in row.steps)
        for row in rows
    )
    return PreparedOPEBatch(
        scenario_id=first.scenario_id,
        shield=first.shield,
        outcome=selected_outcome,
        trajectories=batch,
        target_propensities=logged_target,
    )


def iter_ope_batches(
    logs: Sequence[TrajectoryLog],
    policy: FrozenRequestedPolicy,
    *,
    batch_size: int = 256,
) -> Iterator[PreparedOPEBatch]:
    """Yield return and unsafe-step calls for every complete behavior batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(logs) % batch_size:
        raise ValueError("behavior log count must be divisible by batch_size")
    for start in range(0, len(logs), batch_size):
        chunk = logs[start : start + batch_size]
        for outcome in Outcome:
            yield build_ope_batch(chunk, policy, outcome)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    scenario_id: str
    shield: ShieldVariant
    replicate: int
    seed: int
    total_return: float
    unsafe_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        if self.replicate < 0 or self.seed < 0 or self.unsafe_steps < 0:
            raise ValueError("replicate, seed, and unsafe_steps must be non-negative")
        if not math.isfinite(self.total_return):
            raise ValueError("total_return must be finite")

    @classmethod
    def from_log(cls, log: TrajectoryLog) -> EpisodeOutcome:
        return cls(
            scenario_id=log.scenario_id,
            shield=log.shield,
            replicate=log.trajectory_index,
            seed=log.seed,
            total_return=log.total_return,
            unsafe_steps=log.unsafe_steps,
        )


@dataclass(frozen=True, slots=True)
class PairedOnlineTruth:
    """One common-random-number direct-policy replicate across all shield MDPs."""

    scenario_id: str
    replicate: int
    seed: int
    outcomes: tuple[EpisodeOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if tuple(value.shield for value in self.outcomes) != tuple(ShieldVariant):
            raise ValueError("paired truth must contain off, h1, and h2 outcomes in order")
        if any(
            (value.scenario_id, value.replicate, value.seed)
            != (self.scenario_id, self.replicate, self.seed)
            for value in self.outcomes
        ):
            raise ValueError("paired outcomes must share scenario, replicate, and seed")

    def for_shield(self, shield: ShieldVariant) -> EpisodeOutcome:
        return self.outcomes[tuple(ShieldVariant).index(ShieldVariant(shield))]


def run_paired_online_truth(
    scenario_id: str,
    scenario: ScenarioSpec,
    policy: FrozenRequestedPolicy,
    *,
    replicate: int,
    seed: int,
    horizon: int = 64,
) -> PairedOnlineTruth:
    outcomes = tuple(
        EpisodeOutcome.from_log(
            run_trajectory(
                scenario_id,
                scenario,
                policy,
                shield,
                TrajectoryKind.DIRECT,
                trajectory_index=replicate,
                seed=seed,
                horizon=horizon,
            )
        )
        for shield in ShieldVariant
    )
    return PairedOnlineTruth(scenario_id, replicate, seed, outcomes)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.estimate, self.lower, self.upper)):
            raise ValueError("bootstrap values must be finite")
        if self.lower > self.upper or not 0 < self.confidence < 1:
            raise ValueError("bootstrap interval or confidence is invalid")
        if self.samples < 1 or self.seed < 0:
            raise ValueError("samples must be positive and seed non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    scenario_id: str
    baseline: ShieldVariant
    comparison: ShieldVariant
    return_effect: BootstrapInterval
    unsafe_step_effect: BootstrapInterval

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline", ShieldVariant(self.baseline))
        object.__setattr__(self, "comparison", ShieldVariant(self.comparison))
        if self.baseline is self.comparison:
            raise ValueError("baseline and comparison must differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "baseline": self.baseline.value,
            "comparison": self.comparison.value,
            "return_effect": self.return_effect.to_dict(),
            "unsafe_step_effect": self.unsafe_step_effect.to_dict(),
        }


def _bootstrap_mean(
    values: Sequence[float], *, samples: int, seed: int, confidence: float
) -> BootstrapInterval:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if samples < 1 or seed < 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(array), size=(samples, len(array)))
    replicates = np.mean(array[selections], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, (tail, 1.0 - tail))
    return BootstrapInterval(
        estimate=float(np.mean(array)),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
        seed=seed,
    )


def bootstrap_paired_effect(
    truth: Sequence[PairedOnlineTruth],
    comparison: ShieldVariant,
    *,
    baseline: ShieldVariant = ShieldVariant.OFF,
    samples: int = 1_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> EffectEstimate:
    """Bootstrap paired per-seed effects; never resample shield arms independently."""

    pairs = tuple(truth)
    if not pairs:
        raise ValueError("truth cannot be empty")
    scenario_id = pairs[0].scenario_id
    if any(pair.scenario_id != scenario_id for pair in pairs):
        raise ValueError("one effect estimate cannot mix scenarios")
    baseline_variant = ShieldVariant(baseline)
    comparison_variant = ShieldVariant(comparison)
    return_differences = tuple(
        pair.for_shield(comparison_variant).total_return
        - pair.for_shield(baseline_variant).total_return
        for pair in pairs
    )
    unsafe_differences = tuple(
        float(
            pair.for_shield(comparison_variant).unsafe_steps
            - pair.for_shield(baseline_variant).unsafe_steps
        )
        for pair in pairs
    )
    return EffectEstimate(
        scenario_id=scenario_id,
        baseline=baseline_variant,
        comparison=comparison_variant,
        return_effect=_bootstrap_mean(
            return_differences, samples=samples, seed=seed, confidence=confidence
        ),
        unsafe_step_effect=_bootstrap_mean(
            unsafe_differences, samples=samples, seed=seed + 1, confidence=confidence
        ),
    )


@dataclass(frozen=True, slots=True)
class OnlineTruthSummary:
    scenario_id: str
    shield: ShieldVariant
    trajectories: int
    mean_return: float
    mean_unsafe_steps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        if not self.scenario_id or self.trajectories < 1:
            raise ValueError("online truth summary requires a scenario and trajectories")
        if not math.isfinite(self.mean_return) or not math.isfinite(self.mean_unsafe_steps):
            raise ValueError("online truth means must be finite")
        if self.mean_unsafe_steps < 0:
            raise ValueError("mean_unsafe_steps must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "shield": self.shield.value,
            "trajectories": self.trajectories,
            "mean_return": self.mean_return,
            "mean_unsafe_steps": self.mean_unsafe_steps,
        }


def summarize_online_truth(
    truth: Sequence[PairedOnlineTruth], shield: ShieldVariant
) -> OnlineTruthSummary:
    pairs = tuple(truth)
    if not pairs:
        raise ValueError("truth cannot be empty")
    scenario_id = pairs[0].scenario_id
    if any(pair.scenario_id != scenario_id for pair in pairs):
        raise ValueError("online truth summary cannot mix scenarios")
    variant = ShieldVariant(shield)
    outcomes = tuple(pair.for_shield(variant) for pair in pairs)
    return OnlineTruthSummary(
        scenario_id=scenario_id,
        shield=variant,
        trajectories=len(outcomes),
        mean_return=float(np.mean([value.total_return for value in outcomes])),
        mean_unsafe_steps=float(np.mean([value.unsafe_steps for value in outcomes])),
    )


@dataclass(frozen=True, slots=True)
class OutcomeDiscrepancy:
    outcome: Outcome
    ope_estimate: float
    online_truth: float
    signed_error: float = field(init=False)
    absolute_error: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        if not math.isfinite(self.ope_estimate) or not math.isfinite(self.online_truth):
            raise ValueError("discrepancy inputs must be finite")
        error = self.ope_estimate - self.online_truth
        object.__setattr__(self, "signed_error", error)
        object.__setattr__(self, "absolute_error", abs(error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "ope_estimate": self.ope_estimate,
            "online_truth": self.online_truth,
            "signed_error": self.signed_error,
            "absolute_error": self.absolute_error,
        }


@dataclass(frozen=True, slots=True)
class DiscrepancyReport:
    scenario_id: str
    shield: ShieldVariant
    return_discrepancy: OutcomeDiscrepancy
    unsafe_step_discrepancy: OutcomeDiscrepancy

    def __post_init__(self) -> None:
        object.__setattr__(self, "shield", ShieldVariant(self.shield))
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.return_discrepancy.outcome is not Outcome.RETURN:
            raise ValueError("return_discrepancy must contain the return outcome")
        if self.unsafe_step_discrepancy.outcome is not Outcome.UNSAFE_STEPS:
            raise ValueError("unsafe_step_discrepancy must contain the unsafe-step outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "shield": self.shield.value,
            "return_discrepancy": self.return_discrepancy.to_dict(),
            "unsafe_step_discrepancy": self.unsafe_step_discrepancy.to_dict(),
        }

    @classmethod
    def from_estimates(
        cls,
        summary: OnlineTruthSummary,
        *,
        ope_return: float,
        ope_unsafe_steps: float,
    ) -> DiscrepancyReport:
        return cls(
            scenario_id=summary.scenario_id,
            shield=summary.shield,
            return_discrepancy=OutcomeDiscrepancy(Outcome.RETURN, ope_return, summary.mean_return),
            unsafe_step_discrepancy=OutcomeDiscrepancy(
                Outcome.UNSAFE_STEPS, ope_unsafe_steps, summary.mean_unsafe_steps
            ),
        )
