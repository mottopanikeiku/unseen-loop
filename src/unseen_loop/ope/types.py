"""Frozen data contracts for clear off-policy evaluation.

These types describe clear reference semantics.  They make no claim that policy
training, nuisance-model fitting, or client-side post-processing is private.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _float_tuple(values: npt.ArrayLike) -> tuple[Any, ...]:
    array = np.asarray(values, dtype=np.float64)
    return tuple(_float_tuple(row) if isinstance(row, np.ndarray) else float(row) for row in array)


def _int_tuple(values: npt.ArrayLike) -> tuple[Any, ...]:
    array = np.asarray(values, dtype=np.int64)
    return tuple(_int_tuple(row) if isinstance(row, np.ndarray) else int(row) for row in array)


@dataclass(frozen=True)
class FailureRow:
    """Machine-readable location and reason for a rejected OPE input row."""

    code: str
    field: str
    message: str
    trajectory: int | None = None
    step: int | None = None
    value: float | int | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("failure-row value must be finite or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FailureRow:
        return cls(
            code=str(raw["code"]),
            field=str(raw["field"]),
            message=str(raw["message"]),
            trajectory=(None if raw.get("trajectory") is None else int(raw["trajectory"])),
            step=None if raw.get("step") is None else int(raw["step"]),
            value=None if raw.get("value") is None else float(raw["value"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> FailureRow:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("failure row payload must be a JSON object")
        return cls.from_dict(raw)


class OPEValidationError(ValueError):
    """Validation error retaining every observable failing row."""

    def __init__(self, rows: tuple[FailureRow, ...]) -> None:
        if not rows:
            raise ValueError("OPEValidationError requires at least one failure row")
        self.rows = rows
        summary = "; ".join(
            f"{row.code} at ({row.trajectory},{row.step}) {row.field}" for row in rows
        )
        super().__init__(summary)


@dataclass(frozen=True)
class TrajectorySpec:
    """Fixed batch shape and optional closed state/reward domains."""

    trajectories: int
    horizon: int
    state_dim: int
    action_count: int
    state_min: tuple[float, ...] = ()
    state_max: tuple[float, ...] = ()
    reward_min: float | None = None
    reward_max: float | None = None
    schema_version: str = "unseen-loop/ope-trajectories-v1"

    def __post_init__(self) -> None:
        dimensions = (self.trajectories, self.horizon, self.state_dim, self.action_count)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in dimensions
        ):
            raise ValueError("trajectory dimensions and action_count must be integers")
        if min(self.trajectories, self.horizon, self.state_dim) < 1:
            raise ValueError("trajectories, horizon, and state_dim must be positive")
        if self.action_count < 2:
            raise ValueError("action_count must be at least two")
        object.__setattr__(self, "state_min", tuple(float(x) for x in self.state_min))
        object.__setattr__(self, "state_max", tuple(float(x) for x in self.state_max))
        if bool(self.state_min) != bool(self.state_max):
            raise ValueError("state_min and state_max must either both be set or both be empty")
        if self.state_min:
            if len(self.state_min) != self.state_dim or len(self.state_max) != self.state_dim:
                raise ValueError("state bounds must have state_dim entries")
            if any(
                not np.isfinite(low) or not np.isfinite(high) or low > high
                for low, high in zip(self.state_min, self.state_max, strict=True)
            ):
                raise ValueError("state bounds must be finite and ordered")
        if (self.reward_min is None) != (self.reward_max is None):
            raise ValueError("reward_min and reward_max must either both be set or both be None")
        if (
            self.reward_min is not None
            and self.reward_max is not None
            and (
                not np.isfinite(self.reward_min)
                or not np.isfinite(self.reward_max)
                or self.reward_min > self.reward_max
            )
        ):
            raise ValueError("reward bounds must be finite and ordered")

    @property
    def batch_shape(self) -> tuple[int, int]:
        return self.trajectories, self.horizon

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return self.trajectories, self.horizon, self.state_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrajectorySpec:
        return cls(
            trajectories=int(raw["trajectories"]),
            horizon=int(raw["horizon"]),
            state_dim=int(raw["state_dim"]),
            action_count=int(raw["action_count"]),
            state_min=tuple(float(value) for value in raw.get("state_min", ())),
            state_max=tuple(float(value) for value in raw.get("state_max", ())),
            reward_min=(None if raw.get("reward_min") is None else float(raw["reward_min"])),
            reward_max=(None if raw.get("reward_max") is None else float(raw["reward_max"])),
            schema_version=str(raw.get("schema_version", "unseen-loop/ope-trajectories-v1")),
        )

    @classmethod
    def from_json(cls, payload: str) -> TrajectorySpec:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("trajectory spec payload must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class TrajectoryBatch:
    """Immutable rectangular logged trajectories.

    ``behavior_propensities[i][t]`` is the logging-policy probability of the
    action in ``actions[i][t]``.  Zero is retained long enough to produce an
    observable support failure; estimators never divide by it.
    """

    spec: TrajectorySpec
    states: tuple[tuple[tuple[float, ...], ...], ...]
    actions: tuple[tuple[int, ...], ...]
    rewards: tuple[tuple[float, ...], ...]
    behavior_propensities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        raw_actions = np.asarray(self.actions, dtype=np.float64)
        action_rows: list[FailureRow] = []
        if raw_actions.shape == self.spec.batch_shape:
            finite = np.isfinite(raw_actions)
            integer_like = np.zeros(raw_actions.shape, dtype=np.bool_)
            integer_like[finite] = raw_actions[finite] == np.floor(raw_actions[finite])
            in_storage_range = np.zeros(raw_actions.shape, dtype=np.bool_)
            in_storage_range[finite] = (raw_actions[finite] >= -(2**63)) & (
                raw_actions[finite] < 2**63
            )
            for i, t in np.argwhere(~integer_like | ~in_storage_range):
                value = raw_actions[i, t]
                action_rows.append(
                    FailureRow(
                        "invalid_integer",
                        "actions",
                        "action must be a finite int64 integer",
                        int(i),
                        int(t),
                        None if not np.isfinite(value) else float(value),
                    )
                )
        if action_rows:
            raise OPEValidationError(tuple(action_rows))
        object.__setattr__(self, "states", _float_tuple(self.states))
        object.__setattr__(self, "actions", _int_tuple(self.actions))
        object.__setattr__(self, "rewards", _float_tuple(self.rewards))
        object.__setattr__(self, "behavior_propensities", _float_tuple(self.behavior_propensities))
        rows = self.validation_failures()
        if rows:
            raise OPEValidationError(rows)

    @property
    def state_array(self) -> FloatArray:
        return np.asarray(self.states, dtype=np.float64)

    @property
    def action_array(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.actions, dtype=np.int64)

    @property
    def reward_array(self) -> FloatArray:
        return np.asarray(self.rewards, dtype=np.float64)

    @property
    def behavior_array(self) -> FloatArray:
        return np.asarray(self.behavior_propensities, dtype=np.float64)

    def validation_failures(self) -> tuple[FailureRow, ...]:
        expected_state = self.spec.state_shape
        expected_batch = self.spec.batch_shape
        arrays = {
            "states": np.asarray(self.states),
            "actions": np.asarray(self.actions),
            "rewards": np.asarray(self.rewards),
            "behavior_propensities": np.asarray(self.behavior_propensities),
        }
        rows: list[FailureRow] = []
        if arrays["states"].shape != expected_state:
            rows.append(
                FailureRow(
                    "shape_mismatch",
                    "states",
                    f"expected {expected_state}, got {arrays['states'].shape}",
                )
            )
        for field in ("actions", "rewards", "behavior_propensities"):
            if arrays[field].shape != expected_batch:
                rows.append(
                    FailureRow(
                        "shape_mismatch",
                        field,
                        f"expected {expected_batch}, got {arrays[field].shape}",
                    )
                )
        if rows:
            return tuple(rows)

        states = np.asarray(self.states, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.int64)
        rewards = np.asarray(self.rewards, dtype=np.float64)
        behavior = np.asarray(self.behavior_propensities, dtype=np.float64)
        for i, t, d in np.argwhere(~np.isfinite(states)):
            rows.append(
                FailureRow("non_finite", "states", "state must be finite", int(i), int(t), int(d))
            )
        if self.spec.state_min:
            low = np.asarray(self.spec.state_min)
            high = np.asarray(self.spec.state_max)
            for i, t, d in np.argwhere((states < low) | (states > high)):
                rows.append(
                    FailureRow(
                        "out_of_range",
                        f"states[{int(d)}]",
                        "state is outside its closed domain",
                        int(i),
                        int(t),
                        float(states[i, t, d]),
                    )
                )
        for i, t in np.argwhere((actions < 0) | (actions >= self.spec.action_count)):
            rows.append(
                FailureRow(
                    "out_of_range",
                    "actions",
                    "action is outside [0, action_count)",
                    int(i),
                    int(t),
                    int(actions[i, t]),
                )
            )
        for i, t in np.argwhere(~np.isfinite(rewards)):
            rows.append(
                FailureRow("non_finite", "rewards", "reward must be finite", int(i), int(t))
            )
        if self.spec.reward_min is not None and self.spec.reward_max is not None:
            outside = (rewards < self.spec.reward_min) | (rewards > self.spec.reward_max)
            for i, t in np.argwhere(outside):
                rows.append(
                    FailureRow(
                        "out_of_range",
                        "rewards",
                        "reward is outside its closed domain",
                        int(i),
                        int(t),
                        float(rewards[i, t]),
                    )
                )
        for i, t in np.argwhere(~np.isfinite(behavior)):
            rows.append(
                FailureRow(
                    "non_finite",
                    "behavior_propensities",
                    "behavior propensity must be finite",
                    int(i),
                    int(t),
                )
            )
        for i, t in np.argwhere((behavior < 0) | (behavior > 1)):
            rows.append(
                FailureRow(
                    "out_of_range",
                    "behavior_propensities",
                    "behavior propensity must be in [0, 1]",
                    int(i),
                    int(t),
                    float(behavior[i, t]),
                )
            )
        return tuple(rows)

    def take(self, indices: npt.ArrayLike) -> TrajectoryBatch:
        selected = tuple(int(value) for value in np.asarray(indices, dtype=np.int64))
        if not selected:
            raise ValueError("indices must contain at least one trajectory")
        if any(value < 0 or value >= self.spec.trajectories for value in selected):
            raise IndexError("trajectory index is out of range")
        spec = TrajectorySpec(
            trajectories=len(selected),
            horizon=self.spec.horizon,
            state_dim=self.spec.state_dim,
            action_count=self.spec.action_count,
            state_min=self.spec.state_min,
            state_max=self.spec.state_max,
            reward_min=self.spec.reward_min,
            reward_max=self.spec.reward_max,
            schema_version=self.spec.schema_version,
        )
        return TrajectoryBatch(
            spec,
            tuple(self.states[index] for index in selected),
            tuple(self.actions[index] for index in selected),
            tuple(self.rewards[index] for index in selected),
            tuple(self.behavior_propensities[index] for index in selected),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "behavior_propensities": self.behavior_propensities,
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrajectoryBatch:
        return cls(
            spec=TrajectorySpec.from_dict(raw["spec"]),
            states=raw["states"],
            actions=raw["actions"],
            rewards=raw["rewards"],
            behavior_propensities=raw["behavior_propensities"],
        )

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryBatch:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("trajectory batch payload must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class PolynomialPolicySpec:
    """Public low-degree polynomial action-propensity model.

    Features are ordered ``1``, each state coordinate, then (for degree two)
    ``x_i*x_j`` for ``i <= j`` in lexicographic order.  Coefficients directly
    produce probabilities; no softmax or private training is implied.  For an
    unbiased design, freeze the policy independently of evaluation trajectories.
    """

    action_count: int
    state_dim: int
    degree: int
    coefficients: tuple[tuple[float, ...], ...]
    probability_tolerance: float = 1e-9
    schema_version: str = "unseen-loop/ope-policy-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _float_tuple(self.coefficients))
        dimensions = (self.action_count, self.state_dim, self.degree)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in dimensions
        ):
            raise ValueError("policy dimensions and degree must be integers")
        if self.action_count < 2 or self.state_dim < 1:
            raise ValueError("action_count must be at least two and state_dim must be positive")
        if self.degree not in {1, 2}:
            raise ValueError("policy degree must be one or two")
        expected = 1 + self.state_dim
        if self.degree == 2:
            expected += self.state_dim * (self.state_dim + 1) // 2
        if np.asarray(self.coefficients).shape != (self.action_count, expected):
            raise ValueError(f"coefficients must have shape {(self.action_count, expected)}")
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("policy coefficients must be finite")
        if not np.isfinite(self.probability_tolerance) or self.probability_tolerance < 0:
            raise ValueError("probability_tolerance must be finite and non-negative")

    @property
    def feature_count(self) -> int:
        return len(self.coefficients[0])

    def polynomial_features(self, states: npt.ArrayLike) -> FloatArray:
        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (self.state_dim,):
            raise ValueError(f"states must end in dimension {self.state_dim}")
        if not np.all(np.isfinite(values)):
            raise ValueError("states must be finite")
        features = [np.ones(values.shape[:-1], dtype=np.float64)]
        features.extend(values[..., index] for index in range(self.state_dim))
        if self.degree == 2:
            features.extend(
                values[..., left] * values[..., right]
                for left in range(self.state_dim)
                for right in range(left, self.state_dim)
            )
        return np.stack(features, axis=-1)

    def action_probabilities(self, states: npt.ArrayLike) -> FloatArray:
        probabilities = self.polynomial_features(states) @ np.asarray(self.coefficients).T
        tolerance = self.probability_tolerance
        if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
            raise ValueError("polynomial policy produced a probability outside [0, 1]")
        if not np.allclose(np.sum(probabilities, axis=-1), 1.0, atol=tolerance, rtol=0):
            raise ValueError("polynomial policy probabilities must sum to one")
        return np.asarray(np.clip(probabilities, 0.0, 1.0), dtype=np.float64)

    def logged_action_probabilities(self, batch: TrajectoryBatch) -> FloatArray:
        if batch.spec.state_dim != self.state_dim:
            raise ValueError("batch and policy state dimensions differ")
        if batch.spec.action_count != self.action_count:
            raise ValueError("batch and policy action counts differ")
        probabilities = self.action_probabilities(batch.state_array)
        return np.take_along_axis(probabilities, batch.action_array[..., None], axis=-1)[..., 0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolynomialPolicySpec:
        return cls(
            action_count=int(raw["action_count"]),
            state_dim=int(raw["state_dim"]),
            degree=int(raw["degree"]),
            coefficients=tuple(tuple(float(value) for value in row) for row in raw["coefficients"]),
            probability_tolerance=float(raw.get("probability_tolerance", 1e-9)),
            schema_version=str(raw.get("schema_version", "unseen-loop/ope-policy-v1")),
        )

    @classmethod
    def from_json(cls, payload: str) -> PolynomialPolicySpec:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("policy payload must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class SufficientStatistics:
    """Per-horizon additive statistics; division is client-side semantics."""

    estimator: str
    numerators: tuple[float, ...]
    denominators: tuple[float, ...]
    counts: tuple[int, ...]
    failures: tuple[FailureRow, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerators", tuple(float(x) for x in self.numerators))
        object.__setattr__(self, "denominators", tuple(float(x) for x in self.denominators))
        object.__setattr__(self, "counts", tuple(int(x) for x in self.counts))
        object.__setattr__(self, "failures", tuple(self.failures))
        if not self.estimator:
            raise ValueError("estimator cannot be empty")
        if not self.numerators or not (
            len(self.numerators) == len(self.denominators) == len(self.counts)
        ):
            raise ValueError("statistics vectors must have the same non-zero length")
        if not np.all(np.isfinite(self.numerators)) or not np.all(np.isfinite(self.denominators)):
            raise ValueError("statistics must be finite")
        if any(value < 0 for value in self.denominators) or any(value < 0 for value in self.counts):
            raise ValueError("denominators and counts must be non-negative")

    @property
    def estimate(self) -> float | None:
        if self.failures or any(value == 0 for value in self.denominators):
            return None
        return float(
            sum(
                numerator / denominator
                for numerator, denominator in zip(self.numerators, self.denominators, strict=True)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "numerators": self.numerators,
            "denominators": self.denominators,
            "counts": self.counts,
            "estimate": self.estimate,
            "failures": tuple(row.to_dict() for row in self.failures),
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SufficientStatistics:
        return cls(
            estimator=str(raw["estimator"]),
            numerators=tuple(float(value) for value in raw["numerators"]),
            denominators=tuple(float(value) for value in raw["denominators"]),
            counts=tuple(int(value) for value in raw["counts"]),
            failures=tuple(FailureRow.from_dict(row) for row in raw.get("failures", ())),
        )

    @classmethod
    def from_json(cls, payload: str) -> SufficientStatistics:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("statistics payload must be a JSON object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class BootstrapResult:
    """Deterministic percentile interval over trajectory resamples."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.estimate, self.lower, self.upper))):
            raise ValueError("bootstrap estimates must be finite")
        if self.lower > self.upper:
            raise ValueError("bootstrap interval must be ordered")
        if self.confidence <= 0 or self.confidence >= 1:
            raise ValueError("bootstrap confidence must be in (0, 1)")
        if self.samples < 1 or self.seed < 0:
            raise ValueError("bootstrap samples must be positive and seed non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BootstrapResult:
        return cls(
            estimate=float(raw["estimate"]),
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            confidence=float(raw["confidence"]),
            samples=int(raw["samples"]),
            seed=int(raw["seed"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> BootstrapResult:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("bootstrap payload must be a JSON object")
        return cls.from_dict(raw)
