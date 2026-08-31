"""Deterministic OPE validation jobs for the flagship evidence DAG."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from unseen_loop.ope.circuit import FixedPointScales, OPECircuitSpec
from unseen_loop.ope.estimators import (
    cumulative_importance_weights,
    per_decision_effective_sample_size,
)
from unseen_loop.ope.fhe import OPEConformanceResult, compile_ope_circuit
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

_SCHEMA = "unseen-loop/flagship-ope-evidence-v1"
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_KINDS = frozenset(
    {"analytic_fixture", "fixed_point_reference", "empirical", "fhe_valid", "fhe_invalid"}
)
_MDP_TRANSITIONS = np.asarray(
    (
        ((0.75, 0.20, 0.05), (0.10, 0.75, 0.15)),
        ((0.15, 0.75, 0.10), (0.05, 0.20, 0.75)),
        ((0.10, 0.15, 0.75), (0.75, 0.20, 0.05)),
    ),
    dtype=np.float64,
)
_MDP_REWARDS = np.asarray(((0.10, 0.70), (0.45, -0.20), (0.90, 0.25)), dtype=np.float64)
_INITIAL = np.asarray((0.50, 0.30, 0.20), dtype=np.float64)
_GAMMA = 0.97
_CONFIDENCE = 0.95


class _InvalidJob(ValueError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _InvalidJob(f"{field}_must_be_mapping")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _InvalidJob(f"{field}_must_be_integer")
    result = int(value)
    if result < minimum:
        raise _InvalidJob(f"{field}_out_of_range")
    return result


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _InvalidJob(f"{field}_must_be_number")
    result = float(value)
    if not math.isfinite(result):
        raise _InvalidJob(f"{field}_must_be_finite")
    return result


def _sequence(section: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    value = section.get(field)
    if not isinstance(value, (list, tuple)):
        raise _InvalidJob(f"manifest_ope_{field}_must_be_sequence")
    return tuple(value)


def _job_parts(
    manifest: Mapping[str, Any], job: Mapping[str, Any]
) -> tuple[str, int, str, Mapping[str, Any], Mapping[str, Any]]:
    ope = _mapping(manifest.get("ope"), "manifest_ope")
    stage = job.get("stage")
    if stage != "ope_validation":
        raise _InvalidJob("wrong_stage")
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise _InvalidJob("invalid_job_id")
    seed = _integer(job.get("seed"), "seed")
    coordinates = _mapping(job.get("coordinates"), "coordinates")
    kind = coordinates.get("kind")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise _InvalidJob("unknown_ope_job_kind")
    return job_id, seed, kind, coordinates, ope


def _require_keys(coordinates: Mapping[str, Any], required: set[str]) -> None:
    if set(coordinates) != required | {"kind"}:
        raise _InvalidJob("unexpected_coordinates")


def _validate_coordinates(
    kind: str, coordinates: Mapping[str, Any], ope: Mapping[str, Any]
) -> None:
    reference = _mapping(ope.get("reference"), "manifest_ope_reference")
    challenge = _mapping(ope.get("fhe_challenge"), "manifest_ope_fhe_challenge")
    if kind == "analytic_fixture":
        _require_keys(coordinates, {"case"})
        case = _integer(coordinates["case"], "case")
        if case >= _integer(reference.get("analytic_fixtures"), "analytic_fixtures", minimum=1):
            raise _InvalidJob("case_out_of_range")
        return
    if kind == "fixed_point_reference":
        _require_keys(coordinates, {"case"})
        case = _integer(coordinates["case"], "case")
        if case >= _integer(
            reference.get("random_fixed_point_cases"), "random_fixed_point_cases", minimum=1
        ):
            raise _InvalidJob("case_out_of_range")
        return
    if kind == "empirical":
        _require_keys(
            coordinates, {"horizon", "trajectories", "overlap", "clip", "estimator", "batch"}
        )
        horizon = _integer(coordinates["horizon"], "horizon", minimum=1)
        trajectories = _integer(coordinates["trajectories"], "trajectories", minimum=1)
        overlap = _number(coordinates["overlap"], "overlap")
        clip = coordinates["clip"]
        estimator = coordinates["estimator"]
        batch = _integer(coordinates["batch"], "batch")
        if horizon not in tuple(int(value) for value in _sequence(ope, "horizons")):
            raise _InvalidJob("horizon_not_declared")
        if trajectories not in tuple(int(value) for value in _sequence(ope, "trajectory_counts")):
            raise _InvalidJob("trajectory_count_not_declared")
        if overlap not in tuple(float(value) for value in _sequence(ope, "overlap_lambdas")):
            raise _InvalidJob("overlap_not_declared")
        declared_clips: tuple[object, ...] = _sequence(ope, "clip_values")
        if bool(ope.get("include_unclipped")):
            declared_clips += ("unclipped",)
        if clip not in declared_clips:
            raise _InvalidJob("clip_not_declared")
        if estimator not in _sequence(ope, "estimators"):
            raise _InvalidJob("estimator_not_declared")
        if batch >= _integer(ope.get("independent_batches"), "independent_batches", minimum=1):
            raise _InvalidJob("batch_out_of_range")
        return
    if kind == "fhe_valid":
        _require_keys(coordinates, {"category", "batch"})
        category = coordinates["category"]
        counts = {
            "occupancy": "occupancy_batches",
            "extrema": "extrema_batches",
            "terminal_padding": "terminal_padding_batches",
            "rounding_boundary": "rounding_boundary_batches",
        }
        if not isinstance(category, str) or category not in counts:
            raise _InvalidJob("unknown_fhe_category")
        batch = _integer(coordinates["batch"], "batch")
        if batch >= _integer(
            challenge.get(counts[str(category)]), str(counts[str(category)]), minimum=1
        ):
            raise _InvalidJob("batch_out_of_range")
        return
    if kind == "fhe_invalid":
        _require_keys(coordinates, {"batch"})
        batch = _integer(coordinates["batch"], "batch")
        if batch >= _integer(
            challenge.get("invalid_batch_rejections"),
            "invalid_batch_rejections",
            minimum=1,
        ):
            raise _InvalidJob("batch_out_of_range")
        return
    raise AssertionError(f"unhandled validated OPE job kind {kind!r}")


def _policy() -> PolynomialPolicySpec:
    return PolynomialPolicySpec(
        action_count=2,
        state_dim=6,
        degree=1,
        coefficients=(
            (0.75, -0.50, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.25, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )


def _state_vector(state: int) -> tuple[float, ...]:
    return (state / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _exact_truth(horizon: int) -> tuple[float, tuple[float, ...]]:
    target = _policy().action_probabilities(tuple(_state_vector(state) for state in range(3)))
    distribution = _INITIAL.copy()
    contributions: list[float] = []
    for step in range(horizon):
        contributions.append(
            float((_GAMMA**step) * np.sum(distribution[:, None] * target * _MDP_REWARDS))
        )
        distribution = np.einsum("s,sa,san->n", distribution, target, _MDP_TRANSITIONS)
    return float(sum(contributions)), tuple(contributions)


def _generate_batch(
    seed: int, horizon: int, trajectories: int, overlap: float
) -> tuple[TrajectoryBatch, npt.NDArray[np.float64]]:
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    policy = _policy()
    states: list[tuple[tuple[float, ...], ...]] = []
    actions: list[tuple[int, ...]] = []
    rewards: list[tuple[float, ...]] = []
    behaviors: list[tuple[float, ...]] = []
    targets = np.empty((trajectories, horizon), dtype=np.float64)
    for trajectory in range(trajectories):
        state = int(rng.choice(3, p=_INITIAL))
        state_row: list[tuple[float, ...]] = []
        action_row: list[int] = []
        reward_row: list[float] = []
        behavior_row: list[float] = []
        for step in range(horizon):
            vector = _state_vector(state)
            target = policy.action_probabilities((vector,))[0]
            behavior = overlap * target + (1.0 - overlap) * 0.5
            action = int(rng.choice(2, p=behavior))
            state_row.append(vector)
            action_row.append(action)
            reward_row.append(float(_MDP_REWARDS[state, action]))
            behavior_row.append(float(behavior[action]))
            targets[trajectory, step] = float(target[action])
            state = int(rng.choice(3, p=_MDP_TRANSITIONS[state, action]))
        states.append(tuple(state_row))
        actions.append(tuple(action_row))
        rewards.append(tuple(reward_row))
        behaviors.append(tuple(behavior_row))
    spec = TrajectorySpec(
        trajectories=trajectories,
        horizon=horizon,
        state_dim=6,
        action_count=2,
        state_min=(0.0,) * 6,
        state_max=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        reward_min=-1.0,
        reward_max=1.0,
    )
    return TrajectoryBatch(
        spec, tuple(states), tuple(actions), tuple(rewards), tuple(behaviors)
    ), targets


def _statistics(
    batch: TrajectoryBatch,
    target: npt.NDArray[np.float64],
    clip: float | None,
    *,
    gamma: float = _GAMMA,
) -> tuple[dict[str, object], dict[str, object], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    weights = cumulative_importance_weights(batch, target, weight_clip=clip)
    discounts = np.power(gamma, np.arange(batch.spec.horizon, dtype=np.float64))
    per_row = weights * batch.reward_array * discounts
    numerators = np.sum(per_row, axis=0)
    pdis_denominators = np.full(batch.spec.horizon, batch.spec.trajectories, dtype=np.float64)
    wpdis_denominators = np.sum(weights, axis=0)
    if np.any(wpdis_denominators <= 0):
        raise ValueError("generated supported batch produced a non-positive WPDIS denominator")
    pdis_value = float(np.sum(numerators / pdis_denominators))
    wpdis_value = float(np.sum(numerators / wpdis_denominators))
    common = {"numerators": tuple(float(value) for value in numerators)}
    pdis_row = {
        **common,
        "denominators": tuple(float(value) for value in pdis_denominators),
        "value": pdis_value,
    }
    wpdis_row = {
        **common,
        "denominators": tuple(float(value) for value in wpdis_denominators),
        "value": wpdis_value,
    }
    return pdis_row, wpdis_row, weights, per_row


def _with_errors(row: dict[str, object], truth: float) -> dict[str, object]:
    value = float(cast(float, row["value"]))
    bias = value - truth
    epsilon = float(np.finfo(np.float64).eps)
    return {
        **row,
        "bias": bias,
        "absolute_error": abs(bias),
        "squared_error": bias * bias,
        "normalized_bias": abs(bias) / max(abs(truth), epsilon),
    }


def _bootstrap_interval(
    *,
    per_row: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    estimator: str,
    repetitions: int,
    seed: int,
    truth: float,
) -> dict[str, object]:
    trajectories = per_row.shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(trajectories, np.full(trajectories, 1.0 / trajectories), repetitions)
    numerators = counts @ per_row
    if estimator.endswith("wpdis"):
        denominators = counts @ weights
        if np.any(denominators == 0):
            raise ValueError("bootstrap encountered a zero WPDIS denominator")
        replicates = np.sum(numerators / denominators, axis=1)
    else:
        replicates = np.sum(numerators, axis=1) / trajectories
    tail = (1.0 - _CONFIDENCE) / 2.0
    lower, upper = np.quantile(replicates, (tail, 1.0 - tail))
    return {
        "method": "deterministic_multinomial_whole_trajectory_percentile",
        "repetitions": repetitions,
        "confidence": _CONFIDENCE,
        "seed": seed,
        "estimator": estimator,
        "lower": float(lower),
        "upper": float(upper),
        "covered_truth": bool(lower <= truth <= upper),
        "replicate_sum": float(np.sum(replicates)),
        "replicate_sum_squares": float(np.sum(np.square(replicates))),
    }


def _public_mdp(horizon: int) -> dict[str, object]:
    return {
        "states": 3,
        "actions": 2,
        "horizon": horizon,
        "gamma": _GAMMA,
        "initial_distribution": tuple(float(value) for value in _INITIAL),
        "transition_probabilities": _MDP_TRANSITIONS,
        "rewards": _MDP_REWARDS,
    }


def _clear_payload(
    *, seed: int, horizon: int, trajectories: int, overlap: float, clip: float | None
) -> tuple[dict[str, object], TrajectoryBatch]:
    batch, target = _generate_batch(seed, horizon, trajectories, overlap)
    truth, truth_by_horizon = _exact_truth(horizon)
    pdis_row, wpdis_row, weights, per_row = _statistics(batch, target, clip)
    ess = per_decision_effective_sample_size(batch, target, weight_clip=clip)
    return (
        {
            "mdp": _public_mdp(horizon),
            "policy": {
                "target": _policy().to_dict(),
                "behavior": {
                    "construction": "overlap * target + (1-overlap) * uniform",
                    "overlap": overlap,
                    "minimum_logged_probability": float(np.min(batch.behavior_array)),
                    "support_valid": True,
                },
            },
            "truth": {"value": truth, "per_horizon_contributions": truth_by_horizon},
            "estimates": {
                "pdis": _with_errors(pdis_row, truth),
                "wpdis": _with_errors(wpdis_row, truth),
            },
            "diagnostics": {
                "per_horizon_ess": ess,
                "minimum_ess": min(ess),
                "minimum_ess_fraction": min(ess) / trajectories,
                "positive_horizon_denominators": all(
                    value > 0 for value in cast(tuple[float, ...], wpdis_row["denominators"])
                ),
                "trajectory_count": trajectories,
                "horizon_count": horizon,
                "logged_batch_digest": hashlib.sha256(batch.to_json().encode()).hexdigest(),
                "logged_batch_persisted": False,
            },
            "_bootstrap_inputs": {"weights": weights, "per_row": per_row},
        },
        batch,
    )


def _fixed_point_payload(batch: TrajectoryBatch, clip: float) -> dict[str, object]:
    spec = OPECircuitSpec(
        batch.spec,
        _policy(),
        gamma=_GAMMA,
        weight_clip=clip,
        minimum_behavior_propensity=float(np.min(batch.behavior_array)),
    )
    integers, receipt = spec.integer_reference(batch)
    clear_pdis = spec.clear_statistics(batch, "clipped_pdis")
    clear_wpdis = spec.clear_statistics(batch, "clipped_wpdis")
    decoded_pdis = spec.client_statistics(integers, "clipped_pdis")
    decoded_wpdis = spec.client_statistics(integers, "clipped_wpdis")
    assert clear_pdis.estimate is not None and clear_wpdis.estimate is not None
    assert decoded_pdis.estimate is not None and decoded_wpdis.estimate is not None
    return {
        "integer_statistics": integers,
        "decoded": {
            "pdis": decoded_pdis,
            "wpdis": decoded_wpdis,
        },
        "clear_reference": {"pdis": clear_pdis, "wpdis": clear_wpdis},
        "error": {
            "pdis": decoded_pdis.estimate - clear_pdis.estimate,
            "wpdis": decoded_wpdis.estimate - clear_wpdis.estimate,
            "absolute_pdis": abs(decoded_pdis.estimate - clear_pdis.estimate),
            "absolute_wpdis": abs(decoded_wpdis.estimate - clear_wpdis.estimate),
        },
        "receipt": receipt,
    }


def _canary(seed: int, category: str) -> tuple[OPECircuitSpec, TrajectoryBatch]:
    rng = np.random.default_rng(seed)
    trajectory_spec = TrajectorySpec(
        trajectories=4,
        horizon=4,
        state_dim=6,
        action_count=2,
        state_min=(0.0,) * 6,
        state_max=(0.0,) * 6,
        reward_min=-1.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=6,
        degree=1,
        coefficients=((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * 2,
    )
    action_rewards = {
        "occupancy": (0.0, 0.5),
        "extrema": (-1.0, 1.0),
        "terminal_padding": (0.0, 0.5),
        "rounding_boundary": (-0.5, 1.0),
    }[category]
    action_array = rng.integers(0, 2, size=(4, 4))
    reward_array = np.asarray(
        [[action_rewards[int(action)] for action in row] for row in action_array],
        dtype=np.float64,
    )
    if category == "terminal_padding":
        reward_array[:, 2:] = 0.0
    batch = TrajectoryBatch(
        trajectory_spec,
        states=tuple(tuple((0.0,) * 6 for _ in range(4)) for _ in range(4)),
        actions=tuple(tuple(int(action) for action in row) for row in action_array),
        rewards=tuple(tuple(float(value) for value in row) for row in reward_array),
        behavior_propensities=tuple((0.5,) * 4 for _ in range(4)),
    )
    spec = OPECircuitSpec(
        trajectory_spec,
        policy,
        gamma=1.0,
        weight_clip=2.0,
        minimum_behavior_propensity=0.5,
        scales=FixedPointScales(state=2, coefficient=2, reciprocal=2, reward=2, discount=2),
    )
    return spec, batch


def _run_concrete_canary(
    spec: OPECircuitSpec, batch: TrajectoryBatch, security_level: int
) -> tuple[OPEConformanceResult, object]:
    """Compile and execute the real colocated canary; tests replace this boundary."""
    with tempfile.TemporaryDirectory(prefix="unseen-loop-ope-concrete-") as directory:
        compiled = compile_ope_circuit(
            spec,
            directory,
            security_level=security_level,
            global_p_error=1e-6,
        )
        return compiled.execute(batch, "clipped_wpdis", "REAL"), compiled.receipt


def _fhe_payload(manifest: Mapping[str, Any], seed: int, category: str) -> dict[str, object]:
    cryptography = _mapping(manifest.get("cryptography"), "manifest_cryptography")
    security_level = _integer(
        cryptography.get("required_classical_security_bits"),
        "required_classical_security_bits",
        minimum=1,
    )
    spec, batch = _canary(seed, category)
    expected, integer_receipt = spec.integer_reference(batch)
    result, compilation_receipt = _run_concrete_canary(spec, batch, security_level)
    if result.mode != "REAL":
        raise RuntimeError("OPE Concrete canary did not execute in REAL mode")
    if result.integer_statistics != expected:
        raise RuntimeError("OPE Concrete canary disagreed with the exact integer reference")
    if result.call_evidence is None:
        raise RuntimeError("OPE Concrete canary omitted sanitized call evidence")
    target = spec.target_policy.logged_action_probabilities(batch)
    pdis_row, wpdis_row, weights, _ = _statistics(batch, target, 2.0, gamma=1.0)
    action_rewards = {
        "occupancy": (0.0, 0.5),
        "extrema": (-1.0, 1.0),
        "terminal_padding": (0.0, 0.5),
        "rounding_boundary": (-0.5, 1.0),
    }[category]
    truth_by_horizon = tuple(
        0.0 if category == "terminal_padding" and step >= 2 else sum(action_rewards) / 2.0
        for step in range(4)
    )
    truth = float(sum(truth_by_horizon))
    ess = tuple(
        float(value) for value in per_decision_effective_sample_size(batch, target, weight_clip=2.0)
    )
    clear_pdis = spec.clear_statistics(batch, "clipped_pdis")
    clear_wpdis = spec.clear_statistics(batch, "clipped_wpdis")
    decoded_pdis = spec.client_statistics(result.integer_statistics, "clipped_pdis")
    decoded_wpdis = spec.client_statistics(result.integer_statistics, "clipped_wpdis")
    return {
        "mode": "REAL",
        "backend": "Concrete-Python TFHE",
        "canary_shape": {"trajectories": 4, "horizon": 4, "state_dim": 6},
        "conforms_to_integer_reference": True,
        "integer_statistics": result.integer_statistics,
        "decoded_statistics": {"pdis": decoded_pdis, "wpdis": decoded_wpdis},
        "clear_statistics": {"pdis": clear_pdis, "wpdis": clear_wpdis},
        "integer_receipt": integer_receipt,
        "compilation_receipt": compilation_receipt,
        "call_evidence": result.call_evidence,
        "clear_evidence": {
            "mdp": {
                "states": 1,
                "actions": 2,
                "horizon": 4,
                "gamma": 1.0,
                "initial_distribution": (1.0,),
                "transition_probabilities": (((1.0,), (1.0,)),),
                "action_rewards": action_rewards,
                "terminal_padding_from_step": 2 if category == "terminal_padding" else None,
            },
            "policy": {
                "target": spec.target_policy.to_dict(),
                "behavior": {
                    "construction": "uniform over both actions",
                    "minimum_logged_probability": 0.5,
                    "support_valid": True,
                },
            },
            "truth": {"value": truth, "per_horizon_contributions": truth_by_horizon},
            "estimates": {
                "pdis": _with_errors(pdis_row, truth),
                "wpdis": _with_errors(wpdis_row, truth),
            },
            "diagnostics": {
                "per_horizon_ess": ess,
                "minimum_ess": min(ess),
                "minimum_ess_fraction": min(ess) / 4.0,
                "positive_horizon_denominators": bool(np.all(np.sum(weights, axis=0) > 0)),
                "trajectory_count": 4,
                "horizon_count": 4,
                "logged_batch_digest": hashlib.sha256(batch.to_json().encode()).hexdigest(),
                "logged_batch_persisted": False,
            },
        },
        "trust_scope": (
            "Client, Concrete server, key generation, encryption, evaluation, and decryption are "
            "colocated in one worker; this is real FHE conformance evidence, not a network "
            "confidentiality measurement."
        ),
    }


def _artifact(
    manifest: Mapping[str, Any],
    job_id: str,
    seed: int,
    kind: str,
    coordinates: Mapping[str, Any],
    ope: Mapping[str, Any],
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": _SCHEMA,
        "job_id": job_id,
        "stage": "ope_validation",
        "kind": kind,
        "seed": seed,
        "coordinates": dict(coordinates),
    }
    if kind == "empirical":
        horizon = int(coordinates["horizon"])
        trajectories = int(coordinates["trajectories"])
        overlap = float(coordinates["overlap"])
        clip = None if coordinates["clip"] == "unclipped" else float(coordinates["clip"])
        payload, _ = _clear_payload(
            seed=seed, horizon=horizon, trajectories=trajectories, overlap=overlap, clip=clip
        )
        bootstrap_inputs = payload.pop("_bootstrap_inputs")
        assert isinstance(bootstrap_inputs, Mapping)
        estimator = str(coordinates["estimator"])
        repetitions = _integer(ope.get("bootstrap_repetitions"), "bootstrap_repetitions", minimum=1)
        truth = float(_mapping(payload["truth"], "truth")["value"])
        estimate_key = "wpdis" if estimator.endswith("wpdis") else "pdis"
        estimates = _mapping(payload["estimates"], "estimates")
        estimate = _mapping(estimates[estimate_key], "estimate")
        ci = _bootstrap_interval(
            per_row=np.asarray(bootstrap_inputs["per_row"], dtype=np.float64),
            weights=np.asarray(bootstrap_inputs["weights"], dtype=np.float64),
            estimator=estimator,
            repetitions=repetitions,
            seed=seed ^ 0x5EED5EED,
            truth=truth,
        )
        ci["estimate"] = float(estimate["value"])
        return {
            **base,
            "evidence_class": "CLEAR STATISTICAL EVIDENCE",
            "trust_scope": (
                "Clear production OPE evaluation; no cryptographic confidentiality claim."
            ),
            **payload,
            "ci": ci,
            "fixed_point": None,
            "fhe": None,
        }
    if kind == "analytic_fixture":
        case = int(coordinates["case"])
        horizon = 1 + case % 4
        payload, _ = _clear_payload(
            seed=seed, horizon=horizon, trajectories=32, overlap=(case % 5) / 4.0, clip=None
        )
        payload.pop("_bootstrap_inputs")
        return {
            **base,
            "evidence_class": "CLEAR ANALYTIC REFERENCE",
            "trust_scope": "Clear finite-MDP dynamic-programming truth and clear estimators.",
            **payload,
            "ci": None,
            "fixed_point": None,
            "fhe": None,
        }
    if kind == "fixed_point_reference":
        case = int(coordinates["case"])
        payload, batch = _clear_payload(
            seed=seed, horizon=4, trajectories=4, overlap=0.25 + 0.25 * (case % 3), clip=2.0
        )
        payload.pop("_bootstrap_inputs")
        return {
            **base,
            "evidence_class": "CLEAR EXACT FIXED-POINT REFERENCE",
            "trust_scope": "Clear integer-circuit reference; not an FHE execution.",
            **payload,
            "ci": None,
            "fixed_point": _fixed_point_payload(batch, 2.0),
            "fhe": None,
        }
    if kind != "fhe_valid":
        raise AssertionError(f"unhandled validated OPE job kind {kind!r}")
    category = str(coordinates["category"])
    challenge = _mapping(ope.get("fhe_challenge"), "manifest_ope_fhe_challenge")
    fhe = _fhe_payload(manifest, seed, category)
    clear_evidence = fhe.pop("clear_evidence")
    if not isinstance(clear_evidence, Mapping):
        raise AssertionError("FHE canary clear evidence must be a mapping")
    return {
        **base,
        "evidence_class": "REAL COLOCATED FHE CANARY",
        "trust_scope": (
            "Real Concrete canary with client and server colocated in one worker; production "
            "empirical OPE remains clear statistical evidence."
        ),
        "mdp": clear_evidence["mdp"],
        "policy": clear_evidence["policy"],
        "truth": clear_evidence["truth"],
        "estimates": clear_evidence["estimates"],
        "diagnostics": clear_evidence["diagnostics"],
        "ci": None,
        "fixed_point": None,
        "configured_challenge_shape": {
            "trajectories": _integer(
                challenge.get("trajectories_per_batch"), "trajectories_per_batch", minimum=1
            ),
            "horizon": _integer(challenge.get("horizon"), "challenge_horizon", minimum=1),
        },
        "fhe": fhe,
    }


def _rejected(reason: str) -> dict[str, object]:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason,
    }


def execute_flagship_job(
    manifest: Mapping[str, Any], job: Mapping[str, Any], evidence_root: str | Path
) -> dict[str, object]:
    """Execute one OPE job and close one canonical aggregate-only evidence artifact."""
    try:
        canonical_manifest = _mapping(manifest, "manifest")
        canonical_job = _mapping(job, "job")
        job_id, seed, kind, coordinates, ope = _job_parts(canonical_manifest, canonical_job)
        _validate_coordinates(kind, coordinates, ope)
    except _InvalidJob as error:
        return _rejected(str(error))

    if kind == "fhe_invalid":
        return _rejected("declared_invalid_ope_batch")

    artifact = _artifact(canonical_manifest, job_id, seed, kind, coordinates, ope)
    payload = _canonical_json(_jsonable(artifact))
    root = Path(evidence_root)
    relative = Path("ope_validation") / f"{job_id}.json"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "status": "succeeded",
        "artifact_path": relative.as_posix(),
        "artifact_digest": hashlib.sha256(payload).hexdigest(),
        "reason_code": "completed",
    }
