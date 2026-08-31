"""Closed integration-stage executor for the flagship DAG.

Trajectory jobs release aggregate client evidence, never step-level states or logs.
Real-FHE OPE jobs deterministically bind those releases back to the frozen rollout
seeds, reconstruct one 64-trajectory client batch, and return only client-released
3H statistics, sanitized transport receipts, and an explicitly scoped effect channel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from unseen_loop.crypto.ckks import CKKSContextArtifacts, CKKSParameters, generate_contexts
from unseen_loop.flagship.integration import (
    FrozenRequestedPolicy,
    Outcome,
    ShieldVariant,
    TrajectoryKind,
    TrajectoryLog,
    build_ope_batch,
    run_trajectory,
)
from unseen_loop.flagship.manifest import PLAN_SCHEMA_VERSION, content_digest, derive_seed
from unseen_loop.ope.circuit import FixedPointScales, OPECircuitSpec
from unseen_loop.ope.ckks import (
    OPECKKSClient,
    OPECKKSServer,
    PolynomialApproxOPEReceipt,
    PolynomialApproxOPESpec,
    executable_ckks_parameters,
)
from unseen_loop.ope.fhe import compile_ope_circuit
from unseen_loop.ope.types import (
    FailureRow,
    PolynomialPolicySpec,
    SufficientStatistics,
    TrajectoryBatch,
    TrajectorySpec,
)
from unseen_loop.shield.study import SCENARIO_FACTORIES, make_scenario
from unseen_loop.shield.types import STATE_DIM, Action

SCHEMA_VERSION = "unseen-loop/flagship-integration-evidence-v1"
_STAGE = "integration"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SCENARIOS = tuple(SCENARIO_FACTORIES)
_TRAJECTORY_KINDS = {
    "behavior_trajectory": TrajectoryKind.BEHAVIOR,
    "direct_trajectory": TrajectoryKind.DIRECT,
}

Result = dict[str, str | None]


def _rejected(reason: str) -> Result:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _persist(root: Path, job_id: str, payload: Mapping[str, Any]) -> Result:
    directory = root / _STAGE
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{job_id}.json"
    encoded = _canonical_bytes(payload)
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    return {
        "status": "succeeded",
        "artifact_path": destination.relative_to(root).as_posix(),
        "artifact_digest": hashlib.sha256(encoded).hexdigest(),
        "reason_code": None,
    }


def _config_digest(root: Path) -> str:
    registry = root / "registry.jsonl"
    if not registry.is_file() or registry.is_symlink():
        raise ValueError("registry is unavailable")
    with registry.open("rb") as handle:
        line = handle.readline()
    raw = json.loads(line)
    provenance = _mapping(_mapping(raw, "registry").get("provenance"), "registry.provenance")
    digest = provenance.get("config_digest")
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise ValueError("registry config digest is invalid")
    return digest


def _planned_identity(
    manifest: Mapping[str, Any], config_digest: str, coordinates: Mapping[str, str | int | float]
) -> tuple[str, int]:
    canonical_coordinates = tuple(sorted(coordinates.items()))
    digest = content_digest(
        {
            "schema": PLAN_SCHEMA_VERSION,
            "manifest": config_digest,
            "stage": _STAGE,
            "coordinates": canonical_coordinates,
        }
    )
    job_id = f"job-{_STAGE}-{digest[:24]}"
    seed_root = manifest.get("seed_root")
    if not isinstance(seed_root, str) or not seed_root:
        raise ValueError("manifest.seed_root is invalid")
    return job_id, derive_seed(seed_root, job_id)


def _validate_common(
    manifest: object, job: object, root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]:
    manifest_map = _mapping(manifest, "manifest")
    job_map = _mapping(job, "job")
    if set(job_map) != {"job_id", "stage", "seed", "coordinates"}:
        raise ValueError("job envelope keys differ from the flagship contract")
    if job_map.get("stage") != _STAGE:
        raise ValueError("job stage is not integration")
    job_id = job_map.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"job-integration-[0-9a-f]{24}", job_id) is None:
        raise ValueError("job_id is invalid")
    coordinates = _mapping(job_map.get("coordinates"), "job.coordinates")
    allowed_coordinate_types = (str, int, float)
    if any(type(value) not in allowed_coordinate_types for value in coordinates.values()):
        raise ValueError("job coordinates contain an invalid value")
    config_digest = _config_digest(root)
    expected_id, expected_seed = _planned_identity(manifest_map, config_digest, coordinates)
    if job_id != expected_id or job_map.get("seed") != expected_seed:
        raise ValueError("job identity is not bound to the registry manifest")
    return manifest_map, job_map, coordinates, job_id


def _integration_spec(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = _mapping(manifest.get("integration"), "manifest.integration")
    if _integer(spec.get("action_count"), "integration.action_count", minimum=1) != len(Action):
        raise ValueError("integration action count differs from the frozen environment")
    if _integer(spec.get("scenarios"), "integration.scenarios", minimum=1) != len(_SCENARIOS):
        raise ValueError("integration scenario count differs from the frozen registry")
    modes = spec.get("shield_modes")
    if not isinstance(modes, Sequence) or isinstance(modes, (str, bytes)):
        raise ValueError("integration.shield_modes is invalid")
    if tuple(modes) != tuple(item.value for item in ShieldVariant):
        raise ValueError("integration shield modes differ from frozen semantics")
    mixture = spec.get("behavior_mixture_target_weight")
    minimum = spec.get("minimum_behavior_probability")
    if (
        isinstance(mixture, bool)
        or not isinstance(mixture, (int, float))
        or not math.isclose(float(mixture), 0.5)
    ):
        raise ValueError("integration behavior mixture is not frozen at one half")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isclose(float(minimum), 0.1)
    ):
        raise ValueError("integration minimum behavior probability is not frozen at 0.1")
    return spec


def _policy() -> FrozenRequestedPolicy:
    return FrozenRequestedPolicy.constant((1.0 / len(Action),) * len(Action))


def _polynomial_policy() -> PolynomialPolicySpec:
    row = (1.0 / len(Action),) + (0.0,) * STATE_DIM
    return PolynomialPolicySpec(len(Action), STATE_DIM, 1, (row,) * len(Action))


def _canary_polynomial_policy() -> PolynomialPolicySpec:
    selected = (1.0, 0.0)
    unselected = (0.0, 0.0)
    return PolynomialPolicySpec(
        len(Action),
        1,
        1,
        (selected,) + (unselected,) * (len(Action) - 1),
    )


def _trajectory_coordinates(
    coordinates: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[TrajectoryKind, int, ShieldVariant, int, int]:
    kind_raw = coordinates.get("kind")
    if kind_raw not in _TRAJECTORY_KINDS or set(coordinates) != {
        "kind",
        "scenario",
        "shield_mode",
        "trajectory",
    }:
        raise ValueError("trajectory coordinates are invalid")
    kind = _TRAJECTORY_KINDS[str(kind_raw)]
    scenario = _integer(coordinates.get("scenario"), "scenario")
    if scenario >= len(_SCENARIOS):
        raise ValueError("scenario is outside the frozen registry")
    shield_raw = coordinates.get("shield_mode")
    if not isinstance(shield_raw, str):
        raise ValueError("shield_mode is invalid")
    try:
        shield = ShieldVariant(shield_raw)
    except ValueError as exc:
        raise ValueError("shield_mode is invalid") from exc
    trajectory = _integer(coordinates.get("trajectory"), "trajectory")
    count_key = (
        "behavior_trajectories_per_cell"
        if kind is TrajectoryKind.BEHAVIOR
        else "direct_target_trajectories_per_cell"
    )
    count = _integer(spec.get(count_key), f"integration.{count_key}", minimum=1)
    if trajectory >= count:
        raise ValueError("trajectory is outside the configured cell")
    horizon = _integer(spec.get("horizon"), "integration.horizon", minimum=1)
    return kind, scenario, shield, trajectory, horizon


def _aggregate_log(log: TrajectoryLog) -> dict[str, Any]:
    requested = [0] * len(Action)
    executed = [0] * len(Action)
    requested_to_executed = [[0] * len(Action) for _ in Action]
    mu_by_requested: list[set[float]] = [set() for _ in Action]
    for step in log.steps:
        request = int(step.requested_action)
        execution = int(step.executed_action)
        requested[request] += 1
        executed[execution] += 1
        requested_to_executed[request][execution] += 1
        mu_by_requested[request].add(step.mu_propensity)
    return {
        "scenario_id": log.scenario_id,
        "shield_mode": log.shield.value,
        "kind": log.kind.value,
        "trajectory_index": log.trajectory_index,
        "seed": log.seed,
        "policy_digest": log.policy_digest,
        "horizon": len(log.steps),
        "total_return": log.total_return,
        "unsafe_steps": log.unsafe_steps,
        "requested_action_counts": requested,
        "executed_action_counts": executed,
        "requested_to_executed_counts": requested_to_executed,
        "mu_by_requested_action": [sorted(values) for values in mu_by_requested],
    }


def _trajectory_artifact(
    job_id: str,
    coordinates: Mapping[str, Any],
    log: TrajectoryLog,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "client_released_trajectory_aggregate",
        "job_id": job_id,
        "stage": _STAGE,
        "coordinates": dict(coordinates),
        "release_scope": (
            "CLIENT-RELEASED aggregate outcomes and requested-to-executed action counts; "
            "no state, per-step reward, safety margin, or executed-action propensity is released"
        ),
        "privacy_claim": "none for these client-released aggregates",
        "trajectory": _aggregate_log(log),
    }


def _run_trajectory_job(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    job_id: str,
    root: Path,
) -> Result:
    spec = _integration_spec(manifest)
    kind, scenario_index, shield, trajectory, horizon = _trajectory_coordinates(coordinates, spec)
    policy = _policy()
    log = run_trajectory(
        f"scenario-{scenario_index + 1:02d}",
        make_scenario(_SCENARIOS[scenario_index]),
        policy,
        shield,
        kind,
        trajectory_index=trajectory,
        seed=int(job["seed"]),
        horizon=horizon,
    )
    return _persist(root, job_id, _trajectory_artifact(job_id, coordinates, log))


def _expected_trajectory_job(
    manifest: Mapping[str, Any],
    config_digest: str,
    *,
    kind: str,
    scenario: int,
    shield: ShieldVariant,
    trajectory: int,
) -> tuple[str, int, dict[str, str | int | float]]:
    coordinates: dict[str, str | int | float] = {
        "kind": kind,
        "scenario": scenario,
        "shield_mode": shield.value,
        "trajectory": trajectory,
    }
    job_id, seed = _planned_identity(manifest, config_digest, coordinates)
    return job_id, seed, coordinates


def _load_trajectory_release(
    root: Path,
    manifest: Mapping[str, Any],
    config_digest: str,
    *,
    kind: str,
    scenario: int,
    shield: ShieldVariant,
    trajectory: int,
) -> tuple[Mapping[str, Any], int]:
    job_id, seed, coordinates = _expected_trajectory_job(
        manifest,
        config_digest,
        kind=kind,
        scenario=scenario,
        shield=shield,
        trajectory=trajectory,
    )
    path = root / _STAGE / f"{job_id}.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("required trajectory release is unavailable")
    raw = json.loads(path.read_bytes())
    payload = _mapping(raw, "trajectory release")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evidence_type") != "client_released_trajectory_aggregate"
        or payload.get("job_id") != job_id
        or payload.get("coordinates") != coordinates
    ):
        raise ValueError("trajectory release is not bound to its planned coordinates")
    trajectory_payload = _mapping(payload.get("trajectory"), "trajectory release aggregate")
    if trajectory_payload.get("seed") != seed:
        raise ValueError("trajectory release seed is invalid")
    return trajectory_payload, seed


def _reconstruct_and_verify(
    release: Mapping[str, Any],
    *,
    scenario_index: int,
    shield: ShieldVariant,
    kind: TrajectoryKind,
    trajectory: int,
    seed: int,
    horizon: int,
) -> TrajectoryLog:
    log = run_trajectory(
        f"scenario-{scenario_index + 1:02d}",
        make_scenario(_SCENARIOS[scenario_index]),
        _policy(),
        shield,
        kind,
        trajectory_index=trajectory,
        seed=seed,
        horizon=horizon,
    )
    if _aggregate_log(log) != dict(release):
        raise ValueError("reconstructed trajectory does not match its client release")
    return log


def _bounded_batch(batch: TrajectoryBatch) -> TrajectoryBatch:
    states = batch.state_array
    rewards = batch.reward_array
    spec = TrajectorySpec(
        trajectories=batch.spec.trajectories,
        horizon=batch.spec.horizon,
        state_dim=batch.spec.state_dim,
        action_count=batch.spec.action_count,
        state_min=tuple(float(value) for value in np.min(states, axis=(0, 1))),
        state_max=tuple(float(value) for value in np.max(states, axis=(0, 1))),
        reward_min=float(np.min(rewards)),
        reward_max=float(np.max(rewards)),
    )
    return TrajectoryBatch(
        spec,
        batch.states,
        batch.actions,
        batch.rewards,
        batch.behavior_propensities,
    )


def _ckks_spec(batch: TrajectoryBatch) -> PolynomialApproxOPESpec:
    minimum = float(np.min(batch.behavior_array))
    raw_bound = math.prod(1.0 / minimum for _ in range(batch.spec.horizon))
    return PolynomialApproxOPESpec(
        batch.spec,
        _polynomial_policy(),
        gamma=1.0,
        weight_clip=raw_bound,
        minimum_behavior_propensity=minimum,
    )


def _ckks_receipt(spec: PolynomialApproxOPESpec) -> PolynomialApproxOPEReceipt:
    parameters = executable_ckks_parameters(
        spec.trajectories.trajectories,
        spec.trajectories.horizon,
    )
    return spec.receipt(parameters)


@lru_cache(maxsize=2)
def _cached_ckks_contexts(parameters: CKKSParameters) -> CKKSContextArtifacts:
    artifacts = generate_contexts(parameters)
    if (
        not artifacts.receipt.security_enforced
        or artifacts.receipt.effective_security_level != "tc128"
        or artifacts.receipt.server_context_is_private
    ):
        raise RuntimeError("integration CKKS context failed tc128 public-server validation")
    return artifacts


def _run_ckks(
    spec: PolynomialApproxOPESpec,
    batch: TrajectoryBatch,
    receipt: PolynomialApproxOPEReceipt,
) -> tuple[SufficientStatistics, list[dict[str, Any]]]:
    receipt.require_executable()
    artifacts = _cached_ckks_contexts(receipt.parameters)
    client = OPECKKSClient.from_serialized(
        artifacts.client_context,
        parameters=receipt.parameters,
        spec=spec,
    )
    server = OPECKKSServer.from_serialized(
        artifacts.server_context,
        parameters=receipt.parameters,
        spec=spec,
    )
    request, encrypt_receipt = client.encrypt_batch(batch)
    response, server_receipt = server.evaluate(request)
    statistics, decrypt_receipt = client.decrypt_statistics(response, "clipped_wpdis")
    return statistics, [
        dataclasses.asdict(encrypt_receipt),
        dataclasses.asdict(server_receipt),
        dataclasses.asdict(decrypt_receipt),
    ]


def _concrete_canary(
    logs: Sequence[TrajectoryLog], outcome: Outcome
) -> tuple[OPECircuitSpec, TrajectoryBatch]:
    selected = tuple(logs[:1])
    if len(selected) != 1 or len(selected[0].steps) < 2:
        raise ValueError("Concrete fallback requires a 1x2 canary")
    rewards = (
        tuple(
            step.reward if outcome is Outcome.RETURN else float(step.unsafe_step)
            for step in selected[0].steps[:2]
        ),
    )
    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=2,
        state_dim=1,
        action_count=len(Action),
        state_min=(0.0,),
        state_max=(0.0,),
        reward_min=float(min(rewards[0])),
        reward_max=float(max(rewards[0])),
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states=(((0.0,), (0.0,)),),
        actions=((0, 0),),
        rewards=rewards,
        behavior_propensities=((1.0, 1.0),),
    )
    spec = OPECircuitSpec(
        trajectory_spec,
        _canary_polynomial_policy(),
        gamma=1.0,
        weight_clip=1.0,
        minimum_behavior_propensity=1.0,
        scales=FixedPointScales(
            state=2,
            coefficient=2,
            reciprocal=2,
            reward=2,
            discount=2,
        ),
    )
    return spec, batch


def _run_concrete(
    spec: OPECircuitSpec, batch: TrajectoryBatch
) -> tuple[SufficientStatistics, list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="unseen-loop-ope-canary-") as temporary:
        compiled = compile_ope_circuit(spec, temporary, global_p_error=1e-6)
        result = compiled.real_roundtrip(batch, "clipped_wpdis")
        if result.call_evidence is None:
            raise RuntimeError("Concrete REAL execution omitted sanitized call evidence")
        return result.client_statistics, [
            dataclasses.asdict(compiled.receipt),
            dataclasses.asdict(result.call_evidence),
        ]


def _failure_receipts(rows: Sequence[FailureRow]) -> list[dict[str, Any]]:
    return [
        {
            "code": row.code,
            "field": row.field,
            "trajectory": row.trajectory,
            "step": row.step,
        }
        for row in rows
    ]


def _bootstrap_effect(
    fhe_estimate: float | None,
    behavior: Sequence[float],
    direct: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    left = np.asarray(behavior, dtype=np.float64)
    right = np.asarray(direct, dtype=np.float64)
    if (
        fhe_estimate is None
        or not math.isfinite(fhe_estimate)
        or not len(left)
        or not len(right)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("effect channel requires a finite FHE estimate and finite samples")
    rng = np.random.default_rng(seed)
    repetitions = max(1, samples)
    left_indices = rng.integers(0, len(left), size=(repetitions, len(left)))
    right_indices = rng.integers(0, len(right), size=(repetitions, len(right)))
    differences = np.mean(left[left_indices], axis=1) - np.mean(right[right_indices], axis=1)
    # The resampling spread is client-released uncertainty evidence. Center it on
    # the decrypted FHE estimate so it can never replace that estimate.
    differences += fhe_estimate - float(np.mean(left))
    lower, upper = np.quantile(differences, (0.025, 0.975))
    direct_replicates = np.mean(right[right_indices], axis=1)
    truth_lower, truth_upper = np.quantile(direct_replicates, (0.025, 0.975))
    direct_mean = float(np.mean(right))
    return {
        "comparison": "fhe_ope_minus_direct_target",
        "online_truth": direct_mean,
        "discrepancy": fhe_estimate - direct_mean,
        "lower": float(lower),
        "upper": float(upper),
        "contains_zero": bool(lower <= 0 <= upper),
        "truth_lower": float(truth_lower),
        "truth_upper": float(truth_upper),
        "truth_excludes_zero": bool(truth_lower > 0 or truth_upper < 0),
        "sign_error": bool(
            (truth_lower > 0 and fhe_estimate < 0) or (truth_upper < 0 and fhe_estimate > 0)
        ),
        "bootstrap_samples": repetitions,
        "bootstrap_seed": seed,
        "interval_method": "client bootstrap spread centered on the decrypted FHE estimate",
    }


def _ope_coordinates(
    coordinates: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[int, ShieldVariant, Outcome, str, int, int, int]:
    expected = {"kind", "scenario", "shield_mode", "outcome", "batch"}
    if set(coordinates) != expected or coordinates.get("kind") != "real_fhe_ope":
        raise ValueError("real_fhe_ope coordinates are invalid")
    scenario = _integer(coordinates.get("scenario"), "scenario")
    if scenario >= len(_SCENARIOS):
        raise ValueError("scenario is outside the frozen registry")
    shield_raw = coordinates.get("shield_mode")
    if not isinstance(shield_raw, str):
        raise ValueError("shield_mode is invalid")
    try:
        shield = ShieldVariant(shield_raw)
    except ValueError as exc:
        raise ValueError("shield_mode is invalid") from exc
    outcome_label = coordinates.get("outcome")
    if outcome_label == "return":
        outcome = Outcome.RETURN
    elif outcome_label in {"unsafe_steps", "unsafe_step_cost"}:
        outcome = Outcome.UNSAFE_STEPS
    else:
        raise ValueError("outcome is invalid")
    batch_size = _integer(
        spec.get("ope_batch_trajectories"),
        "integration.ope_batch_trajectories",
        minimum=1,
    )
    if batch_size != 64:
        raise ValueError("integration REAL FHE batches must contain exactly 64 trajectories")
    behavior_count = _integer(
        spec.get("behavior_trajectories_per_cell"),
        "integration.behavior_trajectories_per_cell",
        minimum=1,
    )
    if behavior_count % batch_size:
        raise ValueError("behavior trajectory count is not divisible by 64")
    batch = _integer(coordinates.get("batch"), "batch")
    if batch >= behavior_count // batch_size:
        raise ValueError("batch is outside the configured cell")
    direct_count = _integer(
        spec.get("direct_target_trajectories_per_cell"),
        "integration.direct_target_trajectories_per_cell",
        minimum=1,
    )
    return scenario, shield, outcome, str(outcome_label), batch, batch_size, direct_count


def _run_ope_job(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    job_id: str,
    root: Path,
) -> Result:
    integration = _integration_spec(manifest)
    (
        scenario,
        shield,
        outcome,
        outcome_label,
        batch_index,
        batch_size,
        direct_count,
    ) = _ope_coordinates(coordinates, integration)
    horizon = _integer(integration.get("horizon"), "integration.horizon", minimum=1)
    config_digest = _config_digest(root)
    start = batch_index * batch_size
    behavior_logs: list[TrajectoryLog] = []
    behavior_outcomes: list[float] = []
    for trajectory in range(start, start + batch_size):
        release, seed = _load_trajectory_release(
            root,
            manifest,
            config_digest,
            kind="behavior_trajectory",
            scenario=scenario,
            shield=shield,
            trajectory=trajectory,
        )
        log = _reconstruct_and_verify(
            release,
            scenario_index=scenario,
            shield=shield,
            kind=TrajectoryKind.BEHAVIOR,
            trajectory=trajectory,
            seed=seed,
            horizon=horizon,
        )
        behavior_logs.append(log)
        behavior_outcomes.append(
            log.total_return if outcome is Outcome.RETURN else float(log.unsafe_steps)
        )
    if len(behavior_logs) != 64:
        raise ValueError("REAL FHE OPE did not load exactly one 64-trajectory batch")

    canary_behavior_outcomes = [
        sum(
            step.reward if outcome is Outcome.RETURN else float(step.unsafe_step)
            for step in behavior_logs[0].steps[:2]
        )
    ]
    direct_outcomes: list[float] = []
    direct_canary_outcomes: list[float] = []
    for trajectory in range(direct_count):
        release, seed = _load_trajectory_release(
            root,
            manifest,
            config_digest,
            kind="direct_trajectory",
            scenario=scenario,
            shield=shield,
            trajectory=trajectory,
        )
        log = _reconstruct_and_verify(
            release,
            scenario_index=scenario,
            shield=shield,
            kind=TrajectoryKind.DIRECT,
            trajectory=trajectory,
            seed=seed,
            horizon=horizon,
        )
        direct_outcomes.append(
            log.total_return if outcome is Outcome.RETURN else float(log.unsafe_steps)
        )
        direct_canary_outcomes.append(
            sum(
                step.reward if outcome is Outcome.RETURN else float(step.unsafe_step)
                for step in log.steps[:2]
            )
        )

    prepared = build_ope_batch(behavior_logs, _policy(), outcome)
    batch = _bounded_batch(prepared.trajectories)
    ckks_spec = _ckks_spec(batch)
    ckks_receipt = _ckks_receipt(ckks_spec)
    ckks_failure_label: str | None = None
    try:
        ckks_receipt.require_executable()
    except ValueError as exc:
        text = str(exc)
        ckks_failure_label = (
            "ckks.insufficient-multiplicative-depth"
            if "multiplication levels" in text
            else "ckks.insufficient-scale-modulus"
        )
        canary_spec, canary_batch = _concrete_canary(behavior_logs, outcome)
        statistics, transport = _run_concrete(canary_spec, canary_batch)
        backend = {
            "label": "CONCRETE_EXACT_SMALL_CANARY",
            "real_fhe": True,
            "statistics_scope": "1 trajectory x 2 horizons canary; not a clear substitute",
            "trust_scope": "colocated-client-server",
            "trust_scope_detail": (
                "Concrete client and server execute in one Modal worker; REAL FHE attests backend "
                "execution but does not claim input privacy from the colocated worker"
            ),
            "context_scope": "one compiled Concrete context for this bounded canary call",
            "ckks_failure_label": ckks_failure_label,
        }
        effect_behavior = canary_behavior_outcomes
        effect_direct = direct_canary_outcomes[:1]
    else:
        statistics, transport = _run_ckks(ckks_spec, batch, ckks_receipt)
        backend = {
            "label": "CKKS_POLYNOMIAL_APPROX_OPE",
            "real_fhe": True,
            "statistics_scope": "full 64-trajectory batch",
            "trust_scope": "colocated-client-server",
            "trust_scope_detail": (
                "TenSEAL client and server execute in one Modal worker; REAL FHE attests backend "
                "execution but does not claim input privacy from the colocated worker"
            ),
            "context_scope": "one parameter-set CKKS context cached per Modal worker",
            "ckks_failure_label": None,
        }
        effect_behavior = behavior_outcomes
        effect_direct = direct_outcomes

    effect = _bootstrap_effect(
        statistics.estimate,
        effect_behavior,
        effect_direct,
        seed=int(job["seed"]) & ((1 << 64) - 1),
        samples=_integer(
            _mapping(manifest.get("ope"), "manifest.ope").get("bootstrap_repetitions"),
            "ope.bootstrap_repetitions",
            minimum=1,
        ),
    )
    effect.update(
        {
            "outcome": outcome_label,
            "estimate": statistics.estimate,
            "fhe_estimate": statistics.estimate,
            "scope": backend["statistics_scope"],
        }
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "real_fhe_ope_receipt",
        "job_id": job_id,
        "stage": _STAGE,
        "coordinates": dict(coordinates),
        "batch": {
            "trajectory_count": 64,
            "horizon": horizon,
            "trajectory_start": start,
            "source": "deterministically reconstructed from 64 bound client releases",
        },
        "backend": backend,
        "statistics": {
            "shape": [3, len(statistics.numerators)],
            "estimator": statistics.estimator,
            "numerators": list(statistics.numerators),
            "denominators": list(statistics.denominators),
            "counts": list(statistics.counts),
            "error_receipts": _failure_receipts(statistics.failures),
            "release_scope": "client-decrypted 3H sufficient statistics; division is client-only",
        },
        "transport_receipts": transport,
        "effect_channel": effect,
        "privacy_claim": (
            "No evaluator evidence contains states, per-step logs, safety margins, secret keys, "
            "or ciphertext payloads"
        ),
    }
    return _persist(root, job_id, artifact)


def execute_flagship_job(
    manifest: Mapping[str, Any], job: Mapping[str, Any], evidence_root: str | Path
) -> Result:
    """Execute one integration job, failing closed to a reason-only rejection."""

    root = Path(evidence_root)
    try:
        manifest_map, job_map, coordinates, job_id = _validate_common(manifest, job, root)
        kind = coordinates.get("kind")
        if kind in _TRAJECTORY_KINDS:
            return _run_trajectory_job(manifest_map, job_map, coordinates, job_id, root)
        if kind == "real_fhe_ope":
            return _run_ope_job(manifest_map, job_map, coordinates, job_id, root)
        return _rejected("integration.unsupported-kind")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return _rejected("integration.invalid-job")
