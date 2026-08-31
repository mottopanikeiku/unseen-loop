"""Fail-closed aggregation of the closed flagship evidence ledger."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import PLAN_SCHEMA_VERSION, canonical_json, content_digest, derive_seed
from .registry import AppendOnlyRegistry, JobStatus, RegistryError

_SCHEMA_VERSION = "unseen-loop/flagship-analysis-v1"
_UPSTREAM_STAGES = (
    "clear_shield_matrix",
    "shield_fhe_challenge",
    "ope_validation",
    "integration",
    "timing",
)
_ALL_STAGES = (*_UPSTREAM_STAGES, "analysis", "evidence_finalizer")


class AnalysisError(ValueError):
    """Raised when the closed evidence set cannot support analysis."""


@dataclass(frozen=True, slots=True)
class _ExpectedJob:
    job_id: str
    stage: str
    seed: int
    coordinates: dict[str, str | int | float]
    expected_terminal: JobStatus


def _rejected(reason: str) -> dict[str, str | None]:
    return {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": reason,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AnalysisError(f"{name} is not a string-keyed mapping")
    return value


def _sequence(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise AnalysisError(f"{name} is not an array")
    return tuple(value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{name} is not a valid integer denominator")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{name} is not finite")
    return result


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError(f"{name} is not boolean")
    return value


def _coordinates(manifest: Mapping[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    shield = _mapping(manifest.get("shield"), "manifest.shield")
    for scenario in range(_integer(shield.get("scenarios"), "shield.scenarios")):
        for controller in _sequence(shield.get("controller_cells"), "shield.controller_cells"):
            if not isinstance(controller, str):
                raise AnalysisError("shield.controller_cells contains a non-string")
            for repetition in range(
                _integer(
                    shield.get("seeds_per_controller_cell"), "shield.seeds_per_controller_cell"
                )
            ):
                yield (
                    "clear_shield_matrix",
                    {
                        "scenario": scenario,
                        "controller": controller,
                        "repetition": repetition,
                    },
                )

    challenge = _mapping(shield.get("fhe_challenge"), "shield.fhe_challenge")
    for category, field in (
        ("occupancy", "occupancy_states"),
        ("extrema", "extrema_states"),
        ("threshold", "threshold_states"),
        ("tie", "tie_states"),
        ("canary", "canary_states"),
    ):
        encryptions = (
            _integer(challenge.get("canary_encryptions_per_state"), "canary_encryptions_per_state")
            if category == "canary"
            else 1
        )
        for state in range(_integer(challenge.get(field), f"shield.fhe_challenge.{field}")):
            for encryption in range(encryptions):
                yield (
                    "shield_fhe_challenge",
                    {
                        "kind": "valid",
                        "category": category,
                        "state": state,
                        "encryption": encryption,
                    },
                )
    for case in range(
        _integer(challenge.get("invalid_domain_rejections"), "invalid_domain_rejections")
    ):
        yield "shield_fhe_challenge", {"kind": "invalid", "case": case}

    ope = _mapping(manifest.get("ope"), "manifest.ope")
    reference = _mapping(ope.get("reference"), "ope.reference")
    for case in range(_integer(reference.get("analytic_fixtures"), "analytic_fixtures")):
        yield "ope_validation", {"kind": "analytic_fixture", "case": case}
    for case in range(
        _integer(reference.get("random_fixed_point_cases"), "random_fixed_point_cases")
    ):
        yield "ope_validation", {"kind": "fixed_point_reference", "case": case}
    clips = list(_sequence(ope.get("clip_values"), "ope.clip_values"))
    if _bool(ope.get("include_unclipped"), "ope.include_unclipped"):
        clips.append("unclipped")
    for horizon in _sequence(ope.get("horizons"), "ope.horizons"):
        for trajectories in _sequence(ope.get("trajectory_counts"), "ope.trajectory_counts"):
            for overlap in _sequence(ope.get("overlap_lambdas"), "ope.overlap_lambdas"):
                for clip in clips:
                    for estimator in _sequence(ope.get("estimators"), "ope.estimators"):
                        for batch in range(
                            _integer(ope.get("independent_batches"), "ope.independent_batches")
                        ):
                            yield (
                                "ope_validation",
                                {
                                    "kind": "empirical",
                                    "horizon": horizon,
                                    "trajectories": trajectories,
                                    "overlap": overlap,
                                    "clip": clip,
                                    "estimator": estimator,
                                    "batch": batch,
                                },
                            )
    ope_fhe = _mapping(ope.get("fhe_challenge"), "ope.fhe_challenge")
    for category, field in (
        ("occupancy", "occupancy_batches"),
        ("extrema", "extrema_batches"),
        ("terminal_padding", "terminal_padding_batches"),
        ("rounding_boundary", "rounding_boundary_batches"),
    ):
        for batch in range(_integer(ope_fhe.get(field), f"ope.fhe_challenge.{field}")):
            yield "ope_validation", {"kind": "fhe_valid", "category": category, "batch": batch}
    for batch in range(
        _integer(ope_fhe.get("invalid_batch_rejections"), "invalid_batch_rejections")
    ):
        yield "ope_validation", {"kind": "fhe_invalid", "batch": batch}

    integration = _mapping(manifest.get("integration"), "manifest.integration")
    behavior_count = _integer(
        integration.get("behavior_trajectories_per_cell"), "behavior_trajectories_per_cell"
    )
    batch_size = _integer(
        integration.get("ope_batch_trajectories"), "ope_batch_trajectories", minimum=1
    )
    if behavior_count % batch_size:
        raise AnalysisError("integration behavior denominator is not divisible by its batch size")
    for scenario in range(_integer(integration.get("scenarios"), "integration.scenarios")):
        for mode in _sequence(integration.get("shield_modes"), "integration.shield_modes"):
            if not isinstance(mode, str):
                raise AnalysisError("integration.shield_modes contains a non-string")
            for trajectory in range(behavior_count):
                yield (
                    "integration",
                    {
                        "kind": "behavior_trajectory",
                        "scenario": scenario,
                        "shield_mode": mode,
                        "trajectory": trajectory,
                    },
                )
            for trajectory in range(
                _integer(
                    integration.get("direct_target_trajectories_per_cell"),
                    "direct_target_trajectories_per_cell",
                )
            ):
                yield (
                    "integration",
                    {
                        "kind": "direct_trajectory",
                        "scenario": scenario,
                        "shield_mode": mode,
                        "trajectory": trajectory,
                    },
                )
            for outcome in _sequence(integration.get("outcomes"), "integration.outcomes"):
                if not isinstance(outcome, str):
                    raise AnalysisError("integration.outcomes contains a non-string")
                for batch in range(behavior_count // batch_size):
                    yield (
                        "integration",
                        {
                            "kind": "real_fhe_ope",
                            "scenario": scenario,
                            "shield_mode": mode,
                            "outcome": outcome,
                            "batch": batch,
                        },
                    )

    systems = _mapping(manifest.get("systems"), "manifest.systems")
    for container in range(
        _integer(systems.get("shield_timing_containers"), "shield_timing_containers")
    ):
        yield "timing", {"kind": "shield", "container": container}
    for container in range(_integer(systems.get("ope_timing_containers"), "ope_timing_containers")):
        yield "timing", {"kind": "ope", "container": container}
    for trajectories in _sequence(
        systems.get("scale_trajectory_counts"), "scale_trajectory_counts"
    ):
        for horizon in _sequence(systems.get("scale_horizons"), "scale_horizons"):
            for container in range(
                _integer(systems.get("scale_containers_per_cell"), "scale_containers_per_cell")
            ):
                yield (
                    "timing",
                    {
                        "kind": "scale",
                        "trajectories": trajectories,
                        "horizon": horizon,
                        "container": container,
                    },
                )
    for client in range(_integer(systems.get("concurrent_clients"), "concurrent_clients")):
        yield "timing", {"kind": "concurrent_client", "client": client}
    yield "analysis", {"kind": "singleton"}
    yield "evidence_finalizer", {"kind": "singleton"}


def _expected_jobs(
    manifest: Mapping[str, Any], manifest_digest: str | None = None
) -> tuple[_ExpectedJob, ...]:
    if manifest_digest is None:
        manifest_digest = content_digest(manifest)
    seed_root = manifest.get("seed_root")
    if not isinstance(seed_root, str) or not seed_root:
        raise AnalysisError("manifest.seed_root is missing")
    expected: list[_ExpectedJob] = []
    for stage, raw_coordinates in _coordinates(manifest):
        coordinates = dict(sorted(raw_coordinates.items()))
        digest = content_digest(
            {
                "schema": PLAN_SCHEMA_VERSION,
                "manifest": manifest_digest,
                "stage": stage,
                "coordinates": tuple(coordinates.items()),
            }
        )
        job_id = f"job-{stage}-{digest[:24]}"
        terminal = (
            JobStatus.REJECTED
            if coordinates.get("kind") in {"invalid", "fhe_invalid"}
            else JobStatus.SUCCEEDED
        )
        expected.append(
            _ExpectedJob(job_id, stage, derive_seed(seed_root, job_id), coordinates, terminal)
        )
    return tuple(expected)


def _parse_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError("a registered artifact is missing or not a regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AnalysisError("a registered artifact is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise AnalysisError("a registered artifact is not a JSON object")
    encoded = canonical_json(payload)
    if raw not in {encoded, encoded + b"\n"}:
        raise AnalysisError("a registered artifact is not canonical JSON")
    return payload, raw


def _verify_evidence(
    manifest: Mapping[str, Any], job: Mapping[str, Any], root: Path
) -> tuple[dict[str, list[tuple[_ExpectedJob, dict[str, Any], str]]], dict[str, Any]]:
    try:
        registry = AppendOnlyRegistry(root / "registry.jsonl")
        snapshot = registry.snapshot()
    except RegistryError as exc:
        raise AnalysisError("registry is absent or unverifiable") from exc
    config_digest = snapshot.provenance.config_digest
    expected = _expected_jobs(manifest, config_digest)
    expected_by_id = {item.job_id: item for item in expected}
    if len(expected_by_id) != len(expected):
        raise AnalysisError("derived plan contains duplicate jobs")
    analysis_jobs = [item for item in expected if item.stage == "analysis"]
    if len(analysis_jobs) != 1:
        raise AnalysisError("analysis denominator is not singleton")
    singleton = analysis_jobs[0]
    if set(job) != {"job_id", "stage", "seed", "coordinates"}:
        raise AnalysisError("analysis job envelope has unknown or missing fields")
    if (
        job.get("job_id") != singleton.job_id
        or job.get("stage") != "analysis"
        or job.get("seed") != singleton.seed
        or job.get("coordinates") != singleton.coordinates
    ):
        raise AnalysisError("analysis job does not match the manifest-derived singleton")
    manifest_digest = content_digest(manifest)
    expected_plan = sorted((item.job_id, item.stage, item.expected_terminal) for item in expected)
    observed_plan = sorted(
        (item.job_id, item.stage, item.expected_terminal) for item in snapshot.plan
    )
    if observed_plan != expected_plan:
        raise AnalysisError("registry plan has a missing or extra manifest-derived job")
    records = {record.job_id: record for record in snapshot.records}
    if set(records) != set(expected_by_id):
        raise AnalysisError("registry records have a missing or extra job")

    artifacts: dict[str, list[tuple[_ExpectedJob, dict[str, Any], str]]] = defaultdict(list)
    allowed_files = {"registry.jsonl"}
    for job_id, expected_job in expected_by_id.items():
        record = records[job_id]
        if expected_job.stage in _UPSTREAM_STAGES:
            if record.status != expected_job.expected_terminal:
                raise AnalysisError(
                    "an upstream job is incomplete or has the wrong terminal status"
                )
        elif expected_job.stage == "analysis":
            if record.status != JobStatus.STARTED:
                raise AnalysisError("analysis registry state is not started")
            continue
        else:
            if record.status is not None:
                raise AnalysisError("evidence finalizer ran before analysis closed")
            continue
        if expected_job.expected_terminal == JobStatus.REJECTED:
            if record.artifact_path is not None or record.artifact_digest is not None:
                raise AnalysisError("a rejected job claims an artifact")
            if not record.reason_code:
                raise AnalysisError("a rejected job lacks a reason code")
            continue
        expected_path = f"{expected_job.stage}/{job_id}.json"
        if record.artifact_path != expected_path or not isinstance(record.artifact_digest, str):
            raise AnalysisError("a succeeded job does not reference its canonical artifact path")
        path = root / expected_path
        payload, raw = _parse_artifact(path)
        observed_digest = hashlib.sha256(raw).hexdigest()
        if observed_digest != record.artifact_digest:
            raise AnalysisError("a registered artifact digest does not match its bytes")
        if payload.get("job_id") != job_id or payload.get("stage") != expected_job.stage:
            raise AnalysisError("an artifact is not bound to its registry job and stage")
        if "coordinates" in payload and payload["coordinates"] != expected_job.coordinates:
            raise AnalysisError("an artifact coordinates field does not match its planned job")
        if "seed" in payload and payload["seed"] != expected_job.seed:
            raise AnalysisError("an artifact seed field does not match its planned job")
        if "manifest_digest" in payload and payload["manifest_digest"] != manifest_digest:
            raise AnalysisError("an artifact manifest digest does not match the run")
        artifacts[expected_job.stage].append((expected_job, payload, observed_digest))
        allowed_files.add(expected_path)

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AnalysisError("evidence tree contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    transient_required = {
        "shared/shield-fhe/shield-server.zip",
        "shared/shield-fhe/shield-client-specs.bin",
        "shared/shield-fhe/shield-receipt.json",
    }
    transient_optional = {"shared/shield-fhe.lock"}
    extras = actual_files - allowed_files
    transient_digests: dict[str, str] = {}
    if extras:
        if not transient_required.issubset(extras) or not extras.issubset(
            transient_required | transient_optional
        ):
            raise AnalysisError("evidence tree contains missing or extra files")
        for relative in sorted(extras):
            transient = root / relative
            transient_digests[relative] = hashlib.sha256(transient.read_bytes()).hexdigest()
            transient.unlink()
        (root / "shared/shield-fhe").rmdir()
        (root / "shared").rmdir()
        actual_files -= extras
    if actual_files != allowed_files:
        raise AnalysisError("evidence tree contains missing or extra files")

    stage_roots = {
        stage: content_digest(
            sorted((item.job_id, digest) for item, _payload, digest in artifacts.get(stage, ()))
        )
        for stage in _UPSTREAM_STAGES
    }
    closure = {
        "registry_id": snapshot.registry_id,
        "registry_tail_hash": snapshot.tail_hash,
        "registry_event_count": snapshot.event_count,
        "manifest_digest": config_digest,
        "manifest_payload_digest": manifest_digest,
        "planned_jobs": len(expected),
        "upstream_artifacts": sum(len(values) for values in artifacts.values()),
        "stage_artifact_counts": {
            stage: len(artifacts.get(stage, ())) for stage in _UPSTREAM_STAGES
        },
        "stage_artifact_set_sha256": stage_roots,
        "consumed_transient_compile_cache_sha256": transient_digests,
    }
    return artifacts, closure


def _ratio(numerator: int, denominator: int, name: str) -> float:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise AnalysisError(f"{name} has an invalid numerator/denominator")
    return numerator / denominator


def _wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float:
    rate = _ratio(successes, total, "Wilson interval")
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center + radius) / denominator


def _gate(
    name: str, observed: float | int | None, comparison: str, threshold: float | int, passed: bool
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _shield_summary(
    manifest: Mapping[str, Any], rows: list[tuple[_ExpectedJob, dict[str, Any], str]]
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    integer_fields = (
        "episode_denominator",
        "step_denominator",
        "unsafe_steps",
        "unsafe_episode",
        "benign_interventions",
        "benign_intervention_denominator",
        "fallbacks",
        "fallback_denominator",
        "requested_preserved",
        "requested_certified_denominator",
        "selected_certified",
        "selected_denominator",
        "certified_candidates",
        "candidate_evaluations",
    )
    for expected, payload, _digest in rows:
        episode = _mapping(payload.get("episode"), "clear-shield episode")
        controller = expected.coordinates["controller"]
        if episode.get("controller") != controller:
            raise AnalysisError("clear-shield controller binding is invalid")
        category_denominators = _mapping(
            payload.get("category_denominators"), "clear-shield category denominators"
        )
        category = payload.get("scenario_category")
        if (
            not isinstance(category, str)
            or category_denominators != {category: 1}
            or payload.get("step_denominator") != episode.get("step_denominator")
        ):
            raise AnalysisError("clear-shield released denominators are inconsistent")
        steps = _sequence(payload.get("steps"), "clear-shield public steps")
        if len(steps) != episode.get("step_denominator"):
            raise AnalysisError("clear-shield step evidence does not close its denominator")
        values = {
            field: _integer(episode.get(field), f"episode.{field}") for field in integer_fields
        }
        if values["episode_denominator"] != 1:
            raise AnalysisError("clear-shield episode denominator is not one")
        for numerator, denominator in (
            ("unsafe_steps", "step_denominator"),
            ("unsafe_episode", "episode_denominator"),
            ("benign_interventions", "benign_intervention_denominator"),
            ("fallbacks", "fallback_denominator"),
            ("requested_preserved", "requested_certified_denominator"),
            ("selected_certified", "selected_denominator"),
            ("certified_candidates", "candidate_evaluations"),
        ):
            if values[numerator] > values[denominator]:
                raise AnalysisError("clear-shield numerator exceeds its denominator")
        for field, value in values.items():
            totals[str(controller)][field] += value
        totals[str(controller)]["total_return"] += _number(
            episode.get("total_return"), "episode.total_return"
        )
        totals[str(controller)]["goals"] += int(
            _bool(episode.get("goal_reached"), "episode.goal_reached")
        )
    shield = _mapping(manifest.get("shield"), "manifest.shield")
    gates = _mapping(shield.get("gates"), "shield.gates")
    for required in ("disabled", "h1", "h2"):
        if required not in totals:
            raise AnalysisError(f"clear-shield evidence lacks required controller {required}")
    unsafe = {
        name: _ratio(int(value["unsafe_steps"]), int(value["step_denominator"]), f"{name} unsafe")
        for name, value in totals.items()
    }
    disabled_rate = unsafe["disabled"]
    h1_rate = unsafe["h1"]
    h2_rate = unsafe["h2"]
    reduction_disabled = (
        1.0
        if disabled_rate == 0.0 and h2_rate == 0.0
        else (None if disabled_rate == 0.0 else 1.0 - h2_rate / disabled_rate)
    )
    reduction_h1 = (
        1.0
        if h1_rate == 0.0 and h2_rate == 0.0
        else (None if h1_rate == 0.0 else 1.0 - h2_rate / h1_rate)
    )
    h2 = totals["h2"]
    benign = _ratio(
        int(h2["benign_interventions"]),
        int(h2["benign_intervention_denominator"]),
        "h2 benign interventions",
    )
    fallback = _ratio(int(h2["fallbacks"]), int(h2["fallback_denominator"]), "h2 fallback")
    preservation = _ratio(
        int(h2["requested_preserved"]),
        int(h2["requested_certified_denominator"]),
        "h2 requested preservation",
    )
    candidate = _ratio(
        int(h2["certified_candidates"]), int(h2["candidate_evaluations"]), "h2 certified candidates"
    )
    return_disabled = totals["disabled"]["total_return"] / totals["disabled"]["episode_denominator"]
    return_h2 = h2["total_return"] / h2["episode_denominator"]
    normalized_return_delta = (return_h2 - return_disabled) / max(abs(return_disabled), 1.0)
    goal_disabled = totals["disabled"]["goals"] / totals["disabled"]["episode_denominator"]
    goal_h2 = h2["goals"] / h2["episode_denominator"]
    goal_delta = goal_h2 - goal_disabled
    nominal = unsafe.get("model_nominal")
    mismatch_values = [
        unsafe[name] for name in ("model_minus_10pct", "model_plus_10pct") if name in unsafe
    ]
    mismatch_multiplier: float | None
    if nominal is None or len(mismatch_values) != 2:
        mismatch_multiplier = None
    elif nominal == 0.0:
        mismatch_multiplier = 1.0 if max(mismatch_values) == 0.0 else None
    else:
        mismatch_multiplier = max(mismatch_values) / nominal
    observed_gates = [
        _gate(
            "minimum_unsafe_reduction_vs_disabled",
            reduction_disabled,
            ">=",
            _number(gates.get("minimum_unsafe_reduction_vs_disabled"), "gate"),
            reduction_disabled is not None
            and reduction_disabled >= float(gates["minimum_unsafe_reduction_vs_disabled"]),
        ),
        _gate(
            "minimum_lookahead_reduction_vs_h1",
            reduction_h1,
            ">=",
            _number(gates.get("minimum_lookahead_reduction_vs_h1"), "gate"),
            reduction_h1 is not None
            and reduction_h1 >= float(gates["minimum_lookahead_reduction_vs_h1"]),
        ),
        _gate(
            "maximum_benign_intervention_rate",
            benign,
            "<=",
            _number(gates.get("maximum_benign_intervention_rate"), "gate"),
            benign <= float(gates["maximum_benign_intervention_rate"]),
        ),
        _gate(
            "maximum_benign_intervention_wilson_upper",
            _wilson_upper(
                int(h2["benign_interventions"]), int(h2["benign_intervention_denominator"])
            ),
            "<=",
            _number(gates.get("maximum_benign_intervention_wilson_upper"), "gate"),
            _wilson_upper(
                int(h2["benign_interventions"]), int(h2["benign_intervention_denominator"])
            )
            <= float(gates["maximum_benign_intervention_wilson_upper"]),
        ),
        _gate(
            "maximum_fallback_rate",
            fallback,
            "<=",
            _number(gates.get("maximum_fallback_rate"), "gate"),
            fallback <= float(gates["maximum_fallback_rate"]),
        ),
        _gate(
            "maximum_fallback_wilson_upper",
            _wilson_upper(int(h2["fallbacks"]), int(h2["fallback_denominator"])),
            "<=",
            _number(gates.get("maximum_fallback_wilson_upper"), "gate"),
            _wilson_upper(int(h2["fallbacks"]), int(h2["fallback_denominator"]))
            <= float(gates["maximum_fallback_wilson_upper"]),
        ),
        _gate(
            "minimum_certified_requested_action_preservation",
            preservation,
            ">=",
            _number(gates.get("minimum_certified_requested_action_preservation"), "gate"),
            preservation >= float(gates["minimum_certified_requested_action_preservation"]),
        ),
        _gate(
            "minimum_certified_candidate_rate",
            candidate,
            ">=",
            _number(gates.get("minimum_certified_candidate_rate"), "gate"),
            candidate >= float(gates["minimum_certified_candidate_rate"]),
        ),
        _gate(
            "normalized_return_delta_lower_bound",
            normalized_return_delta,
            ">=",
            _number(gates.get("normalized_return_delta_lower_bound"), "gate"),
            normalized_return_delta >= float(gates["normalized_return_delta_lower_bound"]),
        ),
        _gate(
            "goal_rate_delta_lower_bound",
            goal_delta,
            ">=",
            _number(gates.get("goal_rate_delta_lower_bound"), "gate"),
            goal_delta >= float(gates["goal_rate_delta_lower_bound"]),
        ),
        _gate(
            "model_mismatch_max_unsafe_multiplier",
            mismatch_multiplier,
            "<=",
            _number(gates.get("model_mismatch_max_unsafe_multiplier"), "gate"),
            mismatch_multiplier is not None
            and mismatch_multiplier <= float(gates["model_mismatch_max_unsafe_multiplier"]),
        ),
    ]
    return {
        "artifact_denominator": len(rows),
        "controller_episode_denominators": {
            name: int(values["episode_denominator"]) for name, values in sorted(totals.items())
        },
        "controller_unsafe_step_rates": dict(sorted(unsafe.items())),
        "gates": observed_gates,
        "all_gates_passed": all(gate["passed"] for gate in observed_gates),
    }


def _shield_fhe_summary(
    manifest: Mapping[str, Any], rows: list[tuple[_ExpectedJob, dict[str, Any], str]]
) -> dict[str, Any]:
    challenge = _mapping(
        _mapping(manifest.get("shield"), "shield").get("fhe_challenge"), "shield.fhe_challenge"
    )
    primary = [(job, payload) for job, payload, _ in rows if job.coordinates.get("encryption") == 0]
    totals: defaultdict[str, int] = defaultdict(int)
    canary_hashes: dict[int, set[str]] = defaultdict(set)
    scope_ok = True
    for job, payload, _digest in rows:
        execution = _mapping(payload.get("execution"), "shield FHE execution")
        trust_scope = execution.get("trust_scope")
        scope_ok &= (
            execution.get("mode") == "REAL FHE"
            and isinstance(trust_scope, str)
            and "colocated" in trust_scope.lower()
            and execution.get("privacy_evidence") is False
            and execution.get("server_selected_action") is False
            and execution.get("client_selected_action") is True
        )
        accounting = _mapping(payload.get("accounting"), "shield FHE accounting")
        if job.coordinates.get("encryption") == 0:
            for field in (
                "valid_calls",
                "decoded_margins",
                "margin_matches",
                "margin_mismatches",
                "action_matches",
                "action_mismatches",
            ):
                totals[field] += _integer(accounting.get(field), f"accounting.{field}")
        if job.coordinates.get("category") == "canary":
            canary = _mapping(payload.get("canary"), "shield FHE canary")
            digest = canary.get("ciphertext_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise AnalysisError("shield FHE canary lacks a ciphertext digest")
            canary_hashes[int(job.coordinates["state"])].add(digest)
    expected_calls = _integer(challenge.get("valid_calls"), "valid_calls")
    expected_margins = _integer(
        challenge.get("expected_decoded_margins"), "expected_decoded_margins"
    )
    expected_actions = _integer(challenge.get("expected_action_matches"), "expected_action_matches")
    conformant = (
        len(primary) == expected_calls
        and totals["valid_calls"] == expected_calls
        and totals["decoded_margins"] == expected_margins
        and totals["margin_matches"] == expected_margins
        and totals["margin_mismatches"] == 0
        and totals["action_matches"] == expected_actions
        and totals["action_mismatches"] == 0
    )
    canary_repeats = _integer(challenge.get("canary_encryptions_per_state"), "canary repeats")
    canaries_distinct = all(len(values) == canary_repeats for values in canary_hashes.values())
    return {
        "artifact_denominator": len(rows),
        "primary_call_denominator": len(primary),
        "accounting": dict(sorted(totals.items())),
        "exact_clear_conformance": conformant,
        "canary_ciphertexts_distinct": canaries_distinct,
        "trust_scope_verified": scope_ok,
        "trust_scope": "colocated-client-server; no colocated input-privacy claim",
        "client_only_action_selection_verified": scope_ok,
    }


def _ope_summary(
    manifest: Mapping[str, Any], rows: list[tuple[_ExpectedJob, dict[str, Any], str]]
) -> dict[str, Any]:
    empirical: list[dict[str, Any]] = []
    fhe_conformance: list[bool] = []
    reference_count = 0
    ope = _mapping(manifest.get("ope"), "manifest.ope")
    gates = _mapping(ope.get("gates"), "ope.gates")
    minimum_ess_fraction = _number(ope.get("minimum_ess_fraction"), "minimum_ess_fraction")
    for job, payload, _digest in rows:
        kind = job.coordinates.get("kind")
        if kind == "empirical":
            truth = _number(_mapping(payload.get("truth"), "OPE truth").get("value"), "truth.value")
            estimates = _mapping(payload.get("estimates"), "OPE estimates")
            estimator = str(job.coordinates["estimator"])
            key = "wpdis" if estimator.endswith("wpdis") else "pdis"
            estimate = _mapping(estimates.get(key), f"estimates.{key}")
            value = _number(estimate.get("value"), "estimate.value")
            normalized_bias = abs(
                _number(estimate.get("normalized_bias"), "estimate.normalized_bias")
            )
            diagnostics = _mapping(payload.get("diagnostics"), "OPE diagnostics")
            positive = _bool(
                diagnostics.get("positive_horizon_denominators"), "positive_horizon_denominators"
            )
            horizon = _integer(job.coordinates.get("horizon"), "empirical horizon", minimum=1)
            trajectories = _integer(
                job.coordinates.get("trajectories"), "empirical trajectories", minimum=1
            )
            if (
                diagnostics.get("trajectory_count") != trajectories
                or diagnostics.get("horizon_count") != horizon
            ):
                raise AnalysisError("OPE diagnostics do not close the planned denominator")
            numerators = _sequence(estimate.get("numerators"), "OPE numerators")
            denominators = _sequence(estimate.get("denominators"), "OPE denominators")
            if len(numerators) != horizon or len(denominators) != horizon:
                raise AnalysisError("OPE sufficient-statistic horizon denominator is incomplete")
            minimum_ess = _number(diagnostics.get("minimum_ess_fraction"), "minimum ESS fraction")
            ci = _mapping(payload.get("ci"), "OPE confidence interval")
            lower = _number(ci.get("lower"), "ci.lower")
            upper = _number(ci.get("upper"), "ci.upper")
            if lower > upper:
                raise AnalysisError("OPE confidence interval is reversed")
            covered = _bool(ci.get("covered_truth"), "ci.covered_truth")
            if covered != (lower <= truth <= upper):
                raise AnalysisError("OPE confidence interval coverage flag is unverifiable")
            empirical.append(
                {
                    "job": job,
                    "error": value - truth,
                    "normalized_bias": normalized_bias,
                    "positive": positive,
                    "covered": covered,
                    "width": upper - lower,
                    "minimum_ess_fraction": minimum_ess,
                }
            )
        elif kind in {"analytic_fixture", "fixed_point_reference"}:
            reference_count += 1
        elif kind == "fhe_valid":
            fhe = _mapping(payload.get("fhe"), "OPE FHE evidence")
            fhe_conformance.append(
                _bool(
                    fhe.get("conforms_to_integer_reference"),
                    "fhe.conforms_to_integer_reference",
                )
            )
    if not empirical:
        raise AnalysisError("OPE empirical calibration denominator is empty")
    # The mapping and gates were validated before inspecting any empirical row.
    primary_clip = gates.get("primary_clip")
    primary_count = gates.get("primary_trajectory_count")
    primary = [
        item
        for item in empirical
        if item["job"].coordinates.get("clip") == primary_clip
        and item["job"].coordinates.get("trajectories") == primary_count
    ]
    selected = primary or empirical
    max_bias = max(item["normalized_bias"] for item in selected)
    rmse = math.sqrt(sum(item["error"] ** 2 for item in selected) / len(selected))
    coverage = sum(item["covered"] for item in selected)
    median_width = statistics.median(item["width"] for item in selected)
    all_positive = all(item["positive"] for item in selected)
    observed_minimum_ess = min(item["minimum_ess_fraction"] for item in selected)
    observed_gates = [
        _gate(
            "maximum_normalized_bias",
            max_bias,
            "<=",
            _number(gates.get("maximum_normalized_bias"), "gate"),
            max_bias <= float(gates["maximum_normalized_bias"]),
        ),
        _gate(
            "maximum_rmse",
            rmse,
            "<=",
            _number(gates.get("maximum_rmse"), "gate"),
            rmse <= float(gates["maximum_rmse"]),
        ),
        _gate(
            "minimum_interval_coverage_count",
            coverage,
            ">=",
            _integer(gates.get("minimum_interval_coverage_count"), "gate"),
            coverage >= int(gates["minimum_interval_coverage_count"]),
        ),
        _gate(
            "maximum_interval_coverage_count",
            coverage,
            "<=",
            _integer(gates.get("maximum_interval_coverage_count"), "gate"),
            coverage <= int(gates["maximum_interval_coverage_count"]),
        ),
        _gate(
            "maximum_median_interval_width",
            median_width,
            "<=",
            _number(gates.get("maximum_median_interval_width"), "gate"),
            median_width <= float(gates["maximum_median_interval_width"]),
        ),
        _gate("require_positive_horizon_denominators", int(all_positive), "==", 1, all_positive),
        _gate(
            "minimum_ess_fraction",
            observed_minimum_ess,
            ">=",
            minimum_ess_fraction,
            observed_minimum_ess >= minimum_ess_fraction,
        ),
    ]
    return {
        "artifact_denominator": len(rows),
        "reference_case_denominator": reference_count,
        "empirical_batch_denominator": len(empirical),
        "calibration_denominator": len(selected),
        "configured_primary_cell_available": bool(primary),
        "gates": observed_gates,
        "all_statistical_gates_passed": all(gate["passed"] for gate in observed_gates),
        "fhe_valid_batch_denominator": len(fhe_conformance),
        "exact_clear_conformance": bool(fhe_conformance) and all(fhe_conformance),
    }


def _integration_summary(
    manifest: Mapping[str, Any],
    rows: list[tuple[_ExpectedJob, dict[str, Any], str]],
) -> dict[str, Any]:
    direct: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    estimates: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    online_truths: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    interval_zero: list[bool] = []
    sign_errors: list[bool] = []
    behavior_count = 0
    trust_ok = True
    for job, payload, _digest in rows:
        kind = job.coordinates.get("kind")
        scenario = int(job.coordinates["scenario"])
        mode = str(job.coordinates["shield_mode"])
        if kind in {"behavior_trajectory", "direct_trajectory"}:
            released = _mapping(payload.get("trajectory"), "integration released trajectory")
            if "steps" in released:
                raise AnalysisError(
                    "integration evidence contains prohibited plain trajectory logs"
                )
            total_return = _number(released.get("total_return"), "trajectory.total_return")
            unsafe_steps = _integer(released.get("unsafe_steps"), "trajectory.unsafe_steps")
            if kind == "behavior_trajectory":
                behavior_count += 1
            else:
                direct[(scenario, mode, "return")].append(total_return)
                direct[(scenario, mode, "unsafe_step_cost")].append(float(unsafe_steps))
            continue
        backend = _mapping(payload.get("backend"), "integration backend")
        trust_scope = backend.get("trust_scope")
        trust_detail = backend.get("trust_scope_detail")
        trust_ok &= (
            backend.get("real_fhe") is True
            and trust_scope == "colocated-client-server"
            and isinstance(trust_detail, str)
            and "does not claim input privacy" in trust_detail.lower()
        )
        effect = _mapping(payload.get("effect_channel"), "integration effect channel")
        outcome = str(job.coordinates["outcome"])
        if effect.get("outcome") != outcome:
            raise AnalysisError("integration outcome binding is invalid")
        key = (scenario, mode, outcome)
        estimates[key].append(_number(effect.get("fhe_estimate"), "FHE effect estimate"))
        online_truths[key].append(_number(effect.get("online_truth"), "online truth"))
        lower = _number(effect.get("lower"), "effect lower")
        upper = _number(effect.get("upper"), "effect upper")
        contains_zero = _bool(effect.get("contains_zero"), "effect contains_zero")
        if lower > upper or contains_zero != (lower <= 0.0 <= upper):
            raise AnalysisError("integration discrepancy interval is unverifiable")
        interval_zero.append(contains_zero)
        sign_errors.append(_bool(effect.get("sign_error"), "effect sign_error"))
        sufficient = _mapping(payload.get("statistics"), "integration statistics")
        denominators = _sequence(sufficient.get("denominators"), "integration denominators")
        counts = _sequence(sufficient.get("counts"), "integration counts")
        if not denominators or len(denominators) != len(counts):
            raise AnalysisError("integration OPE denominator vectors are missing or mismatched")
        if any(_number(value, "integration denominator") <= 0 for value in denominators):
            raise AnalysisError("integration OPE has a non-positive horizon denominator")
    if set(direct) != set(estimates) or not direct:
        raise AnalysisError("integration direct and OPE cells do not close to the same denominator")
    for key, truths in online_truths.items():
        mean = statistics.fmean(direct[key])
        if any(not math.isclose(value, mean, rel_tol=0.0, abs_tol=1e-12) for value in truths):
            raise AnalysisError("integration online truth does not match direct releases")
    discrepancies = {
        key: abs(statistics.fmean(direct[key]) - statistics.fmean(estimates[key])) for key in direct
    }
    by_outcome = {
        outcome: abs(
            statistics.fmean(
                value for key, values in direct.items() if key[2] == outcome for value in values
            )
            - statistics.fmean(
                value for key, values in estimates.items() if key[2] == outcome for value in values
            )
        )
        for outcome in {key[2] for key in discrepancies}
    }
    integration = _mapping(manifest.get("integration"), "integration")
    gates = _mapping(integration.get("gates"), "integration.gates")
    return_discrepancy = by_outcome.get("return")
    unsafe_discrepancy = by_outcome.get("unsafe_step_cost", by_outcome.get("unsafe_steps"))
    if return_discrepancy is None or unsafe_discrepancy is None:
        raise AnalysisError("integration outcomes do not expose return and unsafe discrepancies")
    scenarios = {key[0] for key in discrepancies}
    scenario_passes = sum(
        all(value < 0.05 for key, value in discrepancies.items() if key[0] == scenario)
        for scenario in scenarios
    )
    require_zero = _bool(
        gates.get("require_zero_containing_discrepancy_intervals"),
        "require_zero_containing_discrepancy_intervals",
    )
    require_no_sign = _bool(
        gates.get("require_no_sign_error_when_truth_excludes_zero"),
        "require_no_sign_error_when_truth_excludes_zero",
    )
    observed_gates = [
        _gate(
            "maximum_pooled_return_discrepancy",
            return_discrepancy,
            "<=",
            _number(gates.get("maximum_pooled_return_discrepancy"), "gate"),
            return_discrepancy <= float(gates["maximum_pooled_return_discrepancy"]),
        ),
        _gate(
            "maximum_pooled_unsafe_probability_discrepancy",
            unsafe_discrepancy,
            "<=",
            _number(gates.get("maximum_pooled_unsafe_probability_discrepancy"), "gate"),
            unsafe_discrepancy <= float(gates["maximum_pooled_unsafe_probability_discrepancy"]),
        ),
        _gate(
            "minimum_scenarios_with_discrepancy_below_005",
            scenario_passes,
            ">=",
            _integer(gates.get("minimum_scenarios_with_discrepancy_below_005"), "gate"),
            scenario_passes >= int(gates["minimum_scenarios_with_discrepancy_below_005"]),
        ),
        _gate(
            "require_zero_containing_discrepancy_intervals",
            int(all(interval_zero)),
            "==",
            int(require_zero),
            (not require_zero) or all(interval_zero),
        ),
        _gate(
            "require_no_sign_error_when_truth_excludes_zero",
            int(not any(sign_errors)),
            "==",
            int(require_no_sign),
            (not require_no_sign) or not any(sign_errors),
        ),
    ]
    return {
        "artifact_denominator": len(rows),
        "behavior_trajectory_denominator": behavior_count,
        "direct_cell_denominator": len(direct),
        "ope_cell_denominator": len(estimates),
        "pooled_discrepancy": {
            "return": return_discrepancy,
            "unsafe_step_cost": unsafe_discrepancy,
        },
        "gates": observed_gates,
        "all_observable_gates_passed": all(gate["passed"] for gate in observed_gates),
        "real_fhe_trust_scope_verified": trust_ok,
    }


def _p95(values: list[int], name: str) -> float:
    if not values:
        raise AnalysisError(f"{name} has no successful measured requests")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] / 1_000_000_000.0


def _timing_summary(
    manifest: Mapping[str, Any],
    rows: list[tuple[_ExpectedJob, dict[str, Any], str]],
    shield_fhe_rows: list[tuple[_ExpectedJob, dict[str, Any], str]],
) -> dict[str, Any]:
    systems = _mapping(manifest.get("systems"), "systems")
    ope = _mapping(manifest.get("ope"), "ope")
    ope_challenge = _mapping(ope.get("fhe_challenge"), "ope.fhe_challenge")
    samples: dict[tuple[str, int, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    key_setup: list[int] = []
    evaluation_key_bytes: list[int] = []
    expected_successful_rows = 0
    observed_successful_rows = 0
    for job, payload, _digest in rows:
        kind = str(job.coordinates["kind"])
        if payload.get("kind") != kind or payload.get("trust_scope") is None:
            raise AnalysisError("timing artifact is not bound to its public workload")
        groups = _mapping(payload.get("groups"), "timing groups")
        expected_groups = (
            {"shield", "ope"}
            if kind == "concurrent_client"
            else {"shield"}
            if kind == "shield"
            else {"ope"}
        )
        if set(groups) != expected_groups:
            raise AnalysisError("timing artifact has a missing or extra workload group")
        for workload, raw_group in groups.items():
            group = _mapping(raw_group, "timing group")
            context = _mapping(group.get("context"), "timing context")
            key_setup.append(_integer(context.get("key_setup_ns"), "timing key_setup_ns"))
            evaluation_key_bytes.append(
                _integer(context.get("evaluation_key_bytes"), "timing evaluation_key_bytes")
            )
            raw_trajectories = context.get("trajectories")
            raw_horizon = context.get("horizon")
            if workload == "shield":
                if raw_trajectories is not None or raw_horizon is not None:
                    raise AnalysisError("shield timing context has an unexpected OPE shape")
                trajectories = horizon = 0
            else:
                trajectories = _integer(raw_trajectories, "timing trajectories", minimum=1)
                horizon = _integer(raw_horizon, "timing horizon", minimum=1)
            if kind == "shield":
                warmups = _integer(systems.get("shield_warmups_per_container"), "shield warmups")
                measured = _integer(systems.get("shield_measured_per_container"), "shield measured")
            elif kind == "ope":
                warmups = _integer(systems.get("ope_warmups_per_container"), "ope warmups")
                measured = _integer(systems.get("ope_measured_per_container"), "ope measured")
            elif kind == "scale":
                warmups = _integer(systems.get("scale_warmups_per_container"), "scale warmups")
                measured = _integer(systems.get("scale_measured_per_container"), "scale measured")
            else:
                warmups = 0
                measured = _integer(
                    systems.get(
                        "concurrent_shield_calls_per_client"
                        if workload == "shield"
                        else "concurrent_ope_calls_per_client"
                    ),
                    "concurrent measured",
                )
            public_rows = _sequence(group.get("rows"), "timing rows")
            if len(public_rows) != warmups + measured:
                raise AnalysisError("timing row denominator does not match the manifest")
            measured_rows = [
                _mapping(row, "timing row")
                for row in public_rows
                if _mapping(row, "timing row").get("is_warmup") is False
            ]
            if len(measured_rows) != measured:
                raise AnalysisError("timing measured denominator is inconsistent")
            expected_successful_rows += measured
            bucket_kind = f"concurrent_{workload}" if kind == "concurrent_client" else workload
            if kind == "ope":
                trajectories = _integer(
                    ope_challenge.get("trajectories_per_batch"), "OPE timing trajectories"
                )
                horizon = _integer(ope_challenge.get("horizon"), "OPE timing horizon")
            bucket = samples[(bucket_kind, trajectories, horizon)]
            for row in measured_rows:
                if row.get("success") is not True:
                    continue
                timing_ns = _mapping(row.get("timing_ns"), "timing row metrics")
                server = _integer(timing_ns.get("server_evaluate"), "server timing")
                encrypt = _integer(timing_ns.get("client_encrypt"), "client encrypt timing")
                decrypt = _integer(timing_ns.get("client_decrypt"), "client decrypt timing")
                end_to_end = _integer(timing_ns.get("end_to_end"), "end-to-end timing")
                bucket["server"].append(server)
                bucket["client"].append(encrypt + decrypt)
                bucket["end_to_end"].append(end_to_end)
                observed_successful_rows += 1

    shield_server = _p95(samples[("shield", 0, 0)]["server"], "shield server timing")
    shield_client = _p95(samples[("shield", 0, 0)]["client"], "shield client timing")
    cell_p95: dict[str, float] = {}
    cell_throughput: dict[str, float] = {}
    for (workload, trajectories, horizon), metrics in samples.items():
        if workload not in {"ope"} or not metrics["end_to_end"]:
            continue
        label = f"{trajectories}x{horizon}"
        cell_p95[label] = _p95(metrics["end_to_end"], f"OPE {label}")
        cell_throughput[label] = (
            trajectories * horizon / (statistics.fmean(metrics["end_to_end"]) / 1_000_000_000.0)
        )
    baseline_shield = statistics.fmean(samples[("shield", 0, 0)]["end_to_end"])
    concurrent_shield = samples[("concurrent_shield", 0, 0)]["end_to_end"]
    ope_shape = (
        _integer(ope_challenge.get("trajectories_per_batch"), "OPE trajectories"),
        _integer(ope_challenge.get("horizon"), "OPE horizon"),
    )
    baseline_ope = statistics.fmean(samples[("ope", *ope_shape)]["end_to_end"])
    concurrent_ope = samples[("concurrent_ope", *ope_shape)]["end_to_end"]
    if not concurrent_shield or not concurrent_ope:
        raise AnalysisError("concurrent timing denominator is empty")
    throughput_fraction = min(
        baseline_shield / statistics.fmean(concurrent_shield),
        baseline_ope / statistics.fmean(concurrent_ope),
    )
    p95_multiplier = max(
        _p95(concurrent_shield, "concurrent shield")
        / _p95(samples[("shield", 0, 0)]["end_to_end"], "shield baseline"),
        _p95(concurrent_ope, "concurrent OPE")
        / _p95(samples[("ope", *ope_shape)]["end_to_end"], "OPE baseline"),
    )
    compile_ns = [
        _integer(_mapping(payload.get("compile"), "shield compile").get("compile_ns"), "compile_ns")
        for _job, payload, _digest in shield_fhe_rows
    ]
    call_keygen = [
        _integer(_mapping(payload.get("call"), "shield call").get("keygen_ns"), "keygen_ns")
        for _job, payload, _digest in shield_fhe_rows
    ]
    call_keys = [
        _integer(
            _mapping(payload.get("call"), "shield call").get("evaluation_key_bytes"),
            "evaluation_key_bytes",
        )
        for _job, payload, _digest in shield_fhe_rows
    ]
    gates = _mapping(systems.get("gates"), "systems.gates")
    observed = {
        "shield_server_p95_seconds": shield_server,
        "shield_client_p95_seconds": shield_client,
        "ope_256x64_p95_seconds": cell_p95.get("256x64"),
        "ope_1024x64_p95_seconds": cell_p95.get("1024x64"),
        "minimum_1024x64_state_steps_per_second": cell_throughput.get("1024x64"),
        "minimum_concurrent_throughput_fraction": throughput_fraction,
        "maximum_concurrent_p95_multiplier": p95_multiplier,
        "maximum_compile_seconds": max(compile_ns) / 1_000_000_000.0,
        "maximum_keygen_seconds": max((*key_setup, *call_keygen)) / 1_000_000_000.0,
        "maximum_evaluation_key_bytes": max((*evaluation_key_bytes, *call_keys)),
        "maximum_memory_fraction": None,
    }
    directions = {
        "minimum_1024x64_state_steps_per_second": ">=",
        "minimum_concurrent_throughput_fraction": ">=",
    }
    gate_results = []
    for name, value in observed.items():
        threshold = _number(gates.get(name), name)
        direction = directions.get(name, "<=")
        passed = value is not None and (
            value >= threshold if direction == ">=" else value <= threshold
        )
        gate_results.append(_gate(name, value, direction, threshold, passed))
    return {
        "artifact_denominator": len(rows),
        "measured_request_denominator": expected_successful_rows,
        "successful_measured_requests": observed_successful_rows,
        "ope_cell_p95_seconds": dict(sorted(cell_p95.items())),
        "ope_cell_state_steps_per_second": dict(sorted(cell_throughput.items())),
        "gates": gate_results,
        "all_observed_gates_passed": all(gate["passed"] for gate in gate_results),
        "claim_scope": "colocated timing only; no network or remote-server privacy claim",
    }


def execute_flagship_job(
    manifest: Mapping[str, Any], job: Mapping[str, Any], evidence_root: str | Path
) -> dict[str, str | None]:
    """Verify the complete upstream ledger and write its sole analysis artifact."""

    try:
        manifest_map = _mapping(manifest, "manifest")
        job_map = _mapping(job, "job")
        root = Path(evidence_root)
        if not root.is_dir() or root.is_symlink():
            raise AnalysisError("evidence root is not a regular directory")
        artifacts, closure = _verify_evidence(manifest_map, job_map, root)
        shield = _shield_summary(manifest_map, artifacts["clear_shield_matrix"])
        shield_fhe = _shield_fhe_summary(manifest_map, artifacts["shield_fhe_challenge"])
        ope = _ope_summary(manifest_map, artifacts["ope_validation"])
        integration = _integration_summary(manifest_map, artifacts["integration"])
        systems = _timing_summary(
            manifest_map,
            artifacts["timing"],
            artifacts["shield_fhe_challenge"],
        )
    except AnalysisError:
        return _rejected("analysis.unverifiable-evidence")

    claims = _mapping(manifest_map.get("claims"), "manifest.claims")
    forbidden = _sequence(claims.get("forbidden"), "claims.forbidden")
    publication = {
        "document_name": "publication.json",
        "study_name": manifest_map.get("name"),
        "novel_conjunction": claims.get("novel_conjunction"),
        "forbidden_claims": list(forbidden),
        "claim_scope": {
            "clear_shield_results": "clear simulation aggregates; not privacy evidence",
            "fhe_execution": "real FHE conformance evidence",
            "deployment_trust": (
                "colocated client/server evaluation; no colocated input-privacy claim"
            ),
            "action_selection": "client-only after encrypted shield margins are returned",
            "ope_division": "client-only after encrypted sufficient statistics are returned",
            "integrity": "no malicious-server integrity claim",
        },
        "gate_pass": {
            "clear_shield": shield["all_gates_passed"],
            "ope_calibration": ope["all_statistical_gates_passed"],
            "integration": integration["all_observable_gates_passed"],
            "systems": systems["all_observed_gates_passed"],
            "shield_fhe_conformance": shield_fhe["exact_clear_conformance"],
            "ope_fhe_conformance": ope["exact_clear_conformance"],
        },
    }
    evidence_summary = {
        "document_name": "evidence-summary.json",
        "closure": closure,
        "clear_shield": shield,
        "shield_fhe": shield_fhe,
        "ope": ope,
        "integration": integration,
        "systems": systems,
    }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "manifest_digest": closure["manifest_digest"],
        "manifest_payload_digest": closure["manifest_payload_digest"],
        "job_id": job_map["job_id"],
        "stage": "analysis",
        "coordinates": job_map["coordinates"],
        "evidence_summary": evidence_summary,
        "publication": publication,
    }
    relative = f"analysis/{job_map['job_id']}.json"
    destination = root / relative
    if destination.exists() or destination.is_symlink():
        return _rejected("analysis.artifact-exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(payload) + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return _rejected("analysis.artifact-exists")
    return {
        "status": "succeeded",
        "artifact_path": relative,
        "artifact_digest": hashlib.sha256(raw).hexdigest(),
        "reason_code": None,
    }
