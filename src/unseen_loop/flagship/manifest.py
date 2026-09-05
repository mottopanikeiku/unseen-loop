"""Strict, immutable parsing and deterministic planning for flagship studies."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

SCHEMA_VERSION = "unseen-loop/flagship-study-v1"
PLAN_SCHEMA_VERSION = "unseen-loop/flagship-plan-v1"


class ManifestError(ValueError):
    """Raised when a flagship manifest is incomplete, inconsistent, or ambiguous."""


@dataclass(frozen=True, slots=True)
class Claims:
    novel_conjunction: str
    forbidden: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShieldGates:
    minimum_unsafe_reduction_vs_disabled: float
    minimum_lookahead_reduction_vs_h1: float
    maximum_benign_intervention_rate: float
    maximum_benign_intervention_wilson_upper: float
    maximum_fallback_rate: float
    maximum_fallback_wilson_upper: float
    minimum_certified_requested_action_preservation: float
    minimum_certified_candidate_rate: float
    normalized_return_delta_lower_bound: float
    goal_rate_delta_lower_bound: float
    model_mismatch_max_unsafe_multiplier: float


@dataclass(frozen=True, slots=True)
class ShieldFheChallenge:
    valid_calls: int
    occupancy_states: int
    extrema_states: int
    threshold_states: int
    tie_states: int
    canary_states: int
    canary_encryptions_per_state: int
    expected_decoded_margins: int
    expected_action_matches: int
    invalid_domain_rejections: int
    security_level: int
    global_p_error: float


@dataclass(frozen=True, slots=True)
class Shield:
    state_features: tuple[str, ...]
    actions: tuple[str, ...]
    horizons: int
    margins: tuple[str, ...]
    output_shape: tuple[int, ...]
    scenarios: int
    seeds_per_controller_cell: int
    controller_cells: tuple[str, ...]
    gates: ShieldGates
    fhe_challenge: ShieldFheChallenge


@dataclass(frozen=True, slots=True)
class OpeGates:
    primary_clip: float
    primary_trajectory_count: int
    maximum_normalized_bias: float
    maximum_rmse: float
    minimum_interval_coverage_count: int
    maximum_interval_coverage_count: int
    maximum_median_interval_width: float
    require_positive_horizon_denominators: bool
    require_exact_counts: bool


@dataclass(frozen=True, slots=True)
class OpeReference:
    analytic_fixtures: int
    random_fixed_point_cases: int


@dataclass(frozen=True, slots=True)
class OpeFheChallenge:
    valid_batches: int
    occupancy_batches: int
    extrema_batches: int
    terminal_padding_batches: int
    rounding_boundary_batches: int
    horizon: int
    trajectories_per_batch: int
    expected_outputs: int
    expected_count_entries: int
    invalid_batch_rejections: int


@dataclass(frozen=True, slots=True)
class Ope:
    horizons: tuple[int, ...]
    trajectory_counts: tuple[int, ...]
    overlap_lambdas: tuple[float, ...]
    clip_values: tuple[float, ...]
    include_unclipped: bool
    independent_batches: int
    bootstrap_repetitions: int
    estimators: tuple[str, ...]
    minimum_ess_fraction: float
    gates: OpeGates
    reference: OpeReference
    fhe_challenge: OpeFheChallenge


@dataclass(frozen=True, slots=True)
class IntegrationGates:
    maximum_pooled_return_discrepancy: float
    maximum_pooled_unsafe_probability_discrepancy: float
    minimum_scenarios_with_discrepancy_below_005: int
    require_zero_containing_discrepancy_intervals: bool
    require_no_sign_error_when_truth_excludes_zero: bool


@dataclass(frozen=True, slots=True)
class Integration:
    behavior_mixture_target_weight: float
    action_count: int
    minimum_behavior_probability: float
    scenarios: int
    shield_modes: tuple[str, ...]
    behavior_trajectories_per_cell: int
    direct_target_trajectories_per_cell: int
    horizon: int
    ope_batch_trajectories: int
    outcomes: tuple[str, ...]
    expected_behavior_trajectories: int
    expected_direct_trajectories: int
    expected_real_fhe_calls: int
    expected_returned_scalars: int
    gates: IntegrationGates


@dataclass(frozen=True, slots=True)
class SystemsGates:
    shield_server_p95_seconds: float
    shield_client_p95_seconds: float
    ope_256x64_p95_seconds: float
    ope_1024x64_p95_seconds: float
    minimum_1024x64_state_steps_per_second: float
    minimum_concurrent_throughput_fraction: float
    maximum_concurrent_p95_multiplier: float
    maximum_compile_seconds: int
    maximum_keygen_seconds: int
    maximum_evaluation_key_bytes: int
    maximum_memory_fraction: float


@dataclass(frozen=True, slots=True)
class Systems:
    shield_timing_containers: int
    shield_warmups_per_container: int
    shield_measured_per_container: int
    ope_timing_containers: int
    ope_warmups_per_container: int
    ope_measured_per_container: int
    concurrent_clients: int
    concurrent_shield_calls_per_client: int
    concurrent_ope_calls_per_client: int
    scale_trajectory_counts: tuple[int, ...]
    scale_horizons: tuple[int, ...]
    scale_containers_per_cell: int
    scale_warmups_per_container: int
    scale_measured_per_container: int
    gates: SystemsGates


@dataclass(frozen=True, slots=True)
class Cryptography:
    required_classical_security_bits: int
    same_plaintext_canary_pairs: int
    ckks_max_analytical_error_fixed_point_units: float
    ckks_max_observed_error_fixed_point_units: float
    require_server_secret_absence: bool
    require_compiled_feedback: bool
    require_client_only_action_selection: bool
    require_client_only_ope_division: bool


@dataclass(frozen=True, slots=True)
class Reproducibility:
    require_clean_commit: bool
    require_lockfile_digest: bool
    require_container_digest: bool
    require_closed_root_ledger: bool
    reject_extra_files: bool
    retain_all_attempts: bool
    require_two_clean_replays: bool
    cryptographic_randomness_deterministic: bool


@dataclass(frozen=True, slots=True)
class FlagshipManifest:
    schema_version: str
    name: str
    seed_root: str
    minimum_pillar_score: int
    claims: Claims
    shield: Shield
    ope: Ope
    integration: Integration
    systems: Systems
    cryptography: Cryptography
    reproducibility: Reproducibility
    digest: str

    def canonical_payload(self) -> dict[str, Any]:
        payload = _jsonable(self)
        assert isinstance(payload, dict)
        payload.pop("digest")
        return payload


T = TypeVar("T")


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        instance = cast(Any, value)
        return {
            field.name: _jsonable(getattr(instance, field.name))
            for field in dataclasses.fields(instance)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def content_digest(value: object) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


def _table(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{name} must be a TOML table")
    return cast(dict[str, Any], value)


def _construct(
    cls: type[T], value: object, name: str, nested: Mapping[str, type[Any]] | None = None
) -> T:
    table = _table(value, name)
    fields = {field.name: field for field in dataclasses.fields(cast(Any, cls))}
    expected = set(fields)
    actual = set(table)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(f"{name} keys differ: missing={missing}, extra={extra}")
    kwargs: dict[str, Any] = {}
    nested = nested or {}
    for key, raw in table.items():
        if key in nested:
            kwargs[key] = _construct(nested[key], raw, f"{name}.{key}")
        else:
            kwargs[key] = _coerce_field(raw, fields[key].type, f"{name}.{key}")
    return cls(**kwargs)


def _coerce_field(value: object, annotation: object, name: str) -> object:
    # With postponed annotations, dataclass field types are strings.
    text = str(annotation).replace("typing.", "")
    if annotation is str or text in {"str", "'str'"}:
        if not isinstance(value, str) or not value:
            raise ManifestError(f"{name} must be a non-empty string")
        return value
    if annotation is bool or text in {"bool", "'bool'"}:
        if type(value) is not bool:
            raise ManifestError(f"{name} must be a boolean")
        return value
    if annotation is int or text in {"int", "'int'"}:
        if type(value) is not int:
            raise ManifestError(f"{name} must be an integer")
        return value
    if annotation is float or text in {"float", "'float'"}:
        if type(value) not in (int, float):
            raise ManifestError(f"{name} must be numeric")
        try:
            numeric_value = float(cast(int | float, value))
        except OverflowError as exc:
            raise ManifestError(f"{name} must be finite") from exc
        if not math.isfinite(numeric_value):
            raise ManifestError(f"{name} must be finite")
        return numeric_value
    if "tuple[str" in text:
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise ManifestError(f"{name} must be a non-empty string array")
        if len(set(value)) != len(value):
            raise ManifestError(f"{name} contains duplicates")
        return tuple(value)
    if "tuple[int" in text:
        if not isinstance(value, list) or not value or not all(type(item) is int for item in value):
            raise ManifestError(f"{name} must be a non-empty integer array")
        if len(set(value)) != len(value):
            raise ManifestError(f"{name} contains duplicates")
        return tuple(value)
    if "tuple[float" in text:
        if (
            not isinstance(value, list)
            or not value
            or not all(type(item) in (int, float) for item in value)
        ):
            raise ManifestError(f"{name} must be a non-empty numeric array")
        try:
            result = tuple(float(item) for item in value)
        except OverflowError as exc:
            raise ManifestError(f"{name} must contain only finite numbers") from exc
        if not all(math.isfinite(item) for item in result):
            raise ManifestError(f"{name} must contain only finite numbers")
        if len(set(result)) != len(result):
            raise ManifestError(f"{name} contains duplicates")
        return result
    raise AssertionError(f"unsupported manifest field {name}: {annotation!r}")


def parse_manifest_bytes(data: bytes) -> FlagshipManifest:
    """Parse a TOML manifest, rejecting unknown keys and inconsistent denominators."""

    try:
        root = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"invalid UTF-8 TOML manifest: {exc}") from exc
    expected = {
        "schema_version",
        "name",
        "seed_root",
        "minimum_pillar_score",
        "claims",
        "shield",
        "ope",
        "integration",
        "systems",
        "cryptography",
        "reproducibility",
    }
    if set(root) != expected:
        raise ManifestError(
            f"manifest keys differ: missing={sorted(expected - set(root))}, "
            f"extra={sorted(set(root) - expected)}"
        )
    schema_version = _required_string(root, "schema_version")
    name = _required_string(root, "name")
    seed_root = _required_string(root, "seed_root")
    minimum_pillar_score = _required_integer(root, "minimum_pillar_score")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must equal {SCHEMA_VERSION!r}")
    claims = _construct(Claims, root["claims"], "claims")
    shield = _construct(
        Shield,
        root["shield"],
        "shield",
        {"gates": ShieldGates, "fhe_challenge": ShieldFheChallenge},
    )
    ope = _construct(
        Ope,
        root["ope"],
        "ope",
        {
            "gates": OpeGates,
            "reference": OpeReference,
            "fhe_challenge": OpeFheChallenge,
        },
    )
    integration = _construct(
        Integration,
        root["integration"],
        "integration",
        {"gates": IntegrationGates},
    )
    systems = _construct(
        Systems,
        root["systems"],
        "systems",
        {"gates": SystemsGates},
    )
    cryptography = _construct(Cryptography, root["cryptography"], "cryptography")
    reproducibility = _construct(Reproducibility, root["reproducibility"], "reproducibility")
    manifest = FlagshipManifest(
        schema_version=schema_version,
        name=name,
        seed_root=seed_root,
        minimum_pillar_score=minimum_pillar_score,
        claims=claims,
        shield=shield,
        ope=ope,
        integration=integration,
        systems=systems,
        cryptography=cryptography,
        reproducibility=reproducibility,
        digest=hashlib.sha256(data).hexdigest(),
    )
    _validate_manifest(manifest)
    return manifest


def _required_string(root: Mapping[str, object], key: str) -> str:
    value = root[key]
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _required_integer(root: Mapping[str, object], key: str) -> int:
    value = root[key]
    if type(value) is not int:
        raise ManifestError(f"{key} must be an integer")
    return value


def load_manifest(path: str | Path) -> FlagshipManifest:
    return parse_manifest_bytes(Path(path).read_bytes())


def _positive(name: str, *values: int | float) -> None:
    if any(value <= 0 for value in values):
        raise ManifestError(f"{name} values must all be positive")


def _probability(name: str, *values: float) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ManifestError(f"{name} values must be in [0, 1]")


def _validate_manifest(m: FlagshipManifest) -> None:
    _positive("manifest", m.minimum_pillar_score)
    if m.minimum_pillar_score > 100:
        raise ManifestError("minimum_pillar_score must not exceed 100")
    _positive(
        "shield",
        m.shield.horizons,
        m.shield.scenarios,
        m.shield.seeds_per_controller_cell,
    )
    if tuple(m.shield.output_shape) != (
        len(m.shield.actions),
        m.shield.horizons,
        len(m.shield.margins),
    ):
        raise ManifestError("shield.output_shape does not match action/horizon/margin dimensions")
    if len(m.shield.state_features) != 6:
        raise ManifestError("shield.state_features must describe the six encrypted state wires")
    challenge = m.shield.fhe_challenge
    _positive(
        "shield.fhe_challenge",
        challenge.valid_calls,
        challenge.occupancy_states,
        challenge.extrema_states,
        challenge.threshold_states,
        challenge.tie_states,
        challenge.canary_states,
        challenge.canary_encryptions_per_state,
        challenge.expected_decoded_margins,
        challenge.expected_action_matches,
        challenge.invalid_domain_rejections,
        challenge.security_level,
        challenge.global_p_error,
    )
    valid = (
        challenge.occupancy_states
        + challenge.extrema_states
        + challenge.threshold_states
        + challenge.tie_states
        + challenge.canary_states * challenge.canary_encryptions_per_state
    )
    if valid != challenge.valid_calls:
        raise ManifestError("shield.fhe_challenge valid-call denominator is inconsistent")
    if challenge.expected_action_matches != challenge.valid_calls:
        raise ManifestError("shield expected_action_matches must equal valid_calls")
    expected_margins = challenge.valid_calls * _product(m.shield.output_shape)
    if challenge.expected_decoded_margins != expected_margins:
        raise ManifestError("shield expected_decoded_margins is inconsistent with output_shape")
    _positive(
        "ope",
        *m.ope.horizons,
        *m.ope.trajectory_counts,
        *m.ope.clip_values,
        m.ope.independent_batches,
        m.ope.bootstrap_repetitions,
    )
    _probability("ope.overlap_lambdas", *m.ope.overlap_lambdas)
    ofhe = m.ope.fhe_challenge
    _positive(
        "ope.fhe_challenge",
        ofhe.valid_batches,
        ofhe.occupancy_batches,
        ofhe.extrema_batches,
        ofhe.terminal_padding_batches,
        ofhe.rounding_boundary_batches,
        ofhe.horizon,
        ofhe.trajectories_per_batch,
        ofhe.expected_outputs,
        ofhe.expected_count_entries,
        ofhe.invalid_batch_rejections,
    )
    if (
        ofhe.valid_batches
        != ofhe.occupancy_batches
        + ofhe.extrema_batches
        + ofhe.terminal_padding_batches
        + ofhe.rounding_boundary_batches
    ):
        raise ManifestError("ope.fhe_challenge valid-batch denominator is inconsistent")
    if ofhe.expected_outputs != ofhe.valid_batches * ofhe.horizon * 3:
        raise ManifestError("ope expected_outputs must equal valid_batches * horizon * 3")
    if ofhe.expected_count_entries != ofhe.valid_batches * ofhe.horizon:
        raise ManifestError("ope expected_count_entries must equal valid_batches * horizon")
    _positive(
        "integration",
        m.integration.action_count,
        m.integration.scenarios,
        m.integration.behavior_trajectories_per_cell,
        m.integration.direct_target_trajectories_per_cell,
        m.integration.horizon,
        m.integration.ope_batch_trajectories,
        m.integration.expected_behavior_trajectories,
        m.integration.expected_direct_trajectories,
        m.integration.expected_real_fhe_calls,
        m.integration.expected_returned_scalars,
    )
    if m.integration.scenarios > m.shield.scenarios:
        raise ManifestError("integration scenarios must be a registered shield-scenario prefix")
    integration_cells = m.integration.scenarios * len(m.integration.shield_modes)
    if (
        m.integration.expected_behavior_trajectories
        != integration_cells * m.integration.behavior_trajectories_per_cell
    ):
        raise ManifestError("integration behavior-trajectory denominator is inconsistent")
    if (
        m.integration.expected_direct_trajectories
        != integration_cells * m.integration.direct_target_trajectories_per_cell
    ):
        raise ManifestError("integration direct-trajectory denominator is inconsistent")
    expected_calls = (
        integration_cells
        * (m.integration.behavior_trajectories_per_cell // m.integration.ope_batch_trajectories)
        * len(m.integration.outcomes)
    )
    if (
        m.integration.behavior_trajectories_per_cell % m.integration.ope_batch_trajectories
        or m.integration.expected_real_fhe_calls != expected_calls
    ):
        raise ManifestError("integration real-FHE call denominator is inconsistent")
    if m.integration.expected_returned_scalars != (
        m.integration.expected_behavior_trajectories + m.integration.expected_direct_trajectories
    ):
        raise ManifestError("integration returned-scalar denominator is inconsistent")
    _probability(
        "probability",
        m.integration.behavior_mixture_target_weight,
        m.integration.minimum_behavior_probability,
        m.ope.minimum_ess_fraction,
    )
    systems = m.systems
    _positive(
        "systems",
        systems.shield_timing_containers,
        systems.shield_warmups_per_container,
        systems.shield_measured_per_container,
        systems.ope_timing_containers,
        systems.ope_warmups_per_container,
        systems.ope_measured_per_container,
        systems.concurrent_clients,
        systems.concurrent_shield_calls_per_client,
        systems.concurrent_ope_calls_per_client,
        *systems.scale_trajectory_counts,
        *systems.scale_horizons,
        systems.scale_containers_per_cell,
        systems.scale_warmups_per_container,
        systems.scale_measured_per_container,
    )
    if m.cryptography.required_classical_security_bits != m.shield.fhe_challenge.security_level:
        raise ManifestError("cryptography security bits disagree with shield FHE challenge")


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    stage_id: str
    dependencies: tuple[str, ...]
    max_parallel: int


@dataclass(frozen=True, slots=True)
class PlannedJob:
    job_id: str
    stage: str
    seed: int
    coordinates: tuple[tuple[str, str | int | float], ...]

    def coordinate_dict(self) -> dict[str, str | int | float]:
        return dict(self.coordinates)


STAGE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "clear_shield_matrix": (),
    "shield_fhe_challenge": (),
    "ope_validation": (),
    "integration": ("clear_shield_matrix", "shield_fhe_challenge", "ope_validation"),
    "timing": ("shield_fhe_challenge", "ope_validation"),
    "analysis": (
        "clear_shield_matrix",
        "shield_fhe_challenge",
        "ope_validation",
        "integration",
        "timing",
    ),
    "evidence_finalizer": ("analysis",),
}

DEFAULT_STAGE_POOLS: Mapping[str, int] = {
    "clear_shield_matrix": 64,
    "shield_fhe_challenge": 8,
    "ope_validation": 32,
    "integration": 32,
    "timing": 8,
    "analysis": 1,
    "evidence_finalizer": 1,
}


def stage_dag(
    manifest: FlagshipManifest, pools: Mapping[str, int] | None = None
) -> tuple[Stage, ...]:
    limits = dict(DEFAULT_STAGE_POOLS if pools is None else pools)
    if set(limits) != set(STAGE_DEPENDENCIES) or any(
        type(value) is not int or value <= 0 for value in limits.values()
    ):
        raise ManifestError("stage pools must give every stage one positive integer bound")
    stages: list[Stage] = []
    ids: dict[str, str] = {}
    for name, dependencies in STAGE_DEPENDENCIES.items():
        if any(dependency not in ids for dependency in dependencies):
            raise AssertionError("stage DAG is not topologically ordered")
        stage_content = {
            "manifest": manifest.digest,
            "name": name,
            "dependencies": dependencies,
        }
        stage_id = f"stage-{name}-{content_digest(stage_content)[:16]}"
        ids[name] = stage_id
        stages.append(Stage(name, stage_id, dependencies, limits[name]))
    return tuple(stages)


def derive_seed(seed_root: str, job_id: str) -> int:
    """Derive a deterministic 128-bit, job-domain-separated simulation seed."""
    return int.from_bytes(hashlib.sha256(f"{seed_root}\0{job_id}".encode()).digest()[:16], "big")


def _job(manifest: FlagshipManifest, stage: str, **coordinates: str | int | float) -> PlannedJob:
    canonical_coordinates = tuple(sorted(coordinates.items()))
    digest = content_digest(
        {
            "schema": PLAN_SCHEMA_VERSION,
            "manifest": manifest.digest,
            "stage": stage,
            "coordinates": canonical_coordinates,
        }
    )
    job_id = f"job-{stage}-{digest[:24]}"
    return PlannedJob(job_id, stage, derive_seed(manifest.seed_root, job_id), canonical_coordinates)


def iter_stage_jobs(manifest: FlagshipManifest, stage: str) -> Iterator[PlannedJob]:
    """Expand all configured experimental denominators into stable, explicit jobs."""
    if stage == "clear_shield_matrix":
        for scenario in range(manifest.shield.scenarios):
            for controller in manifest.shield.controller_cells:
                for repetition in range(manifest.shield.seeds_per_controller_cell):
                    yield _job(
                        manifest,
                        stage,
                        scenario=scenario,
                        controller=controller,
                        repetition=repetition,
                    )
    elif stage == "shield_fhe_challenge":
        challenge = manifest.shield.fhe_challenge
        categories = (
            ("occupancy", challenge.occupancy_states, 1),
            ("extrema", challenge.extrema_states, 1),
            ("threshold", challenge.threshold_states, 1),
            ("tie", challenge.tie_states, 1),
            ("canary", challenge.canary_states, challenge.canary_encryptions_per_state),
        )
        for category, states, encryptions in categories:
            for state in range(states):
                for encryption in range(encryptions):
                    yield _job(
                        manifest,
                        stage,
                        kind="valid",
                        category=category,
                        state=state,
                        encryption=encryption,
                    )
        for case in range(challenge.invalid_domain_rejections):
            yield _job(manifest, stage, kind="invalid", case=case)
    elif stage == "ope_validation":
        for fixture in range(manifest.ope.reference.analytic_fixtures):
            yield _job(manifest, stage, kind="analytic_fixture", case=fixture)
        for case in range(manifest.ope.reference.random_fixed_point_cases):
            yield _job(manifest, stage, kind="fixed_point_reference", case=case)
        clips: tuple[str | float, ...] = manifest.ope.clip_values + (
            ("unclipped",) if manifest.ope.include_unclipped else ()
        )
        for horizon in manifest.ope.horizons:
            for trajectories in manifest.ope.trajectory_counts:
                for overlap in manifest.ope.overlap_lambdas:
                    for clip in clips:
                        for estimator in manifest.ope.estimators:
                            for batch in range(manifest.ope.independent_batches):
                                yield _job(
                                    manifest,
                                    stage,
                                    kind="empirical",
                                    horizon=horizon,
                                    trajectories=trajectories,
                                    overlap=overlap,
                                    clip=clip,
                                    estimator=estimator,
                                    batch=batch,
                                )
        ope_challenge = manifest.ope.fhe_challenge
        for category, count in (
            ("occupancy", ope_challenge.occupancy_batches),
            ("extrema", ope_challenge.extrema_batches),
            ("terminal_padding", ope_challenge.terminal_padding_batches),
            ("rounding_boundary", ope_challenge.rounding_boundary_batches),
        ):
            for batch in range(count):
                yield _job(manifest, stage, kind="fhe_valid", category=category, batch=batch)
        for batch in range(ope_challenge.invalid_batch_rejections):
            yield _job(manifest, stage, kind="fhe_invalid", batch=batch)
    elif stage == "integration":
        spec = manifest.integration
        # Materialize and commit every client trajectory release before any OPE
        # job can load a 64-trajectory batch from the shared Volume.
        for scenario in range(spec.scenarios):
            for mode in spec.shield_modes:
                for trajectory in range(spec.behavior_trajectories_per_cell):
                    yield _job(
                        manifest,
                        stage,
                        kind="behavior_trajectory",
                        scenario=scenario,
                        shield_mode=mode,
                        trajectory=trajectory,
                    )
                for trajectory in range(spec.direct_target_trajectories_per_cell):
                    yield _job(
                        manifest,
                        stage,
                        kind="direct_trajectory",
                        scenario=scenario,
                        shield_mode=mode,
                        trajectory=trajectory,
                    )
        batch_count = spec.behavior_trajectories_per_cell // spec.ope_batch_trajectories
        for scenario in range(spec.scenarios):
            for mode in spec.shield_modes:
                for outcome in spec.outcomes:
                    for batch in range(batch_count):
                        yield _job(
                            manifest,
                            stage,
                            kind="real_fhe_ope",
                            scenario=scenario,
                            shield_mode=mode,
                            outcome=outcome,
                            batch=batch,
                        )
    elif stage == "timing":
        systems = manifest.systems
        for container in range(systems.shield_timing_containers):
            yield _job(manifest, stage, kind="shield", container=container)
        for container in range(systems.ope_timing_containers):
            yield _job(manifest, stage, kind="ope", container=container)
        for trajectories in systems.scale_trajectory_counts:
            for horizon in systems.scale_horizons:
                for container in range(systems.scale_containers_per_cell):
                    yield _job(
                        manifest,
                        stage,
                        kind="scale",
                        trajectories=trajectories,
                        horizon=horizon,
                        container=container,
                    )
        for client in range(systems.concurrent_clients):
            yield _job(manifest, stage, kind="concurrent_client", client=client)
    elif stage in {"analysis", "evidence_finalizer"}:
        yield _job(manifest, stage, kind="singleton")
    else:
        raise ManifestError(f"unknown stage {stage!r}")


def planned_job_ids(manifest: FlagshipManifest, stage: str | None = None) -> tuple[str, ...]:
    stages = (stage,) if stage is not None else tuple(STAGE_DEPENDENCIES)
    return tuple(job.job_id for name in stages for job in iter_stage_jobs(manifest, name))


def assert_disjoint_job_seeds(jobs: Sequence[PlannedJob]) -> None:
    ids = {job.job_id for job in jobs}
    seeds = {job.seed for job in jobs}
    if len(ids) != len(jobs) or len(seeds) != len(jobs):
        raise ManifestError("planned jobs contain a duplicate ID or seed")


PRIVATE_OPE_SCHEMA_VERSION = "unseen-loop/private-ope-study-v1"
PRIVATE_OPE_METHOD = "UNCLIPPED_RATIO_LIFT_WPDIS_V1"
PRIVATE_OPE_BASELINE_IDS = (
    "is",
    "pdis",
    "wpdis",
    "clipped_wpdis_2",
    "clipped_wpdis_10",
    "dm",
    "dr",
    "wdr",
    "mis",
)


@dataclass(frozen=True, slots=True)
class PrivateOPEDomain:
    identifier: str
    queue_capacity: int
    initial_queue: int
    horizon: int
    gamma: float
    arrival_zero: float
    arrival_one: float
    arrival_two: float
    service_economy: float
    service_surge: float
    effort_cost: float
    overflow_cost: float
    reward_scale: float
    clear_trajectories: int
    cipher_trajectories: int
    legacy_trajectories: int
    legacy_horizon: int


@dataclass(frozen=True, slots=True)
class PrivateOPEPolicies:
    identifier: str
    a_surge_intercept: float
    a_surge_slope: float
    b_surge_intercept: float
    b_surge_slope: float
    behavior_a_fraction: float
    stress_primary_fraction: float
    primary_propensity_floor: float
    primary_ratio_bound: float
    stress_propensity_floor: float
    stress_ratio_bound: float
    policy_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrivateOPEStatistics:
    bootstrap_repetitions: int
    bootstrap_block_size: int
    interval_level: float
    quantile_method: str
    folds: int
    transition_prior_per_cell: float
    reward_prior_count: float
    reward_prior_sum: float
    bias_interval: str
    rmse_bootstrap_repetitions: int
    timing_bootstrap_repetitions: int
    normalization: str
    baseline_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrivateOPECrypto:
    security_bits: int
    poly_modulus_degree: int
    scale_bits: int
    first_prime_bits: int
    special_prime_bits: int
    raw_scale_bits: int
    raw_h64_depth: int
    legacy_scale_bits: int
    legacy_endpoint_bits: int
    legacy_h8_depth: int
    maximum_horizon: int
    range_reserve_bits: int
    ablation_horizons: tuple[int, ...]
    count_diagnostic_sizes: tuple[int, ...]
    count_replicas: int
    counts_source: str
    bootstrap_enabled: bool


@dataclass(frozen=True, slots=True)
class PrivateOPEExecution:
    app_name: str
    environment_name: str
    deployment_version: int
    code_commit: str
    candidate_code_sha256: str
    baseline_code_sha256: str
    domain_code_sha256: str
    analysis_code_sha256: str
    lockfile_sha256: str
    image_spec_sha256: str
    budget_envelope_usd: float
    budget_guard_sha256: str
    max_in_flight: int
    startup_timeout_s: int
    artifact_grace_s: int
    wave_timeout_s: int
    clear_timeout_s: int
    clear_deadline_s: int
    crypto_timeout_s: int
    crypto_deadline_s: int
    verification_timeout_s: int
    verification_deadline_s: int
    analysis_timeout_s: int
    analysis_deadline_s: int
    probe_timeout_s: int
    probe_deadline_s: int
    crypto_cpu: int
    crypto_memory_mib: int
    clear_cpu: int
    clear_memory_mib: int
    coordinator_cpu: float
    coordinator_memory_mib: int


@dataclass(frozen=True, slots=True)
class PrivateOPEGates:
    diagnostic_sum_abs_error: float
    verification_reuse_tolerance: float
    minimum_normalized_gap: float
    pilot_coverage_successes: int
    pilot_choice_successes: int
    median_ess_fraction: float
    p05_ess_fraction: float
    maximum_normalized_cipher_error: float
    maximum_cipher_error_se_fraction: float
    maximum_mean_statistic_abs_error: float
    maximum_denominator_relative_error: float
    minimum_median_speedup: float
    pilot_speedup_lower_exclusive: float
    confirmation_speedup_lower_inclusive: float
    confirmation_coverage_lower: float
    confirmation_choice_lower: float
    confirmation_bias_equivalence_margin: float
    confirmation_maximum_rmse: float
    maximum_peak_rss_gib: float
    maximum_context_gib_exclusive: float


@dataclass(frozen=True, slots=True)
class PrivateOPEPredecessors:
    historical_confirmation_id: str
    historical_summary_sha256: str
    historical_ledger_sha256: str
    previous_run_id: str
    previous_config_sha256: str
    previous_index_sha256: str
    pilot_kernel_sha256: str
    pilot_policies_sha256: str


@dataclass(frozen=True, slots=True)
class PrivateOPEManifest:
    schema_version: str
    method: str
    phase: str
    seed_root: str
    domain: PrivateOPEDomain
    policies: PrivateOPEPolicies
    statistics: PrivateOPEStatistics
    crypto: PrivateOPECrypto
    execution: PrivateOPEExecution
    gates: PrivateOPEGates
    predecessors: PrivateOPEPredecessors
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return scientific/configuration fields; digest binds the original TOML bytes."""
        payload = cast(dict[str, Any], _jsonable(self))
        payload.pop("digest")
        return payload


def private_ope_fixed_tables(phase: str) -> dict[str, Any]:
    """Return fresh frozen-value tables, without invented source or predecessor evidence.

    Execution deliberately omits its deployment/source/budget-guard fields. Callers must
    supply those verified values and the complete predecessors table before parsing.
    """
    if phase not in {"diagnostic", "pilot", "confirmation"}:
        raise ManifestError("unknown private OPE phase")
    return {
        "schema_version": PRIVATE_OPE_SCHEMA_VERSION,
        "method": PRIVATE_OPE_METHOD,
        "phase": phase,
        "seed_root": f"unseen-loop-private-ope-ratio-lift-v1-{phase}",
        "domain": _jsonable(
            PrivateOPEDomain(
                "QUEUE_POLICY_COMPARISON_V1",
                15,
                8,
                64,
                0.99,
                0.55,
                0.35,
                0.10,
                0.35,
                0.80,
                0.15,
                0.50,
                2.15,
                2048,
                4096,
                64,
                8,
            )
        ),
        "policies": _jsonable(
            PrivateOPEPolicies(
                "QUEUE_ORDERED_POLICIES_V1",
                0.20,
                0.50,
                0.30,
                0.50,
                0.50,
                0.50,
                0.25,
                1.20,
                0.375,
                1.28,
                ("A", "B"),
            )
        ),
        "statistics": _jsonable(
            PrivateOPEStatistics(
                1000,
                64,
                0.95,
                "linear",
                4,
                0.5,
                0.5,
                -0.25,
                "student_t",
                10000,
                10000,
                "discounted_unit_reward_bound",
                PRIVATE_OPE_BASELINE_IDS,
            )
        ),
        "crypto": _jsonable(
            PrivateOPECrypto(
                128,
                16384,
                40,
                60,
                58,
                32,
                10,
                24,
                40,
                14,
                64,
                2,
                (8, 32, 64),
                (64, 4096),
                3,
                "public_fixed_shape",
                False,
            )
        ),
        "execution": {
            "app_name": "unseen-loop-flagship",
            "environment_name": "main",
            "budget_envelope_usd": {"diagnostic": 10.0, "pilot": 30.0, "confirmation": 100.0}[
                phase
            ],
            "max_in_flight": 2,
            "startup_timeout_s": 300,
            "artifact_grace_s": 60,
            "wave_timeout_s": 79200,
            "clear_timeout_s": 600,
            "clear_deadline_s": 900,
            "crypto_timeout_s": 1800,
            "crypto_deadline_s": 2100,
            "verification_timeout_s": 7200,
            "verification_deadline_s": 7500,
            "analysis_timeout_s": 1800,
            "analysis_deadline_s": 2100,
            "probe_timeout_s": 2,
            "probe_deadline_s": 300,
            "crypto_cpu": 8,
            "crypto_memory_mib": 32768,
            "clear_cpu": 2,
            "clear_memory_mib": 8192,
            "coordinator_cpu": 0.5,
            "coordinator_memory_mib": 1024,
        },
        "gates": _jsonable(
            PrivateOPEGates(
                0.25,
                1e-9,
                0.01,
                58,
                55,
                0.10,
                0.02,
                0.001,
                0.10,
                0.001,
                0.01,
                1.25,
                1.0,
                1.10,
                0.90,
                0.85,
                0.005,
                0.02,
                24.0,
                1.5,
            )
        ),
    }


def parse_private_ope_manifest_bytes(data: bytes) -> PrivateOPEManifest:
    """Strictly parse the source-bound fixed study; this performs no empirical work."""
    try:
        root = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"invalid UTF-8 TOML manifest: {exc}") from exc
    if "digest" in root:
        raise ManifestError("digest is derived from exact input bytes, not supplied")
    manifest = _construct(
        PrivateOPEManifest,
        {**root, "digest": content_digest(data)},
        "private_ope",
        {
            "domain": PrivateOPEDomain,
            "policies": PrivateOPEPolicies,
            "statistics": PrivateOPEStatistics,
            "crypto": PrivateOPECrypto,
            "execution": PrivateOPEExecution,
            "gates": PrivateOPEGates,
            "predecessors": PrivateOPEPredecessors,
        },
    )
    _validate_private_ope_manifest(manifest)
    return manifest


def _private_digest(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestError(f"{name} must be a lowercase SHA-256 digest")


def _validate_private_ope_manifest(m: PrivateOPEManifest) -> None:
    frozen = private_ope_fixed_tables(m.phase)
    supplied = m.to_dict()
    for key, expected in frozen.items():
        if isinstance(expected, dict):
            for field, value in expected.items():
                if supplied[key][field] != value:
                    raise ManifestError(f"{key}.{field} is fixed for {PRIVATE_OPE_METHOD}")
        elif supplied[key] != expected:
            raise ManifestError(f"{key} is fixed for {PRIVATE_OPE_METHOD}")
    e, p = m.execution, m.predecessors
    if e.deployment_version <= 0 or re.fullmatch(r"[0-9a-f]{40}", e.code_commit) is None:
        raise ManifestError("execution requires an immutable commit and positive deployed version")
    for field in dataclasses.fields(e):
        if field.name.endswith("_sha256"):
            _private_digest(getattr(e, field.name), f"execution.{field.name}")
    historical = {
        "historical_confirmation_id": "independent-confirmation-20260904-001",
        "historical_summary_sha256": (
            "9233207bce45cdc8cd95fa026f2026571715185d055a0bcd83655d33c87d6b17"
        ),
        "historical_ledger_sha256": (
            "4227d8e154979d49031547a32c026d63f0feeebb7e79cef3e04607114f1f6885"
        ),
    }
    if any(getattr(p, key) != value for key, value in historical.items()):
        raise ManifestError("historical predecessor must match the immutable confirmation archive")
    previous = (p.previous_run_id, p.previous_config_sha256, p.previous_index_sha256)
    if m.phase == "diagnostic":
        if previous != ("not-applicable",) * 3:
            raise ManifestError("diagnostic predecessor fields must be not-applicable")
    else:
        phase = "diagnostic" if m.phase == "pilot" else "pilot"
        _private_digest(p.previous_config_sha256, "predecessors.previous_config_sha256")
        _private_digest(p.previous_index_sha256, "predecessors.previous_index_sha256")
        if p.previous_run_id != f"private-ope-{phase}-{p.previous_config_sha256[:24]}":
            raise ManifestError("predecessor ID must bind the exact preceding phase/config digest")
    for name in ("pilot_kernel_sha256", "pilot_policies_sha256"):
        value = getattr(p, name)
        if m.phase == "confirmation":
            _private_digest(value, f"predecessors.{name}")
        elif value != "not-applicable":
            raise ManifestError(f"predecessors.{name} is only applicable to confirmation")
    _validate_private_ope_arithmetic(m)
    jobs = iter_private_ope_jobs(m)
    expected_count = {"diagnostic": 16, "pilot": 156, "confirmation": 241}[m.phase]
    if len(jobs) != expected_count:
        raise ManifestError("private OPE plan denominator mismatch")
    _validate_private_ope_wave_bounds(m, jobs)


def _validate_private_ope_arithmetic(m: PrivateOPEManifest) -> None:
    """Algebraic checks only: no random draws, nuisance fitting or truth DP."""
    d, p, c = m.domain, m.policies, m.crypto
    if not math.isclose(d.arrival_zero + d.arrival_one + d.arrival_two, 1.0, abs_tol=1e-15):
        raise ManifestError("arrival probabilities must sum to one")
    if not math.isclose(1.0 + d.effort_cost + 2 * d.overflow_cost, d.reward_scale):
        raise ManifestError("reward normalization does not cover the queue domain")
    primary_floor = stress_floor = 1.0
    primary_ratio = stress_ratio = 0.0
    for x in (0.0, 1.0):
        a = p.a_surge_intercept + p.a_surge_slope * x
        b = p.b_surge_intercept + p.b_surge_slope * x
        mu = p.behavior_a_fraction * a + (1 - p.behavior_a_fraction) * b
        stress = p.stress_primary_fraction * mu + (1 - p.stress_primary_fraction) * 0.5
        primary_floor = min(primary_floor, mu, 1 - mu)
        stress_floor = min(stress_floor, stress, 1 - stress)
        primary_ratio = max(primary_ratio, a / mu, b / mu, (1 - a) / (1 - mu), (1 - b) / (1 - mu))
        stress_ratio = max(
            stress_ratio, a / stress, b / stress, (1 - a) / (1 - stress), (1 - b) / (1 - stress)
        )
    # Ratios of affine functions are monotone (or constant), so endpoints suffice.
    for observed, bound in (
        (primary_floor, p.primary_propensity_floor),
        (stress_floor, p.stress_propensity_floor),
        (primary_ratio, p.primary_ratio_bound),
        (stress_ratio, p.stress_ratio_bound),
    ):
        if not math.isclose(observed, bound, rel_tol=0.0, abs_tol=1e-14):
            raise ManifestError("policy/behavior support frontier disagrees with its frozen bound")
    for horizon, total in ((8, 318), (32, 398), (64, 438)):
        depth = (horizon - 1).bit_length() + 2
        bits = c.first_prime_bits + c.scale_bits * depth + c.special_prime_bits
        if bits != total or bits > 438:
            raise ManifestError("candidate chain violates frozen tc128 bit budget")
    if c.first_prime_bits + c.raw_scale_bits * c.raw_h64_depth + c.special_prime_bits != 438:
        raise ManifestError("raw prefix chain violates frozen tc128 bit budget")
    if 2 * c.legacy_endpoint_bits + c.legacy_scale_bits * c.legacy_h8_depth > 438:
        raise ManifestError("legacy chain violates tc128 bit budget")
    if p.primary_ratio_bound**d.horizon >= 2 ** (
        c.first_prime_bits - 1 - c.scale_bits - c.range_reserve_bits
    ):
        raise ManifestError("cumulative ratio violates the final-prime range reserve")
    # The protocol owns the complete exact-DAG intermediate proof, shared with runtime.
    from unseen_loop.ope.lifted import RatioLiftWPDISSpec
    from unseen_loop.ope.types import PolynomialPolicySpec, TrajectorySpec

    policies = tuple(
        PolynomialPolicySpec(
            coefficients=((1 - intercept, -slope), (intercept, slope)),
            action_count=2,
            state_dim=1,
            degree=1,
        )
        for intercept, slope in (
            (p.a_surge_intercept, p.a_surge_slope),
            (p.b_surge_intercept, p.b_surge_slope),
        )
    )
    for horizon in c.ablation_horizons:
        spec = RatioLiftWPDISSpec(
            trajectories=TrajectorySpec(
                trajectories=d.cipher_trajectories,
                horizon=horizon,
                state_dim=1,
                action_count=2,
                state_min=(0.0,),
                state_max=(1.0,),
                reward_min=-1.0,
                reward_max=0.0,
            ),
            target_policies=policies,
            gamma=d.gamma,
            minimum_behavior_propensity=p.primary_propensity_floor,
            maximum_importance_ratio=p.primary_ratio_bound,
        )
        spec.computation_receipt()


def iter_private_ope_jobs(manifest: PrivateOPEManifest) -> tuple[PlannedJob, ...]:
    """Materialize every fixed slot, preserving pair order and shared case seeds."""
    phase = manifest.phase
    if phase not in {"diagnostic", "pilot", "confirmation"}:
        raise ManifestError("unknown private OPE phase")
    jobs: list[PlannedJob] = []
    case_seeds: dict[str, tuple[int, int]] = {}

    def add(
        kind: str,
        cohort: str,
        index: int,
        n: int,
        h: int,
        behavior: str,
        arm: str,
        wave: int,
    ) -> None:
        coordinates: dict[str, str | int | float] = {
            "kind": kind,
            "cohort": cohort,
            "case_index": index,
            "trajectories": n,
            "horizon": h,
            "behavior": behavior,
            "arm": arm,
            "wave_index": wave,
        }
        identity = {
            "schema_version": PRIVATE_OPE_SCHEMA_VERSION,
            "seed_root": manifest.seed_root,
            "phase": phase,
            "coordinates": coordinates,
        }
        job_id = f"job-private_ope_{phase}-{content_digest(identity)[:24]}"
        if kind in {
            "clear_batch",
            "paired_context",
            "ablation_context",
            "historical_context",
            "statistical_context",
            "timing_context",
        }:
            case_identity = {
                "phase": phase,
                "cohort": cohort,
                "case_index": index,
                "trajectories": n,
                "horizon": h,
                "behavior": behavior,
            }
            case_id = f"case-{content_digest(case_identity)[:24]}"
            seeds = (
                derive_seed(manifest.seed_root, "data:" + case_id),
                derive_seed(manifest.seed_root, "bootstrap:" + case_id),
            )
            case_seeds[case_id] = seeds
            coordinates.update(
                case_id=case_id, data_seed=str(seeds[0]), bootstrap_seed=str(seeds[1])
            )
        else:
            coordinates.update(
                case_id="not-applicable",
                data_seed="not-applicable",
                bootstrap_seed="not-applicable",
            )
        jobs.append(
            PlannedJob(
                job_id,
                f"private_ope_{phase}",
                derive_seed(manifest.seed_root, job_id),
                tuple(sorted(coordinates.items())),
            )
        )

    if phase == "diagnostic":
        add("protocol_verification", "verification", 0, 0, 0, "none", "none", 0)
        for index in range(3):
            for n in (64, 4096):
                for arm, h in (("old24", 8), ("candidate40", 64)):
                    add("count_precision", "count", index, n, h, "none", arm, 0)
        for kind in ("smoke_error", "smoke_timeout"):
            add(kind, "infrastructure", 0, 0, 0, "none", "none", 0)
        final_wave = 0
    elif phase == "pilot":
        for cohort, behavior in (("screen_primary", "primary"), ("screen_stress", "stress")):
            for index in range(64):
                add("clear_batch", cohort, index, 2048, 64, behavior, "clear", 0)
        for index in range(10):
            arms = (
                ("lifted_prefix", "raw_prefix")
                if index % 2 == 0
                else ("raw_prefix", "lifted_prefix")
            )
            for arm in arms:
                add("paired_context", "timing", index, 4096, 64, "primary", arm, 1)
        for h in (8, 32, 64):
            for arm in ("lifted_prefix", "lifted_independent_products"):
                add("ablation_context", "ablation", 0, 4096, h, "primary", arm, 1)
        add("historical_context", "legacy", 0, 64, 8, "historical", "raw_sequential_v1", 1)
        final_wave = 1
    else:
        for index in range(200):
            add(
                "statistical_context",
                "statistical",
                index,
                4096,
                64,
                "primary",
                "lifted_prefix",
                index // 40,
            )
        for index in range(20):
            arms = (
                ("lifted_prefix", "raw_prefix")
                if index % 2 == 0
                else ("raw_prefix", "lifted_prefix")
            )
            for arm in arms:
                add("timing_context", "timing", index, 4096, 64, "primary", arm, 5 + index // 10)
        final_wave = 6
    add("analysis", "analysis", 0, 0, 0, "none", "none", final_wave)
    assert_disjoint_job_seeds(jobs)
    all_seeds = [job.seed for job in jobs]
    all_seeds.extend(seed for pair in case_seeds.values() for seed in pair)
    if len(all_seeds) != len(set(all_seeds)):
        raise ManifestError("private OPE job/data/bootstrap seed collision")
    # Frozen prior phases can be expanded without trusting or executing predecessor code.
    for prior in ("diagnostic", "pilot")[: {"diagnostic": 0, "pilot": 1, "confirmation": 2}[phase]]:
        predecessor = dataclasses.replace(
            manifest, phase=prior, seed_root=f"unseen-loop-private-ope-ratio-lift-v1-{prior}"
        )
        prior_jobs = iter_private_ope_jobs(predecessor)
        prior_seeds = {job.seed for job in prior_jobs}
        prior_seeds.update(
            int(value)
            for job in prior_jobs
            for name, value in job.coordinates
            if name in {"data_seed", "bootstrap_seed"} and value != "not-applicable"
        )
        if set(all_seeds) & prior_seeds:
            raise ManifestError("private OPE predecessor seed collision")
    return tuple(jobs)


def _validate_private_ope_wave_bounds(m: PrivateOPEManifest, jobs: Sequence[PlannedJob]) -> None:
    e = m.execution
    for worker in ("clear", "crypto", "verification", "analysis", "probe"):
        timeout = getattr(e, f"{worker}_timeout_s")
        deadline = getattr(e, f"{worker}_deadline_s")
        if timeout > deadline or (worker != "probe" and timeout + e.startup_timeout_s > deadline):
            raise ManifestError("worker deadline does not cover its execution/startup contract")
    waves: dict[int, list[dict[str, str | int | float]]] = {}
    for job in jobs:
        c = job.coordinate_dict()
        waves.setdefault(int(c["wave_index"]), []).append(c)
    for wave in waves.values():
        serial = parallel_clear = parallel_crypto = 0
        for c in wave:
            kind = c["kind"]
            if kind == "analysis":
                serial += e.analysis_deadline_s + e.artifact_grace_s
            elif kind == "protocol_verification":
                serial += e.verification_deadline_s + e.artifact_grace_s
            elif kind in {"paired_context", "timing_context"}:
                serial += e.crypto_deadline_s + e.artifact_grace_s
            elif kind == "smoke_timeout":
                serial += e.probe_deadline_s + e.artifact_grace_s
            elif kind in {"clear_batch", "smoke_error"}:
                parallel_clear += 1
            else:
                parallel_crypto += 1
        bound = (
            serial
            + math.ceil(parallel_clear / e.max_in_flight)
            * (e.clear_deadline_s + e.artifact_grace_s)
            + math.ceil(parallel_crypto / e.max_in_flight)
            * (e.crypto_deadline_s + e.artifact_grace_s)
        )
        if bound > e.wave_timeout_s:
            raise ManifestError("fixed wave cannot fit the absolute deadline-plus-grace contract")
