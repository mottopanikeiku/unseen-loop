"""Typed executor for the preregistered multi-environment release matrix."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from unseen_loop.artifacts import ArtifactLedger, dataclass_dict
from unseen_loop.experiment import Backend, ResearchPreset, run_experiment
from unseen_loop.search import SearchConfig


@dataclass(frozen=True)
class ReleaseSuiteConfig:
    """Executable subset of the immutable release preregistration."""

    schema_version: str
    name: str
    seed_root: str
    fhe_runtime: str
    security_level: int
    global_p_error: float
    stable_argmax: str
    environments: tuple[str, ...]
    checkpoints_per_environment: int
    selection_episodes: int
    evaluation_episodes: int
    degrees: tuple[int, ...]
    input_bits: tuple[int, ...]
    coefficient_bits: tuple[int, ...]
    ridge_values: tuple[float, ...]
    refinement_rounds: int
    calibration_padding: float
    teacher_population: int
    teacher_iterations: int
    episodes_per_candidate: int
    hidden_size: int
    minimum_certified_occupancy: float
    maximum_certified_mismatches: int
    certificate_weighting: bool = True
    student_occupancy_refinement: bool = True

    @property
    def expected_runs(self) -> int:
        return len(self.environments) * self.checkpoints_per_environment

    @property
    def candidates_per_run(self) -> int:
        return (
            len(self.degrees)
            * len(self.input_bits)
            * len(self.coefficient_bits)
            * len(self.ridge_values)
        )

    @property
    def expected_candidate_rows(self) -> int:
        return self.expected_runs * self.candidates_per_run

    @property
    def expected_selection_episodes(self) -> int:
        return self.expected_runs * self.candidates_per_run * self.selection_episodes

    @property
    def expected_paired_episodes(self) -> int:
        return self.expected_runs * self.evaluation_episodes

    @property
    def expected_episode_rows(self) -> int:
        return self.expected_paired_episodes * 2


@dataclass(frozen=True)
class _CompletedRun:
    row: dict[str, Any]
    paired_rows: tuple[dict[str, Any], ...]
    deltas: tuple[float, ...]
    suite_gates_passed: bool
    selection_episode_rows: int


def _table(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"release TOML field {key!r} must be a table")
    return value


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"release TOML field {key!r} must be a nonempty string")
    return value


def _integer(raw: Mapping[str, Any], key: str, *, minimum: int = 1) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"release TOML field {key!r} must be an integer >= {minimum}")
    return value


def _number(raw: Mapping[str, Any], key: str, *, positive: bool = True) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"release TOML field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"release TOML field {key!r} must be {qualifier}")
    return result


def _optional_boolean(raw: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"release TOML field {key!r} must be a boolean")
    return value


def _integer_tuple(raw: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"release TOML field {key!r} must be a nonempty integer array")
    result = tuple(
        item for item in value if isinstance(item, int) and not isinstance(item, bool) and item > 0
    )
    if len(result) != len(value) or len(set(result)) != len(result):
        raise ValueError(f"release TOML field {key!r} must contain unique positive integers")
    return result


def _number_tuple(raw: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"release TOML field {key!r} must be a nonempty numeric array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"release TOML field {key!r} must contain only numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"release TOML field {key!r} must contain finite positive numbers")
    if len(set(result)) != len(result):
        raise ValueError(f"release TOML field {key!r} must not contain duplicates")
    return result


def load_release_config(path: str | Path) -> tuple[ReleaseSuiteConfig, str]:
    """Parse and validate a release TOML, returning its exact source digest."""

    source = Path(path).read_bytes()
    raw = tomllib.loads(source.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("release TOML root must be a table")
    workloads = _table(raw, "workloads")
    search = _table(raw, "search")
    training = _table(_table(raw, "training"), "gpu")
    gates = _table(raw, "gates")

    environments_raw = workloads.get("environments")
    if not isinstance(environments_raw, list) or not environments_raw:
        raise ValueError("release TOML field 'environments' must be a nonempty string array")
    if any(not isinstance(item, str) or not item for item in environments_raw):
        raise ValueError("release TOML field 'environments' must contain nonempty strings")
    environments = tuple(environments_raw)
    if len(set(environments)) != len(environments):
        raise ValueError("release TOML environments must be unique")

    config = ReleaseSuiteConfig(
        schema_version=_string(raw, "schema_version"),
        name=_string(raw, "name"),
        seed_root=_string(raw, "seed_root"),
        fhe_runtime=_string(raw, "fhe_runtime"),
        security_level=_integer(raw, "security_level"),
        global_p_error=_number(raw, "global_p_error"),
        stable_argmax=_string(raw, "stable_argmax"),
        environments=environments,
        checkpoints_per_environment=_integer(workloads, "checkpoints_per_environment"),
        selection_episodes=_integer(workloads, "selection_episodes"),
        evaluation_episodes=_integer(workloads, "evaluation_episodes"),
        degrees=_integer_tuple(search, "degrees"),
        input_bits=_integer_tuple(search, "input_bits"),
        coefficient_bits=_integer_tuple(search, "coefficient_bits"),
        ridge_values=_number_tuple(search, "ridge"),
        refinement_rounds=_integer(search, "refinement_rounds", minimum=0),
        calibration_padding=_number(search, "calibration_padding"),
        teacher_population=_integer(training, "population"),
        teacher_iterations=_integer(training, "iterations"),
        episodes_per_candidate=_integer(training, "episodes_per_candidate"),
        hidden_size=_integer(training, "hidden_size"),
        minimum_certified_occupancy=_number(gates, "minimum_certified_occupancy"),
        maximum_certified_mismatches=_integer(gates, "maximum_certified_mismatches", minimum=0),
        certificate_weighting=_optional_boolean(search, "certificate_weighting", default=True),
        student_occupancy_refinement=_optional_boolean(
            search, "student_occupancy_refinement", default=True
        ),
    )
    if config.schema_version != "unseen-loop/release-suite-v1":
        raise ValueError("unsupported release config schema")
    if config.security_level < 128:
        raise ValueError("release security level must be at least 128 bits")
    if not 0.0 < config.global_p_error < 1.0:
        raise ValueError("release global_p_error must be between zero and one")
    if config.stable_argmax != "lowest-index":
        raise ValueError("release stable_argmax must be 'lowest-index'")
    if not 0.0 <= config.minimum_certified_occupancy <= 1.0:
        raise ValueError("minimum_certified_occupancy must be between zero and one")
    return config, hashlib.sha256(source).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"release artifact {path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"release artifact {path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def _validate_seed_plan(run_path: Path, config: ReleaseSuiteConfig) -> tuple[str, tuple[int, ...]]:
    seed_plan = _read_json_object(run_path / "seeds.json")
    selection = seed_plan.get("selection")
    evaluation = seed_plan.get("evaluation")
    if not isinstance(selection, list) or len(selection) != config.selection_episodes:
        raise RuntimeError("release run has the wrong selection seed count")
    if not isinstance(evaluation, list) or len(evaluation) != config.evaluation_episodes:
        raise RuntimeError("release run has the wrong evaluation seed count")
    training = seed_plan.get("training")
    if isinstance(training, bool) or not isinstance(training, int):
        raise RuntimeError("release run training seed is malformed")
    seed_groups: list[set[int]] = [{training}]
    for purpose in ("distillation", "refinement", "selection", "evaluation", "real_fhe"):
        values = seed_plan.get(purpose)
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise RuntimeError(f"release run seed namespace {purpose!r} is malformed")
        group = set(values)
        if len(group) != len(values):
            raise RuntimeError(f"release run seed namespace {purpose!r} contains duplicates")
        seed_groups.append(group)
    for index, left in enumerate(seed_groups):
        if any(left & right for right in seed_groups[index + 1 :]):
            raise RuntimeError("release run seed namespaces are not disjoint")
    namespace = seed_plan.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise RuntimeError("release run seed namespace label is missing")
    return namespace, tuple(selection)


def _paired_episode_rows(
    run_path: Path,
    *,
    environment: str,
    checkpoint_index: int,
    run_id: str,
    expected_pairs: int,
) -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl(run_path / "evaluation" / "episodes.jsonl")
    expected_modes = {"FLOAT TEACHER", "QUANTIZED CLEAR"}
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        mode = row.get("mode")
        seed = row.get("seed")
        total_return = row.get("total_return")
        if mode not in expected_modes:
            raise RuntimeError(f"release episode has unexpected mode: {mode!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError("release episode seed must be an integer")
        if isinstance(total_return, bool) or not isinstance(total_return, (int, float)):
            raise RuntimeError("release episode return must be numeric")
        if not math.isfinite(float(total_return)):
            raise RuntimeError("release episode return must be finite")
        modes = by_seed.setdefault(seed, {})
        if mode in modes:
            raise RuntimeError("release episode mode/seed pair is duplicated")
        modes[mode] = row
    if len(rows) != expected_pairs * 2 or len(by_seed) != expected_pairs:
        raise RuntimeError("release run has the wrong number of retained evaluation rows")

    paired: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        modes = by_seed[seed]
        if set(modes) != expected_modes:
            raise RuntimeError("paired release episode modes are incomplete")
        teacher = modes["FLOAT TEACHER"]
        student = modes["QUANTIZED CLEAR"]
        teacher_return = float(teacher["total_return"])
        student_return = float(student["total_return"])
        paired.append(
            {
                "environment": environment,
                "checkpoint_index": checkpoint_index,
                "run_id": run_id,
                "seed": seed,
                "teacher_return": teacher_return,
                "integer_student_return": student_return,
                "paired_return_delta": student_return - teacher_return,
                "teacher_constraint_cost": float(teacher["constraint_cost"]),
                "integer_student_constraint_cost": float(student["constraint_cost"]),
                "teacher_action_digest": str(teacher["action_digest"]),
                "integer_student_action_digest": str(student["action_digest"]),
            }
        )
    return tuple(paired)


def _validate_selection_episode_rows(
    run_path: Path,
    *,
    candidate_metrics: Mapping[str, tuple[float, float, bool]],
    selection_seeds: tuple[int, ...],
) -> int:
    rows = _read_jsonl(run_path / "search" / "selection-episodes.jsonl")
    expected_keys = {
        (candidate_digest, seed)
        for candidate_digest in candidate_metrics
        for seed in selection_seeds
    }
    observed_keys: set[tuple[str, int]] = set()
    totals = {
        candidate_digest: {
            "steps": 0,
            "teacher_agreement_count": 0,
            "certified_count": 0,
            "certified_mismatch_count": 0,
            "saturation_count": 0,
        }
        for candidate_digest in candidate_metrics
    }
    for row in rows:
        candidate_digest = row.get("candidate_digest")
        seed = row.get("seed")
        if not isinstance(candidate_digest, str) or not candidate_digest:
            raise RuntimeError("release selection episode candidate digest is malformed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError("release selection episode seed is malformed")
        key = (candidate_digest, seed)
        if key not in expected_keys:
            raise RuntimeError("release selection episode candidate/seed pair is unexpected")
        if key in observed_keys:
            raise RuntimeError("release selection episode candidate/seed pair is duplicated")
        observed_keys.add(key)
        if row.get("mode") != "QUANTIZED CLEAR SELECTION":
            raise RuntimeError("release selection episode mode is malformed")
        for field in ("total_return", "constraint_cost"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"release selection episode {field} is malformed")
        counters: dict[str, int] = {}
        for field in totals[candidate_digest]:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"release selection episode {field} is malformed")
            counters[field] = value
        steps = counters["steps"]
        if steps < 1 or any(
            counters[field] > steps
            for field in (
                "teacher_agreement_count",
                "certified_count",
                "certified_mismatch_count",
                "saturation_count",
            )
        ):
            raise RuntimeError("release selection episode counters exceed episode steps")
        if counters["certified_mismatch_count"] > counters["certified_count"]:
            raise RuntimeError("release certified mismatch count exceeds certified count")
        range_valid = row.get("range_valid")
        if not isinstance(range_valid, bool) or range_valid is not (
            counters["saturation_count"] == 0
        ):
            raise RuntimeError("release selection episode range_valid is inconsistent")
        action_digest = row.get("action_digest")
        if not isinstance(action_digest, str) or not action_digest:
            raise RuntimeError("release selection episode action digest is malformed")
        for field, value in counters.items():
            totals[candidate_digest][field] += value
    if observed_keys != expected_keys:
        raise RuntimeError("release selection episode candidate/seed pairs are incomplete")
    for candidate_digest, counts in totals.items():
        teacher_agreement, certified_coverage, range_valid = candidate_metrics[candidate_digest]
        steps = counts["steps"]
        if not math.isclose(
            counts["teacher_agreement_count"] / steps,
            teacher_agreement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("release candidate agreement disagrees with selection episode rows")
        if not math.isclose(
            counts["certified_count"] / steps,
            certified_coverage,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("release candidate coverage disagrees with selection episode rows")
        if range_valid is not (counts["saturation_count"] == 0):
            raise RuntimeError(
                "release candidate range_valid disagrees with selection episode rows"
            )
    return len(rows)


def _validate_completed_run(
    run_path: Path,
    *,
    destination: Path,
    environment: str,
    checkpoint_index: int,
    backend: Backend,
    config: ReleaseSuiteConfig,
    config_digest: str,
    returned_summary: dict[str, Any],
) -> _CompletedRun:
    valid, failures = ArtifactLedger(run_path).verify()
    if not valid:
        raise RuntimeError(f"release child ledger failed: {failures}")
    recorded_summary = _read_json_object(run_path / "summary.json")
    if recorded_summary != returned_summary:
        raise RuntimeError("release child returned summary differs from its recorded summary")
    run_id = f"{environment}--checkpoint-{checkpoint_index:02d}"
    if (
        recorded_summary.get("run_id") != run_id
        or recorded_summary.get("env_id") != environment
        or recorded_summary.get("backend") != backend
    ):
        raise RuntimeError("release child summary identity is inconsistent")

    seed_namespace, selection_seeds = _validate_seed_plan(run_path, config)
    paired_rows = _paired_episode_rows(
        run_path,
        environment=environment,
        checkpoint_index=checkpoint_index,
        run_id=run_id,
        expected_pairs=config.evaluation_episodes,
    )
    expected_summary_values = {
        "teacher_return_mean": math.fsum(float(row["teacher_return"]) for row in paired_rows)
        / len(paired_rows),
        "champion_return_mean": math.fsum(
            float(row["integer_student_return"]) for row in paired_rows
        )
        / len(paired_rows),
        "champion_return_delta": math.fsum(float(row["paired_return_delta"]) for row in paired_rows)
        / len(paired_rows),
    }
    for field, expected in expected_summary_values.items():
        value = recorded_summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError(
                f"release child summary {field} disagrees with retained episode rows"
            )
    candidates = _read_jsonl(run_path / "search" / "candidates.jsonl")
    if len(candidates) != config.candidates_per_run:
        raise RuntimeError("release run has the wrong number of retained candidate rows")
    if recorded_summary.get("candidates") != config.candidates_per_run:
        raise RuntimeError("release child summary candidate count is inconsistent")
    champion_digest = recorded_summary.get("champion_policy_digest")
    champions = [
        row
        for row in candidates
        if isinstance(row.get("metrics"), dict)
        and row["metrics"].get("policy_digest") == champion_digest
    ]
    if len(champions) != 1 or champions[0]["metrics"].get("range_valid") is not True:
        raise RuntimeError("release champion is missing or range-invalid")
    candidate_metrics: dict[str, tuple[float, float, bool]] = {}
    for candidate in candidates:
        metrics = candidate.get("metrics")
        candidate_digest = metrics.get("policy_digest") if isinstance(metrics, dict) else None
        range_valid = metrics.get("range_valid") if isinstance(metrics, dict) else None
        teacher_agreement = metrics.get("teacher_agreement") if isinstance(metrics, dict) else None
        certified_coverage = (
            metrics.get("certified_coverage") if isinstance(metrics, dict) else None
        )
        if not isinstance(candidate_digest, str) or not candidate_digest:
            raise RuntimeError("release candidate policy digest is malformed")
        if not isinstance(range_valid, bool):
            raise RuntimeError("release candidate range_valid is malformed")
        for field, value in (
            ("teacher_agreement", teacher_agreement),
            ("certified_coverage", certified_coverage),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RuntimeError(f"release candidate {field} is malformed")
        if candidate_digest in candidate_metrics:
            raise RuntimeError("release candidate policy digests are duplicated")
        candidate_metrics[candidate_digest] = (
            float(cast(float, teacher_agreement)),
            float(cast(float, certified_coverage)),
            range_valid,
        )
    selection_episode_rows = _validate_selection_episode_rows(
        run_path,
        candidate_metrics=candidate_metrics,
        selection_seeds=selection_seeds,
    )

    certificate = _read_json_object(run_path / "certificates" / "heldout.json")
    coverage = certificate.get("coverage")
    certified_mismatches = certificate.get("certified_mismatches")
    if (
        not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or not math.isfinite(float(coverage))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise RuntimeError("release held-out certificate coverage is malformed")
    summary_coverage = recorded_summary.get("certified_coverage")
    if (
        isinstance(summary_coverage, bool)
        or not isinstance(summary_coverage, (int, float))
        or not math.isclose(float(coverage), float(summary_coverage), rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise RuntimeError("release summary and held-out certificate coverage disagree")
    if not isinstance(certified_mismatches, int) or isinstance(certified_mismatches, bool):
        raise RuntimeError("release held-out certified mismatch count is malformed")
    gates_passed = (
        float(coverage) >= config.minimum_certified_occupancy
        and certified_mismatches <= config.maximum_certified_mismatches
    )
    manifest = run_path / "checksums.sha256"
    row = {
        "environment": environment,
        "checkpoint_index": checkpoint_index,
        "run_id": run_id,
        "path": run_path.relative_to(destination).as_posix(),
        "config_sha256": config_digest,
        "seed_namespace": seed_namespace,
        "selection_episodes": config.selection_episodes,
        "evaluation_episodes": config.evaluation_episodes,
        "retained_selection_episode_rows": selection_episode_rows,
        "certificate_weighting": config.certificate_weighting,
        "student_occupancy_refinement": config.student_occupancy_refinement,
        "retained_episode_rows": len(paired_rows) * 2,
        "child_ledger_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "suite_gates_passed": gates_passed,
        "suite_gates": {
            "certified_occupancy": {
                "observed": float(coverage),
                "minimum": config.minimum_certified_occupancy,
                "passed": float(coverage) >= config.minimum_certified_occupancy,
            },
            "certified_mismatches": {
                "observed": certified_mismatches,
                "maximum": config.maximum_certified_mismatches,
                "passed": (certified_mismatches <= config.maximum_certified_mismatches),
            },
        },
        "summary": recorded_summary,
    }
    return _CompletedRun(
        row=row,
        paired_rows=paired_rows,
        deltas=tuple(float(item["paired_return_delta"]) for item in paired_rows),
        suite_gates_passed=gates_passed,
        selection_episode_rows=selection_episode_rows,
    )


def _hierarchical_interval(
    deltas_by_environment: Mapping[str, Sequence[Sequence[float]]],
    *,
    seed: int,
    repetitions: int = 10_000,
) -> tuple[float, float, float]:
    """Bootstrap tasks, checkpoints within tasks, then paired episodes."""

    if repetitions < 1 or not deltas_by_environment:
        raise ValueError("hierarchical interval requires tasks and positive repetitions")
    task_values = tuple(
        tuple(tuple(run) for run in runs) for runs in deltas_by_environment.values()
    )
    if any(not runs or any(not run for run in runs) for runs in task_values):
        raise ValueError("hierarchical interval requires populated checkpoint rows")
    observed = float(np.mean([np.mean([np.mean(run) for run in runs]) for runs in task_values]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        task_indices = rng.integers(0, len(task_values), size=len(task_values))
        task_means: list[float] = []
        for task_index in task_indices:
            runs = task_values[int(task_index)]
            checkpoint_indices = rng.integers(0, len(runs), size=len(runs))
            checkpoint_means: list[float] = []
            for checkpoint_index in checkpoint_indices:
                values = np.asarray(runs[int(checkpoint_index)], dtype=np.float64)
                episode_indices = rng.integers(0, len(values), size=len(values))
                checkpoint_means.append(float(np.mean(values[episode_indices])))
            task_means.append(float(np.mean(checkpoint_means)))
        estimates[repetition] = float(np.mean(task_means))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return observed, float(low), float(high)


def _finalize_transitive_ledger(ledger: ArtifactLedger, runs_root: Path) -> None:
    """Seal suite files plus every child file into one verifiable manifest."""

    manifest = ledger.finalize()
    checksums = manifest.read_text().splitlines()
    for child in sorted(path for path in runs_root.rglob("*") if path.is_file()):
        relative = child.relative_to(ledger.root)
        with child.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        checksums.append(f"{digest}  {relative.as_posix()}")
    ledger.write_text("checksums.sha256", "\n".join(sorted(checksums)) + "\n")


def run_release_suite(
    *,
    config_path: str | Path,
    output: str | Path,
    backend: Backend = "clear",
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    """Execute and aggregate every environment/checkpoint declared by the TOML."""

    config, config_digest = load_release_config(config_path)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("release suite output directory must be absent or empty")
    runs_root = destination / "runs"
    preset = ResearchPreset(
        full=True,
        teacher_iterations=config.teacher_iterations,
        teacher_population=config.teacher_population,
        episodes_per_candidate=config.episodes_per_candidate,
        selection_episodes=config.selection_episodes,
        evaluation_episodes=config.evaluation_episodes,
        hidden_size=config.hidden_size,
        search=SearchConfig(
            degrees=config.degrees,
            input_bits=config.input_bits,
            coefficient_bits=config.coefficient_bits,
            ridge_values=config.ridge_values,
            refinement_rounds=config.refinement_rounds,
            calibration_padding=config.calibration_padding,
            certificate_weighting=config.certificate_weighting,
            student_occupancy_refinement=config.student_occupancy_refinement,
            global_p_error=config.global_p_error,
        ),
    )
    completed: list[_CompletedRun] = []
    deltas_by_environment: dict[str, list[tuple[float, ...]]] = {
        environment: [] for environment in config.environments
    }
    for environment in config.environments:
        for checkpoint_index in range(config.checkpoints_per_environment):
            run_id = f"{environment}--checkpoint-{checkpoint_index:02d}"
            run_path = runs_root / run_id
            experiment_summary = run_experiment(
                env_id=environment,
                output=run_path,
                backend=backend,
                preset=preset,
                seed_root=(f"{config.seed_root}|{environment}|checkpoint-{checkpoint_index:02d}"),
                run_id=run_id,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
            result = _validate_completed_run(
                run_path,
                destination=destination,
                environment=environment,
                checkpoint_index=checkpoint_index,
                backend=backend,
                config=config,
                config_digest=config_digest,
                returned_summary=dataclass_dict(experiment_summary),
            )
            completed.append(result)
            deltas_by_environment[environment].append(result.deltas)

    if len(completed) != config.expected_runs:
        raise RuntimeError("release suite did not materialize every preregistered run")
    paired_rows = [row for result in completed for row in result.paired_rows]
    if len(paired_rows) != config.expected_paired_episodes:
        raise RuntimeError("release suite retained paired rows are incomplete")
    selection_episode_rows = sum(result.selection_episode_rows for result in completed)
    if selection_episode_rows != config.expected_selection_episodes:
        raise RuntimeError("release suite retained selection rows are incomplete")
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"{config.seed_root}|paired-return-bootstrap-v1".encode()).digest()[:8],
        "little",
    )
    observed, interval_low, interval_high = _hierarchical_interval(
        deltas_by_environment, seed=bootstrap_seed
    )
    summary: dict[str, Any] = {
        "schema_version": "unseen-loop/release-suite-v1",
        "name": config.name,
        "config_sha256": config_digest,
        "backend": backend,
        "expected_runs": config.expected_runs,
        "completed_runs": len(completed),
        "candidates_per_run": config.candidates_per_run,
        "expected_candidate_rows": config.expected_candidate_rows,
        "expected_selection_episodes": config.expected_selection_episodes,
        "retained_selection_episode_rows": selection_episode_rows,
        "selection_evidence_scope": "complete candidate-by-seed long-form episode rows",
        "environments": list(config.environments),
        "checkpoints_per_environment": config.checkpoints_per_environment,
        "selection_episodes_per_candidate": config.selection_episodes,
        "evaluation_episodes_per_checkpoint": config.evaluation_episodes,
        "certificate_weighting": config.certificate_weighting,
        "student_occupancy_refinement": config.student_occupancy_refinement,
        "expected_paired_episodes": config.expected_paired_episodes,
        "retained_paired_episodes": len(paired_rows),
        "retained_episode_rows": len(paired_rows) * 2,
        "all_runs_complete": True,
        "all_champions_range_valid": True,
        "all_suite_gates_passed": all(result.suite_gates_passed for result in completed),
        "suite_gate_scope": [
            "minimum_certified_occupancy",
            "maximum_certified_mismatches",
        ],
        "paired_return_delta": {
            "mean": observed,
            "ci95_low": interval_low,
            "ci95_high": interval_high,
            "method": ("tasks; checkpoints within tasks; paired episodes within checkpoints"),
            "repetitions": 10_000,
            "seed": bootstrap_seed,
            "seed_namespace": f"{config.seed_root}|paired-return-bootstrap-v1",
        },
    }
    ledger = ArtifactLedger(destination)
    ledger.write_json("suite-summary.json", summary)
    ledger.write_jsonl("suite-runs.jsonl", (result.row for result in completed))
    ledger.write_jsonl("suite-episodes.jsonl", paired_rows)
    ledger.write_bytes("release.toml", Path(config_path).read_bytes())
    _finalize_transitive_ledger(ledger, runs_root)
    valid, failures = ledger.verify()
    if not valid:
        raise RuntimeError(f"release suite ledger failed: {failures}")
    return summary
