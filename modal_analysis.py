"""Modal-only aggregation of the checksummed unseen-loop release evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop-release-analysis"
VOLUME_NAME = "unseen-loop-artifacts"
STUDY_ROOT = Path("/artifacts/studies")
OUTPUT_ID = "unseen-loop-release-analysis-001"
OUTPUT_ROOT = STUDY_ROOT / OUTPUT_ID
EXPANDED_ID = "expanded-multitask-modal-001"
ABLATION_PREFIX = "expanded-cartpole-ablation-modal-003--"
ABLATION_SUFFIXES = (
    "ablation-cartpole-unweighted-refined",
    "ablation-cartpole-unweighted-unrefined",
    "ablation-cartpole-weighted-refined",
    "ablation-cartpole-weighted-unrefined",
)
NONLINEAR_ID = "modal-nonlinear-qmax2-002"
TIMING_ID = "modal-fhe-timing-003"
BOOTSTRAP_REPETITIONS = 10_000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
analysis_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==1.26.4")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} must be a nonempty string")
    return value


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{message}: observed {actual!r}, expected {expected!r}")


def _verify_ledger(root: Path) -> dict[str, Any]:
    """Verify every byte and require the manifest to cover every other file exactly."""
    manifest = root / "checksums.sha256"
    if not root.is_dir() or not manifest.is_file():
        raise RuntimeError(f"checksummed source bundle is missing: {root}")
    expected: dict[Path, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative_text = line.partition("  ")
        relative = Path(relative_text)
        if (
            not separator
            or SHA256_RE.fullmatch(digest) is None
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative == Path("checksums.sha256")
            or relative in expected
        ):
            raise RuntimeError(f"malformed or duplicate ledger row {manifest}:{line_number}")
        expected[relative] = digest
    actual: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"source bundle contains a symlink: {path}")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(root))
    if set(expected) != actual:
        missing = sorted(item.as_posix() for item in set(expected) - actual)
        extra = sorted(item.as_posix() for item in actual - set(expected))
        raise RuntimeError(f"ledger closure failed for {root}: missing={missing}, extra={extra}")
    for relative, expected_digest in expected.items():
        if _file_sha256(root / relative) != expected_digest:
            raise RuntimeError(f"checksum mismatch: {root / relative}")
    return {
        "path": str(root),
        "ledger_sha256": _file_sha256(manifest),
        "ledgered_files": len(expected),
        "failures": 0,
    }


def _suite_root(study_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    study = STUDY_ROOT / study_id
    outer = _object(study / "modal-study-summary.json")
    _require_equal(outer.get("study_id"), study_id, "Modal study identity mismatch")
    execution = outer.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError(f"{study_id} is missing its execution scope")
    _require_equal(execution.get("backend"), "clear", "suite backend must be clear")
    _require_equal(execution.get("privacy_claim"), "none", "clear suite privacy scope mismatch")
    suite = study / "suite"
    ledger = _verify_ledger(suite)
    recorded = _object(suite / "suite-summary.json")
    _require_equal(
        outer.get("suite_summary"), recorded, "outer and persisted suite summaries differ"
    )
    config_meta = outer.get("config")
    if not isinstance(config_meta, dict):
        raise RuntimeError(f"{study_id} lacks config provenance")
    _require_equal(
        config_meta.get("sha256"),
        _file_sha256(suite / "release.toml"),
        "outer config digest mismatch",
    )
    _require_equal(
        recorded.get("config_sha256"), config_meta.get("sha256"), "suite config digest mismatch"
    )
    return suite, outer, ledger


def _config(suite: Path) -> dict[str, Any]:
    value = tomllib.loads((suite / "release.toml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("release.toml root must be a table")
    return value


def _table(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"release config field {key!r} must be a table")
    return value


def _expected_counts(config: Mapping[str, Any]) -> dict[str, int]:
    workloads = _table(config, "workloads")
    search = _table(config, "search")
    environments = workloads.get("environments")
    if (
        not isinstance(environments, list)
        or not environments
        or not all(isinstance(value, str) and value for value in environments)
    ):
        raise RuntimeError("release environments are malformed")
    checkpoints = _integer(workloads.get("checkpoints_per_environment"), "checkpoints", minimum=1)
    selection = _integer(workloads.get("selection_episodes"), "selection episodes", minimum=1)
    evaluation = _integer(workloads.get("evaluation_episodes"), "evaluation episodes", minimum=1)
    grid_lengths: list[int] = []
    for key in ("degrees", "input_bits", "coefficient_bits", "ridge"):
        values = search.get(key)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"search grid {key!r} is malformed")
        grid_lengths.append(len(values))
    candidates = math.prod(grid_lengths)
    runs = len(environments) * checkpoints
    return {
        "runs": runs,
        "candidates_per_run": candidates,
        "candidate_rows": runs * candidates,
        "selection_rows": runs * candidates * selection,
        "paired_episodes": runs * evaluation,
        "episode_rows": runs * evaluation * 2,
        "selection_per_candidate": selection,
        "evaluation_per_checkpoint": evaluation,
        "checkpoints_per_environment": checkpoints,
    }


def _ratio_count(ratio: Any, denominator: int, name: str) -> int:
    value = _number(ratio, name)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"{name} must lie in [0, 1]")
    count = round(value * denominator)
    if not math.isclose(value, count / denominator, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{name} does not encode an exact count over denominator {denominator}")
    return count


def _describe(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("a descriptive statistic received missing or nonfinite values")
    q1, median, q3 = np.quantile(array, (0.25, 0.5, 0.75))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=0)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{OUTPUT_ID}|{label}".encode()).digest()[:8], "little")


def _interval(estimates: Any) -> dict[str, float]:
    import numpy as np

    low, high = np.quantile(estimates, (0.025, 0.975))
    return {"ci95_low": float(low), "ci95_high": float(high)}


def _episode_interval(values: Sequence[float], label: str) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(_seed(label))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPETITIONS, len(array)))
    result = {
        "estimate": float(np.mean(array)),
        **_interval(np.mean(array[indices], axis=1)),
        "method": "paired-episode percentile bootstrap",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": _seed(label),
    }
    return result


def _checkpoint_episode_interval(runs: Sequence[Sequence[float]], label: str) -> dict[str, Any]:
    import numpy as np

    arrays = tuple(np.asarray(run, dtype=np.float64) for run in runs)
    if not arrays or any(array.ndim != 1 or array.size == 0 for array in arrays):
        raise RuntimeError(
            "checkpoint-to-episode bootstrap requires populated one-dimensional runs"
        )
    rng = np.random.default_rng(_seed(label))
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        checkpoint_indices = rng.integers(0, len(arrays), size=len(arrays))
        checkpoint_means: list[float] = []
        for checkpoint_index in checkpoint_indices:
            values = arrays[int(checkpoint_index)]
            episode_indices = rng.integers(0, len(values), size=len(values))
            checkpoint_means.append(float(np.mean(values[episode_indices])))
        estimates[repetition] = float(np.mean(checkpoint_means))
    return {
        "estimate": float(np.mean([np.mean(array) for array in arrays])),
        **_interval(estimates),
        "method": "checkpoint then paired-episode percentile bootstrap",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": _seed(label),
    }


def _policy_digest(policy: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(policy).encode("utf-8"))


def _child_data(
    suite: Path,
    run_row: Mapping[str, Any],
    *,
    expected_candidates: int,
    expected_selection: int,
    expected_evaluation: int,
) -> dict[str, Any]:
    relative = Path(_string(run_row.get("path"), "run path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("suite run path escapes its source bundle")
    run = suite / relative
    child_ledger = _verify_ledger(run)
    _require_equal(
        run_row.get("child_ledger_sha256"),
        child_ledger["ledger_sha256"],
        "child ledger digest mismatch",
    )
    summary = _object(run / "summary.json")
    seeds = _object(run / "seeds.json")
    config = _object(run / "config.json")
    candidates = _jsonl(run / "search" / "candidates.jsonl")
    selection_rows = _jsonl(run / "search" / "selection-episodes.jsonl")
    evaluation_rows = _jsonl(run / "evaluation" / "episodes.jsonl")
    certificate = _object(run / "certificates" / "heldout.json")
    _require_equal(len(candidates), expected_candidates, "candidate denominator mismatch")
    _require_equal(
        len(selection_rows),
        expected_candidates * expected_selection,
        "selection denominator mismatch",
    )
    _require_equal(len(evaluation_rows), 2 * expected_evaluation, "evaluation denominator mismatch")
    selection_seeds = seeds.get("selection")
    evaluation_seeds = seeds.get("evaluation")
    if not isinstance(selection_seeds, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in selection_seeds
    ):
        raise RuntimeError("selection seed plan is malformed")
    if not isinstance(evaluation_seeds, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in evaluation_seeds
    ):
        raise RuntimeError("evaluation seed plan is malformed")
    _require_equal(len(selection_seeds), expected_selection, "selection seed denominator mismatch")
    _require_equal(
        len(evaluation_seeds), expected_evaluation, "evaluation seed denominator mismatch"
    )
    _require_equal(
        len(set(selection_seeds)), len(selection_seeds), "selection seeds are duplicated"
    )
    _require_equal(
        len(set(evaluation_seeds)), len(evaluation_seeds), "evaluation seeds are duplicated"
    )
    if set(selection_seeds) & set(evaluation_seeds):
        raise RuntimeError("selection and evaluation seed namespaces overlap")

    champion_digest = _string(summary.get("champion_policy_digest"), "champion policy digest")
    champion_candidates = [
        row
        for row in candidates
        if isinstance(row.get("metrics"), dict)
        and row["metrics"].get("policy_digest") == champion_digest
    ]
    _require_equal(len(champion_candidates), 1, "champion candidate row mismatch")
    champion = champion_candidates[0]
    metrics = champion["metrics"]
    if metrics.get("range_valid") is not True:
        raise RuntimeError("champion is not range valid")
    policy_path = run / "policies" / f"{champion_digest}.json"
    policy = _object(policy_path)
    _require_equal(
        _policy_digest(policy), champion_digest, "champion policy content digest mismatch"
    )
    _require_equal(policy.get("degree"), metrics.get("degree"), "champion degree mismatch")

    selection_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in selection_rows:
        digest = _string(row.get("candidate_digest"), "selection candidate digest")
        seed_value = _integer(row.get("seed"), "selection seed")
        key = (digest, seed_value)
        if key in selection_by_key:
            raise RuntimeError("duplicate candidate/selection-seed row")
        selection_by_key[key] = row
    expected_selection_keys = {
        (_string(row["metrics"].get("policy_digest"), "candidate digest"), seed_value)
        for row in candidates
        for seed_value in selection_seeds
        if isinstance(row.get("metrics"), dict)
    }
    _require_equal(
        set(selection_by_key), expected_selection_keys, "candidate selection keys mismatch"
    )
    champion_selection = [
        selection_by_key[(champion_digest, seed_value)] for seed_value in selection_seeds
    ]
    selection_steps = 0
    selection_certified = 0
    selection_mismatches = 0
    selection_saturations = 0
    for row in champion_selection:
        steps = _integer(row.get("steps"), "selection steps", minimum=1)
        certified = _integer(row.get("certified_count"), "selection certified count")
        mismatches = _integer(row.get("certified_mismatch_count"), "selection mismatch count")
        saturations = _integer(row.get("saturation_count"), "selection saturation count")
        if certified > steps or mismatches > certified or saturations > steps:
            raise RuntimeError("selection counters exceed their verified denominator")
        selection_steps += steps
        selection_certified += certified
        selection_mismatches += mismatches
        selection_saturations += saturations

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in evaluation_rows:
        seed_value = _integer(row.get("seed"), "evaluation row seed")
        mode = _string(row.get("mode"), "evaluation mode")
        if mode not in {"FLOAT TEACHER", "QUANTIZED CLEAR"}:
            raise RuntimeError("evaluation row has an unexpected execution mode")
        modes = by_seed.setdefault(seed_value, {})
        if mode in modes:
            raise RuntimeError("duplicate evaluation mode/seed row")
        for field in ("total_return", "constraint_cost"):
            _number(row.get(field), f"evaluation {field}")
        modes[mode] = row
    _require_equal(set(by_seed), set(evaluation_seeds), "evaluation row seed keys mismatch")
    if any(set(modes) != {"FLOAT TEACHER", "QUANTIZED CLEAR"} for modes in by_seed.values()):
        raise RuntimeError("paired evaluation rows are incomplete")
    paired: list[dict[str, Any]] = []
    for seed_value in evaluation_seeds:
        teacher = by_seed[seed_value]["FLOAT TEACHER"]
        student = by_seed[seed_value]["QUANTIZED CLEAR"]
        _require_equal(
            teacher.get("policy_digest"), summary.get("teacher_digest"), "teacher digest mismatch"
        )
        _require_equal(student.get("policy_digest"), champion_digest, "student digest mismatch")
        teacher_return = _number(teacher.get("total_return"), "teacher return")
        student_return = _number(student.get("total_return"), "student return")
        paired.append(
            {
                "seed": seed_value,
                "teacher_return": teacher_return,
                "student_return": student_return,
                "return_delta": student_return - teacher_return,
                "teacher_cost": _number(teacher.get("constraint_cost"), "teacher cost"),
                "student_cost": _number(student.get("constraint_cost"), "student cost"),
                "teacher_action_digest": _string(
                    teacher.get("action_digest"), "teacher action digest"
                ),
                "student_action_digest": _string(
                    student.get("action_digest"), "student action digest"
                ),
            }
        )
    teacher_mean = math.fsum(row["teacher_return"] for row in paired) / len(paired)
    student_mean = math.fsum(row["student_return"] for row in paired) / len(paired)
    if not math.isclose(
        _number(summary.get("teacher_return_mean"), "teacher mean"),
        teacher_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("teacher summary mean disagrees with raw evaluation rows")
    if not math.isclose(
        _number(summary.get("champion_return_mean"), "student mean"),
        student_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("student summary mean disagrees with raw evaluation rows")

    observations = _integer(certificate.get("observations"), "held-out observations", minimum=1)
    certified_count = _ratio_count(
        certificate.get("coverage"), observations, "certificate coverage"
    )
    _require_equal(
        certificate.get("policy_digest"), champion_digest, "certificate policy digest mismatch"
    )
    _require_equal(
        certificate.get("coverage"),
        summary.get("certified_coverage"),
        "certificate summary mismatch",
    )
    certificate_mismatches = _integer(
        certificate.get("certified_mismatches"), "certified mismatches"
    )
    total_float_integer_mismatches = _integer(
        certificate.get("mismatches"), "float/integer mismatches"
    )
    agreement_count = _ratio_count(
        summary.get("teacher_agreement"), observations, "teacher agreement"
    )
    return {
        "path": run,
        "summary": summary,
        "seeds": seeds,
        "config": config,
        "paired": paired,
        "champion_selection": champion_selection,
        "champion": {
            "policy_digest": champion_digest,
            "teacher_digest": _string(summary.get("teacher_digest"), "teacher digest"),
            "name": _string(summary.get("champion_name"), "champion name"),
            "degree": _integer(metrics.get("degree"), "champion degree", minimum=1),
            "input_bits": _integer(metrics.get("input_bits"), "champion input bits", minimum=1),
            "coefficient_bits": _integer(
                metrics.get("coefficient_bits"), "champion coefficient bits", minimum=1
            ),
            "estimated_output_bits": _integer(
                summary.get("estimated_output_bits"), "estimated output bits", minimum=1
            ),
        },
        "agreement": {
            "numerator": agreement_count,
            "denominator": observations,
            "rate": agreement_count / observations,
        },
        "certificate": {
            "numerator": certified_count,
            "denominator": observations,
            "rate": certified_count / observations,
            "certified_mismatches": certificate_mismatches,
            "float_integer_mismatches": total_float_integer_mismatches,
        },
        "selection_certificate": {
            "numerator": selection_certified,
            "denominator": selection_steps,
            "rate": selection_certified / selection_steps,
            "certified_mismatches": selection_mismatches,
            "saturation_count": selection_saturations,
        },
        "ledger": child_ledger,
    }


def _load_suite(study_id: str) -> dict[str, Any]:
    suite, outer, ledger = _suite_root(study_id)
    config = _config(suite)
    counts = _expected_counts(config)
    summary = _object(suite / "suite-summary.json")
    runs = _jsonl(suite / "suite-runs.jsonl")
    suite_episodes = _jsonl(suite / "suite-episodes.jsonl")
    for key, expected in (
        ("expected_runs", counts["runs"]),
        ("completed_runs", counts["runs"]),
        ("candidates_per_run", counts["candidates_per_run"]),
        ("expected_candidate_rows", counts["candidate_rows"]),
        ("expected_selection_episodes", counts["selection_rows"]),
        ("retained_selection_episode_rows", counts["selection_rows"]),
        ("expected_paired_episodes", counts["paired_episodes"]),
        ("retained_paired_episodes", counts["paired_episodes"]),
        ("retained_episode_rows", counts["episode_rows"]),
    ):
        _require_equal(summary.get(key), expected, f"suite denominator {key} mismatch")
    _require_equal(len(runs), counts["runs"], "suite run row denominator mismatch")
    _require_equal(
        len(suite_episodes), counts["paired_episodes"], "suite paired row denominator mismatch"
    )
    children = [
        _child_data(
            suite,
            row,
            expected_candidates=counts["candidates_per_run"],
            expected_selection=counts["selection_per_candidate"],
            expected_evaluation=counts["evaluation_per_checkpoint"],
        )
        for row in runs
    ]
    identities = {
        (
            _string(child["summary"].get("env_id"), "environment"),
            int(child["summary"]["run_id"].rsplit("-", 1)[1]),
        )
        for child in children
    }
    workloads = _table(config, "workloads")
    expected_identities = {
        (environment, checkpoint)
        for environment in workloads["environments"]
        for checkpoint in range(counts["checkpoints_per_environment"])
    }
    _require_equal(
        identities, expected_identities, "suite environment/checkpoint identities mismatch"
    )
    return {
        "study_id": study_id,
        "root": suite,
        "outer": outer,
        "summary": summary,
        "config": config,
        "counts": counts,
        "children": children,
        "ledger": ledger,
    }


def _checkpoint_row(child: Mapping[str, Any], study_id: str) -> dict[str, Any]:
    summary = child["summary"]
    paired = child["paired"]
    environment = _string(summary.get("env_id"), "environment")
    checkpoint = int(_string(summary.get("run_id"), "run id").rsplit("-", 1)[1])
    return {
        "study_id": study_id,
        "environment": environment,
        "checkpoint_index": checkpoint,
        "run_id": summary["run_id"],
        "evaluation_pairs": len(paired),
        "teacher_return": _describe([row["teacher_return"] for row in paired]),
        "student_return": _describe([row["student_return"] for row in paired]),
        "teacher_cost": _describe([row["teacher_cost"] for row in paired]),
        "student_cost": _describe([row["student_cost"] for row in paired]),
        "paired_return_delta": {
            **_describe([row["return_delta"] for row in paired]),
            "bootstrap": _episode_interval(
                [row["return_delta"] for row in paired],
                f"expanded|{environment}|{checkpoint}|return-delta",
            ),
        },
        "teacher_agreement": child["agreement"],
        "action_certificate": child["certificate"],
        "champion_selection": child["selection_certificate"],
        "champion": child["champion"],
        "child_ledger_sha256": child["ledger"]["ledger_sha256"],
    }


def _expanded_tables(
    expanded: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows = sorted(
        (_checkpoint_row(child, expanded["study_id"]) for child in expanded["children"]),
        key=lambda row: (row["environment"], row["checkpoint_index"]),
    )
    _require_equal(len(checkpoint_rows), 15, "expanded checkpoint row count mismatch")
    environments: list[dict[str, Any]] = []
    declared = _table(expanded["config"], "workloads")["environments"]
    for environment in declared:
        selected = [
            child for child in expanded["children"] if child["summary"].get("env_id") == environment
        ]
        selected.sort(key=lambda child: child["summary"]["run_id"])
        _require_equal(len(selected), 5, "expanded environment checkpoint denominator mismatch")
        paired = [row for child in selected for row in child["paired"]]
        agreement_numerator = sum(child["agreement"]["numerator"] for child in selected)
        agreement_denominator = sum(child["agreement"]["denominator"] for child in selected)
        certificate_numerator = sum(child["certificate"]["numerator"] for child in selected)
        certificate_denominator = sum(child["certificate"]["denominator"] for child in selected)
        environments.append(
            {
                "study_id": expanded["study_id"],
                "environment": environment,
                "checkpoints": len(selected),
                "evaluation_pairs": len(paired),
                "teacher_return": _describe([row["teacher_return"] for row in paired]),
                "student_return": _describe([row["student_return"] for row in paired]),
                "teacher_cost": _describe([row["teacher_cost"] for row in paired]),
                "student_cost": _describe([row["student_cost"] for row in paired]),
                "paired_return_delta": {
                    **_describe([row["return_delta"] for row in paired]),
                    "bootstrap": _checkpoint_episode_interval(
                        [[row["return_delta"] for row in child["paired"]] for child in selected],
                        f"expanded|{environment}|checkpoint-episode|return-delta",
                    ),
                },
                "teacher_agreement": {
                    "numerator": agreement_numerator,
                    "denominator": agreement_denominator,
                    "rate": agreement_numerator / agreement_denominator,
                },
                "action_certificate": {
                    "numerator": certificate_numerator,
                    "denominator": certificate_denominator,
                    "rate": certificate_numerator / certificate_denominator,
                    "certified_mismatches": sum(
                        child["certificate"]["certified_mismatches"] for child in selected
                    ),
                    "float_integer_mismatches": sum(
                        child["certificate"]["float_integer_mismatches"] for child in selected
                    ),
                },
                "champion_selection_saturation_count": sum(
                    child["selection_certificate"]["saturation_count"] for child in selected
                ),
                "champion_digests": [child["champion"] for child in selected],
            }
        )
    _require_equal(len(environments), 3, "expanded environment row count mismatch")
    return checkpoint_rows, environments


def _ablation_key(config: Mapping[str, Any]) -> tuple[bool, bool]:
    search = _table(config, "search")
    weighting = search.get("certificate_weighting")
    refinement = search.get("student_occupancy_refinement")
    if not isinstance(weighting, bool) or not isinstance(refinement, bool):
        raise RuntimeError("ablation switches must be booleans")
    return weighting, refinement


def _matched_ablation_inputs(
    cells: Mapping[tuple[bool, bool], Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {(False, False), (False, True), (True, False), (True, True)}
    _require_equal(set(cells), expected_keys, "four-cell ablation design is incomplete")
    projections: list[dict[str, Any]] = []
    run_config_projections: list[dict[str, Any]] = []
    teacher_digests: dict[int, set[str]] = {index: set() for index in range(5)}
    evaluation_keys: dict[int, set[tuple[int, ...]]] = {index: set() for index in range(5)}
    selection_keys: dict[int, set[tuple[int, ...]]] = {index: set() for index in range(5)}
    for cell_key, cell in cells.items():
        raw = cell["config"]
        workloads = _table(raw, "workloads")
        search = _table(raw, "search")
        training = _table(_table(raw, "training"), "gpu")
        projections.append(
            {
                "schema_version": raw.get("schema_version"),
                "seed_root": raw.get("seed_root"),
                "fhe_runtime": raw.get("fhe_runtime"),
                "security_level": raw.get("security_level"),
                "global_p_error": raw.get("global_p_error"),
                "stable_argmax": raw.get("stable_argmax"),
                "workloads": workloads,
                "search_grid_and_budget": {
                    key: search.get(key)
                    for key in (
                        "degrees",
                        "input_bits",
                        "coefficient_bits",
                        "ridge",
                        "refinement_rounds",
                        "calibration_padding",
                    )
                },
                "training_budget": training,
            }
        )
        for child in cell["children"]:
            run_config = child["config"]
            preset = run_config.get("preset")
            if not isinstance(preset, dict):
                raise RuntimeError("ablation child run config lacks its preset")
            run_search = preset.get("search")
            if not isinstance(run_search, dict):
                raise RuntimeError("ablation child run config lacks its search grid")
            _require_equal(
                run_search.get("certificate_weighting"),
                cell_key[0],
                "ablation child weighting switch mismatch",
            )
            _require_equal(
                run_search.get("student_occupancy_refinement"),
                cell_key[1],
                "ablation child occupancy-refinement switch mismatch",
            )
            run_projection = json.loads(_canonical_json(run_config))
            projected_search = run_projection["preset"]["search"]
            del projected_search["certificate_weighting"]
            del projected_search["student_occupancy_refinement"]
            run_config_projections.append(run_projection)
            checkpoint = int(child["summary"]["run_id"].rsplit("-", 1)[1])
            teacher_digests[checkpoint].add(child["champion"]["teacher_digest"])
            evaluation_keys[checkpoint].add(tuple(child["seeds"]["evaluation"]))
            selection_keys[checkpoint].add(tuple(child["seeds"]["selection"]))
    if any(projection != projections[0] for projection in projections[1:]):
        raise RuntimeError("ablation search grids, budgets, seed root, or non-switch config differ")
    if any(projection != run_config_projections[0] for projection in run_config_projections[1:]):
        raise RuntimeError("persisted ablation child search grids or budgets differ")
    if any(len(values) != 1 for values in teacher_digests.values()):
        raise RuntimeError("ablation teacher digests are not identical within checkpoint")
    if any(len(values) != 1 for values in evaluation_keys.values()):
        raise RuntimeError("ablation evaluation seed keys are not identical within checkpoint")
    if any(len(values) != 1 for values in selection_keys.values()):
        raise RuntimeError("ablation selection seed keys are not identical within checkpoint")
    return {
        "matched": True,
        "teacher_digest_by_checkpoint": {
            str(index): next(iter(values)) for index, values in teacher_digests.items()
        },
        "evaluation_seed_key_sha256_by_checkpoint": {
            str(index): _sha256(_canonical_json(list(next(iter(values)))).encode())
            for index, values in evaluation_keys.items()
        },
        "selection_seed_key_sha256_by_checkpoint": {
            str(index): _sha256(_canonical_json(list(next(iter(values)))).encode())
            for index, values in selection_keys.items()
        },
        "matched_projection": projections[0],
        "matched_child_run_config_sha256": _sha256(
            _canonical_json(run_config_projections[0]).encode()
        ),
    }


def _cell_children(cell: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for child in cell["children"]:
        checkpoint = int(child["summary"]["run_id"].rsplit("-", 1)[1])
        if checkpoint in result:
            raise RuntimeError("duplicate ablation checkpoint")
        result[checkpoint] = child
    _require_equal(set(result), set(range(5)), "ablation checkpoint set mismatch")
    return result


def _coverage_episode_pairs(child: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [
        (
            _integer(row.get("certified_count"), "certified count"),
            _integer(row.get("steps"), "coverage denominator", minimum=1),
        )
        for row in child["champion_selection"]
    ]


def _cell_bootstrap(
    children: Sequence[Mapping[str, Any]], outcome: str, label: str
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(_seed(label))
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        checkpoint_indices = rng.integers(0, len(children), size=len(children))
        if outcome == "return_delta":
            checkpoint_means: list[float] = []
            for index in checkpoint_indices:
                values = [row["return_delta"] for row in children[int(index)]["paired"]]
                episode_indices = rng.integers(0, len(values), size=len(values))
                checkpoint_means.append(float(np.mean(np.asarray(values)[episode_indices])))
            estimates[repetition] = float(np.mean(checkpoint_means))
        else:
            numerator = 0
            denominator = 0
            for index in checkpoint_indices:
                values = _coverage_episode_pairs(children[int(index)])
                episode_indices = rng.integers(0, len(values), size=len(values))
                numerator += sum(values[int(item)][0] for item in episode_indices)
                denominator += sum(values[int(item)][1] for item in episode_indices)
            estimates[repetition] = numerator / denominator
    if outcome == "return_delta":
        estimate = float(
            np.mean(
                [np.mean([row["return_delta"] for row in child["paired"]]) for child in children]
            )
        )
    else:
        pairs = [pair for child in children for pair in _coverage_episode_pairs(child)]
        estimate = sum(pair[0] for pair in pairs) / sum(pair[1] for pair in pairs)
    return {
        "estimate": estimate,
        **_interval(estimates),
        "method": "matched checkpoint then episode percentile bootstrap",
        "episode_scope": (
            "post-selection paired evaluation"
            if outcome == "return_delta"
            else "champion selection occupancy"
        ),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": _seed(label),
    }


def _contrast(values: Mapping[tuple[bool, bool], float], effect: str) -> float:
    y00, y01, y10, y11 = (
        values[(False, False)],
        values[(False, True)],
        values[(True, False)],
        values[(True, True)],
    )
    if effect == "weighting_main_effect":
        return 0.5 * ((y10 - y00) + (y11 - y01))
    if effect == "occupancy_refinement_bundle_main_effect":
        return 0.5 * ((y01 - y00) + (y11 - y10))
    if effect == "interaction":
        return (y11 - y10) - (y01 - y00)
    raise ValueError(effect)


def _effect_bootstrap(
    children: Mapping[tuple[bool, bool], Mapping[int, Mapping[str, Any]]],
    *,
    effect: str,
    outcome: str,
    label: str,
    checkpoints: Sequence[int] = tuple(range(5)),
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(_seed(label))
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        checkpoint_indices = rng.choice(
            np.asarray(checkpoints), size=len(checkpoints), replace=True
        )
        cell_values: dict[tuple[bool, bool], float] = {}
        if outcome == "return_delta":
            totals: dict[tuple[bool, bool], list[float]] = {key: [] for key in children}
            for checkpoint in checkpoint_indices:
                sample_child = children[(False, False)][int(checkpoint)]
                count = len(sample_child["paired"])
                episode_indices = rng.integers(0, count, size=count)
                for key in children:
                    rows = children[key][int(checkpoint)]["paired"]
                    totals[key].append(
                        float(
                            np.mean([rows[int(index)]["return_delta"] for index in episode_indices])
                        )
                    )
            cell_values = {key: float(np.mean(values)) for key, values in totals.items()}
        else:
            numerators = {key: 0 for key in children}
            denominators = {key: 0 for key in children}
            for checkpoint in checkpoint_indices:
                count = len(_coverage_episode_pairs(children[(False, False)][int(checkpoint)]))
                episode_indices = rng.integers(0, count, size=count)
                for key in children:
                    pairs = _coverage_episode_pairs(children[key][int(checkpoint)])
                    numerators[key] += sum(pairs[int(index)][0] for index in episode_indices)
                    denominators[key] += sum(pairs[int(index)][1] for index in episode_indices)
            cell_values = {key: numerators[key] / denominators[key] for key in children}
        estimates[repetition] = _contrast(cell_values, effect)
    observed: dict[tuple[bool, bool], float] = {}
    for key in children:
        chosen = [children[key][checkpoint] for checkpoint in checkpoints]
        if outcome == "return_delta":
            observed[key] = float(
                np.mean(
                    [np.mean([row["return_delta"] for row in child["paired"]]) for child in chosen]
                )
            )
        else:
            pairs = [pair for child in chosen for pair in _coverage_episode_pairs(child)]
            observed[key] = sum(pair[0] for pair in pairs) / sum(pair[1] for pair in pairs)
    return {
        "estimate": _contrast(observed, effect),
        **_interval(estimates),
        "method": (
            "matched checkpoint then episode percentile bootstrap"
            if len(checkpoints) > 1
            else "matched episode percentile bootstrap within checkpoint"
        ),
        "episode_scope": (
            "post-selection paired evaluation"
            if outcome == "return_delta"
            else "champion selection occupancy"
        ),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": _seed(label),
    }


def _ablation_tables(
    cells: Mapping[tuple[bool, bool], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matching = _matched_ablation_inputs(cells)
    by_checkpoint = {key: _cell_children(cell) for key, cell in cells.items()}
    cell_rows: list[dict[str, Any]] = []
    for key in sorted(cells):
        weighting, refinement = key
        children = [by_checkpoint[key][index] for index in range(5)]
        paired = [row for child in children for row in child["paired"]]
        heldout_num = sum(child["certificate"]["numerator"] for child in children)
        heldout_den = sum(child["certificate"]["denominator"] for child in children)
        selection_pairs = [pair for child in children for pair in _coverage_episode_pairs(child)]
        cell_rows.append(
            {
                "study_id": cells[key]["study_id"],
                "certificate_weighting": weighting,
                "occupancy_refinement_bundle": refinement,
                "checkpoints": 5,
                "evaluation_pairs": len(paired),
                "paired_return_delta": _cell_bootstrap(
                    children, "return_delta", f"ablation|{int(weighting)}|{int(refinement)}|return"
                ),
                "champion_selection_certificate_coverage": {
                    "numerator": sum(pair[0] for pair in selection_pairs),
                    "denominator": sum(pair[1] for pair in selection_pairs),
                    "bootstrap": _cell_bootstrap(
                        children,
                        "coverage",
                        f"ablation|{int(weighting)}|{int(refinement)}|coverage",
                    ),
                },
                "postselection_heldout_certificate_coverage": {
                    "numerator": heldout_num,
                    "denominator": heldout_den,
                    "certified_mismatches": sum(
                        child["certificate"]["certified_mismatches"] for child in children
                    ),
                    "bootstrap_not_computed": (
                        "held-out receipts retain checkpoint totals but no "
                        "per-episode certificate rows"
                    ),
                },
            }
        )

    effects = (
        "weighting_main_effect",
        "occupancy_refinement_bundle_main_effect",
        "interaction",
    )
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint in range(5):
        row: dict[str, Any] = {
            "checkpoint_index": checkpoint,
            "teacher_digest": matching["teacher_digest_by_checkpoint"][str(checkpoint)],
            "contrasts": {},
        }
        for effect in effects:
            row["contrasts"][effect] = {
                "paired_return_delta": _effect_bootstrap(
                    by_checkpoint,
                    effect=effect,
                    outcome="return_delta",
                    label=f"ablation|checkpoint-{checkpoint}|{effect}|return",
                    checkpoints=(checkpoint,),
                ),
                "champion_selection_certificate_coverage": _effect_bootstrap(
                    by_checkpoint,
                    effect=effect,
                    outcome="coverage",
                    label=f"ablation|checkpoint-{checkpoint}|{effect}|coverage",
                    checkpoints=(checkpoint,),
                ),
            }
        checkpoint_rows.append(row)
    _require_equal(len(checkpoint_rows), 5, "ablation checkpoint contrast count mismatch")

    effect_rows = [
        {
            "effect": effect,
            "definition": {
                "weighting_main_effect": "mean(weighted-unweighted) across refinement levels",
                "occupancy_refinement_bundle_main_effect": (
                    "mean(refined-unrefined) across weighting levels"
                ),
                "interaction": "(weighted effect when refined) - (weighted effect when unrefined)",
            }[effect],
            "scope": "matched clear CartPole checkpoints only",
            "paired_return_delta": _effect_bootstrap(
                by_checkpoint,
                effect=effect,
                outcome="return_delta",
                label=f"ablation|all-checkpoints|{effect}|return",
            ),
            "champion_selection_certificate_coverage": _effect_bootstrap(
                by_checkpoint,
                effect=effect,
                outcome="coverage",
                label=f"ablation|all-checkpoints|{effect}|coverage",
            ),
        }
        for effect in effects
    ]
    return cell_rows, checkpoint_rows, effect_rows, matching


def _scoped_fhe() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nonlinear_root = STUDY_ROOT / NONLINEAR_ID
    nonlinear_ledger = _verify_ledger(nonlinear_root)
    nonlinear = _object(nonlinear_root / "summary.json")
    _require_equal(nonlinear.get("study_id"), NONLINEAR_ID, "nonlinear study identity mismatch")
    nonlinear_rows = _jsonl(nonlinear_root / "raw.jsonl")
    challenge = nonlinear.get("challenge_summary")
    configuration = nonlinear.get("configuration")
    if not isinstance(challenge, dict) or not isinstance(configuration, dict):
        raise RuntimeError("nonlinear study lacks configuration or exact challenge summary")
    planned_nonlinear = _integer(
        configuration.get("expected_real_fhe_calls"), "planned nonlinear calls", minimum=1
    )
    nonlinear_source = nonlinear.get("source_provenance")
    if not isinstance(nonlinear_source, dict):
        raise RuntimeError("nonlinear study lacks source provenance")
    _require_equal(
        len(nonlinear_rows), planned_nonlinear, "nonlinear raw call denominator mismatch"
    )
    _require_equal(
        challenge.get("real_fhe_rows"), planned_nonlinear, "nonlinear summary denominator mismatch"
    )
    nonlinear_failures = sum(
        1
        for row in nonlinear_rows
        if row.get("real_fhe_matches_integer_clear") is not True
        or row.get("simulation_matches_integer_clear") is not True
    )
    _require_equal(
        challenge.get("real_fhe_all_match"),
        nonlinear_failures == 0,
        "nonlinear match summary mismatch",
    )

    timing_root = STUDY_ROOT / TIMING_ID
    timing_ledger = _verify_ledger(timing_root)
    timing = _object(timing_root / "summary.json")
    context = _object(timing_root / "context.json")
    timing_rows = _jsonl(timing_root / "raw.jsonl")
    timing_source = timing.get("source_provenance")
    if not isinstance(timing_source, dict):
        raise RuntimeError("timing study lacks source provenance")
    _require_equal(timing.get("study_id"), TIMING_ID, "timing study identity mismatch")
    context_copy = dict(context)
    recorded_context_digest = _string(
        context_copy.pop("context_digest", None), "timing context digest"
    )
    _require_equal(
        _sha256(_canonical_json(context_copy).encode()),
        recorded_context_digest,
        "timing context digest mismatch",
    )
    _require_equal(
        timing.get("context_digest"),
        recorded_context_digest,
        "timing summary context digest mismatch",
    )
    if any(row.get("context_digest") != recorded_context_digest for row in timing_rows):
        raise RuntimeError("a timing row is not bound to the verified context")
    execution = timing.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("timing execution summary is missing")
    timing_configuration = context.get("configuration")
    if not isinstance(timing_configuration, dict):
        raise RuntimeError("timing context lacks its planned configuration")
    schedule = timing_configuration.get("schedule")
    if not isinstance(schedule, dict):
        raise RuntimeError("timing context lacks its planned schedule")
    planned_timing = _integer(
        schedule.get("total_real_fhe_attempts"), "planned timing attempts", minimum=1
    )
    planned_containers = _integer(
        schedule.get("containers"), "planned timing containers", minimum=1
    )
    planned_warmups = planned_containers * _integer(
        schedule.get("warmups_per_container"), "planned warmups per container"
    )
    planned_measured = planned_containers * _integer(
        schedule.get("measured_requests_per_container"), "planned measurements per container"
    )
    _require_equal(
        planned_warmups + planned_measured, planned_timing, "timing schedule arithmetic mismatch"
    )
    _require_equal(
        execution.get("real_fhe_attempts"), planned_timing, "timing execution plan mismatch"
    )
    _require_equal(len(timing_rows), planned_timing, "timing raw attempt denominator mismatch")
    warmups = sum(row.get("is_warmup") is True for row in timing_rows)
    measured = len(timing_rows) - warmups
    _require_equal(warmups, planned_warmups, "timing warmup denominator mismatch")
    _require_equal(measured, planned_measured, "timing measured denominator mismatch")
    _require_equal(
        execution.get("warmup_attempts"), planned_warmups, "timing execution warmup mismatch"
    )
    _require_equal(
        execution.get("measured_attempts"),
        planned_measured,
        "timing execution measurement mismatch",
    )
    distinct_containers = {
        _string(row.get("container_id"), "timing container id") for row in timing_rows
    }
    _require_equal(
        len(distinct_containers), planned_containers, "timing planned container count mismatch"
    )
    _require_equal(
        len(distinct_containers),
        execution.get("actual_distinct_containers"),
        "timing container count mismatch",
    )
    timing_failures = sum(row.get("success") is not True for row in timing_rows)

    summaries = {
        "schema_version": "unseen-loop/scoped-fhe-evidence-v1",
        "nonlinear": {
            "source_summary_exact": nonlinear,
            "raw_accounting": {
                "planned_attempts": planned_nonlinear,
                "observed_attempts": len(nonlinear_rows),
                "failures": nonlinear_failures,
            },
            "scope": (
                "degree-2 qmax=2 synthetic complete-domain REAL-FHE challenge in one colocated "
                "Modal client/server research worker; not local-client/remote-server "
                "secrecy evidence"
            ),
        },
        "timing": {
            "source_summary_exact": timing,
            "raw_accounting": {
                "planned_attempts": planned_timing,
                "observed_attempts": len(timing_rows),
                "warmup_attempts": warmups,
                "measured_attempts": measured,
                "failures": timing_failures,
                "distinct_containers": len(distinct_containers),
            },
            "scope": (
                "four independent colocated Modal client/server cryptographic contexts; "
                "warmups excluded as specified by the source summary; not a shared "
                "client/server context"
            ),
        },
    }
    evidence = [
        {
            "study_id": NONLINEAR_ID,
            "source_path": str(nonlinear_root),
            "source_summary_sha256": _file_sha256(nonlinear_root / "summary.json"),
            "configuration_sha256": _sha256(_canonical_json(configuration).encode()),
            "source_entrypoint_sha256": nonlinear_source.get("entrypoint_sha256"),
            **nonlinear_ledger,
            "backend": "REAL FHE",
            "trust_label": "colocated Modal client/server research worker",
            "planned": {"real_fhe_attempts": planned_nonlinear},
            "observed": {"real_fhe_attempts": len(nonlinear_rows)},
            "failures": nonlinear_failures,
        },
        {
            "study_id": TIMING_ID,
            "source_path": str(timing_root),
            "source_summary_sha256": _file_sha256(timing_root / "summary.json"),
            "configuration_sha256": _sha256(_canonical_json(timing_configuration).encode()),
            "context_sha256": recorded_context_digest,
            "source_entrypoint_sha256": timing_source.get("entrypoint_sha256"),
            **timing_ledger,
            "backend": "REAL FHE",
            "trust_label": "four independent colocated Modal client/server research contexts",
            "planned": {"all_attempts": planned_timing, "warmups": warmups, "measured": measured},
            "observed": {
                "all_attempts": len(timing_rows),
                "distinct_containers": len(distinct_containers),
            },
            "failures": timing_failures,
        },
    ]
    return summaries, evidence


def _suite_evidence(suite: Mapping[str, Any]) -> dict[str, Any]:
    summary = suite["summary"]
    counts = suite["counts"]
    gates = _table(suite["config"], "gates")
    minimum_coverage = _number(
        gates.get("minimum_certified_occupancy"), "minimum certified occupancy"
    )
    maximum_mismatches = _integer(
        gates.get("maximum_certified_mismatches"), "maximum certified mismatches"
    )
    gates_failed = sum(
        child["certificate"]["numerator"] / child["certificate"]["denominator"] < minimum_coverage
        or child["certificate"]["certified_mismatches"] > maximum_mismatches
        for child in suite["children"]
    )
    child_ledgers = {
        child["summary"]["run_id"]: child["ledger"]["ledger_sha256"] for child in suite["children"]
    }
    source = suite["outer"].get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Modal suite source provenance is missing")
    return {
        "study_id": suite["study_id"],
        "source_path": str(suite["root"]),
        "source_summary_sha256": _file_sha256(suite["root"] / "suite-summary.json"),
        "modal_study_summary_sha256": _file_sha256(
            STUDY_ROOT / suite["study_id"] / "modal-study-summary.json"
        ),
        "config_sha256": _file_sha256(suite["root"] / "release.toml"),
        "python_source_sha256": source.get("python_source_sha256"),
        **suite["ledger"],
        "child_ledger_count": len(child_ledgers),
        "child_ledger_sha256": child_ledgers,
        "backend": "QUANTIZED CLEAR",
        "trust_label": "clear Modal CPU research worker; no privacy evidence",
        "planned": {
            "runs": counts["runs"],
            "candidate_rows": counts["candidate_rows"],
            "selection_episode_rows": counts["selection_rows"],
            "paired_evaluation_episodes": counts["paired_episodes"],
            "long_form_evaluation_rows": counts["episode_rows"],
        },
        "observed": {
            "runs": summary["completed_runs"],
            "candidate_rows": counts["candidate_rows"],
            "selection_episode_rows": summary["retained_selection_episode_rows"],
            "paired_evaluation_episodes": summary["retained_paired_episodes"],
            "long_form_evaluation_rows": summary["retained_episode_rows"],
        },
        "failures": {
            "checksum": 0,
            "incomplete_denominators": 0,
            "suite_gate_failures": gates_failed,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.write_text("".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        flattened.append(
            {
                key: _canonical_json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    fieldnames = sorted({key for row in flattened for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _close_output_ledger(destination: Path) -> dict[str, str]:
    files = sorted(
        path for path in destination.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    checksums = {path.name: _file_sha256(path) for path in files}
    (destination / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8"
    )
    _verify_ledger(destination)
    return checksums


@app.function(
    image=analysis_image,
    cpu=(4.0, 4.0),
    memory=(8_192, 8_192),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=3 * 3_600,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def analyze_remote() -> str:
    """Verify and aggregate all source evidence inside one bounded Modal container."""
    import numpy as np

    artifacts.reload()
    if OUTPUT_ROOT.exists() and (not OUTPUT_ROOT.is_dir() or any(OUTPUT_ROOT.iterdir())):
        raise RuntimeError(f"analysis destination is not empty: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    expanded = _load_suite(EXPANDED_ID)
    expanded_checkpoint_rows, expanded_environment_rows = _expanded_tables(expanded)

    ablation_suites = [_load_suite(ABLATION_PREFIX + suffix) for suffix in ABLATION_SUFFIXES]
    ablation_cells: dict[tuple[bool, bool], dict[str, Any]] = {}
    for suite in ablation_suites:
        key = _ablation_key(suite["config"])
        if key in ablation_cells:
            raise RuntimeError("duplicate ablation factor cell")
        ablation_cells[key] = suite
    cell_rows, contrast_rows, effect_rows, matching = _ablation_tables(ablation_cells)
    scoped_fhe, fhe_evidence = _scoped_fhe()

    source_evidence = (
        [_suite_evidence(expanded)]
        + [_suite_evidence(suite) for suite in ablation_suites]
        + fhe_evidence
    )
    evidence_index = {
        "schema_version": "unseen-loop/evidence-index-v1",
        "analysis_id": OUTPUT_ID,
        "execution": {
            "location": "Modal",
            "backend": "NumPy analysis over persisted evidence",
            "python": "3.12",
            "numpy": np.__version__,
            "max_containers": 1,
            "retries": 0,
            "public_endpoint": False,
        },
        "sources": source_evidence,
        "ablation_matching": matching,
        "allowed_claims": [
            (
                "descriptive paired clear evaluation for the verified expanded "
                "three-environment matrix"
            ),
            (
                "matched-factorial clear CartPole comparisons for certificate "
                "weighting and the occupancy-refinement bundle"
            ),
            (
                "causal interpretation, if made, is restricted to the CartPole "
                "occupancy-refinement-bundle intervention represented by these four "
                "matched cells"
            ),
            "the exact source-scoped degree-2 qmax=2 nonlinear REAL-FHE challenge summary",
            "the exact source-scoped four-context colocated Modal REAL-FHE timing summary",
        ],
        "forbidden_claims": [
            (
                "privacy, confidentiality, or REAL-FHE evidence from any clear expanded "
                "or ablation study"
            ),
            "a causal effect outside CartPole or outside the tested refinement bundle",
            "local-client/remote-server secrecy from colocated nonlinear or timing studies",
            (
                "shared-context, production-service, throughput, or real-time latency "
                "from the timing study"
            ),
            "empirical validation of global_p_error",
            "preregistration-wide or release-wide completion",
            (
                "private training, malicious-server integrity, endpoint security, or "
                "traffic-flow confidentiality"
            ),
        ],
    }
    analysis = {
        "schema_version": "unseen-loop/release-analysis-v1",
        "analysis_id": OUTPUT_ID,
        "statistics": {
            "sd": "population SD (ddof=0)",
            "quartiles": "NumPy linear-interpolation quantiles",
            "bootstrap": (
                "deterministic percentile intervals; matched checkpoint then episode "
                "where both levels exist"
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "ablation_return_scope": "post-selection paired evaluation episodes",
            "ablation_bootstrap_certificate_scope": "champion selection-occupancy episode counters",
            "heldout_certificate_limit": (
                "held-out receipts retain exact checkpoint numerators/denominators but "
                "not per-episode certificate rows; no unverified episode-level heldout "
                "denominator was manufactured"
            ),
        },
        "tables": {
            "expanded_checkpoints": "expanded-checkpoints.jsonl",
            "expanded_environments": "expanded-environments.jsonl",
            "ablation_cells": "ablation-cells.jsonl",
            "ablation_checkpoint_contrasts": "ablation-checkpoint-contrasts.jsonl",
            "ablation_effects": "ablation-effects.jsonl",
            "scoped_fhe_summaries": "scoped-fhe-summaries.json",
        },
        "observed_rows": {
            "expanded_checkpoints": len(expanded_checkpoint_rows),
            "expanded_environments": len(expanded_environment_rows),
            "ablation_cells": len(cell_rows),
            "ablation_checkpoint_contrasts": len(contrast_rows),
            "ablation_factorial_effects": len(effect_rows),
        },
        "claim_scope": {
            "clear_privacy_claim": "none",
            "causal_scope": "CartPole occupancy-refinement bundle only",
            "release_label": "bounded evidence analysis only",
        },
    }

    _write_jsonl(OUTPUT_ROOT / "expanded-checkpoints.jsonl", expanded_checkpoint_rows)
    _write_jsonl(OUTPUT_ROOT / "expanded-environments.jsonl", expanded_environment_rows)
    _write_jsonl(OUTPUT_ROOT / "ablation-cells.jsonl", cell_rows)
    _write_jsonl(OUTPUT_ROOT / "ablation-checkpoint-contrasts.jsonl", contrast_rows)
    _write_jsonl(OUTPUT_ROOT / "ablation-effects.jsonl", effect_rows)
    _write_json(OUTPUT_ROOT / "scoped-fhe-summaries.json", scoped_fhe)
    _write_json(OUTPUT_ROOT / "analysis.json", analysis)
    _write_json(OUTPUT_ROOT / "evidence-index.json", evidence_index)
    _write_csv(OUTPUT_ROOT / "expanded-checkpoints.csv", expanded_checkpoint_rows)
    _write_csv(OUTPUT_ROOT / "expanded-environments.csv", expanded_environment_rows)
    _write_csv(OUTPUT_ROOT / "ablation-cells.csv", cell_rows)
    output_checksums = _close_output_ledger(OUTPUT_ROOT)
    artifacts.commit()
    return _canonical_json(
        {
            "schema_version": "unseen-loop/modal-analysis-result-v1",
            "analysis_id": OUTPUT_ID,
            "artifact_path": str(OUTPUT_ROOT),
            "ledger_sha256": _file_sha256(OUTPUT_ROOT / "checksums.sha256"),
            "files": [*sorted(output_checksums), "checksums.sha256"],
            "source_studies": [source["study_id"] for source in source_evidence],
            "status": "complete",
        }
    )


@app.local_entrypoint()
def main() -> str:
    """Run the complete evidence analysis remotely; local execution only submits the call."""
    return analyze_remote.remote()
