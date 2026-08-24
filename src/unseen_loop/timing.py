"""Deterministic summaries for sanitized, repeated FHE timing measurements.

The input schema is deliberately narrow: callers must remove observations, plaintexts,
ciphertexts, exception messages, and other free-form values before constructing rows.
Each row's context digest must bind the separately ledgered study/configuration,
policy/circuit/server artifacts, Concrete security settings and version, hardware and
container identity, batch/concurrency settings, schedule, and metric definitions.
Unknown fields are rejected rather than silently copied into research artifacts.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]

TIMING_ROW_SCHEMA = "unseen-loop/sanitized-timing-row-v1"
TIMING_SUMMARY_SCHEMA = "unseen-loop/timing-summary-v1"
MINIMUM_P95_SAMPLE_COUNT = 20
RELEASE_MINIMUM_MEASURED_REQUESTS = 64
RELEASE_MINIMUM_CONTAINERS = 4
RELEASE_MINIMUM_REQUESTS_PER_CONTAINER = 16
RELEASE_MINIMUM_WARMUPS_PER_CONTAINER = 3

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "container_id",
        "trial_id",
        "context_digest",
        "request_id",
        "is_warmup",
        "success",
        "timing_ns",
        "byte_metrics",
    }
)
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"failure_code"}
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_SAFE_METRIC = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_CONTEXT_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def summarize_timing_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    minimum_p95_samples: int = MINIMUM_P95_SAMPLE_COUNT,
) -> JSONDict:
    """Summarize sanitized request rows with a two-stage cluster bootstrap.

    Warmups and failures remain visible in counts, grouping, and the exact sanitized
    failure-row list, but neither contributes performance values. Bootstrap replicates
    first resample containers and then resample measured requests inside each selected
    container. Order statistics use the nearest-rank definition.
    """

    _validate_configuration(
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
        minimum_p95_samples=minimum_p95_samples,
    )
    validated = [_validate_row(row) for row in rows]
    context_digest = _validate_single_context(validated)
    _validate_unique_requests(validated)

    measured = [row for row in validated if not cast(bool, row["is_warmup"])]
    successful = [row for row in measured if cast(bool, row["success"])]
    _validate_metric_schema(successful)

    failures = [row for row in validated if not cast(bool, row["success"])]
    failures.sort(key=_row_sort_key)
    measured_failures = len(measured) - len(successful)
    if not measured:
        quantiles_conditioning = "no_measured_requests"
    elif measured_failures:
        quantiles_conditioning = "conditional_on_success"
    else:
        quantiles_conditioning = "all_measured_requests_succeeded"

    summary: JSONDict = {
        "schema_version": TIMING_SUMMARY_SCHEMA,
        "context_digest": context_digest,
        "method": {
            "warmups_excluded": True,
            "statistics_population": "non_warmup_successful_requests",
            "order_statistic": "nearest_rank",
            "bootstrap": "resample_containers_then_requests",
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": seed,
            "quantiles_conditioning": quantiles_conditioning,
            "minimum_p95_samples": minimum_p95_samples,
        },
        "row_counts": _row_counts(validated),
        "denominators": {
            "attempted_measured_requests": len(measured),
            "successful_measured_requests": len(successful),
            "failed_measured_requests": measured_failures,
            "success_fraction": f"{len(successful)}/{len(measured)}",
        },
        "release_quality": _release_quality(validated),
        "grouping": _grouping(validated),
        "timing_ns": _summarize_metric_family(
            successful,
            family="timing_ns",
            unit="ns",
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
            minimum_p95_samples=minimum_p95_samples,
        ),
        "byte_metrics": _summarize_metric_family(
            successful,
            family="byte_metrics",
            unit="bytes",
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
            minimum_p95_samples=minimum_p95_samples,
        ),
        "failures": cast(list[JSONValue], failures),
    }
    return summary


def _validate_configuration(
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    minimum_p95_samples: int,
) -> None:
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise TypeError("bootstrap_samples must be an integer")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float)):
        raise TypeError("confidence_level must be numeric")
    if not math.isfinite(float(confidence_level)) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    if isinstance(minimum_p95_samples, bool) or not isinstance(minimum_p95_samples, int):
        raise TypeError("minimum_p95_samples must be an integer")
    if minimum_p95_samples < MINIMUM_P95_SAMPLE_COUNT:
        raise ValueError(f"minimum_p95_samples cannot be below {MINIMUM_P95_SAMPLE_COUNT}")


def _validate_row(row: Mapping[str, object]) -> JSONDict:
    if not isinstance(row, Mapping):
        raise TypeError("each timing row must be a mapping")
    if any(not isinstance(field, str) for field in row):
        raise ValueError("timing row field names must be strings")
    fields = set(row)
    missing = _REQUIRED_FIELDS - fields
    unknown = fields - _ALLOWED_FIELDS
    if missing:
        raise ValueError(f"timing row is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError("timing row contains non-sanitized fields: " + ", ".join(sorted(unknown)))
    schema_version = row["schema_version"]
    if not isinstance(schema_version, str) or schema_version != TIMING_ROW_SCHEMA:
        raise ValueError(f"schema_version must be {TIMING_ROW_SCHEMA!r}")
    context_digest = row["context_digest"]
    if not isinstance(context_digest, str) or _CONTEXT_DIGEST.fullmatch(context_digest) is None:
        raise ValueError("context_digest must be a lowercase SHA-256 digest")

    for field in ("container_id", "trial_id", "request_id"):
        value = row[field]
        if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
            raise ValueError(f"{field} must be a non-empty sanitized token")
    for field in ("is_warmup", "success"):
        if not isinstance(row[field], bool):
            raise TypeError(f"{field} must be a boolean")

    timing_ns = _validate_metrics(row["timing_ns"], field="timing_ns")
    byte_metrics = _validate_metrics(row["byte_metrics"], field="byte_metrics")
    success = cast(bool, row["success"])
    failure_code = row.get("failure_code")
    if success:
        if not timing_ns:
            raise ValueError("successful timing rows require at least one timing_ns metric")
        if failure_code is not None:
            raise ValueError("successful timing rows cannot have a failure_code")
    elif not isinstance(failure_code, str) or _SAFE_TOKEN.fullmatch(failure_code) is None:
        raise ValueError("failed timing rows require a sanitized failure_code")

    cleaned: JSONDict = {
        "schema_version": TIMING_ROW_SCHEMA,
        "container_id": cast(str, row["container_id"]),
        "trial_id": cast(str, row["trial_id"]),
        "request_id": cast(str, row["request_id"]),
        "context_digest": cast(str, row["context_digest"]),
        "is_warmup": cast(bool, row["is_warmup"]),
        "success": success,
        "timing_ns": timing_ns,
        "byte_metrics": byte_metrics,
    }
    if "failure_code" in row:
        cleaned["failure_code"] = failure_code
    return cleaned


def _validate_metrics(value: object, *, field: str) -> JSONDict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    cleaned: JSONDict = {}
    for name, measurement in value.items():
        if not isinstance(name, str) or _SAFE_METRIC.fullmatch(name) is None:
            raise ValueError(f"{field} names must be sanitized identifiers")
        if isinstance(measurement, bool) or not isinstance(measurement, int):
            raise TypeError(f"{field}.{name} must be an integer")
        if measurement < 0:
            raise ValueError(f"{field}.{name} cannot be negative")
        cleaned[name] = measurement
    return dict(sorted(cleaned.items()))


def _validate_unique_requests(rows: Sequence[JSONDict]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (
            cast(str, row["container_id"]),
            cast(str, row["trial_id"]),
            cast(str, row["request_id"]),
        )
        if identity in seen:
            raise ValueError("duplicate timing request identity: " + "/".join(identity))
        seen.add(identity)


def _validate_single_context(rows: Sequence[JSONDict]) -> str | None:
    contexts = {cast(str, row["context_digest"]) for row in rows}
    if len(contexts) > 1:
        raise ValueError("timing rows with different context digests cannot be pooled")
    return next(iter(contexts), None)


def _validate_metric_schema(rows: Sequence[JSONDict]) -> None:
    if not rows:
        return
    expected_timing = set(cast(dict[str, JSONValue], rows[0]["timing_ns"]))
    expected_bytes = set(cast(dict[str, JSONValue], rows[0]["byte_metrics"]))
    for row in rows[1:]:
        if set(cast(dict[str, JSONValue], row["timing_ns"])) != expected_timing:
            raise ValueError("successful measured rows must share the same timing_ns metrics")
        if set(cast(dict[str, JSONValue], row["byte_metrics"])) != expected_bytes:
            raise ValueError("successful measured rows must share the same byte_metrics")


def _row_counts(rows: Sequence[JSONDict]) -> JSONDict:
    warmup_rows = [row for row in rows if cast(bool, row["is_warmup"])]
    measured = [row for row in rows if not cast(bool, row["is_warmup"])]
    successes = sum(cast(bool, row["success"]) for row in measured)
    warmup_failures = sum(not cast(bool, row["success"]) for row in warmup_rows)
    measured_failures = len(measured) - successes
    return {
        "total": len(rows),
        "warmup_excluded": len(warmup_rows),
        "measured": len(measured),
        "successful": successes,
        "failed": measured_failures,
        "warmup_failed": warmup_failures,
        "failures_retained": measured_failures + warmup_failures,
    }


def _release_quality(rows: Sequence[JSONDict]) -> JSONDict:
    measured = [row for row in rows if not cast(bool, row["is_warmup"])]
    warmups = [row for row in rows if cast(bool, row["is_warmup"])]
    by_container: dict[str, list[JSONDict]] = defaultdict(list)
    for row in measured:
        by_container[cast(str, row["container_id"])].append(row)
    warmups_per_container: dict[str, int] = defaultdict(int)
    for row in warmups:
        warmups_per_container[cast(str, row["container_id"])] += 1
    successful_per_container = {
        container_id: sum(cast(bool, row["success"]) for row in container_rows)
        for container_id, container_rows in by_container.items()
    }
    successful = sum(successful_per_container.values())
    failed = len(measured) - successful
    warmup_failures = sum(not cast(bool, row["success"]) for row in warmups)
    minimum_successful = min(successful_per_container.values(), default=0)
    minimum_warmups = min(
        (warmups_per_container[container_id] for container_id in by_container),
        default=0,
    )

    reasons: list[JSONValue] = []
    if len(measured) < RELEASE_MINIMUM_MEASURED_REQUESTS:
        reasons.append("fewer_than_64_measured_requests")
    if failed:
        reasons.append("measured_failures_present")
    if warmup_failures:
        reasons.append("warmup_failures_present")
    if len(by_container) < RELEASE_MINIMUM_CONTAINERS:
        reasons.append("fewer_than_4_measured_containers")
    if minimum_successful < RELEASE_MINIMUM_REQUESTS_PER_CONTAINER:
        reasons.append("fewer_than_16_successes_in_a_container")
    if minimum_warmups < RELEASE_MINIMUM_WARMUPS_PER_CONTAINER:
        reasons.append("fewer_than_3_warmups_in_a_container")

    return {
        "eligible": not reasons,
        "requirements": {
            "minimum_measured_requests": RELEASE_MINIMUM_MEASURED_REQUESTS,
            "zero_measured_failures": True,
            "zero_warmup_failures": True,
            "minimum_measured_containers": RELEASE_MINIMUM_CONTAINERS,
            "minimum_successful_requests_per_container": (RELEASE_MINIMUM_REQUESTS_PER_CONTAINER),
            "minimum_excluded_warmups_per_container": (RELEASE_MINIMUM_WARMUPS_PER_CONTAINER),
        },
        "observed": {
            "measured_requests": len(measured),
            "successful_requests": successful,
            "failed_requests": failed,
            "warmup_failures": warmup_failures,
            "measured_containers": len(by_container),
            "minimum_successful_requests_per_container": minimum_successful,
            "minimum_excluded_warmups_per_container": minimum_warmups,
        },
        "ineligibility_reasons": reasons,
    }


def _grouping(rows: Sequence[JSONDict]) -> JSONDict:
    by_container: dict[str, list[JSONDict]] = defaultdict(list)
    for row in rows:
        by_container[cast(str, row["container_id"])].append(row)

    containers: list[JSONValue] = []
    trial_total = 0
    for container_id in sorted(by_container):
        container_rows = by_container[container_id]
        by_trial: dict[str, list[JSONDict]] = defaultdict(list)
        for row in container_rows:
            by_trial[cast(str, row["trial_id"])].append(row)
        trial_total += len(by_trial)
        trials: list[JSONValue] = []
        for trial_id in sorted(by_trial):
            trial_rows = by_trial[trial_id]
            trials.append({"trial_id": trial_id, **_row_counts(trial_rows)})
        containers.append(
            {
                "container_id": container_id,
                **_row_counts(container_rows),
                "trial_count": len(by_trial),
                "trials": trials,
            }
        )
    return {
        "container_count": len(by_container),
        "trial_count": trial_total,
        "containers": containers,
    }


def _summarize_metric_family(
    rows: Sequence[JSONDict],
    *,
    family: str,
    unit: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    minimum_p95_samples: int,
) -> JSONDict:
    if not rows:
        return {}
    metric_names = sorted(cast(dict[str, JSONValue], rows[0][family]))
    result: JSONDict = {}
    for metric_name in metric_names:
        by_container: dict[str, list[int]] = defaultdict(list)
        for row in sorted(rows, key=_row_sort_key):
            metrics = cast(dict[str, JSONValue], row[family])
            by_container[cast(str, row["container_id"])].append(cast(int, metrics[metric_name]))
        values = [value for group in by_container.values() for value in group]
        p50 = _nearest_rank(values, 0.50)
        can_report_p95 = len(values) >= minimum_p95_samples
        p95 = _nearest_rank(values, 0.95) if can_report_p95 else None
        bootstrap = _hierarchical_bootstrap(
            by_container,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=_metric_seed(seed, family, metric_name),
            include_p95=can_report_p95,
        )
        result[metric_name] = {
            "unit": unit,
            "n": len(values),
            "container_count": len(by_container),
            "p50": p50,
            "p95": p95,
            "p95_status": "reported" if can_report_p95 else "insufficient_sample_count",
            "minimum_p95_samples": minimum_p95_samples,
            "confidence_interval": {
                "method": "hierarchical_container_request_bootstrap",
                "confidence_level": confidence_level,
                "replicates": bootstrap_samples,
                "p50": cast(JSONValue, bootstrap["p50"]),
                "p95": cast(JSONValue, bootstrap["p95"]),
            },
        }
    return result


def _hierarchical_bootstrap(
    by_container: Mapping[str, Sequence[int]],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    include_p95: bool,
) -> dict[str, list[int] | None]:
    containers = sorted(by_container)
    rng = random.Random(seed)
    p50_replicates: list[int] = []
    p95_replicates: list[int] = []
    for _ in range(bootstrap_samples):
        sampled: list[int] = []
        for _ in containers:
            container_id = containers[rng.randrange(len(containers))]
            requests = by_container[container_id]
            sampled.extend(requests[rng.randrange(len(requests))] for _ in requests)
        p50_replicates.append(_nearest_rank(sampled, 0.50))
        if include_p95:
            p95_replicates.append(_nearest_rank(sampled, 0.95))

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "p50": [
            _nearest_rank(p50_replicates, alpha),
            _nearest_rank(p50_replicates, 1.0 - alpha),
        ],
        "p95": (
            [
                _nearest_rank(p95_replicates, alpha),
                _nearest_rank(p95_replicates, 1.0 - alpha),
            ]
            if include_p95
            else None
        ),
    }


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("an order statistic requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _metric_seed(seed: int, family: str, metric_name: str) -> int:
    payload = f"{seed}\0{family}\0{metric_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _row_sort_key(row: JSONDict) -> tuple[str, str, str]:
    return (
        cast(str, row["container_id"]),
        cast(str, row["trial_id"]),
        cast(str, row["request_id"]),
    )
