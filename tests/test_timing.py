from __future__ import annotations

import json

import pytest

from unseen_loop.timing import TIMING_ROW_SCHEMA, summarize_timing_rows


def timing_row(
    request: int,
    *,
    container: str = "container-a",
    trial: str = "trial-0",
    warmup: bool = False,
    success: bool = True,
    evaluate_ns: int | None = None,
    request_bytes: int = 200,
    response_bytes: int = 80,
    failure_code: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": TIMING_ROW_SCHEMA,
        "context_digest": "a" * 64,
        "container_id": container,
        "trial_id": trial,
        "request_id": f"request-{request:03d}",
        "is_warmup": warmup,
        "success": success,
        "timing_ns": (
            {"server_evaluate": evaluate_ns if evaluate_ns is not None else request}
            if success
            else {}
        ),
        "byte_metrics": (
            {"request_bytes": request_bytes, "response_bytes": response_bytes} if success else {}
        ),
    }
    if failure_code is not None:
        row["failure_code"] = failure_code
    return row


def test_failures_are_retained_exactly_but_excluded_from_metrics() -> None:
    failed = timing_row(2, success=False, failure_code="server.timeout")
    summary = summarize_timing_rows([timing_row(1, evaluate_ns=10), failed], bootstrap_samples=100)

    assert summary["row_counts"] == {
        "total": 2,
        "warmup_excluded": 0,
        "measured": 2,
        "successful": 1,
        "failed": 1,
        "warmup_failed": 0,
        "failures_retained": 1,
    }
    assert summary["failures"] == [failed]
    assert summary["timing_ns"]["server_evaluate"]["n"] == 1
    assert summary["timing_ns"]["server_evaluate"]["p50"] == 10
    assert summary["denominators"] == {
        "attempted_measured_requests": 2,
        "successful_measured_requests": 1,
        "failed_measured_requests": 1,
        "success_fraction": "1/2",
    }
    assert summary["method"]["quantiles_conditioning"] == "conditional_on_success"
    assert summary["release_quality"]["eligible"] is False
    assert summary["context_digest"] == "a" * 64
    json.dumps(summary, allow_nan=False)


def test_warmups_are_explicitly_counted_and_excluded() -> None:
    warmup_failure = timing_row(1, warmup=True, success=False, failure_code="warmup.transient")
    summary = summarize_timing_rows(
        [
            timing_row(0, warmup=True, evaluate_ns=1_000_000),
            warmup_failure,
            timing_row(2, evaluate_ns=10),
            timing_row(3, evaluate_ns=20),
        ],
        bootstrap_samples=100,
    )

    assert summary["method"]["warmups_excluded"] is True
    assert summary["row_counts"]["warmup_excluded"] == 2
    assert summary["row_counts"]["measured"] == 2
    assert summary["row_counts"]["failed"] == 0
    assert summary["row_counts"]["warmup_failed"] == 1
    assert summary["row_counts"]["failures_retained"] == 1
    assert "warmup_failures_present" in summary["release_quality"]["ineligibility_reasons"]
    assert summary["timing_ns"]["server_evaluate"]["n"] == 2
    assert summary["timing_ns"]["server_evaluate"]["p50"] == 10
    assert summary["failures"] == [warmup_failure]


def test_p95_is_refused_until_nearest_rank_has_twenty_samples() -> None:
    insufficient = summarize_timing_rows(
        [timing_row(index, evaluate_ns=index) for index in range(1, 20)],
        bootstrap_samples=100,
    )["timing_ns"]["server_evaluate"]

    assert insufficient["n"] == 19
    assert insufficient["p95"] is None
    assert insufficient["p95_status"] == "insufficient_sample_count"
    assert insufficient["minimum_p95_samples"] == 20
    assert insufficient["confidence_interval"]["p95"] is None

    sufficient = summarize_timing_rows(
        [timing_row(index, evaluate_ns=index) for index in range(1, 21)],
        bootstrap_samples=100,
    )["timing_ns"]["server_evaluate"]
    assert sufficient["p50"] == 10
    assert sufficient["p95"] == 19
    assert sufficient["p95_status"] == "reported"
    assert sufficient["confidence_interval"]["p95"] is not None


def test_container_and_trial_groups_are_preserved() -> None:
    rows = [
        timing_row(1, container="container-b", trial="trial-2"),
        timing_row(2, container="container-a", trial="trial-1"),
        timing_row(3, container="container-a", trial="trial-0", warmup=True),
        timing_row(
            4,
            container="container-a",
            trial="trial-0",
            success=False,
            failure_code="request.rejected",
        ),
    ]
    grouping = summarize_timing_rows(rows, bootstrap_samples=100)["grouping"]

    assert grouping["container_count"] == 2
    assert grouping["trial_count"] == 3
    assert [group["container_id"] for group in grouping["containers"]] == [
        "container-a",
        "container-b",
    ]
    container_a = grouping["containers"][0]
    assert container_a["trial_count"] == 2
    assert container_a["failed"] == 1
    assert [trial["trial_id"] for trial in container_a["trials"]] == [
        "trial-0",
        "trial-1",
    ]
    assert container_a["trials"][0]["warmup_excluded"] == 1
    assert container_a["trials"][0]["failed"] == 1


def test_hierarchical_bootstrap_is_seeded_and_input_order_independent() -> None:
    rows = [
        timing_row(
            container_index * 10 + request,
            container=f"container-{container_index}",
            trial=f"trial-{request // 2}",
            evaluate_ns=container_index * 1_000 + request * 10,
        )
        for container_index in range(3)
        for request in range(5)
    ]

    first = summarize_timing_rows(rows, bootstrap_samples=200, seed=2718)
    second = summarize_timing_rows(reversed(rows), bootstrap_samples=200, seed=2718)

    assert first == second
    interval = first["timing_ns"]["server_evaluate"]["confidence_interval"]
    assert interval["method"] == "hierarchical_container_request_bootstrap"
    assert interval["replicates"] == 200
    assert first["method"]["bootstrap"] == "resample_containers_then_requests"


def test_release_quality_requires_four_complete_container_groups_and_warmups() -> None:
    rows = [
        timing_row(
            request,
            container=f"container-{container_index}",
            trial="measured",
            evaluate_ns=container_index * 1_000 + request,
        )
        for container_index in range(4)
        for request in range(16)
    ]
    rows.extend(
        timing_row(
            100 + warmup,
            container=f"container-{container_index}",
            trial="warmup",
            warmup=True,
            evaluate_ns=container_index * 1_000 + warmup,
        )
        for container_index in range(4)
        for warmup in range(3)
    )

    summary = summarize_timing_rows(rows, bootstrap_samples=100)

    assert summary["release_quality"]["eligible"] is True
    assert summary["release_quality"]["ineligibility_reasons"] == []
    assert summary["release_quality"]["observed"] == {
        "measured_requests": 64,
        "successful_requests": 64,
        "failed_requests": 0,
        "warmup_failures": 0,
        "measured_containers": 4,
        "minimum_successful_requests_per_container": 16,
        "minimum_excluded_warmups_per_container": 3,
    }
    assert summary["method"]["quantiles_conditioning"] == ("all_measured_requests_succeeded")


def test_byte_metrics_use_the_same_order_statistics_and_p95_gate() -> None:
    summary = summarize_timing_rows(
        [
            timing_row(
                index,
                request_bytes=100 + index,
                response_bytes=40 + 2 * index,
            )
            for index in range(1, 21)
        ],
        bootstrap_samples=100,
    )

    requests = summary["byte_metrics"]["request_bytes"]
    responses = summary["byte_metrics"]["response_bytes"]
    assert requests["unit"] == "bytes"
    assert requests["p50"] == 110
    assert requests["p95"] == 119
    assert responses["p50"] == 60
    assert responses["p95"] == 78


def test_only_strictly_sanitized_rows_are_accepted() -> None:
    with_plaintext = timing_row(1)
    with_plaintext["observation"] = [0.5, -1.0]
    with pytest.raises(ValueError, match="non-sanitized fields: observation"):
        summarize_timing_rows([with_plaintext], bootstrap_samples=100)

    wrong_schema = timing_row(1)
    wrong_schema["schema_version"] = "untrusted"
    with pytest.raises(ValueError, match="schema_version"):
        summarize_timing_rows([wrong_schema], bootstrap_samples=100)

    bad_failure = timing_row(1, success=False, failure_code="raw exception text")
    with pytest.raises(ValueError, match="sanitized failure_code"):
        summarize_timing_rows([bad_failure], bootstrap_samples=100)


def test_rows_from_different_fixed_contexts_cannot_be_pooled() -> None:
    first = timing_row(1)
    second = timing_row(2)
    second["context_digest"] = "b" * 64

    with pytest.raises(ValueError, match="different context digests cannot be pooled"):
        summarize_timing_rows([first, second], bootstrap_samples=100)


def test_duplicate_request_identity_is_rejected() -> None:
    duplicate = timing_row(1)
    with pytest.raises(ValueError, match="duplicate timing request identity"):
        summarize_timing_rows([duplicate, duplicate], bootstrap_samples=100)
