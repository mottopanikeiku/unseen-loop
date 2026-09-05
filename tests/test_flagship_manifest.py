from __future__ import annotations

import copy
import dataclasses
import json
from collections import Counter

import pytest

from unseen_loop.flagship import manifest as module
from unseen_loop.flagship.manifest import (
    ManifestError,
    canonical_json,
    content_digest,
    iter_private_ope_jobs,
    parse_private_ope_manifest_bytes,
    private_ope_fixed_tables,
)


def _payload(phase: str = "diagnostic") -> dict:
    result = private_ope_fixed_tables(phase)
    result["execution"].update(
        deployment_version=1,
        code_commit="a" * 40,
        **{
            field: "b" * 64
            for field in (
                "candidate_code_sha256",
                "baseline_code_sha256",
                "domain_code_sha256",
                "analysis_code_sha256",
                "lockfile_sha256",
                "image_spec_sha256",
                "budget_guard_sha256",
            )
        },
    )
    previous_phase = "diagnostic" if phase == "pilot" else "pilot"
    result["predecessors"] = {
        "historical_confirmation_id": "independent-confirmation-20260904-001",
        "historical_summary_sha256": (
            "9233207bce45cdc8cd95fa026f2026571715185d055a0bcd83655d33c87d6b17"
        ),
        "historical_ledger_sha256": (
            "4227d8e154979d49031547a32c026d63f0feeebb7e79cef3e04607114f1f6885"
        ),
        "previous_run_id": "not-applicable"
        if phase == "diagnostic"
        else f"private-ope-{previous_phase}-{'c' * 24}",
        "previous_config_sha256": "not-applicable" if phase == "diagnostic" else "c" * 64,
        "previous_index_sha256": "not-applicable" if phase == "diagnostic" else "d" * 64,
        "pilot_kernel_sha256": "e" * 64 if phase == "confirmation" else "not-applicable",
        "pilot_policies_sha256": "f" * 64 if phase == "confirmation" else "not-applicable",
    }
    return result


def _toml(payload: dict) -> bytes:
    # Test-only serializer: the frozen schema has one table level and JSON scalar/array values.
    rows = [
        f"{key} = {json.dumps(value)}"
        for key, value in payload.items()
        if not isinstance(value, dict)
    ]
    for key, value in payload.items():
        if isinstance(value, dict):
            rows.append(f"[{key}]")
            rows.extend(f"{field} = {json.dumps(item)}" for field, item in value.items())
    return ("\n".join(rows) + "\n").encode()


@pytest.mark.parametrize(
    "phase,total,waves,kind_counts",
    [
        (
            "diagnostic",
            16,
            {0: 16},
            {
                "protocol_verification": 1,
                "count_precision": 12,
                "smoke_error": 1,
                "smoke_timeout": 1,
                "analysis": 1,
            },
        ),
        (
            "pilot",
            156,
            {0: 128, 1: 28},
            {
                "clear_batch": 128,
                "paired_context": 20,
                "ablation_context": 6,
                "historical_context": 1,
                "analysis": 1,
            },
        ),
        (
            "confirmation",
            241,
            {0: 40, 1: 40, 2: 40, 3: 40, 4: 40, 5: 20, 6: 21},
            {
                "statistical_context": 200,
                "timing_context": 40,
                "analysis": 1,
            },
        ),
    ],
)
def test_fixed_slots_waves_and_exact_digest(phase, total, waves, kind_counts) -> None:
    raw = _toml(_payload(phase))
    manifest = parse_private_ope_manifest_bytes(raw)
    assert manifest.digest == content_digest(raw)
    assert manifest.to_dict() == _payload(phase)
    jobs = iter_private_ope_jobs(manifest)
    assert len(jobs) == total
    assert Counter(job.coordinate_dict()["wave_index"] for job in jobs) == waves
    assert Counter(job.coordinate_dict()["kind"] for job in jobs) == kind_counts
    assert len({job.job_id for job in jobs}) == total
    assert len({job.seed for job in jobs}) == total
    assert jobs[-1].coordinate_dict()["kind"] == "analysis"
    assert iter_private_ope_jobs(manifest) == jobs
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.domain.horizon = 8


@pytest.mark.parametrize("phase", ["pilot", "confirmation"])
def test_paired_arms_share_cases_not_job_seeds_and_alternate_order(phase: str) -> None:
    manifest = parse_private_ope_manifest_bytes(_toml(_payload(phase)))
    jobs = [
        job
        for job in iter_private_ope_jobs(manifest)
        if job.coordinate_dict()["cohort"] == "timing"
    ]
    seen_cases = set()
    for index in range(len(jobs) // 2):
        left, right = jobs[2 * index : 2 * index + 2]
        a, b = left.coordinate_dict(), right.coordinate_dict()
        assert [a["arm"], b["arm"]] == (
            ["lifted_prefix", "raw_prefix"] if index % 2 == 0 else ["raw_prefix", "lifted_prefix"]
        )
        assert a["case_id"] == b["case_id"]
        assert a["case_id"] not in seen_cases
        seen_cases.add(a["case_id"])
        assert a["data_seed"] == b["data_seed"]
        assert a["bootstrap_seed"] == b["bootstrap_seed"]
        assert isinstance(a["data_seed"], str) and a["data_seed"].isdigit()
        assert a["data_seed"] != a["bootstrap_seed"]
        assert left.seed != right.seed
    # Formatting changes bind a different config, not a different scientific case/job.
    reformatted = parse_private_ope_manifest_bytes(b"# identical study\n" + _toml(_payload(phase)))
    assert reformatted.digest != manifest.digest
    assert iter_private_ope_jobs(reformatted) == iter_private_ope_jobs(manifest)


def test_distinct_cases_and_phases_have_disjoint_seed_domains() -> None:
    seen = set()
    for phase in ("diagnostic", "pilot", "confirmation"):
        jobs = iter_private_ope_jobs(parse_private_ope_manifest_bytes(_toml(_payload(phase))))
        job_seeds = {job.seed for job in jobs}
        cases = {}
        for job in jobs:
            row = job.coordinate_dict()
            assert set(row) == {
                "kind",
                "cohort",
                "case_index",
                "trajectories",
                "horizon",
                "behavior",
                "arm",
                "wave_index",
                "case_id",
                "data_seed",
                "bootstrap_seed",
            }
            if row["case_id"] != "not-applicable":
                cases[row["case_id"]] = (int(row["data_seed"]), int(row["bootstrap_seed"]))
            else:
                assert row["data_seed"] == row["bootstrap_seed"] == "not-applicable"
        seeds = [seed for pair in cases.values() for seed in pair]
        assert len(seeds) == len(set(seeds))
        assert not job_seeds.intersection(seeds)
        current = job_seeds.union(seeds)
        assert not current.intersection(seen)
        seen.update(current)


def test_seed_collisions_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = parse_private_ope_manifest_bytes(_toml(_payload("pilot")))
    original = module.derive_seed
    monkeypatch.setattr(module, "derive_seed", lambda root, key: 1)
    with pytest.raises(ManifestError):
        iter_private_ope_jobs(manifest)
    # Distinct jobs can still collide in the independently domain-separated data seeds.
    monkeypatch.setattr(
        module,
        "derive_seed",
        lambda root, key: 1 if key.startswith("data:") else original(root, key),
    )
    with pytest.raises(ManifestError):
        iter_private_ope_jobs(manifest)


def test_every_scientific_field_is_frozen_not_a_search_parameter() -> None:
    frozen = private_ope_fixed_tables("diagnostic")
    for table, fields in frozen.items():
        if not isinstance(fields, dict):
            continue
        for field, value in fields.items():
            payload = _payload()
            if type(value) is bool:
                replacement = not value
            elif type(value) in (int, float):
                replacement = value + 1
            elif isinstance(value, list):
                replacement = list(reversed(value)) if len(value) > 1 else ["changed"]
            else:
                replacement = "changed"
            payload[table][field] = replacement
            with pytest.raises(ManifestError):
                parse_private_ope_manifest_bytes(_toml(payload))


def test_missing_and_unknown_fields_are_rejected_at_every_table() -> None:
    baseline = _payload()
    for key, value in baseline.items():
        missing_top = copy.deepcopy(baseline)
        del missing_top[key]
        with pytest.raises(ManifestError):
            parse_private_ope_manifest_bytes(_toml(missing_top))
        if isinstance(value, dict):
            for field in value:
                missing = copy.deepcopy(baseline)
                del missing[key][field]
                with pytest.raises(ManifestError):
                    parse_private_ope_manifest_bytes(_toml(missing))
            extra = copy.deepcopy(baseline)
            extra[key]["unexpected"] = 1
            with pytest.raises(ManifestError):
                parse_private_ope_manifest_bytes(_toml(extra))
    extra = copy.deepcopy(baseline)
    extra["digest"] = "0" * 64
    with pytest.raises(ManifestError):
        parse_private_ope_manifest_bytes(_toml(extra))


@pytest.mark.parametrize(
    "table,field,value",
    [
        ("domain", "horizon", True),
        ("domain", "gamma", True),
        ("crypto", "bootstrap_enabled", 0),
        ("crypto", "count_diagnostic_sizes", [64, 64]),
        ("execution", "deployment_version", 0),
        ("execution", "deployment_version", True),
        ("execution", "code_commit", "main"),
        ("execution", "candidate_code_sha256", "B" * 64),
        ("execution", "budget_guard_sha256", "not-applicable"),
        ("policies", "policy_order", ["A", "A"]),
    ],
)
def test_wrong_types_shapes_and_mutable_source_aliases_are_rejected(table, field, value) -> None:
    payload = _payload()
    payload[table][field] = value
    with pytest.raises(ManifestError):
        parse_private_ope_manifest_bytes(_toml(payload))


@pytest.mark.parametrize("literal", ["nan", "+inf", "-inf"])
def test_nonfinite_numeric_fields_are_rejected(literal: str) -> None:
    raw = _toml(_payload()).replace(b"gamma = 0.99", f"gamma = {literal}".encode())
    with pytest.raises(ManifestError):
        parse_private_ope_manifest_bytes(raw)
    with pytest.raises(ValueError):
        canonical_json({"value": float(literal)})


@pytest.mark.parametrize("raw", [b"\xff", b"[domain", b"phase = 1\nphase = 2\n"])
def test_malformed_manifest_bytes_are_rejected(raw: bytes) -> None:
    with pytest.raises(ManifestError):
        parse_private_ope_manifest_bytes(raw)


@pytest.mark.parametrize(
    "phase,field,value",
    [
        ("diagnostic", "previous_run_id", "latest"),
        ("pilot", "previous_run_id", "../private-ope-diagnostic-" + "c" * 24),
        ("pilot", "previous_run_id", "private-ope-pilot-" + "c" * 24),
        ("pilot", "previous_config_sha256", "e" * 64),
        ("pilot", "previous_index_sha256", "not-applicable"),
        ("pilot", "pilot_kernel_sha256", "e" * 64),
        ("confirmation", "pilot_policies_sha256", "not-applicable"),
        ("confirmation", "previous_run_id", "private-ope-pilot-latest"),
        ("diagnostic", "historical_summary_sha256", "a" * 64),
    ],
)
def test_predecessors_bind_fixed_history_and_phase_config(phase, field, value) -> None:
    payload = _payload(phase)
    payload["predecessors"][field] = value
    with pytest.raises(ManifestError):
        parse_private_ope_manifest_bytes(_toml(payload))


def test_fixed_empirical_shapes_retain_every_ablation_and_count_replica() -> None:
    diagnostic = [
        job.coordinate_dict()
        for job in iter_private_ope_jobs(
            parse_private_ope_manifest_bytes(_toml(_payload("diagnostic")))
        )
    ]
    counts = [row for row in diagnostic if row["kind"] == "count_precision"]
    assert {
        (row["case_index"], row["trajectories"], row["horizon"], row["arm"]) for row in counts
    } == {
        (index, n, h, arm)
        for index in range(3)
        for n in (64, 4096)
        for h, arm in ((8, "old24"), (64, "candidate40"))
    }
    assert diagnostic[0]["kind"] == "protocol_verification"
    pilot = [
        job.coordinate_dict()
        for job in iter_private_ope_jobs(parse_private_ope_manifest_bytes(_toml(_payload("pilot"))))
    ]
    clear = [row for row in pilot if row["kind"] == "clear_batch"]
    assert {
        (row["cohort"], row["case_index"], row["trajectories"], row["horizon"], row["behavior"])
        for row in clear
    } == {
        (cohort, index, 2048, 64, behavior)
        for cohort, behavior in (("screen_primary", "primary"), ("screen_stress", "stress"))
        for index in range(64)
    }
    ablations = [row for row in pilot if row["kind"] == "ablation_context"]
    assert {
        (row["case_index"], row["trajectories"], row["horizon"], row["arm"]) for row in ablations
    } == {
        (0, 4096, h, arm)
        for h in (8, 32, 64)
        for arm in ("lifted_prefix", "lifted_independent_products")
    }
    assert len({row["case_id"] for row in ablations}) == 3
    legacy = [row for row in pilot if row["kind"] == "historical_context"]
    assert [(row["trajectories"], row["horizon"], row["arm"]) for row in legacy] == [
        (64, 8, "raw_sequential_v1"),
    ]
