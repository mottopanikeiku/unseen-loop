from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from unseen_loop.flagship.executor_private_ope import PrivateOPEAttempt, VerificationMetrics
from unseen_loop.flagship.manifest import PlannedJob
from unseen_loop.flagship.registry import AppendOnlyRegistry, JobStatus, Provenance

_module_spec = importlib.util.spec_from_file_location(
    "modal_flagship",
    Path(__file__).resolve().parents[1] / "modal_flagship.py",
)
assert _module_spec is not None and _module_spec.loader is not None
runtime = importlib.util.module_from_spec(_module_spec)
sys.modules[_module_spec.name] = runtime
_module_spec.loader.exec_module(runtime)


def runtime_receipt():
    return {
        "image_id": "im-test",
        "image_spec_sha256": "b" * 64,
        "code_commit": "a" * 40,
        **{
            f"{name}_code_sha256": "c" * 64
            for name in ("candidate", "baseline", "domain", "analysis")
        },
        "lockfile_sha256": "d" * 64,
        "python_version": "3.12.13",
        "numpy_version": "1.26.4",
        "scipy_version": "1.14.1",
        "tenseal_version": "0.3.17",
        "seal_version": "not-exposed",
        "modal_version": "1.5.4",
        "source_match": True,
        "execution_site": "Modal",
    }


class Claims:
    def __init__(self):
        self.values = {}

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.fixture
def volume(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime, "evidence_volume", SimpleNamespace(commit=lambda: None, reload=lambda: None)
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    return tmp_path


def job(index=0):
    return PlannedJob(
        f"job-private_ope_diagnostic-{index:024x}",
        "private_ope_diagnostic",
        index + 1,
        tuple(
            sorted(
                {
                    "kind": "protocol_verification",
                    "cohort": "verification",
                    "case_index": index,
                    "trajectories": 0,
                    "horizon": 0,
                    "behavior": "none",
                    "arm": "none",
                    "wave_index": 0,
                    "case_id": "not-applicable",
                    "data_seed": "not-applicable",
                    "bootstrap_seed": "not-applicable",
                }.items()
            )
        ),
    )


def setup_run(volume, jobs):
    config = b'phase = "diagnostic"\n'
    root = volume / "private-ope" / runtime._private_run_id(config, "diagnostic")
    runtime._private_publish(root / "config.toml", config)
    runtime._private_publish(
        root / "provenance.json", runtime._private_json({"runtime": runtime_receipt()})
    )
    registry = AppendOnlyRegistry.create(
        root / "registry.jsonl",
        jobs=jobs,
        provenance=Provenance.from_mapping(
            source_digest="a" * 64,
            config_digest=runtime._private_digest(config),
            image_digests={"private_ope": "b" * 64},
        ),
    )
    return root, registry


def entry_for(root, item, now, *, deadline=None):
    deadline = deadline if deadline is not None else now + timedelta(seconds=2100)
    intent = runtime._private_json(
        {"job_id": item.job_id, "deadline_utc": runtime._private_utc(deadline)}
    )
    runtime._private_publish(root / "dispatch" / f"{item.job_id}.json", intent)
    entry = {
        "schema_version": "unseen-loop/private-ope-worker-entry-v1",
        "run_id": root.name,
        "job_id": item.job_id,
        "input_id": f"in-{item.seed}",
        "function_call_id": f"fc-{item.seed}",
        "config_sha256": runtime._private_digest((root / "config.toml").read_bytes()),
        "entered_at_utc": runtime._private_utc(now),
        "dispatch_intent_sha256": runtime._private_digest(intent),
        "deadline_utc": runtime._private_utc(deadline),
        "claim_acquired": True,
    }
    path = (
        runtime.VOLUME_ROOT / "private-ope-transport" / root.name / item.job_id / entry["input_id"]
    )
    runtime._private_publish(path / "entry.json", runtime._private_json(entry))
    return entry, path


def publish_success(root, item, now, *, deadline=None, entered_at=None):
    entry, path = entry_for(
        root, item, entered_at if entered_at is not None else now, deadline=deadline
    )
    failed = runtime._private_failure(
        root, item, "verification.failed", entry=entry, runtime=runtime_receipt()
    )
    metrics = VerificationMetrics.from_dict(
        {
            "kind": "verification",
            "test_source_sha256": "c" * 64,
            "collected_node_ids": ["tests/test_ratio_lift_wpdis.py::test_binding"],
            "outcomes": [
                {"node_id": "tests/test_ratio_lift_wpdis.py::test_binding", "outcome": "passed"}
            ],
            "exit_code": 0,
            "required_cases_passed": True,
            "elapsed_ns": 10,
        }
    )
    row = dataclasses.replace(failed, completed=True, failure_code=None, metrics=metrics)
    data = runtime._private_json(row.to_dict())
    digest = runtime._private_digest(data)
    runtime._private_publish(path / "result.json", data)
    runtime._private_publish(
        path / "finished.json",
        runtime._private_json(
            {
                "finished_at_utc": runtime._private_utc(now),
                "result_sha256": digest,
            }
        ),
    )
    envelope = {
        "schema_version": "unseen-loop/private-ope-transport-v1",
        "run_id": root.name,
        "job_id": item.job_id,
        "function_call_id": entry["function_call_id"],
        "input_id": entry["input_id"],
        "entry_path": (path / "entry.json").relative_to(runtime.VOLUME_ROOT).as_posix(),
        "result_path": (path / "result.json").relative_to(runtime.VOLUME_ROOT).as_posix(),
        "result_sha256": digest,
        "delivery": "result",
    }
    runtime._private_publish(path / "transport.json", runtime._private_json(envelope))
    return row, envelope


def test_dispatch_window_is_unknown_not_not_dispatched(volume):
    item = job()
    root, _ = setup_run(volume, (item,))
    untouched = runtime._private_failure(root, item, "runtime.not_dispatched")
    assert untouched.attempted is False
    runtime._private_publish(root / "dispatch" / f"{item.job_id}.json", b"{}")
    unknown = runtime._private_failure(root, item, "runtime.dispatch_unknown")
    assert unknown.attempted is None
    assert unknown.metrics is None


def test_entry_proves_attempt_without_a_result(volume):
    item = job()
    root, _ = setup_run(volume, (item,))
    entered, _ = entry_for(root, item, datetime.now(UTC))
    interrupted = runtime._private_failure(root, item, "runtime.interrupted", entry=entered)
    assert interrupted.attempted is True
    assert interrupted.completed is False
    assert interrupted.function_call_id == entered["function_call_id"]


def test_late_transport_cannot_revise_closed_attempt(volume):
    item = job()
    root, registry = setup_run(volume, (item,))
    cutoff = datetime.now(UTC)
    entry_for(root, item, cutoff + timedelta(seconds=1))
    original = runtime._private_failure(root, item, "runtime.dispatch_unknown")
    runtime._private_terminalize(root, registry, original)
    later, _ = publish_success(root, item, cutoff + timedelta(seconds=1))
    assert runtime._private_harvest(root, item, cutoff) is None
    with pytest.raises(ValueError):
        runtime._private_terminalize(root, registry, later)
    retained = PrivateOPEAttempt.from_dict(
        json.loads((root / "attempts" / f"{item.job_id}.json").read_bytes())
    )
    assert retained.failure_code == "runtime.dispatch_unknown"


def test_nested_original_ledgers_are_included_in_closure(volume):
    item = job()
    root, registry = setup_run(volume, (item,))
    runtime._private_publish(
        root / "inputs" / "historical" / "checksums.sha256", b"original ledger\n"
    )
    runtime._private_terminalize(
        root, registry, runtime._private_failure(root, item, "runtime.not_dispatched")
    )
    index_path = runtime._private_close_index(root, registry)
    original_index = (root / "evidence-index.json").read_bytes()
    assert "inputs/historical/checksums.sha256" in (root / "checksums.sha256").read_text()
    assert runtime._private_close_index(root, registry) == index_path
    assert (root / "evidence-index.json").read_bytes() == original_index
    (root / "inputs" / "historical" / "checksums.sha256").write_bytes(b"changed")
    with pytest.raises(ValueError):
        runtime._private_close_index(root, registry)


def test_success_error_success_keeps_all_fixed_slots(volume, monkeypatch):
    jobs = tuple(job(index) for index in range(3))
    root, registry = setup_run(volume, jobs)
    clock = [datetime.now(UTC)]

    def now():
        clock[0] += timedelta(seconds=10)
        return clock[0]

    monkeypatch.setattr(runtime, "_private_now", now)
    monkeypatch.setattr(runtime, "_private_budget_available", lambda *args, **kwargs: True)
    claims = Claims()
    claims.put(("seed-root", "fixed"), {"reserved": True})
    monkeypatch.setattr(runtime, "_private_claims", lambda: claims)
    dispatched = []

    def dispatch(config, manifest, run_root, ledger, item):
        dispatched.append(item.job_id)
        ledger.started(item.job_id)
        if item.seed == 2:
            entry_for(run_root, item, clock[0])

            def get(**kwargs):
                raise RuntimeError("sanitized worker interruption")
        else:
            _, envelope = publish_success(run_root, item, clock[0])

            def get(**kwargs):
                return envelope

        return {
            "job": item,
            "call": SimpleNamespace(get=get, cancel=lambda **kwargs: None),
            "deadline": clock[0] + timedelta(seconds=2100),
            "cutoff": clock[0] + timedelta(seconds=2160),
            "failure": None,
            "settled": False,
        }

    monkeypatch.setattr(runtime, "_private_dispatch", dispatch)
    manifest = SimpleNamespace(seed_root="fixed")
    for batch in (jobs[:2], jobs[2:]):
        runtime._private_run_batch(
            b"",
            manifest,
            root,
            registry,
            batch,
            jobs,
            {"cpu_hour_usd": 0.04730, "memory_gib_hour_usd": 0.008},
            clock[0] + timedelta(hours=22),
        )
    rows = [json.loads((root / "attempts" / f"{item.job_id}.json").read_bytes()) for item in jobs]
    assert dispatched == [item.job_id for item in jobs]
    assert [row["completed"] for row in rows] == [True, False, True]
    assert rows[1]["failure_code"] == "runtime.interrupted"
    assert all(row.status == JobStatus.SUCCEEDED for row in registry.snapshot().records)


def test_conflicting_publish_preserves_original_bytes(volume):
    destination = volume / "private-ope-transport" / "receipt.json"
    runtime._private_publish(destination, b"original")
    runtime._private_publish(destination, b"original")
    with pytest.raises(ValueError):
        runtime._private_publish(destination, b"conflict")
    assert destination.read_bytes() == b"original"


def coordinator_fixture(volume, monkeypatch):
    from unseen_loop.flagship import manifest as manifest_module

    first = job()
    coordinates = dict(job(1).coordinates)
    coordinates.update(kind="analysis", cohort="analysis")
    analysis = dataclasses.replace(job(1), coordinates=tuple(sorted(coordinates.items())))
    jobs = (first, analysis)
    manifest = SimpleNamespace(
        phase="diagnostic",
        seed_root="fixed-diagnostic",
        execution=SimpleNamespace(budget_envelope_usd=10.0, wave_timeout_s=79200),
    )
    claims = Claims()
    monkeypatch.setattr(manifest_module, "parse_private_ope_manifest_bytes", lambda _: manifest)
    monkeypatch.setattr(manifest_module, "iter_private_ope_jobs", lambda _: jobs)
    monkeypatch.setattr(
        runtime,
        "_private_runtime",
        lambda _: {"image_id": "im-fixed", "image_spec_sha256": "b" * 64},
    )
    monkeypatch.setattr(runtime, "_private_sources", lambda _: {"fixed": True})
    monkeypatch.setattr(runtime, "_private_verified_inputs", lambda _: {})
    monkeypatch.setattr(
        runtime,
        "_private_budget_guard",
        lambda *args, **kwargs: {
            "projected_stage_with_overhead_usd": 10.0,
            "cpu_hour_usd": 0.04730,
            "memory_gib_hour_usd": 0.008,
        },
    )
    monkeypatch.setattr(runtime, "_private_claims", lambda: claims)
    monkeypatch.setattr(runtime.modal, "current_function_call_id", lambda: "fc-new")
    monkeypatch.setattr(runtime.modal, "current_input_id", lambda: "in-new")
    monkeypatch.setattr(
        runtime, "_private_dispatch", lambda *args: pytest.fail("unexpected worker dispatch")
    )
    config = b'phase = "diagnostic"\n'
    root = volume / "private-ope" / runtime._private_run_id(config, "diagnostic")
    reservation = {
        "run_id": root.name,
        "config_sha256": runtime._private_digest(config),
        "seed_root": manifest.seed_root,
        "origin_function_call_id": "fc-original",
        "origin_input_id": "in-original",
    }
    permanent = (
        volume
        / "private-ope-reservations"
        / f"{runtime._private_digest(manifest.seed_root.encode())}.json"
    )
    return config, root, manifest, jobs, claims, reservation, permanent


def test_second_launch_performs_no_worker_dispatch(volume, monkeypatch):
    config, root, manifest, _jobs, claims, reservation, permanent = coordinator_fixture(
        volume, monkeypatch
    )
    claims.put(("seed-root", manifest.seed_root), reservation)
    assert runtime._private_coordinator_body(config, 0, "launch") == str(permanent)
    assert not (root / "dispatch").exists()
    assert json.loads(permanent.read_bytes()) == reservation


def test_wrong_predecessor_fails_before_namespace_reservation(volume, monkeypatch):
    config, _, _, _, claims, _, _ = coordinator_fixture(volume, monkeypatch)

    def reject(_):
        raise ValueError("wrong predecessor digest")

    monkeypatch.setattr(runtime, "_private_verified_inputs", reject)
    with pytest.raises(ValueError):
        runtime._private_coordinator_body(config, 0, "launch")
    assert claims.values == {}


def test_missing_initialization_marker_recovery_never_dispatches(volume, monkeypatch):
    config, root, manifest, jobs, claims, reservation, permanent = coordinator_fixture(
        volume, monkeypatch
    )
    claims.put(("seed-root", manifest.seed_root), reservation)
    runtime._private_publish(permanent, runtime._private_json(reservation))
    index = runtime._private_coordinator_body(config, 0, "recover")
    assert index == str(root / "evidence-index.json")
    rows = [json.loads((root / "attempts" / f"{item.job_id}.json").read_bytes()) for item in jobs]
    assert all(
        row["attempted"] is False and row["failure_code"] == "runtime.not_dispatched"
        for row in rows
    )
    assert not (root / "dispatch").exists()


def test_visibility_grace_does_not_extend_computation_deadline(volume):
    on_time, late = job(), job(1)
    root, _ = setup_run(volume, (on_time, late))
    deadline = datetime.now(UTC)
    for item, finished in ((on_time, deadline), (late, deadline + timedelta(seconds=1))):
        publish_success(
            root, item, finished, deadline=deadline, entered_at=deadline - timedelta(seconds=5)
        )
    visibility_cutoff = deadline + timedelta(seconds=60)
    assert runtime._private_harvest(root, on_time, visibility_cutoff).completed is True
    assert runtime._private_harvest(root, late, visibility_cutoff) is None


def test_same_worker_input_reentry_preserves_first_entry(volume, monkeypatch):
    item = job()
    root, _ = setup_run(volume, (item,))
    entered, path = entry_for(root, item, datetime.now(UTC))
    original = (path / "entry.json").read_bytes()
    claims = Claims()
    claims.put(
        ("job", root.name, item.job_id),
        {
            key: entered[key]
            for key in ("run_id", "job_id", "config_sha256", "function_call_id", "input_id")
        },
    )
    monkeypatch.setattr(runtime, "_private_claims", lambda: claims)
    monkeypatch.setattr(
        runtime.modal, "current_function_call_id", lambda: entered["function_call_id"]
    )
    monkeypatch.setattr(runtime.modal, "current_input_id", lambda: entered["input_id"])
    result = runtime._private_worker_body(
        (root / "config.toml").read_bytes(),
        {
            "job": {"job_id": item.job_id},
            "dispatch_intent_sha256": entered["dispatch_intent_sha256"],
            "deadline_utc": entered["deadline_utc"],
        },
        str(root),
    )
    assert result["delivery"] == "reentry"
    assert result["result_path"] is None
    assert (path / "entry.json").read_bytes() == original
    assert not (path / "result.json").exists()


def test_rescheduled_coordinator_closes_unknown_dispatch_without_replacement(volume, monkeypatch):
    config, root, manifest, jobs, claims, reservation, permanent = coordinator_fixture(
        volume, monkeypatch
    )
    monkeypatch.setattr(
        runtime.modal, "current_function_call_id", lambda: reservation["origin_function_call_id"]
    )
    monkeypatch.setattr(runtime.modal, "current_input_id", lambda: reservation["origin_input_id"])
    monkeypatch.setattr(runtime, "_private_budget_available", lambda *args, **kwargs: False)
    clock = [datetime.now(UTC)]

    def now():
        clock[0] += timedelta(seconds=10)
        return clock[0]

    monkeypatch.setattr(runtime, "_private_now", now)
    claims.put(("seed-root", manifest.seed_root), reservation)
    runtime._private_publish(permanent, runtime._private_json(reservation))
    registry = runtime._private_initialize(
        config,
        manifest,
        root,
        jobs,
        {},
        runtime._private_runtime(manifest),
        runtime._private_budget_guard(manifest),
    )
    registry.started(jobs[0].job_id)
    runtime._private_publish(
        root / "dispatch" / f"{jobs[0].job_id}.json",
        runtime._private_json(
            {
                "job_id": jobs[0].job_id,
                "deadline_utc": runtime._private_utc(clock[0] + timedelta(hours=1)),
            }
        ),
    )
    assert runtime._private_coordinator_body(config, 0, "launch") == str(
        root / "evidence-index.json"
    )
    rows = [json.loads((root / "attempts" / f"{item.job_id}.json").read_bytes()) for item in jobs]
    assert [row["failure_code"] for row in rows] == [
        "runtime.dispatch_unknown",
        "runtime.not_dispatched",
    ]
    assert [row["attempted"] for row in rows] == [None, False]


def test_batch_budget_and_wave_preflight_leave_slots_undispatched(volume, monkeypatch):
    items = (job(), job(1))
    root, registry = setup_run(volume, items)
    manifest = SimpleNamespace(seed_root="fixed")
    guard = {"cpu_hour_usd": 0.04730, "memory_gib_hour_usd": 0.008}
    monkeypatch.setattr(
        runtime, "_private_dispatch", lambda *args: pytest.fail("unexpected dispatch")
    )
    monkeypatch.setattr(runtime, "_private_budget_available", lambda *args, **kwargs: False)
    with pytest.raises(TimeoutError):
        runtime._private_run_batch(
            b"",
            manifest,
            root,
            registry,
            items,
            items,
            guard,
            datetime.now(UTC) + timedelta(hours=22),
        )
    monkeypatch.setattr(runtime, "_private_budget_available", lambda *args, **kwargs: True)
    with pytest.raises(TimeoutError):
        runtime._private_run_batch(
            b"",
            manifest,
            root,
            registry,
            items,
            items,
            guard,
            datetime.now(UTC) + timedelta(seconds=2160),
        )
    assert all(row.status is None for row in registry.snapshot().records)
    assert not (root / "dispatch").exists()


def test_budget_loss_between_workers_preserves_undispatched_slot(volume, monkeypatch):
    items = (job(), job(1))
    root, registry = setup_run(volume, items)
    manifest = SimpleNamespace(seed_root="fixed")
    guard = {"cpu_hour_usd": 0.04730, "memory_gib_hour_usd": 0.008}
    available = iter((True, False))
    monkeypatch.setattr(
        runtime, "_private_budget_available", lambda *args, **kwargs: next(available)
    )
    claims = Claims()
    claims.put(("seed-root", "fixed"), {"reserved": True})
    monkeypatch.setattr(runtime, "_private_claims", lambda: claims)

    def dispatch(config, manifest, run_root, ledger, item):
        ledger.started(item.job_id)
        runtime._private_publish(
            run_root / "dispatch" / f"{item.job_id}.json",
            runtime._private_json({"job_id": item.job_id}),
        )
        return {}

    monkeypatch.setattr(runtime, "_private_dispatch", dispatch)
    with pytest.raises(TimeoutError):
        runtime._private_run_batch(
            b"",
            manifest,
            root,
            registry,
            items,
            items,
            guard,
            datetime.now(UTC) + timedelta(hours=22),
        )
    assert [row.status for row in registry.snapshot().records] == [JobStatus.STARTED, None]
    assert (root / "dispatch" / f"{items[0].job_id}.json").is_file()
    assert not (root / "dispatch" / f"{items[1].job_id}.json").exists()


def test_wave_budget_requires_complete_current_cycle_window(monkeypatch):
    from decimal import Decimal

    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 10, 1, tzinfo=UTC)
    clock = [end - timedelta(seconds=79260)]
    monkeypatch.setattr(runtime, "_private_now", lambda: clock[0])
    guard = {
        "workspace": "fixed",
        "cycle_start_utc": runtime._private_utc(start),
        "cycle_end_utc": runtime._private_utc(end),
        "cpu_hour_usd": 0.04730,
        "memory_gib_hour_usd": 0.008,
        "usage_limit_usd": 10,
    }
    workspace = SimpleNamespace(
        name="fixed",
        hydrate=lambda: None,
        billing=SimpleNamespace(
            rates=lambda: {
                "cpu_hour_cost": Decimal("0.04730"),
                "mem_gib_hour_cost": Decimal("0.008"),
            },
            summary=lambda _: SimpleNamespace(metered_cost=Decimal("1")),
        ),
    )
    monkeypatch.setattr(runtime.modal.Workspace, "from_context", lambda: workspace)
    manifest = SimpleNamespace(execution=SimpleNamespace(analysis_deadline_s=2100))
    assert runtime._private_budget_available(manifest, guard, 5, required_window_s=79260)
    clock[0] += timedelta(seconds=1)
    assert not runtime._private_budget_available(manifest, guard, 5, required_window_s=79260)
    clock[0] = start - timedelta(seconds=1)
    assert not runtime._private_budget_available(manifest, guard, 5, required_window_s=79260)


def test_recovery_visibility_does_not_admit_post_cutoff_completion(volume, monkeypatch):
    item = job()
    root, registry = setup_run(volume, (item,))
    registry.started(item.job_id)
    clock = [datetime.now(UTC)]
    publish_success(
        root,
        item,
        clock[0] + timedelta(seconds=1),
        deadline=clock[0] + timedelta(hours=1),
        entered_at=clock[0] - timedelta(seconds=10),
    )
    monkeypatch.setattr(runtime, "_private_now", lambda: clock[0])

    def advance(seconds):
        clock[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(runtime.time, "sleep", advance)
    monkeypatch.setattr(
        runtime.modal.FunctionCall,
        "from_id",
        lambda _: SimpleNamespace(cancel=lambda **kwargs: None),
    )
    runtime._private_recover_rows(root, registry, (item,))
    row = json.loads((root / "attempts" / f"{item.job_id}.json").read_bytes())
    assert row["completed"] is False
    assert row["failure_code"] == "runtime.interrupted"
    assert row["attempted"] is True


def test_source_binding_rejects_omitted_and_extra_image_files(tmp_path, monkeypatch):
    for relative in runtime.PRIVATE_OPE_REQUIRED_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "modal_flagship.py"))
    file_manifest = {
        "schema_version": "unseen-loop/private-ope-code-manifest-v1",
        "code_commit": "a" * 40,
        "entries": [
            {"path": path, "sha256": digest}
            for path, digest in runtime._private_source_inventory(tmp_path).items()
        ],
    }
    bundle = {
        "schema_version": "unseen-loop/private-ope-source-bundle-v1",
        "manifests": {
            name: file_manifest for name in ("candidate", "baseline", "domain", "analysis")
        },
        "image_spec": runtime.PRIVATE_OPE_IMAGE_SPEC,
        "lockfile_sha256": runtime._private_digest((tmp_path / "uv.lock").read_bytes()),
    }
    execution = SimpleNamespace(
        code_commit="a" * 40,
        **{
            f"{name}_code_sha256": runtime._private_digest(runtime._private_json(file_manifest))
            for name in bundle["manifests"]
        },
        image_spec_sha256=runtime._private_digest(runtime._private_json(bundle["image_spec"])),
        lockfile_sha256=bundle["lockfile_sha256"],
    )
    manifest = SimpleNamespace(execution=execution)
    destination = tmp_path / runtime.PRIVATE_OPE_SOURCE_FILE
    destination.write_bytes(runtime._private_json(bundle))
    assert runtime._private_sources(manifest) == bundle
    omitted = {**file_manifest, "entries": []}
    bundle["manifests"]["candidate"] = omitted
    execution.candidate_code_sha256 = runtime._private_digest(runtime._private_json(omitted))
    destination.write_bytes(runtime._private_json(bundle))
    with pytest.raises(ValueError):
        runtime._private_sources(manifest)
    bundle["manifests"]["candidate"] = file_manifest
    execution.candidate_code_sha256 = runtime._private_digest(runtime._private_json(file_manifest))
    destination.write_bytes(runtime._private_json(bundle))
    (tmp_path / "src" / "sitecustomize.py").write_text("# unbound startup module")
    with pytest.raises(ValueError):
        runtime._private_sources(manifest)
