from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from unseen_loop.flagship.executor_timing import (
    COLOCATED_TRUST_SCOPE,
    BackendContextReceipt,
    TimingContextRequest,
    TimingObservation,
    UnsupportedTimingCell,
    execute_flagship_job,
)
from unseen_loop.flagship.manifest import PlannedJob, canonical_json, iter_stage_jobs, load_manifest

SMOKE = Path(__file__).parents[1] / "experiments" / "flagship-smoke.toml"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


@dataclass
class FakeSession:
    request: TimingContextRequest
    execution_label: str
    failed_indices: frozenset[int]

    @property
    def context_receipt(self) -> BackendContextReceipt:
        return BackendContextReceipt(
            backend="injected-test-backend",
            backend_version="1.2.3",
            execution_label=self.execution_label,
            implementation_id=self.request.implementation,
            circuit_digest=DIGEST_A,
            server_artifact_digest=DIGEST_B,
            client_artifact_digest=DIGEST_C,
            hardware_digest=DIGEST_D,
            compile_ns=13,
            key_setup_ns=17,
            evaluation_key_bytes=23,
        )

    def measure(self, request_index: int) -> TimingObservation:
        if request_index in self.failed_indices:
            return TimingObservation({}, {}, False, "backend.injected_failure")
        return TimingObservation(
            {
                "client_encrypt": 10 + request_index,
                "server_evaluate": 20 + request_index,
                "client_decrypt": 30 + request_index,
                "end_to_end": 60 + 3 * request_index,
            },
            {"evaluation_keys": 23, "request": 40, "response": 50},
        )


class FakeTimingBackend:
    def __init__(
        self,
        execution_label: str = "FHE SIMULATED",
        failed_indices: frozenset[int] = frozenset(),
        unsupported_shapes: frozenset[tuple[int | None, int | None]] = frozenset(),
    ) -> None:
        self.execution_label = execution_label
        self.failed_indices = failed_indices
        self.unsupported_shapes = unsupported_shapes
        self.requests: list[TimingContextRequest] = []

    def open_context(self, request: TimingContextRequest) -> FakeSession:
        self.requests.append(request)
        if (request.trajectories, request.horizon) in self.unsupported_shapes:
            raise UnsupportedTimingCell("timing.injected_unsupported")
        return FakeSession(request, self.execution_label, self.failed_indices)


def _job(manifest, kind: str, **wanted: int) -> PlannedJob:
    for job in iter_stage_jobs(manifest, "timing"):
        coordinates = job.coordinate_dict()
        if coordinates.get("kind") == kind and all(
            coordinates.get(name) == value for name, value in wanted.items()
        ):
            return job
    raise AssertionError(f"missing timing job {kind} {wanted}")


def _artifact(root: Path, envelope: dict[str, object]) -> tuple[Path, dict[str, object]]:
    assert set(envelope) == {"status", "artifact_path", "artifact_digest", "reason_code"}
    assert envelope["status"] == "succeeded"
    assert envelope["reason_code"] is None
    path = root / str(envelope["artifact_path"])
    encoded = path.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == envelope["artifact_digest"]
    payload = json.loads(encoded)
    assert encoded == canonical_json(payload)
    return path, payload


def test_shield_uses_manifest_warmups_and_retains_failures_without_replacement(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(SMOKE)
    job = _job(manifest, "shield", container=0)
    backend = FakeTimingBackend(failed_indices=frozenset({1, 4}))

    envelope = execute_flagship_job(manifest, job, tmp_path, backend=backend)
    path, payload = _artifact(tmp_path, envelope)

    assert path.relative_to(tmp_path).as_posix() == f"timing/{job.job_id}.json"
    assert len(backend.requests) == 1
    group = payload["groups"]["shield"]
    rows = group["rows"]
    warmups = manifest.systems.shield_warmups_per_container
    assert len(rows) == warmups + manifest.systems.shield_measured_per_container
    assert all(row["is_warmup"] for row in rows[:warmups])
    assert all(not row["is_warmup"] for row in rows[warmups:])
    assert [row["request_id"] for row in rows] == [
        f"shield-{index:04d}" for index in range(len(rows))
    ]
    assert group["summary"]["row_counts"] == {
        "total": len(rows),
        "warmup_excluded": warmups,
        "measured": manifest.systems.shield_measured_per_container,
        "successful": manifest.systems.shield_measured_per_container - 1,
        "failed": 1,
        "warmup_failed": 1,
        "failures_retained": 2,
    }
    assert [row["failure_code"] for row in group["summary"]["failures"]] == [
        "backend.injected_failure",
        "backend.injected_failure",
    ]
    assert payload["retry_policy"].startswith("zero retries")
    assert payload["trust_scope"] == COLOCATED_TRUST_SCOPE
    assert "network" in payload["trust_scope"]


def test_ope_uses_one_fixed_ckks_context_and_preserves_simulation_label(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE)
    job = _job(manifest, "ope", container=0)
    backend = FakeTimingBackend(execution_label="FHE SIMULATED")

    envelope = execute_flagship_job(manifest, job, tmp_path, backend=backend)
    _, payload = _artifact(tmp_path, envelope)

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.implementation == "POLYNOMIAL_APPROX_OPE_V1"
    assert request.trajectories == manifest.ope.fhe_challenge.trajectories_per_batch
    assert request.horizon == manifest.ope.fhe_challenge.horizon
    group = payload["groups"]["ope"]
    assert group["context"]["execution_label"] == "FHE SIMULATED"
    assert len({row["context_digest"] for row in group["rows"]}) == 1
    assert len(group["rows"]) == (
        manifest.systems.ope_warmups_per_container + manifest.systems.ope_measured_per_container
    )


@pytest.mark.parametrize("trajectories,horizon", [(64, 8), (256, 32), (1024, 64)])
def test_scale_executes_each_requested_cell_with_its_exact_shape(
    tmp_path: Path, trajectories: int, horizon: int
) -> None:
    manifest = load_manifest(
        SMOKE
        if trajectories != 1024 or horizon != 64
        else Path(__file__).parents[1] / "experiments" / "flagship.toml"
    )
    job = _job(manifest, "scale", trajectories=trajectories, horizon=horizon, container=0)
    backend = FakeTimingBackend()

    envelope = execute_flagship_job(manifest, job, tmp_path, backend=backend)
    _, payload = _artifact(tmp_path, envelope)

    request = backend.requests[0]
    assert (request.trajectories, request.horizon) == (trajectories, horizon)
    rows = payload["groups"]["ope"]["rows"]
    assert len(rows) == (
        manifest.systems.scale_warmups_per_container + manifest.systems.scale_measured_per_container
    )


def test_scale_explicitly_rejects_an_unsupported_backend_cell_without_artifact(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(SMOKE)
    job = _job(manifest, "scale", trajectories=64, horizon=8, container=0)
    backend = FakeTimingBackend(unsupported_shapes=frozenset({(64, 8)}))

    envelope = execute_flagship_job(manifest, job, tmp_path, backend=backend)

    assert envelope == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "timing.injected_unsupported",
    }
    assert not (tmp_path / "timing").exists()


def test_concurrent_client_measures_its_assigned_shield_and_ope_calls(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE)
    job = _job(manifest, "concurrent_client", client=0)
    backend = FakeTimingBackend()

    envelope = execute_flagship_job(manifest, job, tmp_path, backend=backend)
    _, payload = _artifact(tmp_path, envelope)

    assert [request.workload for request in backend.requests] == ["shield", "ope"]
    shield = payload["groups"]["shield"]
    ope = payload["groups"]["ope"]
    assert len(shield["rows"]) == manifest.systems.concurrent_shield_calls_per_client
    assert len(ope["rows"]) == manifest.systems.concurrent_ope_calls_per_client
    assert not any(row["is_warmup"] for row in (*shield["rows"], *ope["rows"]))
    assert shield["context"]["container_id"] == ope["context"]["container_id"]
    assert shield["context"]["context_digest"] != ope["context"]["context_digest"]
    assert shield["context"]["configured_clients"] == manifest.systems.concurrent_clients


def test_modal_mapping_payloads_use_the_same_executor_contract(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE)
    job = _job(manifest, "shield", container=0)
    manifest_payload = manifest.canonical_payload()
    job_payload = {
        "job_id": job.job_id,
        "stage": job.stage,
        "seed": job.seed,
        "coordinates": job.coordinate_dict(),
    }

    envelope = execute_flagship_job(
        manifest_payload, job_payload, tmp_path, backend=FakeTimingBackend()
    )
    _, payload = _artifact(tmp_path, envelope)

    assert payload["job_id"] == job.job_id
    assert payload["coordinates"] == job.coordinate_dict()
    assert len(payload["groups"]["shield"]["rows"]) == (
        manifest.systems.shield_warmups_per_container
        + manifest.systems.shield_measured_per_container
    )
