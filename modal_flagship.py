"""Modal-only orchestration for the closed CipherShield/private-OPE flagship plan.

There are no web endpoints.  Local entrypoints only validate/submit a plan; every
experimental job, analysis pass, and evidence finalization executes in Modal.
Stage executors are deliberately injected by module name so this scheduler stays
independent of experiment implementation while failing closed if an executor or
its security evidence is absent.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import modal

from unseen_loop.flagship.manifest import (
    PlannedJob,
    iter_stage_jobs,
    load_manifest,
    stage_dag,
)
from unseen_loop.flagship.registry import (
    AppendOnlyRegistry,
    JobStatus,
    Provenance,
    RegistryError,
    Transition,
    finalize_evidence,
)

APP_NAME = "unseen-loop-flagship"
VOLUME_NAME = "unseen-loop-flagship-evidence"
VOLUME_ROOT = Path("/flagship-evidence")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")
EXECUTOR_FUNCTION = "execute_flagship_job"

CORE_PACKAGES = ("numpy==1.26.4",)
FHE_PACKAGES = ("numpy==1.26.4", "concrete-python==2.10.0", "setuptools==75.3.0")
ALL_CRYPTO_PACKAGES = (*FHE_PACKAGES, "tenseal==0.3.17")

app = modal.App(APP_NAME)
evidence_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*CORE_PACKAGES)
    .add_local_python_source("unseen_loop")
)
fhe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*FHE_PACKAGES)
    .add_local_python_source("unseen_loop")
)
integration_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*ALL_CRYPTO_PACKAGES)
    .add_local_python_source("unseen_loop")
)


def _job_payload(job: PlannedJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "seed": job.seed,
        "coordinates": job.coordinate_dict(),
    }


def _load_executor(module_name: str) -> Callable[..., object]:
    if not isinstance(module_name, str) or not module_name.startswith("unseen_loop."):
        raise RuntimeError(
            "executor module must be an unseen_loop module in the pinned source image"
        )
    module = importlib.import_module(module_name)
    executor = getattr(module, EXECUTOR_FUNCTION, None)
    if not callable(executor):
        raise RuntimeError(f"{module_name} does not expose callable {EXECUTOR_FUNCTION}")
    return executor


def _execute_remote(
    executor_module: str,
    manifest_payload: dict[str, object],
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, str | None]:
    """Invoke one pinned executor and accept only a registry-safe result envelope."""
    executor = _load_executor(executor_module)
    try:
        raw = executor(
            manifest=manifest_payload,
            job=job_payload,
            evidence_root=run_root,
        )
    except TimeoutError:
        return {
            "status": "timed-out",
            "artifact_path": None,
            "artifact_digest": None,
            "reason_code": "executor.timeout",
        }
    except Exception as exc:  # Error text/private inputs are intentionally never persisted.
        return {
            "status": "failed",
            "artifact_path": None,
            "artifact_digest": None,
            "reason_code": f"executor.{type(exc).__name__.lower()}"[:128],
        }
    if not isinstance(raw, dict) or set(raw) != {
        "status",
        "artifact_path",
        "artifact_digest",
        "reason_code",
    }:
        return {
            "status": "failed",
            "artifact_path": None,
            "artifact_digest": None,
            "reason_code": "executor.invalid-envelope",
        }
    status = raw.get("status")
    if status not in {"succeeded", "rejected"}:
        return {
            "status": "failed",
            "artifact_path": None,
            "artifact_digest": None,
            "reason_code": "executor.invalid-status",
        }
    result = {
        "status": status,
        "artifact_path": raw.get("artifact_path")
        if isinstance(raw.get("artifact_path"), str)
        else None,
        "artifact_digest": raw.get("artifact_digest")
        if isinstance(raw.get("artifact_digest"), str)
        else None,
        "reason_code": raw.get("reason_code") if isinstance(raw.get("reason_code"), str) else None,
    }
    if status == "succeeded":
        if result["artifact_path"] is None or result["artifact_digest"] is None:
            return {
                "status": "failed",
                "artifact_path": None,
                "artifact_digest": None,
                "reason_code": "executor.missing-artifact-evidence",
            }
        evidence_volume.commit()
    return result


_COMMON = {
    "volumes": {str(VOLUME_ROOT): evidence_volume},
    "retries": 0,
    "timeout": 60 * 60,
}


@app.function(image=core_image, max_containers=64, **_COMMON)
def clear_shield_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    return _execute_remote(executor_module, manifest, job, run_root)


@app.function(
    image=fhe_image,
    cpu=16.0,
    memory=32_768,
    max_containers=1,
    **_COMMON,
)
def shield_fhe_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    return _execute_remote(executor_module, manifest, job, run_root)


@app.function(
    image=integration_image,
    cpu=4.0,
    memory=16_384,
    max_containers=32,
    **_COMMON,
)
def ope_validation_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    return _execute_remote(executor_module, manifest, job, run_root)


@app.function(
    image=integration_image,
    cpu=4.0,
    memory=16_384,
    max_containers=8,
    **_COMMON,
)
def integration_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    evidence_volume.reload()
    return _execute_remote(executor_module, manifest, job, run_root)


@app.function(
    image=integration_image,
    cpu=4.0,
    memory=16_384,
    max_containers=8,
    **_COMMON,
)
def timing_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    return _execute_remote(executor_module, manifest, job, run_root)


@app.function(
    image=core_image,
    cpu=8.0,
    memory=32_768,
    max_containers=1,
    volumes={str(VOLUME_ROOT): evidence_volume},
    retries=3,
    timeout=2 * 60 * 60,
)
def analysis_worker(
    executor_module: str, manifest: dict[str, object], job: dict[str, object], run_root: str
) -> dict[str, str | None]:
    for delay in (2, 4, 8, 16, 32, None):
        evidence_volume.reload()
        result = _execute_remote(executor_module, manifest, job, run_root)
        if (
            result.get("status") != "rejected"
            or result.get("reason_code") != "analysis.unverifiable-evidence"
            or delay is None
        ):
            return result
        time.sleep(delay)
    raise AssertionError("analysis convergence loop is exhaustive")


WORKERS: dict[str, Any] = {
    "clear_shield_matrix": clear_shield_worker,
    "shield_fhe_challenge": shield_fhe_worker,
    "ope_validation": ope_validation_worker,
    "integration": integration_worker,
    "timing": timing_worker,
    "analysis": analysis_worker,
}


def _batched(jobs: Iterator[PlannedJob], size: int) -> Iterator[tuple[PlannedJob, ...]]:
    batch: list[PlannedJob] = []
    for job in jobs:
        batch.append(job)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _result_transition(job: PlannedJob, result: object) -> Transition:
    if isinstance(result, BaseException):
        # Modal infrastructure exceptions are retained as terminal outcomes; never retried.
        timed_out = "timeout" in type(result).__name__.lower()
        return Transition(
            job.job_id,
            JobStatus.TIMED_OUT if timed_out else JobStatus.FAILED,
            reason_code="modal.timeout" if timed_out else "modal.remote-failure",
        )
    if not isinstance(result, dict):
        return Transition(job.job_id, JobStatus.FAILED, reason_code="modal.invalid-result")
    status = result.get("status")
    reason = result.get("reason_code")
    safe_reason = (
        reason
        if isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", reason)
        else None
    )
    if status == "succeeded":
        path, digest = result.get("artifact_path"), result.get("artifact_digest")
        if isinstance(path, str) and isinstance(digest, str):
            return Transition(
                job.job_id,
                JobStatus.SUCCEEDED,
                artifact_path=path,
                artifact_digest=digest,
            )
        return Transition(
            job.job_id,
            JobStatus.FAILED,
            reason_code="executor.missing-artifact-evidence",
        )
    if status == "rejected" and safe_reason is not None:
        return Transition(job.job_id, JobStatus.REJECTED, reason_code=safe_reason)
    if status == "timed-out" and safe_reason is not None:
        return Transition(job.job_id, JobStatus.TIMED_OUT, reason_code=safe_reason)
    return Transition(
        job.job_id,
        JobStatus.FAILED,
        reason_code=safe_reason or "executor.failure",
    )


@app.function(
    image=core_image,
    max_containers=1,
    volumes={str(VOLUME_ROOT): evidence_volume},
    retries=0,
    timeout=60 * 60,
)
def evidence_finalizer(run_id: str, manifest_digest: str) -> str:
    """Close the immutable registry and create the sole root evidence index."""
    evidence_volume.reload()
    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid run_id")
    run_root = VOLUME_ROOT / run_id
    registry = AppendOnlyRegistry(run_root / "registry.jsonl")
    (finalizer_job,) = tuple(
        record for record in registry.snapshot().records if record.stage == "evidence_finalizer"
    )
    if finalizer_job.status is not None:
        raise RegistryError("evidence finalizer already has an attempt")
    incomplete_upstream = [
        record.job_id
        for record in registry.snapshot().records
        if record.stage != "evidence_finalizer" and record.status != record.expected_terminal
    ]
    if incomplete_upstream:
        raise RegistryError("evidence finalizer rejected incomplete upstream jobs")
    registry.started(finalizer_job.job_id)
    candidates = (
        "shared/shield-fhe/shield-server.zip",
        "shared/shield-fhe/shield-client-specs.bin",
        "shared/shield-fhe/shield-receipt.json",
    )
    supporting_artifacts = tuple(
        relative for relative in candidates if (run_root / relative).is_file()
    )
    cache_lock = run_root / "shared" / "shield-fhe.lock"
    if cache_lock.is_symlink():
        raise RegistryError("shield cache lock cannot be a symlink")
    cache_lock.unlink(missing_ok=True)
    receipt_path = run_root / "evidence-finalizer-receipt.json"
    receipt = {
        "manifest_digest": manifest_digest,
        "registry_id": registry.snapshot().registry_id,
        "planned_jobs": len(registry.snapshot().plan),
        "supporting_artifacts": list(supporting_artifacts),
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    registry.succeeded(
        finalizer_job.job_id,
        artifact_path=receipt_path.relative_to(run_root).as_posix(),
        artifact_digest=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    index = finalize_evidence(
        registry,
        evidence_root=run_root,
        reject_extra_files=True,
        supporting_paths=supporting_artifacts,
    )
    evidence_volume.commit()
    return str(index)


@app.function(
    image=core_image,
    max_containers=1,
    volumes={str(VOLUME_ROOT): evidence_volume},
    retries=0,
    timeout=24 * 60 * 60,
)
def orchestrate(
    config_bytes: bytes,
    run_id: str,
    source_digest: str,
    image_digests: dict[str, str],
    executor_modules: dict[str, str],
) -> str:
    """Execute the dependency DAG, stopping after any failed stage."""
    from unseen_loop.flagship.manifest import parse_manifest_bytes

    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must contain 1-80 lowercase letters, digits, or hyphens")
    manifest = parse_manifest_bytes(config_bytes)
    required_executors = set(WORKERS)
    if set(executor_modules) != required_executors:
        raise ValueError("executor_modules must name every compute/analysis stage exactly")
    if set(image_digests) != {"core", "fhe", "integration"}:
        raise ValueError("image_digests must bind core, fhe, and integration images exactly")
    provenance = Provenance.from_mapping(
        source_digest=source_digest,
        config_digest=manifest.digest,
        image_digests=image_digests,
    )
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    all_jobs = (
        job for stage in stage_dag(manifest) for job in iter_stage_jobs(manifest, stage.name)
    )
    registry = AppendOnlyRegistry.create(
        run_root / "registry.jsonl", jobs=all_jobs, provenance=provenance
    )
    manifest_payload = manifest.canonical_payload()
    for stage in stage_dag(manifest):
        if stage.name == "evidence_finalizer":
            break
        jobs = iter_stage_jobs(manifest, stage.name)
        worker = WORKERS[stage.name]
        for batch in _batched(jobs, max(1, stage.max_parallel * 4)):
            registry.apply(Transition(job.job_id, JobStatus.STARTED) for job in batch)
            args = [
                (executor_modules[stage.name], manifest_payload, _job_payload(job), str(run_root))
                for job in batch
            ]
            results = worker.starmap(args, return_exceptions=True)
            registry.apply(
                _result_transition(job, result) for job, result in zip(batch, results, strict=True)
            )
            evidence_volume.commit()
        stage_records = [
            record for record in registry.snapshot().records if record.stage == stage.name
        ]
        if any(record.status != record.expected_terminal for record in stage_records):
            raise RegistryError(
                f"stage {stage.name} did not close successfully; "
                "downstream stages were not submitted"
            )
    evidence_volume.commit()
    return evidence_finalizer.remote(run_id, manifest.digest)


@app.local_entrypoint(name="inspect-plan")
def inspect_plan(config: str = "experiments/flagship.toml") -> None:
    """Validate locally and print IDs/counts only; it performs no empirical compute."""
    manifest = load_manifest(config)
    payload = {
        "manifest_digest": manifest.digest,
        "stages": [
            {
                "name": stage.name,
                "stage_id": stage.stage_id,
                "dependencies": stage.dependencies,
                "max_parallel": stage.max_parallel,
                "planned_jobs": sum(1 for _ in iter_stage_jobs(manifest, stage.name)),
            }
            for stage in stage_dag(manifest)
        ],
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


@app.local_entrypoint(name="launch")
def launch(
    config: str,
    run_id: str,
    source_digest: str,
    image_digests_json: str,
    executor_modules_json: str,
) -> None:
    """Submit the immutable plan; all study work remains remote."""
    image_digests = json.loads(image_digests_json)
    executor_modules = json.loads(executor_modules_json)
    if not isinstance(image_digests, dict) or not isinstance(executor_modules, dict):
        raise TypeError("digest and executor arguments must each encode a JSON object")
    result = orchestrate.remote(
        Path(config).read_bytes(), run_id, source_digest, image_digests, executor_modules
    )
    print(result)
