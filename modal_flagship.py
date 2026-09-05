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
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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


# The private comparison shares the flagship Volume and append-only registry.
# These helpers are deliberately stdlib-only: a worker claims its slot before
# importing an estimator or a cryptographic backend.
PRIVATE_OPE_CLAIMS = "unseen-loop-private-ope-claims"
PRIVATE_OPE_SOURCE_FILE = ".private-ope-source.json"
PRIVATE_OPE_IMAGE_DIR = ".private-ope-image"
PRIVATE_OPE_REQUIRED_SOURCES = frozenset(
    {
        "src/unseen_loop/ope/lifted.py",
        "src/unseen_loop/ope/study.py",
        "src/unseen_loop/ope/types.py",
        "src/unseen_loop/ope/ckks.py",
        "src/unseen_loop/crypto/ckks.py",
        "src/unseen_loop/flagship/executor_private_ope.py",
        "src/unseen_loop/flagship/manifest.py",
        "src/unseen_loop/flagship/registry.py",
        "tests/test_ratio_lift_wpdis.py",
        "tests/test_ope_ckks.py",
        "modal_flagship.py",
        "pyproject.toml",
        "uv.lock",
    }
)
PRIVATE_OPE_PACKAGES = (
    "numpy==1.26.4",
    "scipy==1.14.1",
    "tenseal==0.3.17",
    "modal==1.5.4",
    "pytest==8.4.2",
)
PRIVATE_OPE_IMAGE_SPEC = {
    "schema_version": "unseen-loop/private-ope-image-spec-v1",
    "base": "debian_slim",
    "python_version": "3.12",
    "packages": list(PRIVATE_OPE_PACKAGES),
    "image_builder_version": "2025.06",
    "image_builder_binding": "MODAL_IMAGE_BUILDER_VERSION",
    "source_mode": "committed_snapshot",
    "gpu": False,
}


def _private_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _private_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private_now() -> datetime:
    return datetime.now(UTC)


def _private_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _private_date(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("evidence.invalid_artifact")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != UTC:
        raise ValueError("evidence.invalid_artifact")
    return parsed


def _private_safe_path(root: Path, relative: str) -> Path:
    from pathlib import PurePosixPath

    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError("evidence.invalid_artifact")
    destination = root.joinpath(*path.parts)
    if root.is_symlink() or any(part.is_symlink() for part in (destination, *destination.parents)):
        raise ValueError("evidence.invalid_artifact")
    return destination


def _private_publish(path: Path, data: bytes) -> None:
    """Publish complete bytes without replacing evidence, even on a crash."""
    path = _private_safe_path(VOLUME_ROOT, path.relative_to(VOLUME_ROOT).as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("evidence.invalid_artifact")
        return
    staging = VOLUME_ROOT / "private-ope-staging"
    staging.mkdir(exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="publish-", dir=staging)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # The singleton coordinator (or uniquely claimed worker namespace) is
        # the mutation authority. Do not assume Volume v1 supports hard links.
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError("evidence.invalid_artifact")
        else:
            os.rename(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _private_read_json(path: Path) -> dict[str, Any]:
    path = _private_safe_path(VOLUME_ROOT, path.relative_to(VOLUME_ROOT).as_posix())
    if not path.is_file():
        raise ValueError("evidence.invalid_artifact")
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("evidence.invalid_artifact")
    # Reject NaN/Infinity even in otherwise unused nested fields.
    _private_json(raw)
    return raw


def _private_run_id(config_bytes: bytes, phase: str) -> str:
    if phase not in {"diagnostic", "pilot", "confirmation"}:
        raise ValueError("evidence.invalid_artifact")
    return f"private-ope-{phase}-{_private_digest(config_bytes)[:24]}"


def _private_source_inventory(source_root: Path) -> dict[str, str]:
    paths = {"modal_flagship.py", "pyproject.toml", "uv.lock"}
    for directory in ("src", "tests"):
        for path in (source_root / directory).rglob("*"):
            relative = path.relative_to(source_root)
            if "__pycache__" in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError("evidence.source_mismatch")
            if path.is_file():
                paths.add(relative.as_posix())
    if not PRIVATE_OPE_REQUIRED_SOURCES.issubset(paths):
        raise ValueError("evidence.source_mismatch")
    return {
        relative: _private_digest(_private_safe_path(source_root, relative).read_bytes())
        for relative in sorted(paths)
    }


def private_ope_capture_sources(code_commit: str) -> dict[str, object]:
    """Capture committed bytes into an immutable image tree without building it."""
    import subprocess

    source_root = Path(__file__).resolve().parent
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("code commit must be an exact commit")
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", code_commit],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths = sorted(
        path
        for path in listed
        if path.startswith(("src/", "tests/"))
        or path in {"modal_flagship.py", "pyproject.toml", "uv.lock"}
    )
    if not PRIVATE_OPE_REQUIRED_SOURCES.issubset(paths):
        raise ValueError("source commit lacks required private OPE implementation")
    committed_files = {}
    for path in paths:
        if "__pycache__" in Path(path).parts or Path(path).suffix == ".pyc":
            raise ValueError("compiled Python caches cannot be image source")
        committed = subprocess.run(
            ["git", "show", f"{code_commit}:{path}"],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        if _private_safe_path(source_root, path).read_bytes() != committed:
            raise ValueError("working source differs from the image code commit")
        committed_files[path] = committed
    # Every group conservatively binds the complete executable/fixture inventory.
    entries = [
        {"path": path, "sha256": _private_digest(data)} for path, data in committed_files.items()
    ]
    file_manifest = {
        "schema_version": "unseen-loop/private-ope-code-manifest-v1",
        "code_commit": code_commit,
        "entries": entries,
    }
    bundle: dict[str, object] = {
        "schema_version": "unseen-loop/private-ope-source-bundle-v1",
        "manifests": {
            name: file_manifest for name in ("candidate", "baseline", "domain", "analysis")
        },
        "image_spec": PRIVATE_OPE_IMAGE_SPEC,
        "lockfile_sha256": _private_digest(committed_files["uv.lock"]),
    }
    data = _private_json(bundle)
    snapshot = source_root / PRIVATE_OPE_IMAGE_DIR
    destination = source_root / PRIVATE_OPE_SOURCE_FILE
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError("refusing to replace a different image source bundle")
    if snapshot.exists():
        if (
            _private_source_inventory(snapshot)
            != {entry["path"]: entry["sha256"] for entry in entries}
            or (snapshot / PRIVATE_OPE_SOURCE_FILE).read_bytes() != data
        ):
            raise ValueError("refusing to replace a different image source snapshot")
    else:
        with tempfile.TemporaryDirectory(
            prefix=".private-ope-image-stage-", dir=source_root
        ) as temporary:
            staging = Path(temporary)
            for relative, content in committed_files.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            (staging / PRIVATE_OPE_SOURCE_FILE).write_bytes(data)
            os.rename(staging, snapshot)
    destination.write_bytes(data)
    return bundle


def _private_sources(manifest: Any) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    bundle = json.loads((source_root / PRIVATE_OPE_SOURCE_FILE).read_bytes())
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "manifests", "image_spec", "lockfile_sha256"}
        or bundle["schema_version"] != "unseen-loop/private-ope-source-bundle-v1"
    ):
        raise ValueError("evidence.source_mismatch")
    groups = bundle["manifests"]
    if not isinstance(groups, dict) or set(groups) != {
        "candidate",
        "baseline",
        "domain",
        "analysis",
    }:
        raise ValueError("evidence.source_mismatch")
    inventory = _private_source_inventory(source_root)
    for name, file_manifest in groups.items():
        if (
            not isinstance(file_manifest, dict)
            or set(file_manifest) != {"schema_version", "code_commit", "entries"}
            or file_manifest["schema_version"] != "unseen-loop/private-ope-code-manifest-v1"
            or file_manifest["code_commit"] != manifest.execution.code_commit
            or not isinstance(file_manifest["entries"], list)
            or _private_digest(_private_json(file_manifest))
            != getattr(manifest.execution, f"{name}_code_sha256")
        ):
            raise ValueError("evidence.source_mismatch")
        paths = set()
        for entry in file_manifest["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("evidence.source_mismatch")
            relative = entry["path"]
            if not isinstance(relative, str) or relative in paths:
                raise ValueError("evidence.source_mismatch")
            paths.add(relative)
            if relative not in inventory or inventory[relative] != entry["sha256"]:
                raise ValueError("evidence.source_mismatch")
        if paths != set(inventory):
            raise ValueError("evidence.source_mismatch")
    if (
        bundle["image_spec"] != PRIVATE_OPE_IMAGE_SPEC
        or _private_digest(_private_json(bundle["image_spec"]))
        != manifest.execution.image_spec_sha256
        or _private_digest((source_root / "uv.lock").read_bytes())
        != manifest.execution.lockfile_sha256
        or bundle["lockfile_sha256"] != manifest.execution.lockfile_sha256
    ):
        raise ValueError("evidence.source_mismatch")
    return bundle


def _private_runtime(manifest: Any) -> dict[str, object]:
    import importlib.metadata
    import platform
    import sys

    _private_sources(manifest)
    source_root = Path(__file__).resolve().parent / "src"
    for name, module in tuple(sys.modules.items()):
        if name != "unseen_loop" and not name.startswith("unseen_loop."):
            continue
        filename = getattr(module, "__file__", None)
        expected = source_root.joinpath(*name.split("."))
        if not isinstance(filename, str) or Path(filename).resolve() not in {
            expected.with_suffix(".py"),
            expected / "__init__.py",
        }:
            raise ValueError("evidence.source_mismatch")
    image_id = os.environ.get("MODAL_IMAGE_ID")
    if not image_id:
        from modal.config import config as modal_config

        image_id = modal_config.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise ValueError("evidence.source_mismatch")
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "tenseal", "modal")
    }
    if platform.python_version_tuple()[:2] != ("3", "12") or versions != {
        "numpy": "1.26.4",
        "scipy": "1.14.1",
        "tenseal": "0.3.17",
        "modal": "1.5.4",
    }:
        raise ValueError("evidence.source_mismatch")
    return {
        "image_id": image_id,
        "image_spec_sha256": manifest.execution.image_spec_sha256,
        "code_commit": manifest.execution.code_commit,
        **{
            f"{name}_code_sha256": getattr(manifest.execution, f"{name}_code_sha256")
            for name in ("candidate", "baseline", "domain", "analysis")
        },
        "lockfile_sha256": manifest.execution.lockfile_sha256,
        "python_version": platform.python_version(),
        **{f"{name}_version": version for name, version in versions.items()},
        # TenSEAL's Python distribution does not declare a SEAL package version.
        # The backend executor may replace this only with an exposed runtime value.
        "seal_version": "not-exposed",
        "source_match": True,
        "execution_site": "Modal",
    }


def _private_verify_ledger(root: Path, expected_digest: str) -> dict[str, bytes]:
    ledger = _private_safe_path(root, "checksums.sha256").read_bytes()
    if _private_digest(ledger) != expected_digest:
        raise ValueError("evidence.invalid_artifact")
    result = {"checksums.sha256": ledger}
    for line in ledger.decode("ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative in result
            or relative == "evidence-index.json"
        ):
            raise ValueError("evidence.invalid_artifact")
        path = _private_safe_path(root, relative)
        data = path.read_bytes()
        if _private_digest(data) != digest:
            raise ValueError("evidence.invalid_artifact")
        result[relative] = data
    return result


def _private_verified_inputs(manifest: Any) -> dict[str, bytes]:
    predecessor = manifest.predecessors
    historical = VOLUME_ROOT / "confirmations" / predecessor.historical_confirmation_id
    retained = _private_verify_ledger(historical, predecessor.historical_ledger_sha256)
    if (
        set(retained)
        != {
            "checksums.sha256",
            "summary.json",
            "config.toml",
            "ope.json",
            "ckks.json",
            "shield.json",
        }
        or _private_digest(retained["summary.json"]) != predecessor.historical_summary_sha256
    ):
        raise ValueError("evidence.invalid_artifact")
    originals = json.loads(retained["summary.json"])
    if any(
        originals.get(key) is not False
        for key in (
            "all_tracks_passed",
            "independently_confirmed_positive_result",
            "scale_up_allowed",
        )
    ):
        raise ValueError("evidence.invalid_artifact")
    inputs = {f"inputs/historical/{name}": data for name, data in retained.items()}
    if manifest.phase == "diagnostic":
        return inputs
    prior_root = VOLUME_ROOT / "private-ope" / predecessor.previous_run_id
    index_data = _private_safe_path(prior_root, "evidence-index.json").read_bytes()
    if _private_digest(index_data) != predecessor.previous_index_sha256:
        raise ValueError("evidence.invalid_artifact")
    index = json.loads(index_data)
    support = index.get("supporting_artifacts")
    if not isinstance(support, dict) or "checksums.sha256" not in support:
        raise ValueError("evidence.invalid_artifact")
    prior_files = _private_verify_ledger(prior_root, support["checksums.sha256"])
    prior_config = prior_files.get("config.toml")
    if prior_config is None or _private_digest(prior_config) != predecessor.previous_config_sha256:
        raise ValueError("evidence.invalid_artifact")
    from unseen_loop.flagship.manifest import parse_private_ope_manifest_bytes

    previous_manifest = parse_private_ope_manifest_bytes(prior_config)
    if previous_manifest.phase != {"pilot": "diagnostic", "confirmation": "pilot"}[manifest.phase]:
        raise ValueError("evidence.invalid_artifact")
    previous_registry = AppendOnlyRegistry(prior_root / "registry.jsonl")
    finalize_evidence(
        previous_registry,
        evidence_root=prior_root,
        supporting_paths=tuple(support),
    )
    analysis_rows = [
        json.loads(prior_files[entry["path"]])
        for entry in index["artifacts"].values()
        if json.loads(prior_files[entry["path"]])["job"]["coordinates"]["kind"] == "analysis"
    ]
    if len(analysis_rows) != 1:
        raise ValueError("evidence.invalid_artifact")
    analysis = analysis_rows[0]
    if (
        analysis.get("completed") is not True
        or analysis.get("metrics", {}).get("promotion_allowed") is not True
        or analysis["metrics"].get("status") != "passed"
    ):
        raise ValueError("evidence.invalid_artifact")
    for name in (
        "candidate_code_sha256",
        "baseline_code_sha256",
        "domain_code_sha256",
        "analysis_code_sha256",
        "lockfile_sha256",
        "image_spec_sha256",
        "deployment_version",
        "code_commit",
    ):
        if getattr(previous_manifest.execution, name) != getattr(manifest.execution, name):
            raise ValueError("evidence.source_mismatch")
    if manifest.phase == "confirmation":
        domain = json.loads(prior_files["inputs/queue.json"])
        if (
            domain["kernel_sha256"] != predecessor.pilot_kernel_sha256
            or domain["policies_sha256"] != predecessor.pilot_policies_sha256
        ):
            raise ValueError("evidence.invalid_artifact")
    inputs.update({f"inputs/predecessor/{name}": data for name, data in prior_files.items()})
    inputs["inputs/predecessor/evidence-index.json"] = index_data
    return inputs


def _private_budget_guard(manifest: Any, *, enforce_cycle: bool = True) -> dict[str, Any]:
    digest = manifest.execution.budget_guard_sha256
    path = VOLUME_ROOT / "private-ope-budget-guards" / f"{digest}.json"
    data = _private_safe_path(VOLUME_ROOT, path.relative_to(VOLUME_ROOT).as_posix()).read_bytes()
    if _private_digest(data) != digest:
        raise ValueError("evidence.invalid_artifact")
    guard = json.loads(data)
    fields = {
        "schema_version",
        "workspace",
        "authority",
        "verification_source",
        "verification_artifact_sha256",
        "verified_at_utc",
        "cycle_start_utc",
        "cycle_end_utc",
        "hard_workspace_usage_cap",
        "usage_limit_usd",
        "observed_usage_usd",
        "stage_envelope_usd",
        "projected_stage_with_overhead_usd",
        "cpu_hour_usd",
        "memory_gib_hour_usd",
    }
    if (
        not isinstance(guard, dict)
        or set(guard) != fields
        or guard["schema_version"] != "unseen-loop/private-ope-budget-guard-v1"
        or guard["authority"] not in {"workspace-owner", "workspace-manager"}
        or guard["verification_source"] != "authenticated-modal-usage-dashboard"
        or guard["hard_workspace_usage_cap"] is not True
        or not isinstance(guard["verification_artifact_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", guard["verification_artifact_sha256"]) is None
    ):
        raise ValueError("evidence.invalid_artifact")
    for key in (
        "usage_limit_usd",
        "observed_usage_usd",
        "stage_envelope_usd",
        "projected_stage_with_overhead_usd",
        "cpu_hour_usd",
        "memory_gib_hour_usd",
    ):
        if type(guard[key]) not in (int, float) or not Decimal(str(guard[key])).is_finite():
            raise ValueError("evidence.invalid_artifact")
    now = _private_now()
    start, end = (_private_date(guard[key]) for key in ("cycle_start_utc", "cycle_end_utc"))
    if (
        not start <= _private_date(guard["verified_at_utc"]) < end
        or (enforce_cycle and not start <= now < end)
        or (enforce_cycle and end - now < timedelta(seconds=manifest.execution.wave_timeout_s + 60))
        or guard["stage_envelope_usd"] != manifest.execution.budget_envelope_usd
    ):
        raise ValueError("evidence.invalid_artifact")
    remaining = Decimal(str(guard["usage_limit_usd"])) - Decimal(str(guard["observed_usage_usd"]))
    if (
        not Decimal(str(guard["projected_stage_with_overhead_usd"]))
        <= remaining
        <= Decimal(str(manifest.execution.budget_envelope_usd))
        or remaining <= 0
    ):
        raise ValueError("evidence.invalid_artifact")
    proof_path = VOLUME_ROOT / "private-ope-budget-guards" / guard["verification_artifact_sha256"]
    if _private_digest(proof_path.read_bytes()) != guard["verification_artifact_sha256"]:
        raise ValueError("evidence.invalid_artifact")
    return guard


def _private_budget_available(
    manifest: Any,
    guard: dict[str, Any],
    reserved_usd: float,
    *,
    required_window_s: int | None = None,
) -> bool:
    """The authenticated hard cap is primary; polling never substitutes for it."""
    window = timedelta(
        seconds=required_window_s
        if required_window_s is not None
        else manifest.execution.analysis_deadline_s + 60
    )
    start = _private_date(guard["cycle_start_utc"])
    end = _private_date(guard["cycle_end_utc"])
    now = _private_now()
    if not start <= now < end or now + window > end:
        return False
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != guard["workspace"]:
        return False
    rates = workspace.billing.rates()
    if rates["cpu_hour_cost"] != Decimal(str(guard["cpu_hour_usd"])) or rates[
        "mem_gib_hour_cost"
    ] != Decimal(str(guard["memory_gib_hour_usd"])):
        return False
    usage = workspace.billing.summary(guard["cycle_start_utc"][:7]).metered_cost
    remaining = Decimal(str(guard["usage_limit_usd"])) - usage
    now = _private_now()
    return start <= now < end and now + window <= end and remaining >= Decimal(str(reserved_usd))


def _private_job_dict(job: PlannedJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "seed": str(job.seed),
        "coordinates": job.coordinate_dict(),
    }


def _private_claims() -> Any:
    return modal.Dict.from_name(
        PRIVATE_OPE_CLAIMS,
        environment_name="main",
        create_if_missing=True,
    )


def _private_initialize(
    config_bytes: bytes,
    manifest: Any,
    root: Path,
    jobs: tuple[PlannedJob, ...],
    inputs: dict[str, bytes],
    runtime: dict[str, object],
    guard: dict[str, Any],
) -> AppendOnlyRegistry:
    provenance = {
        "schema_version": "unseen-loop/private-ope-provenance-v1",
        "run_id": root.name,
        "config_sha256": _private_digest(config_bytes),
        "runtime": runtime,
        "source_bundle_sha256": _private_digest(_private_json(_private_sources(manifest))),
    }
    files = {
        **inputs,
        "config.toml": config_bytes,
        "plan.json": _private_json([_private_job_dict(job) for job in jobs]),
        "provenance.json": _private_json(provenance),
        "inputs/source-bundle.json": _private_json(_private_sources(manifest)),
        "inputs/budget-guard.json": _private_json(guard),
    }
    # Public DP inputs are deterministic Modal computations, not fitted nuisances.
    if manifest.phase != "diagnostic":
        from unseen_loop.ope.study import queue_inputs

        files["inputs/queue.json"] = _private_json(queue_inputs(64, 0.99))
    for relative, data in files.items():
        _private_publish(root / relative, data)
    registry_provenance = Provenance.from_mapping(
        source_digest=provenance["source_bundle_sha256"],
        config_digest=_private_digest(config_bytes),
        image_digests={
            "private_ope": _private_digest(
                _private_json(
                    {
                        "image_id": runtime["image_id"],
                        "image_spec_sha256": runtime["image_spec_sha256"],
                    }
                )
            )
        },
    )
    registry_path = root / "registry.jsonl"
    if not registry_path.exists():
        staging = VOLUME_ROOT / "private-ope-staging"
        staging.mkdir(exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="registry-", dir=staging)
        os.close(descriptor)
        Path(temporary).unlink()
        try:
            AppendOnlyRegistry.create(temporary, jobs=jobs, provenance=registry_provenance)
            _private_publish(registry_path, Path(temporary).read_bytes())
        finally:
            Path(temporary).unlink(missing_ok=True)
    registry = AppendOnlyRegistry(registry_path)
    snapshot = registry.snapshot()
    if snapshot.provenance != registry_provenance or {entry.job_id for entry in snapshot.plan} != {
        job.job_id for job in jobs
    }:
        raise ValueError("evidence.invalid_artifact")
    marker = {
        "schema_version": "unseen-loop/private-ope-initialized-v1",
        "run_id": root.name,
        "config_sha256": _private_digest(config_bytes),
        "plan_sha256": _private_digest(files["plan.json"]),
        "provenance_sha256": _private_digest(files["provenance.json"]),
        "registry_id": snapshot.registry_id,
    }
    evidence_volume.commit()
    _private_publish(root / "initialized.json", _private_json(marker))
    evidence_volume.commit()
    return registry


def _private_failure(
    root: Path,
    job: PlannedJob,
    failure_code: str,
    *,
    entry: dict[str, Any] | None = None,
    runtime: dict[str, object] | None = None,
) -> Any:
    from unseen_loop.flagship.executor_private_ope import PrivateOPEAttempt

    intent = (root / "dispatch" / f"{job.job_id}.json").is_file()
    attempted = True if entry is not None else None if intent else False
    kind = job.coordinate_dict()["kind"]
    modern = kind in {"paired_context", "ablation_context", "statistical_context", "timing_context"}
    arity = 2 if modern else 1 if kind == "historical_context" else 0
    metrics = None
    if kind in {"smoke_error", "smoke_timeout"} and attempted is True:
        metrics = {
            "kind": "probe",
            "expected_failure_code": "runtime.timeout"
            if kind == "smoke_timeout"
            else "probe.deliberate_exception",
            "observed_failure_code": failure_code,
            "elapsed_ns": max(
                0,
                int(
                    (_private_now() - _private_date(entry["entered_at_utc"])).total_seconds() * 1e9
                ),
            ),
        }
    return PrivateOPEAttempt.from_dict(
        {
            "schema_version": "unseen-loop/private-ope-attempt-v1",
            "run_id": root.name,
            "config_sha256": _private_digest((root / "config.toml").read_bytes()),
            "provenance_sha256": _private_digest((root / "provenance.json").read_bytes()),
            "job": _private_job_dict(job),
            "function_call_id": None if entry is None else entry["function_call_id"],
            "input_id": None if entry is None else entry["input_id"],
            "attempted": attempted,
            "completed": False,
            "failure_code": failure_code,
            "worker_result_sha256": None,
            "metrics": metrics,
            "receipts": {
                "runtime": runtime,
                "context": None,
                "computation_sha256": None,
                "batch_sha256": None,
                "request_sha256": None,
                "response_sha256": [None] * arity,
                "operations": [],
                "public_context_sha256": None,
                "public_context_bytes": None,
                "client_context_bytes": None,
                "request_bytes": None,
                "response_bytes": [None] * arity,
                "counts_source": "public_fixed_shape"
                if modern
                else "legacy_encrypted"
                if kind == "historical_context"
                else "diagnostic_sum"
                if kind == "count_precision"
                else "not-applicable",
            },
            "private_rows_persisted": False,
            "secret_material_persisted": False,
        }
    )


def _private_worker_body(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
    *,
    timeout_probe: bool = False,
) -> dict[str, object]:
    worker_started_ns = time.perf_counter_ns()
    import tomllib

    phase = tomllib.loads(config_bytes.decode())["phase"]
    run_id = _private_run_id(config_bytes, phase)
    root = VOLUME_ROOT / "private-ope" / run_id
    if str(root) != run_root:
        raise ValueError("evidence.invalid_artifact")
    raw_job = job_payload.get("job")
    if not isinstance(raw_job, dict) or not isinstance(raw_job.get("job_id"), str):
        raise ValueError("evidence.invalid_artifact")
    job_id = raw_job["job_id"]
    call_id, input_id = modal.current_function_call_id(), modal.current_input_id()
    if any(
        not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value) is None
        for value in (job_id, call_id, input_id)
    ):
        raise ValueError("evidence.invalid_artifact")
    identity = {
        "run_id": run_id,
        "job_id": job_id,
        "config_sha256": _private_digest(config_bytes),
        "function_call_id": call_id,
        "input_id": input_id,
    }
    claims = _private_claims()
    acquired = claims.put(("job", run_id, job_id), identity, skip_if_exists=True)
    # No estimator/backend imports or random draws precede the atomic claim.
    transport = VOLUME_ROOT / "private-ope-transport" / run_id / job_id / input_id
    entry = {
        **identity,
        "schema_version": "unseen-loop/private-ope-worker-entry-v1",
        "entered_at_utc": _private_utc(_private_now()),
        "dispatch_intent_sha256": job_payload.get("dispatch_intent_sha256"),
        "deadline_utc": job_payload.get("deadline_utc"),
        "claim_acquired": acquired,
    }
    first = None
    if not acquired:
        first = claims.get(("job", run_id, job_id))
        if not isinstance(first, dict) or any(
            first.get(key) != identity[key] for key in ("run_id", "job_id", "config_sha256")
        ):
            raise ValueError("evidence.invalid_artifact")
        if any(
            not isinstance(first.get(key), str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", first[key]) is None
            for key in ("function_call_id", "input_id")
        ):
            raise ValueError("evidence.invalid_artifact")
        evidence_volume.reload()
        if (transport / "entry.json").is_file():
            original = _private_read_json(transport / "entry.json")
            if (
                set(original) != set(entry)
                or any(
                    original[key] != entry[key]
                    for key in entry
                    if key not in {"entered_at_utc", "claim_acquired"}
                )
                or type(original["claim_acquired"]) is not bool
            ):
                raise ValueError("evidence.invalid_artifact")
            _private_date(original["entered_at_utc"])
            entry = original
    _private_publish(transport / "entry.json", _private_json(entry))
    evidence_volume.commit()
    envelope = {
        "schema_version": "unseen-loop/private-ope-transport-v1",
        "run_id": run_id,
        "job_id": job_id,
        "function_call_id": call_id,
        "input_id": input_id,
        "entry_path": (transport / "entry.json").relative_to(VOLUME_ROOT).as_posix(),
        "result_path": None,
        "result_sha256": None,
        "delivery": "reentry",
    }
    if not acquired:
        assert first is not None
        prior = VOLUME_ROOT / "private-ope-transport" / run_id / job_id / first["input_id"]
        prior_envelope = prior / "transport.json"
        if prior_envelope.is_file():
            return _private_read_json(prior_envelope)
        return envelope
    from dataclasses import replace

    from unseen_loop.flagship.executor_private_ope import (
        PrivateOPETransportResult,
        RuntimeReceipt,
        execute_private_ope_job,
        validate_job_payload,
    )
    from unseen_loop.flagship.manifest import parse_private_ope_manifest_bytes

    payload = validate_job_payload(config_bytes, job_payload, run_root)
    intent_data = (root / "dispatch" / f"{job_id}.json").read_bytes()
    if _private_digest(intent_data) != payload.dispatch_intent_sha256:
        raise ValueError("evidence.invalid_artifact")
    manifest = parse_private_ope_manifest_bytes(config_bytes)
    runtime = None
    try:
        runtime = _private_runtime(manifest)
        if _private_now() >= _private_date(payload.deadline_utc):
            attempt = _private_failure(
                root, payload.job, "runtime.timeout", entry=entry, runtime=runtime
            )
        elif timeout_probe:
            time.sleep(3)
            raise RuntimeError("timeout probe unexpectedly survived its frozen two-second timeout")
        else:
            attempt = execute_private_ope_job(
                config_bytes,
                payload.to_dict(),
                run_root,
                RuntimeReceipt.from_dict(runtime),
            )
            attempt = replace(attempt, function_call_id=call_id, input_id=input_id)
            if attempt.metrics is not None and hasattr(attempt.metrics, "timing"):
                timing = replace(
                    attempt.metrics.timing,
                    job_elapsed_ns=time.perf_counter_ns() - worker_started_ns,
                )
                attempt = replace(attempt, metrics=replace(attempt.metrics, timing=timing))
    except Exception as exc:
        code = (
            "evidence.source_mismatch"
            if isinstance(exc, (ValueError, FileNotFoundError)) and runtime is None
            else "runtime.interrupted"
        )
        attempt = _private_failure(root, payload.job, code, entry=entry, runtime=runtime)
    data = _private_json(attempt.to_dict())
    _private_publish(transport / "result.json", data)
    _private_publish(
        transport / "finished.json",
        _private_json(
            {
                "finished_at_utc": _private_utc(_private_now()),
                "result_sha256": _private_digest(data),
            }
        ),
    )
    envelope.update(
        {
            "result_path": (transport / "result.json").relative_to(VOLUME_ROOT).as_posix(),
            "result_sha256": _private_digest(data),
            "delivery": "result",
        }
    )
    envelope = PrivateOPETransportResult.from_dict(envelope).to_dict()
    _private_publish(transport / "transport.json", _private_json(envelope))
    evidence_volume.commit()
    return envelope


def _private_entries(root: Path, job: PlannedJob) -> tuple[dict[str, Any], ...]:
    transport = VOLUME_ROOT / "private-ope-transport" / root.name / job.job_id
    entries = []
    if transport.is_dir():
        for path in sorted(transport.glob("*/entry.json")):
            entry = _private_read_json(path)
            if (
                entry.get("run_id") != root.name
                or entry.get("job_id") != job.job_id
                or entry.get("input_id") != path.parent.name
                or entry.get("config_sha256")
                != _private_digest((root / "config.toml").read_bytes())
                or not isinstance(entry.get("function_call_id"), str)
            ):
                raise ValueError("evidence.invalid_artifact")
            intent = root / "dispatch" / f"{job.job_id}.json"
            if (
                not intent.is_file()
                or entry.get("dispatch_intent_sha256") != _private_digest(intent.read_bytes())
                or entry.get("deadline_utc") != _private_read_json(intent).get("deadline_utc")
            ):
                raise ValueError("evidence.invalid_artifact")
            entries.append(entry)
    return tuple(entries)


def _private_harvest(root: Path, job: PlannedJob, cutoff: datetime) -> Any | None:
    from dataclasses import replace

    from unseen_loop.flagship.executor_private_ope import (
        PrivateOPEAttempt,
        PrivateOPETransportResult,
    )

    entries = _private_entries(root, job)
    if not entries:
        return None
    intent = _private_read_json(root / "dispatch" / f"{job.job_id}.json")
    cutoff = min(cutoff, _private_date(intent["deadline_utc"]))
    candidates = []
    for entry in entries:
        transport = (
            VOLUME_ROOT / "private-ope-transport" / root.name / job.job_id / entry["input_id"]
        )
        if not (transport / "transport.json").is_file():
            continue
        envelope = PrivateOPETransportResult.from_dict(
            _private_read_json(transport / "transport.json")
        )
        if (
            envelope.run_id != root.name
            or envelope.job_id != job.job_id
            or envelope.input_id != entry["input_id"]
            or envelope.function_call_id != entry["function_call_id"]
            or envelope.delivery != "result"
            or not entry.get("claim_acquired")
        ):
            raise ValueError("evidence.invalid_artifact")
        finish = _private_read_json(transport / "finished.json")
        if _private_date(finish["finished_at_utc"]) > cutoff:
            continue
        data = _private_safe_path(VOLUME_ROOT, envelope.result_path).read_bytes()
        if (
            _private_digest(data) != envelope.result_sha256
            or finish["result_sha256"] != envelope.result_sha256
        ):
            raise ValueError("evidence.invalid_artifact")
        row = PrivateOPEAttempt.from_dict(json.loads(data))
        if (
            row.run_id != root.name
            or _private_job_dict(row.job) != _private_job_dict(job)
            or row.config_sha256 != _private_digest((root / "config.toml").read_bytes())
            or row.provenance_sha256 != _private_digest((root / "provenance.json").read_bytes())
            or row.function_call_id != envelope.function_call_id
            or row.input_id != envelope.input_id
        ):
            raise ValueError("evidence.invalid_artifact")
        if row.worker_result_sha256 is not None or (row.completed and row.receipts.runtime is None):
            raise ValueError("evidence.source_mismatch")
        if row.receipts.runtime is not None:
            expected = _private_read_json(root / "provenance.json")["runtime"]
            observed = row.receipts.runtime.to_dict()
            # A backend may expose SEAL after the initial metadata-only check.
            if any(
                observed[key] != value for key, value in expected.items() if key != "seal_version"
            ):
                raise ValueError("evidence.source_mismatch")
        candidates.append(replace(row, worker_result_sha256=envelope.result_sha256))
    if len(candidates) > 1:
        raise ValueError("evidence.invalid_artifact")
    return candidates[0] if candidates else None


def _private_terminalize(root: Path, registry: AppendOnlyRegistry, row: Any) -> None:
    from unseen_loop.flagship.executor_private_ope import PrivateOPEAttempt

    row = PrivateOPEAttempt.from_dict(row.to_dict())
    data = _private_json(row.to_dict())
    path = root / "attempts" / f"{row.job.job_id}.json"
    _private_publish(path, data)
    current = next(
        record for record in registry.snapshot().records if record.job_id == row.job.job_id
    )
    if current.status == JobStatus.SUCCEEDED:
        if current.artifact_digest != _private_digest(data):
            raise ValueError("evidence.invalid_artifact")
        return
    if current.status is None:
        # STARTED is an evidence lifecycle event, not a claim of worker entry.
        registry.started(row.job.job_id)
    registry.succeeded(
        row.job.job_id,
        artifact_path=path.relative_to(root).as_posix(),
        artifact_digest=_private_digest(data),
    )
    evidence_volume.commit()


def _private_close_index(root: Path, registry: AppendOnlyRegistry) -> str:
    if any(row.status != JobStatus.SUCCEEDED for row in registry.snapshot().records):
        raise ValueError("evidence.invalid_artifact")
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("evidence.invalid_artifact")
        if path.is_file() and path not in {root / "evidence-index.json", root / "checksums.sha256"}:
            paths.append(path.relative_to(root).as_posix())
    ledger = "".join(
        f"{_private_digest((root / relative).read_bytes())}  {relative}\n" for relative in paths
    ).encode()
    evidence_volume.commit()
    _private_publish(root / "checksums.sha256", ledger)
    evidence_volume.commit()
    artifacts = {row.artifact_path for row in registry.snapshot().records}
    supporting = tuple(path for path in paths if path not in artifacts and path != "registry.jsonl")
    index = finalize_evidence(
        registry,
        evidence_root=root,
        supporting_paths=(*supporting, "checksums.sha256"),
    )
    evidence_volume.commit()
    return str(index)


def _private_contract(kind: str) -> tuple[str, int, float, int]:
    if kind in {"clear_batch", "smoke_error"}:
        return "private_ope_clear_worker", 900, 2.0, 8192
    if kind in {
        "count_precision",
        "paired_context",
        "ablation_context",
        "historical_context",
        "statistical_context",
        "timing_context",
    }:
        return "private_ope_ckks_worker", 2100, 8.0, 32768
    if kind == "protocol_verification":
        return "private_ope_verification_worker", 7500, 8.0, 32768
    if kind == "smoke_timeout":
        return "private_ope_timeout_probe", 300, 0.5, 1024
    if kind == "analysis":
        return "private_ope_analysis_worker", 2100, 2.0, 8192
    raise ValueError("evidence.invalid_artifact")


def _private_projected_cost(jobs: tuple[PlannedJob, ...], guard: dict[str, Any]) -> float:
    cpu_rate = float(guard["cpu_hour_usd"])
    memory_rate = float(guard["memory_gib_hour_usd"])
    cost = 0.0
    for job in jobs:
        _, deadline, cpu, memory = _private_contract(str(job.coordinate_dict()["kind"]))
        cost += (deadline + 60) / 3600 * (cpu * cpu_rate + memory / 1024 * memory_rate)
    waves = len({job.coordinate_dict()["wave_index"] for job in jobs})
    # Full coordinator deadlines and a separate build/storage/closure allowance.
    return cost + waves * 22 * (0.5 * cpu_rate + memory_rate) + 2.0


def _private_pending_jobs(
    registry: AppendOnlyRegistry, jobs: tuple[PlannedJob, ...]
) -> tuple[PlannedJob, ...]:
    snapshot = registry.snapshot()
    completed = {row.job_id for row in snapshot.records if row.status == JobStatus.SUCCEEDED}
    return tuple(job for job in jobs if job.job_id not in completed)


def _private_function(manifest: Any, name: str) -> Any:
    return modal.Function.from_name(
        "unseen-loop-flagship",
        name,
        version=manifest.execution.deployment_version,
        environment_name="main",
    )


def _private_dispatch(
    config_bytes: bytes,
    manifest: Any,
    root: Path,
    registry: AppendOnlyRegistry,
    job: PlannedJob,
) -> dict[str, Any]:
    name, deadline_s, _, _ = _private_contract(str(job.coordinate_dict()["kind"]))
    deadline = _private_now() + timedelta(seconds=deadline_s)
    intent = {
        "schema_version": "unseen-loop/private-ope-dispatch-intent-v1",
        "run_id": root.name,
        "config_sha256": _private_digest(config_bytes),
        "job_id": job.job_id,
        "deadline_utc": _private_utc(deadline),
        "deployment_version": manifest.execution.deployment_version,
        "worker": name,
    }
    registry.started(job.job_id)
    data = _private_json(intent)
    _private_publish(root / "dispatch" / f"{job.job_id}.json", data)
    evidence_volume.commit()
    payload = {
        "schema_version": "unseen-loop/private-ope-job-v1",
        "run_id": root.name,
        "config_sha256": _private_digest(config_bytes),
        "job": _private_job_dict(job),
        "dispatch_intent_sha256": _private_digest(data),
        "deadline_utc": intent["deadline_utc"],
    }
    # An exception here is an unknown dispatch window, never a replacement spawn.
    call = _private_function(manifest, name).spawn(config_bytes, payload, str(root))
    _private_publish(
        root / "handles" / f"{job.job_id}.json",
        _private_json(
            {
                "function_call_id": call.object_id,
                "dispatch_intent_sha256": _private_digest(data),
            }
        ),
    )
    evidence_volume.commit()
    return {
        "job": job,
        "call": call,
        "deadline": deadline,
        "cutoff": deadline + timedelta(seconds=60),
        "failure": None,
        "settled": False,
    }


def _private_run_batch(
    config_bytes: bytes,
    manifest: Any,
    root: Path,
    registry: AppendOnlyRegistry,
    jobs: tuple[PlannedJob, ...],
    planned_jobs: tuple[PlannedJob, ...],
    guard: dict[str, Any],
    wave_deadline: datetime,
) -> None:
    from unseen_loop.flagship.executor_private_ope import PrivateOPETransportResult

    if not 1 <= len(jobs) <= 2:
        raise ValueError("evidence.invalid_artifact")
    closure_reserve_s = (
        60 if all(job.coordinate_dict()["kind"] == "analysis" for job in jobs) else 2160
    )
    pending = []
    for job in jobs:
        _, deadline_s, _, _ = _private_contract(str(job.coordinate_dict()["kind"]))
        required_window_s = deadline_s + 60 + closure_reserve_s
        if not _private_budget_available(
            manifest,
            guard,
            _private_projected_cost(_private_pending_jobs(registry, planned_jobs), guard),
            required_window_s=required_window_s,
        ):
            raise TimeoutError("workspace allowance reached before dispatch")
        if _private_now() + timedelta(seconds=required_window_s) > wave_deadline:
            raise TimeoutError("batch cannot fit before coordinator closure reserve")
        if _private_claims().get(("seed-root", manifest.seed_root)) is None:
            raise RuntimeError("active seed reservation disappeared")
        pending.append(_private_dispatch(config_bytes, manifest, root, registry, job))
    next_budget_check = _private_now()
    while pending:
        now = _private_now()
        if now >= wave_deadline - timedelta(seconds=closure_reserve_s):
            raise TimeoutError("coordinator closure reserve reached")
        if now >= next_budget_check:
            if not _private_budget_available(manifest, guard, 0.2):
                raise TimeoutError("workspace allowance reached")
            # Accesses refresh the active Dict reservation before its seven-day expiry.
            if _private_claims().get(("seed-root", manifest.seed_root)) is None:
                raise RuntimeError("active seed reservation disappeared")
            next_budget_check = now + timedelta(seconds=30)
        for active in tuple(pending):
            now = _private_now()
            call, job = active["call"], active["job"]
            if now >= active["deadline"] and not active["settled"]:
                call.cancel(terminate_containers=True)
                active["failure"] = "runtime.timeout"
                active["settled"] = True
            if not active["settled"]:
                try:
                    result = call.get(
                        timeout=max(0.0, min(0.5, (active["deadline"] - now).total_seconds()))
                    )
                    envelope = PrivateOPETransportResult.from_dict(result)
                    if envelope.run_id != root.name or envelope.job_id != job.job_id:
                        raise ValueError("evidence.invalid_artifact")
                    active["settled"] = True
                    active["failure"] = (
                        "runtime.reentry" if envelope.delivery == "reentry" else None
                    )
                    active["cutoff"] = min(active["cutoff"], _private_now() + timedelta(seconds=60))
                except TimeoutError:
                    continue
                except Exception as exc:
                    active["settled"] = True
                    active["failure"] = (
                        "runtime.timeout"
                        if type(exc).__name__ == "FunctionTimeoutError"
                        else "evidence.invalid_artifact"
                        if isinstance(exc, ValueError)
                        else "runtime.interrupted"
                    )
                    active["cutoff"] = min(active["cutoff"], _private_now() + timedelta(seconds=60))
            evidence_volume.reload()
            try:
                row = _private_harvest(root, job, active["cutoff"])
            except (ValueError, KeyError, TypeError, OSError):
                row = None
                active["failure"] = "evidence.invalid_artifact"
                active["cutoff"] = _private_now()
            if row is not None:
                _private_terminalize(root, registry, row)
                pending.remove(active)
            elif _private_now() >= active["cutoff"]:
                entries = _private_entries(root, job)
                row = _private_failure(
                    root,
                    job,
                    active["failure"] or "runtime.interrupted",
                    entry=entries[0] if entries else None,
                )
                _private_terminalize(root, registry, row)
                pending.remove(active)
        if pending:
            time.sleep(0.25)


def _private_recover_rows(
    root: Path,
    registry: AppendOnlyRegistry,
    jobs: tuple[PlannedJob, ...],
) -> None:
    """Cancel known physical calls and close every unclosed logical slot once."""
    records = {record.job_id: record for record in registry.snapshot().records}
    remaining = tuple(job for job in jobs if records[job.job_id].status != JobStatus.SUCCEEDED)
    if not remaining:
        return
    cutoff_path = root / (
        "analysis-recovery.json"
        if all(job.coordinate_dict()["kind"] == "analysis" for job in jobs)
        else "recovery.json"
    )
    if cutoff_path.is_file():
        recovery = _private_read_json(cutoff_path)
        cutoff = _private_date(recovery["artifact_cutoff_utc"])
        result_cutoff = _private_date(recovery["result_cutoff_utc"])
    else:
        dispatched = any((root / "dispatch" / f"{job.job_id}.json").exists() for job in remaining)
        result_cutoff = _private_now()
        cutoff = result_cutoff + timedelta(seconds=60 if dispatched else 0)
        _private_publish(
            cutoff_path,
            _private_json(
                {
                    "schema_version": "unseen-loop/private-ope-recovery-v1",
                    "artifact_cutoff_utc": _private_utc(cutoff),
                    "result_cutoff_utc": _private_utc(result_cutoff),
                }
            ),
        )
        evidence_volume.commit()
    cancelled: set[str] = set()
    invalid_jobs: set[str] = set()
    while True:
        evidence_volume.reload()
        for job in remaining:
            handles = []
            receipt = root / "handles" / f"{job.job_id}.json"
            if receipt.is_file():
                handles.append(_private_read_json(receipt)["function_call_id"])
            try:
                handles.extend(entry["function_call_id"] for entry in _private_entries(root, job))
            except (ValueError, KeyError, TypeError, OSError):
                invalid_jobs.add(job.job_id)
            for handle in handles:
                if handle not in cancelled:
                    try:
                        modal.FunctionCall.from_id(handle).cancel(terminate_containers=True)
                        cancellation = "cancel_requested"
                    except Exception:
                        cancellation = "cancel_failed"
                    _private_publish(
                        root / "cancellations" / f"{handle}-{cancellation}.json",
                        _private_json(
                            {
                                "function_call_id": handle,
                                "outcome": cancellation,
                            }
                        ),
                    )
                    evidence_volume.commit()
                    cancelled.add(handle)
        if _private_now() >= cutoff:
            break
        time.sleep(min(1.0, max(0.0, (cutoff - _private_now()).total_seconds())))
    for job in remaining:
        invalid = job.job_id in invalid_jobs
        try:
            row = _private_harvest(root, job, result_cutoff)
        except (ValueError, KeyError, TypeError, OSError):
            row, invalid = None, True
        if row is None:
            try:
                entries = _private_entries(root, job)
            except (ValueError, KeyError, TypeError, OSError):
                entries, invalid = (), True
            intent = root / "dispatch" / f"{job.job_id}.json"
            code = (
                "evidence.invalid_artifact"
                if invalid
                else (
                    "runtime.interrupted"
                    if entries
                    else "runtime.dispatch_unknown"
                    if intent.exists()
                    else "runtime.not_dispatched"
                )
            )
            row = _private_failure(root, job, code, entry=entries[0] if entries else None)
        _private_terminalize(root, registry, row)


def _private_coordinator_body(config_bytes: bytes, wave_index: int, action: str) -> str:
    from unseen_loop.flagship.manifest import (
        iter_private_ope_jobs,
        parse_private_ope_manifest_bytes,
    )

    if action not in {"launch", "recover"} or type(wave_index) is not int or wave_index < 0:
        raise ValueError("evidence.invalid_artifact")
    manifest = parse_private_ope_manifest_bytes(config_bytes)
    jobs = iter_private_ope_jobs(manifest)
    if wave_index not in {job.coordinate_dict()["wave_index"] for job in jobs}:
        raise ValueError("evidence.invalid_artifact")
    runtime = _private_runtime(manifest)
    evidence_volume.reload()
    inputs = _private_verified_inputs(manifest)
    guard = _private_budget_guard(manifest, enforce_cycle=False)
    projected = _private_projected_cost(jobs, guard)
    if guard["projected_stage_with_overhead_usd"] < projected:
        raise ValueError("budget receipt omits full projected workload and overhead")
    root = VOLUME_ROOT / "private-ope" / _private_run_id(config_bytes, manifest.phase)
    call_id, input_id = modal.current_function_call_id(), modal.current_input_id()
    if any(
        not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value) is None
        for value in (call_id, input_id)
    ):
        raise ValueError("evidence.invalid_artifact")
    reservation = {
        "run_id": root.name,
        "config_sha256": _private_digest(config_bytes),
        "seed_root": manifest.seed_root,
        "origin_function_call_id": call_id,
        "origin_input_id": input_id,
    }
    permanent = (
        VOLUME_ROOT
        / "private-ope-reservations"
        / f"{_private_digest(manifest.seed_root.encode())}.json"
    )
    claims = _private_claims()
    prior_permanent = _private_read_json(permanent) if permanent.exists() else None
    key = ("seed-root", manifest.seed_root)
    prior_claim = claims.get(key)
    if prior_permanent is None and prior_claim is None and action == "recover":
        raise ValueError("cannot recover a study that was never reserved")
    acquired = claims.put(key, prior_permanent or prior_claim or reservation, skip_if_exists=True)
    recorded = claims.get(key)
    if (
        not isinstance(recorded, dict)
        or set(recorded) != set(reservation)
        or any(
            recorded[key] != reservation[key] for key in ("run_id", "config_sha256", "seed_root")
        )
        or any(
            not isinstance(recorded[key], str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", recorded[key]) is None
            for key in ("origin_function_call_id", "origin_input_id")
        )
        or (prior_permanent is not None and recorded != prior_permanent)
    ):
        raise ValueError("seed namespace already belongs to another immutable configuration")
    _private_publish(permanent, _private_json(recorded))
    if action == "launch" and wave_index == 0 and (not acquired or prior_permanent is not None):
        if recorded == reservation:
            # The same Modal input was rescheduled: cancel/close, never redispatch.
            action = "recover"
        else:
            evidence_volume.commit()
            return str(permanent)
    evidence_volume.commit()
    initialized_before = (root / "initialized.json").is_file()
    registry = _private_initialize(config_bytes, manifest, root, jobs, inputs, runtime, guard)
    if (root / "evidence-index.json").is_file():
        return _private_close_index(root, registry)
    if wave_index:
        prior_handoff = root / "waves" / f"{wave_index - 1}-handoff.json"
        if (
            not prior_handoff.is_file()
            or _private_read_json(prior_handoff).get("function_call_id")
            != modal.current_function_call_id()
        ):
            action = "recover"
    wave_root = root / "waves"
    started_path = wave_root / f"{wave_index}-started.json"
    terminal_path = wave_root / f"{wave_index}-terminal.json"
    if terminal_path.is_file() and action == "launch":
        handoff = wave_root / f"{wave_index}-handoff.json"
        if handoff.is_file():
            return str(handoff)
        action = "recover"
    if started_path.exists():
        action = "recover"
    if action == "recover":
        return _private_finish_recovery(
            config_bytes, manifest, root, registry, jobs, guard, allow_analysis=initialized_before
        )
    if not initialized_before and wave_index != 0:
        _private_recover_rows(root, registry, jobs)
        return _private_close_index(root, registry)
    if not _private_budget_available(
        manifest,
        guard,
        _private_projected_cost(_private_pending_jobs(registry, jobs), guard),
        required_window_s=manifest.execution.wave_timeout_s + 60,
    ):
        _private_recover_rows(root, registry, jobs)
        return _private_close_index(root, registry)
    wave_deadline = _private_now() + timedelta(seconds=manifest.execution.wave_timeout_s)
    _private_publish(
        started_path,
        _private_json(
            {
                "wave_index": wave_index,
                "function_call_id": modal.current_function_call_id(),
                "absolute_deadline_utc": _private_utc(wave_deadline),
            }
        ),
    )
    evidence_volume.commit()
    wave_jobs = tuple(
        job
        for job in jobs
        if job.coordinate_dict()["wave_index"] == wave_index
        and job.coordinate_dict()["kind"] != "analysis"
    )
    try:
        index = 0
        while index < len(wave_jobs):
            job = wave_jobs[index]
            coordinates = job.coordinate_dict()
            size = (
                1
                if coordinates["cohort"] == "timing"
                or coordinates["kind"] == "protocol_verification"
                else 2
            )
            batch = wave_jobs[index : index + size]
            _private_run_batch(
                config_bytes, manifest, root, registry, batch, jobs, guard, wave_deadline
            )
            if coordinates["kind"] == "protocol_verification":
                result = _private_read_json(root / "attempts" / f"{job.job_id}.json")
                if not result["completed"] or not result["metrics"]["required_cases_passed"]:
                    return _private_finish_recovery(
                        config_bytes,
                        manifest,
                        root,
                        registry,
                        jobs,
                        guard,
                        allow_analysis=True,
                    )
            index += len(batch)
        final_wave = max(job.coordinate_dict()["wave_index"] for job in jobs)
        if wave_index == final_wave:
            analysis_jobs = tuple(
                job for job in jobs if job.coordinate_dict()["kind"] == "analysis"
            )
            _private_run_batch(
                config_bytes, manifest, root, registry, analysis_jobs, jobs, guard, wave_deadline
            )
            _private_publish(
                terminal_path, _private_json({"wave_index": wave_index, "terminal": True})
            )
            evidence_volume.commit()
            return _private_close_index(root, registry)
        if not _private_budget_available(
            manifest,
            guard,
            _private_projected_cost(_private_pending_jobs(registry, jobs), guard),
            required_window_s=manifest.execution.wave_timeout_s + 60,
        ):
            raise TimeoutError("next wave lacks its complete budget and billing-cycle window")
        _private_publish(terminal_path, _private_json({"wave_index": wave_index, "terminal": True}))
        _private_publish(
            wave_root / f"{wave_index}-handoff-intent.json",
            _private_json(
                {
                    "next_wave_index": wave_index + 1,
                    "deployment_version": manifest.execution.deployment_version,
                    "config_sha256": _private_digest(config_bytes),
                }
            ),
        )
        evidence_volume.commit()
        next_call = _private_function(manifest, "private_ope_orchestrate").spawn(
            config_bytes, wave_index + 1, "launch"
        )
        _private_publish(
            wave_root / f"{wave_index}-handoff.json",
            _private_json(
                {
                    "function_call_id": next_call.object_id,
                    "next_wave_index": wave_index + 1,
                }
            ),
        )
        evidence_volume.commit()
        return next_call.object_id
    except Exception:
        # No exception can turn an unknown dispatch or lost handoff into a retry.
        if "next_call" in locals():
            next_call.cancel(terminate_containers=True)
        return _private_finish_recovery(
            config_bytes,
            manifest,
            root,
            registry,
            jobs,
            guard,
            allow_analysis=True,
        )


def _private_finish_recovery(
    config_bytes: bytes,
    manifest: Any,
    root: Path,
    registry: AppendOnlyRegistry,
    jobs: tuple[PlannedJob, ...],
    guard: dict[str, Any],
    *,
    allow_analysis: bool,
) -> str:
    empirical = tuple(job for job in jobs if job.coordinate_dict()["kind"] != "analysis")
    analysis = tuple(job for job in jobs if job.coordinate_dict()["kind"] == "analysis")
    _private_recover_rows(root, registry, empirical)
    job = analysis[0]
    record = next(row for row in registry.snapshot().records if row.job_id == job.job_id)
    # Deterministic analysis gets its first attempt only. A lost analysis is
    # terminal evidence; later replay cannot rewrite that run's outcome.
    if record.status is None and allow_analysis and _private_budget_available(manifest, guard, 0.2):
        try:
            _private_run_batch(
                config_bytes,
                manifest,
                root,
                registry,
                analysis,
                jobs,
                guard,
                _private_now() + timedelta(seconds=manifest.execution.wave_timeout_s),
            )
        except Exception:
            _private_recover_rows(root, registry, analysis)
    elif record.status != JobStatus.SUCCEEDED:
        _private_recover_rows(root, registry, analysis)
    return _private_close_index(root, registry)


_PRIVATE_SOURCE_ROOT = Path(__file__).resolve().parent
_PRIVATE_IMAGE_ROOT = _PRIVATE_SOURCE_ROOT / PRIVATE_OPE_IMAGE_DIR
if (
    modal.is_local()
    and _PRIVATE_IMAGE_ROOT.exists()
    and (_PRIVATE_IMAGE_ROOT / "modal_flagship.py").read_bytes() != Path(__file__).read_bytes()
):
    raise ValueError("deploying definitions that differ from the captured source is forbidden")
# Modal's documented override selects the actual builder, not a workspace default.
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = str(PRIVATE_OPE_IMAGE_SPEC["image_builder_version"])
private_ope_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PRIVATE_OPE_PACKAGES)
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(
        _PRIVATE_IMAGE_ROOT / "src",
        "/root/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        _PRIVATE_IMAGE_ROOT / "tests",
        "/root/tests",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(_PRIVATE_IMAGE_ROOT / "pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file(_PRIVATE_IMAGE_ROOT / "uv.lock", "/root/uv.lock", copy=True)
    .add_local_file(_PRIVATE_IMAGE_ROOT / "modal_flagship.py", "/root/modal_flagship.py", copy=True)
    .add_local_file(
        _PRIVATE_IMAGE_ROOT / PRIVATE_OPE_SOURCE_FILE,
        f"/root/{PRIVATE_OPE_SOURCE_FILE}",
        copy=True,
    )
)
_PRIVATE_WORKER_OPTIONS = {
    "image": private_ope_image,
    "volumes": {str(VOLUME_ROOT): evidence_volume},
    "startup_timeout": 300,
    "single_use_containers": True,
    "retries": 0,
    "include_source": False,
}


@app.function(
    cpu=(2.0, 2.0), memory=(8192, 8192), timeout=600, max_containers=2, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_clear_worker(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, object]:
    return _private_worker_body(config_bytes, job_payload, run_root)


@app.function(
    cpu=(8.0, 8.0), memory=(32768, 32768), timeout=1800, max_containers=2, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_ckks_worker(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, object]:
    return _private_worker_body(config_bytes, job_payload, run_root)


@app.function(
    cpu=(8.0, 8.0), memory=(32768, 32768), timeout=7200, max_containers=1, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_verification_worker(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, object]:
    return _private_worker_body(config_bytes, job_payload, run_root)


@app.function(
    cpu=(0.5, 0.5), memory=(1024, 1024), timeout=2, max_containers=1, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_timeout_probe(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, object]:
    return _private_worker_body(config_bytes, job_payload, run_root, timeout_probe=True)


@app.function(
    cpu=(2.0, 2.0), memory=(8192, 8192), timeout=1800, max_containers=1, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_analysis_worker(
    config_bytes: bytes,
    job_payload: dict[str, object],
    run_root: str,
) -> dict[str, object]:
    return _private_worker_body(config_bytes, job_payload, run_root)


@app.function(
    cpu=(0.5, 0.5), memory=(1024, 1024), timeout=79200, max_containers=1, **_PRIVATE_WORKER_OPTIONS
)
def private_ope_orchestrate(config_bytes: bytes, wave_index: int, action: str) -> str:
    return _private_coordinator_body(config_bytes, wave_index, action)


@app.local_entrypoint()
def private_ope_inspect(config: str) -> None:
    """Inspect the complete frozen plan without empirical computation."""
    from unseen_loop.flagship.manifest import (
        iter_private_ope_jobs,
        parse_private_ope_manifest_bytes,
    )

    data = Path(config).read_bytes()
    manifest = parse_private_ope_manifest_bytes(data)
    _private_sources(manifest)
    jobs = iter_private_ope_jobs(manifest)
    print(
        json.dumps(
            {
                "run_id": _private_run_id(data, manifest.phase),
                "config_sha256": _private_digest(data),
                "phase": manifest.phase,
                "planned_jobs": len(jobs),
                "context_jobs": {"diagnostic": 12, "pilot": 27, "confirmation": 240}[
                    manifest.phase
                ],
                "ciphertext_policy_evaluations": {
                    "diagnostic": 0,
                    "pilot": 53,
                    "confirmation": 480,
                }[manifest.phase],
                "projected_stage_with_overhead_usd": _private_projected_cost(jobs),
                "stage_envelope_usd": manifest.execution.budget_envelope_usd,
                "jobs": [_private_job_dict(job) for job in jobs],
                "source_match": True,
                "empirical_computation": False,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


def _private_submit(config: str, action: str) -> None:
    from unseen_loop.flagship.manifest import parse_private_ope_manifest_bytes

    data = Path(config).read_bytes()
    manifest = parse_private_ope_manifest_bytes(data)
    _private_sources(manifest)
    call = _private_function(manifest, "private_ope_orchestrate").spawn(data, 0, action)
    print(call.object_id)


@app.local_entrypoint()
def private_ope_launch(config: str) -> None:
    _private_submit(config, "launch")


@app.local_entrypoint()
def private_ope_close(config: str) -> None:
    _private_submit(config, "recover")
