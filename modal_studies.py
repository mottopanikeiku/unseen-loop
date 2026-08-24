"""Bounded Modal runners for release suites and the four CartPole ablations."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop-studies"
VOLUME_NAME = "unseen-loop-artifacts"
ARTIFACT_ROOT = Path("/artifacts/studies")
MAX_CONFIG_BYTES = 1_048_576
ABLATION_CONFIGS = (
    "ablation-cartpole-unweighted-refined.toml",
    "ablation-cartpole-unweighted-unrefined.toml",
    "ablation-cartpole-weighted-refined.toml",
    "ablation-cartpole-weighted-unrefined.toml",
)

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "gymnasium==1.3.0")
    .add_local_python_source("unseen_loop")
)


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _validated_study_id(study_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", study_id):
        raise ValueError(
            "study_id must be 1-128 characters and contain only letters, digits, '.', '_', or '-'"
        )
    return study_id


def _package_source_digest(root: Path) -> tuple[str, int]:
    source_files = sorted(root.rglob("*.py"))
    if not source_files:
        raise RuntimeError("the Modal image contains no unseen_loop Python sources")
    digest = hashlib.sha256()
    for source_file in source_files:
        relative = source_file.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(source_file.read_bytes()).digest())
    return digest.hexdigest(), len(source_files)


def _caller_source_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "modal_sdk_version": modal.__version__,
    }


@app.function(
    image=core_image,
    cpu=(8.0, 8.0),
    memory=(16_384, 16_384),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=4,
    scaledown_window=60,
    timeout=6 * 3_600,
    retries=0,
)
def run_suite(config_text: str, study_id: str, caller_source_json: str) -> str:
    """Run one caller-supplied release TOML wholly on a bounded Modal CPU worker."""
    import tempfile

    import unseen_loop
    from unseen_loop.suite import run_release_suite

    validated_id = _validated_study_id(study_id)
    config_bytes = config_text.encode("utf-8")
    if not config_text.strip():
        raise ValueError("config_text must contain a release-suite TOML document")
    if len(config_bytes) > MAX_CONFIG_BYTES:
        raise ValueError(f"config_text exceeds the {MAX_CONFIG_BYTES}-byte limit")
    caller_source = json.loads(caller_source_json)
    if not isinstance(caller_source, dict):
        raise TypeError("caller_source_json must encode an object")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    package_file = unseen_loop.__file__
    if package_file is None:
        raise RuntimeError("cannot locate unseen_loop package sources in the Modal image")
    source_sha256, source_file_count = _package_source_digest(Path(package_file).parent)

    artifacts.reload()
    destination = ARTIFACT_ROOT / validated_id
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(
            f"study_id {validated_id!r} already has artifacts; choose a fresh study_id"
        )
    suite_destination = destination / "suite"

    with tempfile.TemporaryDirectory(prefix="unseen-loop-release-config-") as temporary:
        config_path = Path(temporary) / "release.toml"
        config_path.write_bytes(config_bytes)
        suite_summary = run_release_suite(
            config_path=config_path,
            output=suite_destination,
            backend="clear",
        )
    if suite_summary.get("config_sha256") != config_sha256:
        raise RuntimeError("release suite reported a config digest different from its input")

    summary: dict[str, Any] = {
        "schema_version": "unseen-loop/modal-study-v1",
        "study_id": validated_id,
        "artifact_path": str(destination),
        "suite_artifact_path": str(suite_destination),
        "config": {
            "artifact_path": str(suite_destination / "release.toml"),
            "bytes": len(config_bytes),
            "sha256": config_sha256,
            "source": "caller-supplied TOML text",
        },
        "source": {
            "executor": "unseen_loop.suite.run_release_suite",
            "package_transport": "modal.Image.add_local_python_source('unseen_loop')",
            "python_source_sha256": source_sha256,
            "python_source_files": source_file_count,
            "python_source_digest_method": (
                "sha256 of sorted relative .py path, NUL, and file sha256 digest tuples"
            ),
            "python": "3.12",
            "numpy": "1.26.4",
            "gymnasium": "1.3.0",
            "caller": caller_source,
        },
        "execution": {
            "backend": "clear",
            "compute": "Modal CPU research worker",
            "privacy_claim": "none",
        },
        "suite_summary": suite_summary,
    }
    payload = _canonical_json(summary)
    (destination / "modal-study-summary.json").write_text(payload, encoding="utf-8")
    artifacts.commit()
    return payload


@app.local_entrypoint()
def suite(config: str, study_id: str) -> str:
    """Run: modal run modal_studies.py::suite --config PATH --study-id FRESH_ID"""
    return run_suite.remote(
        Path(config).read_text(encoding="utf-8"),
        study_id,
        json.dumps(_caller_source_provenance(), sort_keys=True),
    )


@app.local_entrypoint()
def ablations(study_id: str, config_directory: str = "experiments") -> str:
    """Run: modal run modal_studies.py::ablations --study-id FRESH_PREFIX"""
    prefix = _validated_study_id(study_id)
    directory = Path(config_directory)
    config_paths = tuple(directory / name for name in ABLATION_CONFIGS)
    config_texts = tuple(path.read_text(encoding="utf-8") for path in config_paths)
    caller_source = json.dumps(_caller_source_provenance(), sort_keys=True)
    provenance_payloads = tuple(caller_source for _ in config_paths)
    study_ids = tuple(_validated_study_id(f"{prefix}--{path.stem}") for path in config_paths)
    summaries = [
        json.loads(payload)
        for payload in run_suite.map(
            config_texts,
            study_ids,
            provenance_payloads,
            order_outputs=True,
        )
    ]
    return _canonical_json(
        {
            "schema_version": "unseen-loop/modal-ablation-batch-v1",
            "study_id": prefix,
            "config_sources": [str(path) for path in config_paths],
            "summaries": summaries,
        }
    )
