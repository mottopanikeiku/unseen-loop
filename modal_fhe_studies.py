"""Modal-only REAL-FHE challenge and timing studies.

The timing study intentionally uses four independent, colocated client/server research
contexts.  A client secret key is generated and consumed inside each dedicated Modal
worker; no result from this study is evidence of local-client/remote-server secrecy.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop-fhe-studies"
VOLUME_NAME = "unseen-loop-artifacts"
STUDY_ROOT = Path("/artifacts/studies")
STUDY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")

FHE_PACKAGES = (
    "numpy==1.26.4",
    "concrete-python==2.10.0",
    "setuptools==75.3.0",
)
TIMING_CONTAINER_COUNT = 4
TIMING_WARMUPS_PER_CONTAINER = 3
TIMING_MEASURED_PER_CONTAINER = 16
TIMING_ATTEMPTS = TIMING_CONTAINER_COUNT * (
    TIMING_WARMUPS_PER_CONTAINER + TIMING_MEASURED_PER_CONTAINER
)

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Aggregation never imports Concrete. REAL-FHE compilation, key generation, and
# evaluation are confined to the separately pinned FHE image.
core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4")
    .add_local_python_source("unseen_loop")
)
fhe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*FHE_PACKAGES)
    .add_local_python_source("unseen_loop")
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pretty_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_study_id(study_id: str) -> str:
    if not isinstance(study_id, str) or STUDY_ID_PATTERN.fullmatch(study_id) is None:
        raise ValueError(
            "study_id must be 1-80 characters containing only letters, digits, '.', '_', or '-'"
        )
    return study_id


def _parse_object(payload: str, *, name: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must encode a JSON object")
    return value


def _require_empty_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("study destination must be a directory")
        if any(destination.iterdir()):
            raise RuntimeError("study destination is not empty; refusing to overwrite evidence")
    destination.mkdir(parents=True, exist_ok=True)


def _rewrite_checksum_ledger(destination: Path) -> list[str]:
    checksum_path = destination / "checksums.sha256"
    files = sorted(
        path for path in destination.iterdir() if path.is_file() and path != checksum_path
    )
    lines = [f"{_sha256(path.read_bytes())}  {path.name}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n")
    return [path.name for path in files] + [checksum_path.name]


def _source_provenance() -> dict[str, Any]:
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
        "entrypoint_sha256": _sha256(Path(__file__).read_bytes()),
    }


def _challenge_configuration() -> dict[str, Any]:
    return {
        "schema_version": "unseen-loop/modal-nonlinear-challenge-config-v1",
        "qmax": 2,
        "canary_repetitions": 5,
        "global_p_error": 1e-6,
        "security_level": 128,
        "expected_real_fhe_calls": 40,
        "image": {
            "python": "3.12",
            "packages": list(FHE_PACKAGES),
        },
        "resources": {
            "cpu": 16.0,
            "memory_mib": 32_768,
            "timeout_seconds": 21_600,
            "max_containers": 1,
            "retries": 0,
        },
        "trust_scope": {
            "client_and_server_location": "one dedicated Modal research worker",
            "client_secret_key_leaves_worker": False,
            "local_client_remote_server_secrecy_claim": False,
        },
    }


@app.function(
    image=fhe_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=6 * 3_600,
    retries=0,
)
def nonlinear_challenge_remote(study_id: str, source_provenance_json: str) -> str:
    """Run all 40 nonlinear challenge calls in one bounded Modal worker."""
    from dataclasses import asdict

    from unseen_loop.challenge import run_fhe_challenge

    checked_id = _validate_study_id(study_id)
    source = _parse_object(source_provenance_json, name="source_provenance_json")
    configuration = _challenge_configuration()
    destination = STUDY_ROOT / checked_id

    # run_fhe_challenge performs its own empty-destination check and writes the
    # canonical policy, receipt, serialized circuit, rows, summary, and ledger.
    challenge_summary = run_fhe_challenge(
        destination,
        qmax=configuration["qmax"],
        canary_repetitions=configuration["canary_repetitions"],
        global_p_error=configuration["global_p_error"],
        security_level=configuration["security_level"],
    )
    if challenge_summary.real_fhe_rows != configuration["expected_real_fhe_calls"]:
        raise RuntimeError("nonlinear challenge did not execute exactly 40 REAL-FHE calls")

    summary: dict[str, Any] = {
        "schema_version": "unseen-loop/modal-nonlinear-challenge-study-v1",
        "study_id": checked_id,
        "configuration": configuration,
        "source_provenance": source,
        "execution": {
            "location": "Modal",
            "remote_function": "nonlinear_challenge_remote",
            "real_fhe_calls": challenge_summary.real_fhe_rows,
            "all_calls_remote": True,
        },
        "challenge_summary": asdict(challenge_summary),
        "artifact_path": str(destination),
    }
    (destination / "summary.json").write_text(_pretty_json(summary))
    bundle_files = _rewrite_checksum_ledger(destination)
    summary["bundle_files"] = bundle_files
    # Include the final manifest in the persisted summary, then close the ledger
    # over those exact final bytes.
    (destination / "summary.json").write_text(_pretty_json(summary))
    _rewrite_checksum_ledger(destination)
    artifacts.commit()
    return _pretty_json(summary)


def _timing_configuration(study_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "unseen-loop/modal-fhe-timing-config-v1",
        "study_id": study_id,
        "source_provenance": source,
        "circuit": {
            "family": "degree-2 two-feature two-action integer polynomial",
            "policy_factory": "unseen_loop.challenge.challenge_policy_spec",
            "qmax": 2,
            "single_query_input": [1, -2],
            "global_p_error": 1e-6,
            "security_level": 128,
            "calibration": "exhaustive signed integer domain",
        },
        "schedule": {
            "logical_container_slots": [f"slot-{index}" for index in range(4)],
            "containers": TIMING_CONTAINER_COUNT,
            "warmups_per_container": TIMING_WARMUPS_PER_CONTAINER,
            "measured_requests_per_container": TIMING_MEASURED_PER_CONTAINER,
            "total_real_fhe_attempts": TIMING_ATTEMPTS,
            "query_batch_size": 1,
            "input_concurrency_per_container": 1,
        },
        "metrics": {
            "timing_ns": [
                "encrypt_ns",
                "server_evaluate_ns",
                "decrypt_ns",
                "end_to_end_ns",
            ],
            "byte_metrics": ["evaluation_key_bytes", "request_bytes", "response_bytes"],
            "warmups_excluded_from_quantiles": True,
        },
        "image": {
            "python": "3.12",
            "packages": list(FHE_PACKAGES),
        },
        "resources_per_worker": {
            "cpu": 16.0,
            "memory_mib": 32_768,
            "timeout_seconds": 21_600,
            "max_containers": 4,
            "retries": 0,
        },
        "trust_scope": {
            "context_model": "four independent colocated client/server research contexts",
            "client_secret_key_generated_in": "its dedicated Modal timing worker",
            "client_secret_key_leaves_worker": False,
            "client_material_sent_to_aggregator": False,
            "local_client_remote_server_secrecy_claim": False,
            "aggregation_claim": (
                "performance distribution clustered by four independent Modal contexts; "
                "not a shared cryptographic client/server context"
            ),
        },
    }


@app.function(
    image=fhe_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    min_containers=0,
    buffer_containers=0,
    max_containers=4,
    scaledown_window=60,
    timeout=6 * 3_600,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def timing_container_remote(worker_request_json: str) -> str:
    """Compile, key, warm up, and measure one independent REAL-FHE context."""
    import socket
    import tempfile
    from dataclasses import asdict
    from itertools import product

    import numpy as np

    from unseen_loop.challenge import _SerializedChallengeSession, challenge_policy_spec
    from unseen_loop.fhe_backend import compile_policy
    from unseen_loop.policy import PolynomialPolicy

    request = _parse_object(worker_request_json, name="worker_request_json")
    configuration = request.get("configuration")
    if not isinstance(configuration, dict):
        raise TypeError("worker request configuration must be an object")
    slot = request.get("slot")
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 4:
        raise ValueError("worker slot must be an integer in [0, 4)")

    schedule = configuration.get("schedule")
    circuit_configuration = configuration.get("circuit")
    if not isinstance(schedule, dict) or not isinstance(circuit_configuration, dict):
        raise TypeError("worker request is missing timing schedule or circuit configuration")
    if (
        schedule.get("warmups_per_container") != TIMING_WARMUPS_PER_CONTAINER
        or schedule.get("measured_requests_per_container") != TIMING_MEASURED_PER_CONTAINER
    ):
        raise ValueError("worker request does not match the release timing schedule")

    hostname = socket.gethostname()
    physical_container_digest = _sha256(hostname.encode("utf-8"))
    container_id = f"slot-{slot}-{physical_container_digest[:16]}"
    trial_id = f"trial-{slot}"
    qmax = circuit_configuration["qmax"]
    policy = PolynomialPolicy(challenge_policy_spec(qmax=qmax))
    domain = np.asarray(tuple(product(range(-qmax, qmax + 1), repeat=2)), dtype=np.int64)
    fixed_query = np.asarray(circuit_configuration["single_query_input"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"unseen-loop-timing-{slot}-") as temporary:
        compiled = compile_policy(
            policy,
            domain,
            Path(temporary),
            global_p_error=circuit_configuration["global_p_error"],
            security_level=circuit_configuration["security_level"],
        )
        session = _SerializedChallengeSession(compiled)
        receipt = asdict(compiled.receipt)

        attempts = TIMING_WARMUPS_PER_CONTAINER + TIMING_MEASURED_PER_CONTAINER
        for index in range(attempts):
            is_warmup = index < TIMING_WARMUPS_PER_CONTAINER
            phase_index = index if is_warmup else index - TIMING_WARMUPS_PER_CONTAINER
            request_id = ("warmup" if is_warmup else "measured") + f"-{phase_index:02d}"
            try:
                measurement = session.run(fixed_query)
                if measurement.backend != "REAL FHE":
                    raise RuntimeError("backend label is not REAL FHE")
                if not measurement.output_matches_clear:
                    raise RuntimeError("REAL-FHE output mismatch")
                if measurement.server_secret_key_marker_present:
                    raise RuntimeError("server artifact contains a secret-key marker")
                row: dict[str, Any] = {
                    "container_id": container_id,
                    "trial_id": trial_id,
                    "request_id": request_id,
                    "is_warmup": is_warmup,
                    "success": True,
                    "timing_ns": {
                        "encrypt_ns": measurement.encrypt_ns,
                        "server_evaluate_ns": measurement.server_evaluate_ns,
                        "decrypt_ns": measurement.decrypt_ns,
                        "end_to_end_ns": measurement.end_to_end_ns,
                    },
                    "byte_metrics": {
                        "evaluation_key_bytes": measurement.evaluation_key_bytes,
                        "request_bytes": measurement.request_bytes,
                        "response_bytes": measurement.response_bytes,
                    },
                }
            except Exception:  # failure evidence is deliberately sanitized
                row = {
                    "container_id": container_id,
                    "trial_id": trial_id,
                    "request_id": request_id,
                    "is_warmup": is_warmup,
                    "success": False,
                    "failure_code": "real_fhe_evaluation_error",
                    "timing_ns": {},
                    "byte_metrics": {},
                }
            rows.append(row)

    context = {
        "logical_slot": slot,
        "container_id": container_id,
        "physical_container_sha256": physical_container_digest,
        "hostname_retained": False,
        "policy_digest": policy.spec.digest,
        "circuit_receipt": receipt,
        "client_context_sha256": session.client_context_sha256,
        "evaluation_key_sha256": session.evaluation_key_sha256,
        "client_keygen_ns": session.client_keygen_ns,
        "trust_scope": configuration["trust_scope"],
        "attempts": len(rows),
    }
    result = {
        "schema_version": "unseen-loop/modal-fhe-timing-worker-v1",
        "slot": slot,
        "context": context,
        "rows_without_aggregation_context": rows,
    }
    return _canonical_json(result)


@app.function(
    image=core_image,
    cpu=(4.0, 4.0),
    memory=(8_192, 8_192),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=12 * 3_600,
    retries=0,
)
def timing_study_remote(study_id: str, source_provenance_json: str) -> str:
    """Fan out to exactly four FHE workers, summarize remotely, and persist evidence."""
    from unseen_loop.timing import TIMING_ROW_SCHEMA, summarize_timing_rows

    checked_id = _validate_study_id(study_id)
    source = _parse_object(source_provenance_json, name="source_provenance_json")
    configuration = _timing_configuration(checked_id, source)
    worker_requests = [
        _canonical_json({"slot": slot, "configuration": configuration})
        for slot in range(TIMING_CONTAINER_COUNT)
    ]

    # map submits all four long-running calls together. max_inputs=1 and
    # max_containers=4 make each concurrent input occupy its own container.
    worker_payloads = list(timing_container_remote.map(worker_requests, order_outputs=True))
    workers = [_parse_object(payload, name="timing worker result") for payload in worker_payloads]
    workers.sort(key=lambda item: int(item["slot"]))

    if [worker["slot"] for worker in workers] != list(range(TIMING_CONTAINER_COUNT)):
        raise RuntimeError("timing study did not return exactly one result for each worker slot")
    contexts = [worker.get("context") for worker in workers]
    if any(not isinstance(context, dict) for context in contexts):
        raise RuntimeError("timing worker omitted its context record")
    typed_contexts = [context for context in contexts if isinstance(context, dict)]
    physical_ids = {str(context["physical_container_sha256"]) for context in typed_contexts}
    container_ids = {str(context["container_id"]) for context in typed_contexts}
    if len(physical_ids) != TIMING_CONTAINER_COUNT or len(container_ids) != TIMING_CONTAINER_COUNT:
        raise RuntimeError("timing study requires four actual, distinct Modal containers")

    context_record: dict[str, Any] = {
        "schema_version": "unseen-loop/modal-fhe-timing-context-v1",
        "study_id": checked_id,
        "configuration": configuration,
        "source_provenance": source,
        "aggregation_unit": "independent colocated client/server research context",
        "cryptographic_contexts_pooled_as_one": False,
        "container_contexts": typed_contexts,
    }
    context_digest = _sha256(_canonical_json(context_record).encode("utf-8"))
    context_record["context_digest"] = context_digest

    rows: list[dict[str, Any]] = []
    for worker in workers:
        worker_rows = worker.get("rows_without_aggregation_context")
        if not isinstance(worker_rows, list):
            raise RuntimeError("timing worker omitted its sanitized rows")
        for unbound_row in worker_rows:
            if not isinstance(unbound_row, dict):
                raise RuntimeError("timing worker returned a non-object row")
            rows.append(
                {
                    "schema_version": TIMING_ROW_SCHEMA,
                    "context_digest": context_digest,
                    **unbound_row,
                }
            )
    rows.sort(key=lambda row: (str(row["container_id"]), str(row["request_id"])))
    if len(rows) != TIMING_ATTEMPTS:
        raise RuntimeError("timing study did not retain exactly 76 REAL-FHE attempts")

    summary = summarize_timing_rows(rows)
    release_quality = summary.get("release_quality")
    release_eligible = isinstance(release_quality, dict) and release_quality.get("eligible") is True
    study_summary: dict[str, Any] = {
        "schema_version": "unseen-loop/modal-fhe-timing-study-v1",
        "study_id": checked_id,
        "source_provenance": source,
        "context_digest": context_digest,
        "execution": {
            "location": "Modal",
            "remote_worker_function": "timing_container_remote",
            "actual_distinct_containers": len(physical_ids),
            "real_fhe_attempts": len(rows),
            "warmup_attempts": TIMING_CONTAINER_COUNT * TIMING_WARMUPS_PER_CONTAINER,
            "measured_attempts": TIMING_CONTAINER_COUNT * TIMING_MEASURED_PER_CONTAINER,
            "all_attempts_remote": True,
        },
        "trust_scope": configuration["trust_scope"],
        "timing_summary": summary,
        "release_quality_required": True,
        "release_quality": release_eligible,
        "artifact_path": str(STUDY_ROOT / checked_id),
        "bundle_files": ["checksums.sha256", "context.json", "raw.jsonl", "summary.json"],
    }

    destination = STUDY_ROOT / checked_id
    _require_empty_destination(destination)
    (destination / "context.json").write_text(_pretty_json(context_record))
    raw_payload = "".join(_canonical_json(row) + "\n" for row in rows)
    (destination / "raw.jsonl").write_text(raw_payload)
    (destination / "summary.json").write_text(_pretty_json(study_summary))
    _rewrite_checksum_ledger(destination)
    artifacts.commit()

    if not release_eligible:
        raise RuntimeError("timing study failed the required release_quality gate")
    return _pretty_json(study_summary)


@app.local_entrypoint()
def nonlinear_challenge(study_id: str = "modal-nonlinear-qmax2") -> str:
    """Launch the complete nonlinear study; the local process performs no FHE work."""
    return nonlinear_challenge_remote.remote(study_id, _canonical_json(_source_provenance()))


@app.local_entrypoint()
def timing_study(study_id: str = "modal-fhe-timing") -> str:
    """Launch the four-container study; all 76 evaluations and summary work are remote."""
    return timing_study_remote.remote(study_id, _canonical_json(_source_provenance()))
