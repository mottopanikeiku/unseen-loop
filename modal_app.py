"""Modal orchestration for GPU RL training and client-key FHE evaluation.

The local entrypoint owns the FHE client and secret key. Remote functions receive
only a compiled server artifact, ciphertext, and public evaluation material.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop"
VOLUME_NAME = "unseen-loop-artifacts"
ARTIFACT_ROOT = Path("/artifacts/runs")
REPLAY_LEDGER_PATH = Path("/artifacts/protocol/replay-ledger.json")
REPLAY_RETENTION_NS = 600_000_000_000

app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "gymnasium==1.3.0")
    .add_local_python_source("unseen_loop")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "gymnasium==1.3.0", "torch==2.7.1")
    .add_local_python_source("unseen_loop")
)

fhe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==1.26.4",
        "gymnasium==1.3.0",
        "concrete-python==2.10.0",
        "setuptools==75.3.0",
    )
    .add_local_python_source("unseen_loop")
)


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=(4.0, 4.0),
    memory=(16_384, 16_384),
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=3_600,
    retries=0,
)
def train_teacher_gpu(seed: int, full: bool = False) -> dict[str, Any]:
    """Run thousands of CartPole policies concurrently on a fixed L4 GPU."""
    from unseen_loop.artifacts import dataclass_dict
    from unseen_loop.gpu_teacher import train_cartpole_gpu

    result = train_cartpole_gpu(
        seed=seed,
        hidden_size=24 if full else 12,
        iterations=40 if full else 18,
        population=8_192 if full else 2_048,
        episodes_per_candidate=32 if full else 12,
        elite_fraction=0.04 if full else 0.06,
    )
    return dataclass_dict(result)


@app.function(
    image=core_image,
    cpu=(8.0, 8.0),
    memory=(16_384, 16_384),
    volumes={"/artifacts": artifacts},
    min_containers=0,
    buffer_containers=0,
    max_containers=2,
    scaledown_window=60,
    timeout=6 * 3_600,
    retries=0,
)
def search_on_cpu(checkpoint_json: str, run_id: str, full: bool = False) -> dict[str, Any]:
    """Search the return/certificate/circuit frontier and persist every candidate."""
    import shutil
    from unseen_loop.artifacts import dataclass_dict
    from unseen_loop.experiment import ResearchPreset, run_experiment
    from unseen_loop.teacher import TeacherCheckpoint

    checkpoint = TeacherCheckpoint.from_json(checkpoint_json)
    output = ARTIFACT_ROOT / run_id / "clear-search"
    if output.exists():
        shutil.rmtree(output)
    summary = run_experiment(
        env_id=checkpoint.env_id,
        output=output,
        backend="clear",
        preset=ResearchPreset.release() if full else ResearchPreset.quick(),
        seed_root=f"modal:{run_id}",
        run_id=run_id,
        teacher_checkpoint=checkpoint,
    )
    champion_path = output / "policies" / f"{summary.champion_policy_digest}.json"
    artifacts.commit()
    return {
        "summary": dataclass_dict(summary),
        "champion_policy_json": champion_path.read_text(),
        "artifact_path": str(output),
    }


def _domain_validation_rows(policy: Any) -> Any:
    """Return two valid rows; compile_policy constructs the range-sound inputset."""
    import numpy as np

    dimensions = policy.spec.quantizer.n_features
    qmax = policy.spec.quantizer.qmax
    return np.asarray(
        (
            np.full(dimensions, -qmax, dtype=np.int64),
            np.full(dimensions, qmax, dtype=np.int64),
        ),
        dtype=np.int64,
    )


@app.function(
    image=fhe_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    min_containers=0,
    buffer_containers=0,
    max_containers=2,
    scaledown_window=60,
    timeout=6 * 3_600,
    retries=0,
)
def compile_finalist(policy_json: str) -> dict[str, Any]:
    """Compile on the same x86 image family used by the remote evaluator."""
    from unseen_loop.artifacts import dataclass_dict
    from unseen_loop.fhe_backend import compile_policy
    from unseen_loop.policy import PolynomialPolicy
    from unseen_loop.specs import PolicySpec

    policy = PolynomialPolicy(PolicySpec.from_json(policy_json))
    domain_rows = _domain_validation_rows(policy)
    with tempfile.TemporaryDirectory(prefix="unseen-loop-modal-compile-") as temporary:
        compiled = compile_policy(policy, domain_rows, temporary, global_p_error=1e-6)
        return {
            "server_artifact": compiled.server_path.read_bytes(),
            "client_specs": compiled.client_specs_path.read_bytes(),
            "receipt": dataclass_dict(compiled.receipt),
            "calibration_digest": compiled.receipt.calibration_sha256,
        }


def _claim_authenticated_request(request_digest: str, *, now_ns: int) -> None:
    """Atomically claim one authenticated request in the serialized evaluator pool."""
    artifacts.reload()
    try:
        raw = json.loads(REPLAY_LEDGER_PATH.read_text())
    except FileNotFoundError:
        raw = {}
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or type(value) is not int for key, value in raw.items()
    ):
        raise RuntimeError("durable replay ledger is corrupt")
    retained = {
        key: timestamp
        for key, timestamp in raw.items()
        if now_ns - timestamp <= REPLAY_RETENTION_NS
    }
    if request_digest in retained:
        raise RuntimeError("authenticated request replay detected")
    retained[request_digest] = now_ns
    REPLAY_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPLAY_LEDGER_PATH.with_suffix(f".{request_digest}.tmp")
    temporary.write_text(json.dumps(retained, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(REPLAY_LEDGER_PATH)
    artifacts.commit()


@app.function(
    image=fhe_image,
    cpu=(8.0, 8.0),
    memory=(16_384, 16_384),
    min_containers=0,
    volumes={"/artifacts": artifacts},
    buffer_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=3_600,
    retries=0,
)
def evaluate_ciphertext(
    server_artifact: bytes,
    signed_request_json: str,
    evaluation_keys: bytes,
    authentication_key: bytes,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate, bind, and evaluate ciphertext without constructing a client."""
    from concrete import fhe

    from unseen_loop.fhe_backend import server_artifact_secret_markers
    from unseen_loop.protocol import (
        FixedShapeGuard,
        RequestEnvelope,
        ResponseEnvelope,
        SignedEnvelope,
        TranscriptAuthenticator,
    )

    if len(signed_request_json.encode("utf-8")) > 1_500_000:
        raise ValueError("signed request exceeds the authenticated transport cap")
    authenticator = TranscriptAuthenticator(authentication_key)
    signed_request = SignedEnvelope.from_json(signed_request_json)
    verified = authenticator.verify(signed_request, RequestEnvelope)
    if not isinstance(verified, RequestEnvelope):
        raise TypeError("authenticated evaluator request has the wrong type")
    if verified.ciphertext_bytes > 1_048_576:
        raise ValueError("serialized ciphertext exceeds the evaluator safety cap")

    evaluation_key_digest = hashlib.sha256(evaluation_keys).hexdigest()
    actual_server_digest = hashlib.sha256(server_artifact).hexdigest()
    if actual_server_digest != receipt["server_artifact_sha256"]:
        raise ValueError("server artifact does not match its circuit receipt")
    guard = FixedShapeGuard(
        policy_digest=receipt["policy_digest"],
        circuit_digest=actual_server_digest,
        client_context_digest=receipt["client_specs_sha256"],
        evaluation_key_digest=evaluation_key_digest,
        observation_shape=tuple(receipt["input_shape"]),
        output_shape=(len(receipt["integer_output_bound"]),),
        request_bytes=verified.ciphertext_bytes,
        response_bytes=1,
    )
    encrypted_input = guard.validate_request(verified)

    _claim_authenticated_request(verified.digest, now_ns=time.time_ns())
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="unseen-loop-modal-server-") as temporary:
        server_path = Path(temporary) / "server.zip"
        server_path.write_bytes(server_artifact)
        actual_secret_markers = server_artifact_secret_markers(server_path)
        if tuple(receipt["server_secret_key_markers"]) != actual_secret_markers:
            raise ValueError("server artifact secret-marker audit does not match its receipt")
        if actual_secret_markers:
            raise ValueError("server artifact contains secret-key filename markers")

        # No Concrete artifact, ciphertext, or evaluation key is deserialized before
        # authentication and every context digest above has succeeded.
        server = fhe.Server.load(str(server_path))
        value = fhe.Value.deserialize(encrypted_input)
        keys = fhe.EvaluationKeys.deserialize(evaluation_keys)
        result = server.run(value, evaluation_keys=keys)
        serialized = result.serialize()
    response = ResponseEnvelope.create(
        verified,
        serialized,
        output_shape=(len(receipt["integer_output_bound"]),),
    )
    return {
        "signed_response_json": authenticator.sign(response).to_json(),
        "server_evaluate_ns": time.perf_counter_ns() - started,
        "request_sha256": hashlib.sha256(encrypted_input).hexdigest(),
        "response_sha256": hashlib.sha256(serialized).hexdigest(),
        "request_bytes": len(encrypted_input),
        "response_bytes": len(serialized),
        "evaluation_key_bytes": len(evaluation_keys),
        "server_secret_key_marker_present": bool(actual_secret_markers),
        "mode": "REAL FHE",
    }


@app.function(
    image=core_image,
    cpu=(0.25, 0.25),
    memory=(512, 512),
    volumes={"/artifacts": artifacts},
    max_containers=1,
    timeout=300,
    retries=0,
)
def persist_cloud_evidence(
    run_id: str,
    payload: str,
    server_artifact: bytes,
    client_specs: bytes,
    policy_json: str,
    receipt_json: str,
) -> dict[str, Any]:
    destination = ARTIFACT_ROOT / run_id / "modal"
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "evidence.json": payload.encode("utf-8"),
        "server.zip": server_artifact,
        "client-specs.bin": client_specs,
        "policy.json": policy_json.encode("utf-8"),
        "receipt.json": receipt_json.encode("utf-8"),
    }
    checksums: list[str] = []
    for name, content in files.items():
        (destination / name).write_bytes(content)
        checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    checksum_payload = ("\n".join(sorted(checksums)) + "\n").encode("utf-8")
    (destination / "checksums.sha256").write_bytes(checksum_payload)
    artifacts.commit()

    persisted_payload = (destination / "evidence.json").read_bytes()
    bundle_files = sorted((*files, "checksums.sha256"))
    return {
        "evidence_path": str(destination / "evidence.json"),
        "bundle_path": str(destination),
        "bundle_files": bundle_files,
        "evidence_bytes": len(persisted_payload),
        "evidence_sha256": hashlib.sha256(persisted_payload).hexdigest(),
    }


@app.local_entrypoint()
def research(run_id: str = "modal-smoke", full: bool = False) -> str:
    """GPU train → CPU search → remote compile → local encrypt → remote evaluate."""
    import gymnasium as gym
    import numpy as np
    from concrete import fhe

    from unseen_loop.certificate import certify_actions
    from unseen_loop.fhe_backend import server_artifact_secret_markers
    from unseen_loop.policy import PolynomialPolicy
    from unseen_loop.protocol import (
        FixedShapeGuard,
        RequestEnvelope,
        ResponseEnvelope,
        SignedEnvelope,
        TranscriptAuthenticator,
    )
    from unseen_loop.specs import PolicySpec
    from unseen_loop.teacher import TeacherCheckpoint, observation_constraint_cost

    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_commit = None
        source_dirty = None

    seed_digest = hashlib.sha256(f"unseen-loop|{run_id}|gpu-teacher".encode()).digest()
    training_seed = int.from_bytes(seed_digest[:4], "little") & 0x7FFF_FFFF
    gpu_record = train_teacher_gpu.remote(training_seed, full)
    checkpoint = TeacherCheckpoint.from_dict(gpu_record["checkpoint"])
    search_record = search_on_cpu.remote(checkpoint.to_json(), run_id, full)
    policy = PolynomialPolicy(PolicySpec.from_json(search_record["champion_policy_json"]))
    compiled = compile_finalist.remote(search_record["champion_policy_json"])

    receipt = compiled["receipt"]
    server_artifact = compiled["server_artifact"]
    client_specs_payload = compiled["client_specs"]
    actual_server_digest = hashlib.sha256(server_artifact).hexdigest()
    actual_client_context_digest = hashlib.sha256(client_specs_payload).hexdigest()
    if receipt["schema_version"] != "unseen-loop/circuit-receipt-v1":
        raise RuntimeError("compiled policy receipt has an unsupported schema")
    if receipt["policy_digest"] != policy.spec.digest:
        raise RuntimeError("compiled policy receipt does not match the selected policy")
    if receipt["server_artifact_sha256"] != actual_server_digest:
        raise RuntimeError("compiled server artifact does not match its receipt")
    if receipt["client_specs_sha256"] != actual_client_context_digest:
        raise RuntimeError("compiled client context does not match its receipt")
    if receipt["calibration_sha256"] != compiled["calibration_digest"]:
        raise RuntimeError("compiled calibration digest does not match its receipt")
    with tempfile.TemporaryDirectory(prefix="unseen-loop-local-artifact-audit-") as temporary:
        audit_path = Path(temporary) / "server.zip"
        audit_path.write_bytes(server_artifact)
        server_secret_markers = server_artifact_secret_markers(audit_path)
    if tuple(receipt["server_secret_key_markers"]) != server_secret_markers:
        raise RuntimeError("server artifact secret-marker audit does not match its receipt")
    if server_secret_markers:
        raise RuntimeError("server artifact contains secret-key filename markers")

    # Bind and audit serialized client material before Concrete parses it.
    specs = fhe.ClientSpecs.deserialize(client_specs_payload)
    client = fhe.Client(specs)
    keygen_started = time.perf_counter_ns()
    client.keys.generate()
    keygen_ns = time.perf_counter_ns() - keygen_started
    evaluation_keys = client.evaluation_keys.serialize()
    authentication_key = secrets.token_bytes(32)
    authenticator = TranscriptAuthenticator(authentication_key)
    evaluation_key_digest = hashlib.sha256(evaluation_keys).hexdigest()
    request_nonces: set[str] = set()
    fixed_request_bytes: int | None = None
    fixed_response_bytes: int | None = None

    def evaluate_remote(encrypted_input: bytes) -> tuple[dict[str, Any], bytes]:
        nonlocal fixed_request_bytes, fixed_response_bytes
        if fixed_request_bytes is None:
            fixed_request_bytes = len(encrypted_input)
        elif len(encrypted_input) != fixed_request_bytes:
            raise RuntimeError("encrypted request violated the fixed transcript length")
        request = RequestEnvelope.create(
            encrypted_input,
            policy_digest=policy.spec.digest,
            circuit_digest=actual_server_digest,
            client_context_digest=actual_client_context_digest,
            evaluation_key_digest=evaluation_key_digest,
            observation_shape=(policy.spec.quantizer.n_features,),
        )
        if request.nonce in request_nonces:
            raise RuntimeError("client generated a repeated protocol nonce")
        request_nonces.add(request.nonce)
        remote_record = evaluate_ciphertext.remote(
            server_artifact,
            authenticator.sign(request).to_json(),
            evaluation_keys,
            authentication_key,
            receipt,
        )
        expected_remote_fields = {
            "signed_response_json",
            "server_evaluate_ns",
            "request_sha256",
            "response_sha256",
            "request_bytes",
            "response_bytes",
            "evaluation_key_bytes",
            "server_secret_key_marker_present",
            "mode",
        }
        if not isinstance(remote_record, dict) or set(remote_record) != expected_remote_fields:
            raise RuntimeError("remote evaluator returned an invalid measurement schema")
        signed_response = SignedEnvelope.from_json(remote_record["signed_response_json"])
        verified_response = authenticator.verify(signed_response, ResponseEnvelope)
        if not isinstance(verified_response, ResponseEnvelope):
            raise TypeError("authenticated evaluator response has the wrong type")
        if fixed_response_bytes is None:
            fixed_response_bytes = verified_response.ciphertext_bytes
        elif verified_response.ciphertext_bytes != fixed_response_bytes:
            raise RuntimeError("encrypted response violated the fixed transcript length")
        guard = FixedShapeGuard(
            policy_digest=policy.spec.digest,
            circuit_digest=actual_server_digest,
            client_context_digest=actual_client_context_digest,
            evaluation_key_digest=evaluation_key_digest,
            observation_shape=(policy.spec.quantizer.n_features,),
            output_shape=(policy.spec.actions,),
            request_bytes=fixed_request_bytes,
            response_bytes=fixed_response_bytes,
        )
        encrypted_output = guard.validate_response(request, verified_response)
        expected_measurements = {
            "request_sha256": hashlib.sha256(encrypted_input).hexdigest(),
            "response_sha256": hashlib.sha256(encrypted_output).hexdigest(),
            "request_bytes": len(encrypted_input),
            "response_bytes": len(encrypted_output),
            "evaluation_key_bytes": len(evaluation_keys),
            "server_secret_key_marker_present": bool(server_secret_markers),
            "mode": "REAL FHE",
        }
        for name, expected_value in expected_measurements.items():
            if remote_record[name] != expected_value:
                raise RuntimeError(f"remote evaluator measurement mismatch: {name}")
        server_evaluate_ns = remote_record["server_evaluate_ns"]
        if type(server_evaluate_ns) is not int or server_evaluate_ns < 0:
            raise RuntimeError("remote evaluator returned an invalid duration")
        sanitized_record = {
            **expected_measurements,
            "server_evaluate_ns": server_evaluate_ns,
            "protocol": {
                "name": "authenticated-envelope-v1",
                "authentication_algorithm": signed_response.algorithm,
                "request_schema_version": request.schema_version,
                "response_schema_version": verified_response.schema_version,
                "request_envelope_digest": request.digest,
                "response_envelope_digest": verified_response.digest,
                "policy_digest": request.policy_digest,
                "circuit_digest": request.circuit_digest,
                "client_context_digest": request.client_context_digest,
                "evaluation_key_digest": request.evaluation_key_digest,
            },
        }
        return sanitized_record, encrypted_output

    quantized = np.zeros(policy.spec.quantizer.n_features, dtype=np.int64)
    expected = np.atleast_1d(policy.integer_scores_from_quantized(quantized))
    trials: list[dict[str, Any]] = []
    for trial in range(2 if not full else 5):
        encrypt_started = time.perf_counter_ns()
        encrypted_input = client.encrypt(quantized).serialize()
        encrypt_ns = time.perf_counter_ns() - encrypt_started
        server_record, serialized_output = evaluate_remote(encrypted_input)
        decrypt_started = time.perf_counter_ns()
        output = fhe.Value.deserialize(serialized_output)
        decrypted = np.atleast_1d(np.asarray(client.decrypt(output), dtype=np.int64))
        decrypt_ns = time.perf_counter_ns() - decrypt_started
        trials.append(
            {
                **server_record,
                "trial": trial,
                "encrypt_ns": encrypt_ns,
                "decrypt_ns": decrypt_ns,
                "output_shape": list(decrypted.shape),
                "action": int(np.argmax(decrypted)),
                "matches_integer_clear": bool(np.array_equal(decrypted, expected)),
            }
        )
    if not all(row["matches_integer_clear"] for row in trials):
        raise RuntimeError("Modal real-FHE execution diverged from integer-clear semantics")
    if len({row["request_sha256"] for row in trials}) != len(trials):
        raise RuntimeError("fresh encryptions produced repeated ciphertext hashes")
    same_input_canary = {
        "repetitions": len(trials),
        "distinct_ciphertexts": len({row["request_sha256"] for row in trials}),
        "all_match": all(row["matches_integer_clear"] for row in trials),
    }
    episode_seed_digest = hashlib.sha256(
        f"unseen-loop|{run_id}|closed-loop-episode".encode()
    ).digest()
    episode_seed = int.from_bytes(episode_seed_digest[:4], "little") & 0x7FFF_FFFF
    environment = gym.make(policy.spec.env_id)
    observation, _ = environment.reset(seed=episode_seed)
    environment.action_space.seed(episode_seed)
    maximum_steps = 25
    episode_return = 0.0
    constraint_cost = 0.0
    terminated = False
    truncated = False
    trajectory: list[dict[str, Any]] = []
    try:
        while len(trajectory) < maximum_steps and not (terminated or truncated):
            step_started = time.perf_counter_ns()
            quantized_step = policy.quantize(observation, reject=True)
            expected_step = np.atleast_1d(policy.integer_scores_from_quantized(quantized_step))
            certificate = certify_actions(policy, quantized_step)
            encrypt_started = time.perf_counter_ns()
            encrypted_step = client.encrypt(quantized_step).serialize()
            encrypt_ns = time.perf_counter_ns() - encrypt_started
            server_step, serialized_step = evaluate_remote(encrypted_step)
            decrypt_started = time.perf_counter_ns()
            encrypted_output = fhe.Value.deserialize(serialized_step)
            decrypted_step = np.atleast_1d(
                np.asarray(client.decrypt(encrypted_output), dtype=np.int64)
            )
            decrypt_ns = time.perf_counter_ns() - decrypt_started
            if not np.array_equal(decrypted_step, expected_step):
                raise RuntimeError("closed-loop REAL FHE output diverged from integer clear")
            action = int(np.argmax(decrypted_step))
            constraint_cost += observation_constraint_cost(policy.spec.env_id, observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += float(reward)
            trajectory.append(
                {
                    **server_step,
                    "step": len(trajectory),
                    "encrypt_ns": encrypt_ns,
                    "decrypt_ns": decrypt_ns,
                    "online_end_to_end_ns": time.perf_counter_ns() - step_started,
                    "action": action,
                    "reward": float(reward),
                    "certified": bool(certificate.certified[0]),
                    "matches_integer_clear": True,
                }
            )
    finally:
        environment.close()
    closed_loop = {
        "episode_seed": episode_seed,
        "requested_steps": maximum_steps,
        "completed_steps": len(trajectory),
        "return": episode_return,
        "constraint_cost": constraint_cost,
        "terminated": terminated,
        "truncated": truncated,
        "exact_matches": sum(row["matches_integer_clear"] for row in trajectory),
        "certified_steps": sum(row["certified"] for row in trajectory),
        "trajectory": trajectory,
        "global_p_error_union_bound": min(
            1.0, len(trajectory) * compiled["receipt"]["global_p_error"]
        ),
        "server_received_plaintext_observation": False,
    }
    if not trajectory:
        raise RuntimeError("closed-loop REAL FHE trace produced no steps")

    bundle_path = f"/artifacts/runs/{run_id}/modal"
    bundle_files = [
        "checksums.sha256",
        "client-specs.bin",
        "evidence.json",
        "policy.json",
        "receipt.json",
        "server.zip",
    ]
    evidence = {
        "schema_version": "unseen-loop/modal-evidence-v2",
        "run_id": run_id,
        "source": {
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "modal_sdk_version": modal.__version__,
        },
        "gpu_training": gpu_record,
        "clear_search": search_record["summary"],
        "circuit_receipt": receipt,
        "calibration_digest": compiled["calibration_digest"],
        "client": {
            "location": "local entrypoint",
            "secret_key_sent_to_modal": False,
            "keygen_ns": keygen_ns,
            "evaluation_key_bytes": len(evaluation_keys),
        },
        "authenticated_envelope_protocol": {
            "name": "authenticated-envelope-v1",
            "authentication_algorithm": "HMAC-SHA256",
            "request_schema_version": "unseen-loop/request-v1",
            "response_schema_version": "unseen-loop/response-v1",
            "replay_protection": "Volume-backed request-digest ledger within freshness window",
            "bindings": [
                "policy_digest",
                "circuit_digest",
                "client_context_digest",
                "evaluation_key_digest",
                "request_digest",
            ],
        },
        "artifact_secret_marker_audit": {
            "server_secret_key_markers": list(server_secret_markers),
            "server_secret_key_marker_present": bool(server_secret_markers),
        },
        "same_input_canary": same_input_canary,
        "real_fhe_trials": trials,
        "closed_loop_real_fhe": closed_loop,
        "all_real_fhe_match": True,
        "privacy_evidence": True,
        "limitations": [
            "honest-but-curious evaluator only",
            "authentication is not a proof of correct evaluation",
            "client learns score vector and performs argmax",
            "server archive filename-marker audit is not a proof of secret absence",
        ],
        "nonsecret_bundle": {
            "volume_path": bundle_path,
            "evidence_path": f"{bundle_path}/evidence.json",
            "checksums_path": f"{bundle_path}/checksums.sha256",
            "files": bundle_files,
        },
    }
    payload = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    receipt_json = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    persisted = persist_cloud_evidence.remote(
        run_id,
        payload,
        server_artifact,
        client_specs_payload,
        search_record["champion_policy_json"],
        receipt_json,
    )
    expected_persisted = {
        "evidence_path": evidence["nonsecret_bundle"]["evidence_path"],
        "bundle_path": evidence["nonsecret_bundle"]["volume_path"],
        "bundle_files": bundle_files,
        "evidence_bytes": len(payload.encode("utf-8")),
        "evidence_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    if persisted != expected_persisted:
        raise RuntimeError("Modal Volume evidence bytes or bundle manifest do not match locally")
    return payload
