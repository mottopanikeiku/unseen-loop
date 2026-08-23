"""Modal orchestration for GPU RL training and client-key FHE evaluation.

The local entrypoint owns the FHE client and secret key. Remote functions receive
only a compiled server artifact, ciphertext, and public evaluation material.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop"
VOLUME_NAME = "unseen-loop-artifacts"
ARTIFACT_ROOT = Path("/artifacts/runs")

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
    from unseen_loop.artifacts import dataclass_dict
    from unseen_loop.experiment import ResearchPreset, run_experiment
    from unseen_loop.teacher import TeacherCheckpoint

    checkpoint = TeacherCheckpoint.from_json(checkpoint_json)
    output = ARTIFACT_ROOT / run_id / "clear-search"
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


def _corner_inputset(policy: Any) -> Any:
    import itertools

    import numpy as np

    dimensions = policy.spec.quantizer.n_features
    qmax = policy.spec.quantizer.qmax
    corners = tuple(itertools.product((-qmax, qmax), repeat=dimensions))
    axes = [np.zeros(dimensions, dtype=np.int64)]
    for feature in range(dimensions):
        low = np.zeros(dimensions, dtype=np.int64)
        high = low.copy()
        low[feature] = -qmax
        high[feature] = qmax
        axes.extend((low, high))
    return np.unique(np.concatenate((np.asarray(corners, dtype=np.int64), axes), axis=0), axis=0)


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
    inputset = _corner_inputset(policy)
    with tempfile.TemporaryDirectory(prefix="unseen-loop-modal-compile-") as temporary:
        compiled = compile_policy(policy, inputset, temporary, global_p_error=1e-6)
        return {
            "server_artifact": compiled.server_path.read_bytes(),
            "client_specs": compiled.client_specs_path.read_bytes(),
            "receipt": dataclass_dict(compiled.receipt),
            "calibration_digest": hashlib.sha256(inputset.tobytes()).hexdigest(),
        }


@app.function(
    image=fhe_image,
    cpu=(8.0, 8.0),
    memory=(16_384, 16_384),
    min_containers=0,
    buffer_containers=0,
    max_containers=4,
    scaledown_window=60,
    timeout=3_600,
    retries=0,
)
def evaluate_ciphertext(
    server_artifact: bytes,
    encrypted_input: bytes,
    evaluation_keys: bytes,
) -> dict[str, Any]:
    """Evaluate one opaque ciphertext without constructing a client or secret key."""
    from concrete import fhe

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="unseen-loop-modal-server-") as temporary:
        server_path = Path(temporary) / "server.zip"
        server_path.write_bytes(server_artifact)
        server = fhe.Server.load(str(server_path))
        value = fhe.Value.deserialize(encrypted_input)
        keys = fhe.EvaluationKeys.deserialize(evaluation_keys)
        result = server.run(value, evaluation_keys=keys)
        serialized = result.serialize()
    return {
        "encrypted_output": serialized,
        "server_evaluate_ns": time.perf_counter_ns() - started,
        "request_sha256": hashlib.sha256(encrypted_input).hexdigest(),
        "response_sha256": hashlib.sha256(serialized).hexdigest(),
        "request_bytes": len(encrypted_input),
        "response_bytes": len(serialized),
        "evaluation_key_bytes": len(evaluation_keys),
        "server_secret_key_present": False,
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
def persist_cloud_evidence(run_id: str, payload: str, server_artifact: bytes) -> str:
    destination = ARTIFACT_ROOT / run_id / "modal"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "evidence.json"
    server_path = destination / "server.zip"
    manifest_path.write_text(payload)
    server_path.write_bytes(server_artifact)
    artifacts.commit()
    return str(manifest_path)


@app.local_entrypoint()
def research(run_id: str = "modal-smoke", full: bool = False) -> str:
    """GPU train → CPU search → remote compile → local encrypt → remote evaluate."""
    import numpy as np
    from concrete import fhe

    from unseen_loop.policy import PolynomialPolicy
    from unseen_loop.specs import PolicySpec
    from unseen_loop.teacher import TeacherCheckpoint

    seed_digest = hashlib.sha256(f"unseen-loop|{run_id}|gpu-teacher".encode()).digest()
    training_seed = int.from_bytes(seed_digest[:4], "little") & 0x7FFF_FFFF
    gpu_record = train_teacher_gpu.remote(training_seed, full)
    checkpoint = TeacherCheckpoint.from_dict(gpu_record["checkpoint"])
    search_record = search_on_cpu.remote(checkpoint.to_json(), run_id, full)
    policy = PolynomialPolicy(PolicySpec.from_json(search_record["champion_policy_json"]))
    compiled = compile_finalist.remote(search_record["champion_policy_json"])

    specs = fhe.ClientSpecs.deserialize(compiled["client_specs"])
    client = fhe.Client(specs)
    keygen_started = time.perf_counter_ns()
    client.keys.generate()
    keygen_ns = time.perf_counter_ns() - keygen_started
    evaluation_keys = client.evaluation_keys.serialize()
    quantized = np.zeros(policy.spec.quantizer.n_features, dtype=np.int64)
    expected = np.atleast_1d(policy.integer_scores_from_quantized(quantized))
    trials: list[dict[str, Any]] = []
    for trial in range(2 if not full else 8):
        encrypt_started = time.perf_counter_ns()
        encrypted_input = client.encrypt(quantized).serialize()
        encrypt_ns = time.perf_counter_ns() - encrypt_started
        server_record = evaluate_ciphertext.remote(
            compiled["server_artifact"], encrypted_input, evaluation_keys
        )
        decrypt_started = time.perf_counter_ns()
        output = fhe.Value.deserialize(server_record.pop("encrypted_output"))
        decrypted = np.atleast_1d(np.asarray(client.decrypt(output), dtype=np.int64))
        decrypt_ns = time.perf_counter_ns() - decrypt_started
        trials.append(
            {
                **server_record,
                "trial": trial,
                "encrypt_ns": encrypt_ns,
                "decrypt_ns": decrypt_ns,
                "output": decrypted.tolist(),
                "expected": expected.tolist(),
                "matches_integer_clear": bool(np.array_equal(decrypted, expected)),
            }
        )
    if not all(row["matches_integer_clear"] for row in trials):
        raise RuntimeError("Modal real-FHE execution diverged from integer-clear semantics")
    if len({row["request_sha256"] for row in trials}) != len(trials):
        raise RuntimeError("fresh encryptions produced repeated ciphertext hashes")

    evidence = {
        "schema_version": "unseen-loop/modal-evidence-v1",
        "run_id": run_id,
        "gpu_training": gpu_record,
        "clear_search": search_record["summary"],
        "circuit_receipt": compiled["receipt"],
        "calibration_digest": compiled["calibration_digest"],
        "client": {
            "location": "local entrypoint",
            "secret_key_sent_to_modal": False,
            "keygen_ns": keygen_ns,
            "evaluation_key_bytes": len(evaluation_keys),
        },
        "real_fhe_trials": trials,
        "all_real_fhe_match": True,
        "privacy_evidence": True,
        "limitations": [
            "honest-but-curious evaluator only",
            "authentication is not a proof of correct evaluation",
            "client learns score vector and performs argmax",
        ],
    }
    payload = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    persisted = persist_cloud_evidence.remote(run_id, payload, compiled["server_artifact"])
    evidence["modal_artifact_path"] = persisted
    return json.dumps(evidence, sort_keys=True, indent=2)
