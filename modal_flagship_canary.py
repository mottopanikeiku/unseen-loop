"""Modal-only REAL-FHE canaries for CipherShield and exact/CKKS OPE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

app = modal.App("unseen-loop-flagship-canaries")
volume = modal.Volume.from_name("unseen-loop-flagship-evidence", create_if_missing=True)
root = Path("/flagship-evidence/canaries")

fhe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==1.26.4",
        "concrete-python==2.10.0",
        "setuptools==75.3.0",
    )
    .add_local_python_source("unseen_loop")
)
ckks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "tenseal==0.3.17")
    .add_local_python_source("unseen_loop")
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_bundle(study_id: str, payload: dict[str, Any]) -> str:
    destination = root / study_id
    volume.reload()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("canary destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / "summary.json"
    summary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    (destination / "checksums.sha256").write_text(f"{digest}  summary.json\n")
    volume.commit()
    return str(destination)


@app.function(
    image=fhe_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    volumes={"/flagship-evidence": volume},
    max_containers=1,
    timeout=6 * 3_600,
    retries=0,
)
def shield_canary(study_id: str) -> str:
    import tempfile

    from unseen_loop.shield.fhe import (
        ShieldIntegerSpec,
        compile_shield,
        exhaustive_simulation_conformance,
        real_fhe_canary,
    )
    from unseen_loop.shield.types import Action, ShieldState

    with tempfile.TemporaryDirectory(prefix="shield-canary-") as temporary:
        print("shield-canary: compiling", flush=True)
        compiled = compile_shield(ShieldIntegerSpec(), temporary, global_p_error=1e-6)
        print("shield-canary: compile complete", flush=True)
        simulation = exhaustive_simulation_conformance(compiled, workers=16)
        print("shield-canary: complete-domain simulation complete", flush=True)
        real = real_fhe_canary(
            compiled,
            ShieldState(0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
            Action.BRAKE,
        )
        print("shield-canary: REAL FHE roundtrip complete", flush=True)
        payload = {
            "schema_version": "unseen-loop/modal-shield-canary-v1",
            "study_id": study_id,
            "execution": "Modal REAL FHE",
            "simulation": asdict(simulation),
            "real": {
                "action": real.selection.action.name,
                "reason": real.selection.reason,
                "emergency_fallback": real.selection.emergency_fallback,
                "selected_certified": real.selection.selected_certified,
                "call": real.call.to_dict(),
            },
            "receipt": compiled.receipt.to_dict(),
        }
    artifact_path = _write_bundle(study_id, payload)
    return _canonical({"artifact_path": artifact_path, "payload": payload})


@app.function(
    image=fhe_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    volumes={"/flagship-evidence": volume},
    max_containers=1,
    timeout=6 * 3_600,
    retries=0,
)
def exact_ope_canary(study_id: str) -> str:
    import tempfile

    from unseen_loop.ope.circuit import FixedPointScales, OPECircuitSpec
    from unseen_loop.ope.fhe import compile_ope_circuit
    from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

    trajectory_spec = TrajectorySpec(
        trajectories=1,
        horizon=2,
        state_dim=1,
        action_count=2,
        state_min=(0.0,),
        state_max=(0.0,),
        reward_min=-1.0,
        reward_max=1.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.5, 0.0),) * 2,
    )
    spec = OPECircuitSpec(
        trajectory_spec,
        policy,
        gamma=1.0,
        weight_clip=2.0,
        minimum_behavior_propensity=1.0,
        scales=FixedPointScales(state=2, coefficient=2, reciprocal=2, reward=2, discount=2),
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states=(((0.0,), (0.0,)),),
        actions=((0, 1),),
        rewards=((1.0, -1.0),),
        behavior_propensities=((1.0, 1.0),),
    )
    with tempfile.TemporaryDirectory(prefix="ope-canary-") as temporary:
        compiled = compile_ope_circuit(spec, temporary, global_p_error=1e-3)
        simulated = compiled.simulate(batch, "clipped_wpdis")
        real = compiled.real_roundtrip(batch, "clipped_wpdis")
        payload = {
            "schema_version": "unseen-loop/modal-exact-ope-canary-v1",
            "study_id": study_id,
            "execution": "Modal REAL FHE",
            "simulation_matches_real": simulated.integer_statistics == real.integer_statistics,
            "receipt": json.loads(compiled.receipt.to_json()),
            "client_released_integer_statistics": asdict(real.integer_statistics),
            "client_released_statistics": asdict(real.client_statistics),
            "call": (
                None if real.call_evidence is None else json.loads(real.call_evidence.to_json())
            ),
        }
    artifact_path = _write_bundle(study_id, payload)
    return _canonical({"artifact_path": artifact_path, "payload": payload})


@app.function(
    image=ckks_image,
    cpu=(16.0, 16.0),
    memory=(32_768, 32_768),
    volumes={"/flagship-evidence": volume},
    max_containers=1,
    timeout=6 * 3_600,
    retries=0,
)
def ckks_ope_canary(study_id: str) -> str:
    from unseen_loop.crypto.ckks import CKKSClient, CKKSServer
    from unseen_loop.flagship.executor_timing import _ckks_parameters
    from unseen_loop.ope.ckks import (
        OPECKKSClient,
        OPECKKSServer,
        PolynomialApproxOPESpec,
        generate_ope_contexts,
    )
    from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

    trajectories, horizon = 64, 8
    trajectory_spec = TrajectorySpec(
        trajectories=trajectories,
        horizon=horizon,
        state_dim=1,
        action_count=2,
        state_min=(-1.0,),
        state_max=(1.0,),
        reward_min=-2.0,
        reward_max=2.0,
    )
    policy = PolynomialPolicySpec(
        action_count=2,
        state_dim=1,
        degree=1,
        coefficients=((0.6, 0.1), (0.4, -0.1)),
    )
    states = tuple(
        tuple((((row % 3) - 1 + 0.1 * step) / (1 + 0.1 * step),) for step in range(horizon))
        for row in range(trajectories)
    )
    actions = tuple(
        tuple((row + step) % 2 for step in range(horizon)) for row in range(trajectories)
    )
    rewards = tuple(
        tuple(((-1.0) ** (row + step)) * 0.5 for step in range(horizon))
        for row in range(trajectories)
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states,
        actions,
        rewards,
        ((0.5,) * horizon,) * trajectories,
    )
    spec = PolynomialApproxOPESpec(
        trajectory_spec,
        policy,
        gamma=0.9,
        weight_clip=128.0,
        minimum_behavior_propensity=0.5,
    )
    parameters = _ckks_parameters(trajectories, horizon)
    contexts = generate_ope_contexts(spec, parameters)
    client = OPECKKSClient(
        CKKSClient.from_serialized(contexts.ckks.client_context, parameters=parameters), spec
    )
    server = OPECKKSServer(
        CKKSServer.from_serialized(contexts.ckks.server_context, parameters=parameters), spec
    )
    request, encrypt_receipt = client.encrypt_batch(batch)
    response, evaluate_receipt = server.evaluate(request)
    statistics, decrypt_receipt = client.decrypt_statistics(response, "clipped_wpdis")
    clear = spec.clear_oracle(batch, "clipped_wpdis")
    errors = tuple(
        abs(left - right)
        for left, right in zip(statistics.numerators, clear.numerators, strict=True)
    )
    payload = {
        "schema_version": "unseen-loop/modal-ckks-ope-canary-v1",
        "study_id": study_id,
        "execution": "Modal REAL FHE approximate arithmetic",
        "context_receipt": json.loads(contexts.ckks.receipt.to_json()),
        "computation_receipt": json.loads(contexts.computation.to_json()),
        "max_numerator_error": max(errors),
        "client_released_statistics": asdict(statistics),
        "clear_statistics": asdict(clear),
        "absolute_numerator_errors": list(errors),
        "transport": [
            json.loads(encrypt_receipt.to_json()),
            json.loads(evaluate_receipt.to_json()),
            json.loads(decrypt_receipt.to_json()),
        ],
    }
    artifact_path = _write_bundle(study_id, payload)
    return _canonical({"artifact_path": artifact_path, "payload": payload})


@app.local_entrypoint()
def run(prefix: str = "flagship-canary") -> str:
    """Dispatch all cryptographic canaries; local execution performs no FHE work."""
    results = {
        "shield": json.loads(shield_canary.remote(f"{prefix}-shield")),
        "exact_ope": json.loads(exact_ope_canary.remote(f"{prefix}-exact-ope")),
        "ckks_ope": json.loads(ckks_ope_canary.remote(f"{prefix}-ckks-ope")),
    }
    return _canonical(results)
