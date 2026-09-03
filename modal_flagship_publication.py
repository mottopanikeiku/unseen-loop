"""Build the browser publication from Modal-held canary evidence and clear references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

app = modal.App("unseen-loop-flagship-publication")
volume = modal.Volume.from_name("unseen-loop-flagship-evidence", create_if_missing=False)
root = Path("/flagship-evidence")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4")
    .add_local_python_source("unseen_loop")
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_canary(study_id: str) -> tuple[dict[str, Any], str]:
    path = root / "canaries" / study_id / "summary.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular canary summary for {study_id}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"canary summary for {study_id} is not an object")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _margin_values(value: Any) -> dict[str, float]:
    return {name: float(getattr(value, name)) for name in ("obstacle", "speed", "tilt", "battery")}


def _shield_publication(canary: dict[str, Any], canary_digest: str) -> dict[str, Any]:
    from unseen_loop.shield.environment import WarehouseEnvironment
    from unseen_loop.shield.shield import ShieldConfig, rollout_candidates, shield_step
    from unseen_loop.shield.study import (
        DEFAULT_ERROR_BUFFER,
        make_scenario,
        nominal_requested_policy,
    )
    from unseen_loop.shield.types import Action

    scenario = make_scenario("static_obstacle")
    environment = WarehouseEnvironment(scenario, seed=20260823)
    environment.reset(seed=20260823)
    config = ShieldConfig(error_buffer=DEFAULT_ERROR_BUFFER)
    decisions: list[dict[str, Any]] = []
    while not environment.done and len(decisions) < 12:
        step = len(decisions)
        current = environment.state
        requested = nominal_requested_policy(current, scenario)
        rollouts = rollout_candidates(current, scenario.dynamics)
        receipt = shield_step(
            current,
            requested,
            step=step,
            dynamics=scenario.dynamics,
            limits=scenario.safety,
            config=config,
        )
        result = environment.step(receipt.selected_action)
        candidates: list[dict[str, Any]] = []
        for certificate in receipt.candidates:
            candidates.append(
                {
                    "action": int(certificate.action),
                    "active_families": [family.value for family in certificate.active_families],
                    "certified": certificate.certified,
                    "minimum_buffered_margin": certificate.minimum_buffered_margin,
                    "failed_obligations": [
                        [horizon, family.value]
                        for horizon, family in certificate.failed_obligations
                    ],
                    "steps": [
                        {
                            "horizon": horizon.horizon,
                            "raw": _margin_values(horizon.raw),
                            "buffer": _margin_values(horizon.buffer),
                            "buffered": _margin_values(horizon.buffered),
                        }
                        for horizon in certificate.steps
                    ],
                }
            )
        candidates_by_action = {candidate.action: candidate for candidate in rollouts}
        decisions.append(
            {
                "step": step,
                "requested_action": int(receipt.requested_action),
                "selected_action": int(receipt.selected_action),
                "reason": receipt.reason,
                "emergency_fallback": receipt.emergency_fallback,
                "selected_certified": receipt.selected_certified,
                "receipt_digest": receipt.receipt_digest,
                "candidates": candidates,
                "visualization": {
                    "current_position": [current.x, current.y],
                    "candidate_paths": [
                        {
                            "action": int(action),
                            "positions": [
                                [state.x, state.y] for state in candidates_by_action[action].states
                            ],
                        }
                        for action in Action
                    ],
                    "executed_next_position": [result.state.x, result.state.y],
                },
            }
        )
    emergency = sum(bool(item["emergency_fallback"]) for item in decisions)
    retained = sum(
        not item["emergency_fallback"] and item["requested_action"] == item["selected_action"]
        for item in decisions
    )
    override = len(decisions) - retained - emergency
    scenario_payload = scenario.to_dict()
    scenario_payload.pop("initial_state", None)
    compile_receipt = canary.get("receipt")
    if not isinstance(compile_receipt, dict):
        raise RuntimeError("shield canary omitted its compilation receipt")
    real = canary.get("real")
    if (
        not isinstance(real, dict)
        or not isinstance(real.get("call"), dict)
        or real["call"].get("output_matches_clear") is not True
    ):
        raise RuntimeError("shield canary REAL FHE call did not match the exact clear tensor")
    simulation = canary.get("simulation")
    if (
        not isinstance(simulation, dict)
        or simulation.get("domain_points") != 15_625
        or simulation.get("matches") != 15_625
        or simulation.get("mismatches") != 0
        or simulation.get("clear_outputs_sha256") != simulation.get("simulated_outputs_sha256")
    ):
        raise RuntimeError("shield complete-domain Concrete simulation did not close exactly")
    return {
        "schema_version": "unseen-loop/shield-publication-v1",
        "state_features": ["x", "y", "vx", "vy", "battery", "tilt"],
        "actions": [
            {"id": int(action), "name": action.name, "vector": list(action.vector)}
            for action in Action
        ],
        "horizon": 2,
        "margin_families": ["obstacle", "speed", "tilt", "battery"],
        "scenario": {
            **scenario_payload,
            "scenario_sha256": _digest(scenario.to_dict()),
            "dynamics_sha256": _digest(scenario.dynamics.to_dict()),
        },
        "canary": {
            "mode": "REAL FHE",
            "backend": compile_receipt.get("backend"),
            "domain_points": compile_receipt.get("domain_points"),
            "output_shape": compile_receipt.get("output_shape"),
            "exact_complete_domain": True,
            "real_call": canary.get("real"),
            "receipt": compile_receipt,
            "source_sha256": canary_digest,
        },
        "run": {
            "run_id": "static-obstacle-clear-replay-20260823",
            "mode": "CLEAR_REFERENCE",
            "disclosure": "CLIENT_RELEASED_DERIVED_GEOMETRY",
            "decisions": decisions,
            "receipt_sha256": _digest([item["receipt_digest"] for item in decisions]),
        },
        "summary": {
            "total_steps": len(decisions),
            "requested_retained": retained,
            "override_to_certified": override,
            "emergency_brake": emergency,
        },
    }


def _ope_batch() -> tuple[Any, Any, Any]:
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
    return trajectory_spec, policy, batch


def _statistics_payload(statistics: dict[str, Any]) -> dict[str, Any]:
    numerators = [float(value) for value in statistics["numerators"]]
    denominators = [float(value) for value in statistics["denominators"]]
    counts = [int(value) for value in statistics["counts"]]
    if any(value <= 0 for value in denominators):
        estimate: float | None = None
    else:
        estimate = sum(left / right for left, right in zip(numerators, denominators, strict=True))
    return {
        "estimator": str(statistics["estimator"]),
        "numerators": numerators,
        "denominators": denominators,
        "counts": counts,
        "estimate": estimate,
    }


def _ope_publication(
    exact: dict[str, Any], exact_digest: str, ckks: dict[str, Any], ckks_digest: str
) -> dict[str, Any]:
    import numpy as np

    from unseen_loop.ope.estimators import (
        bootstrap_ope,
        cumulative_importance_weights,
        per_decision_effective_sample_size,
    )

    trajectory_spec, policy, batch = _ope_batch()
    target = policy.logged_action_probabilities(batch)
    bootstrap = bootstrap_ope(
        batch,
        target,
        estimator="wpdis",
        gamma=0.9,
        weight_clip=128.0,
        samples=2_000,
        seed=20260823,
    )
    raw_weights = cumulative_importance_weights(batch, target, weight_clip=None)
    encrypted_raw = ckks.get("client_released_statistics")
    clear_raw = ckks.get("clear_statistics")
    if not isinstance(encrypted_raw, dict) or not isinstance(clear_raw, dict):
        raise RuntimeError("CKKS canary omitted client-released aggregate statistics")
    encrypted = _statistics_payload(encrypted_raw)
    clear = _statistics_payload(clear_raw)
    context = ckks.get("context_receipt")
    computation = ckks.get("computation_receipt")
    if not isinstance(context, dict) or not isinstance(computation, dict):
        raise RuntimeError("CKKS canary omitted context or computation receipt")
    exact_receipt = exact.get("receipt")
    exact_call = exact.get("call")
    if not isinstance(exact_receipt, dict) or not isinstance(exact_call, dict):
        raise RuntimeError("exact OPE canary omitted backend evidence")
    return {
        "schema_version": "unseen-loop/ope-publication-v1",
        "batch": {
            "trajectory_spec": asdict(trajectory_spec),
            "private_fields": ["states", "actions", "rewards", "behavior_propensities"],
            "batch_sha256_disclosure": "WITHHELD",
        },
        "target_policy": {
            "degree": policy.degree,
            "action_count": policy.action_count,
            "state_dim": policy.state_dim,
            "coefficients_public": True,
            "policy_sha256": _digest(asdict(policy)),
        },
        "variant": {
            "variant_id": "polynomial-approx-ckks-h8-clip128",
            "estimator": encrypted["estimator"],
            "clip_threshold": 128.0,
            "mode": "REAL FHE (approximate arithmetic)",
            "disclosure": "CLIENT_RELEASED_STATISTICS",
            "statistics": encrypted,
            "clear_reference": clear,
            "absolute_numerator_errors": ckks.get("absolute_numerator_errors"),
            "max_numerator_error": ckks.get("max_numerator_error"),
            "matches_exact_clear": False,
            "receipt": {
                "context": context,
                "computation": computation,
                "source_sha256": ckks_digest,
            },
        },
        "exact_canary": {
            "mode": "REAL FHE",
            "shape": {"trajectories": 1, "horizon": 2, "state_dim": 1},
            "simulation_matches_real": exact.get("simulation_matches_real"),
            "integer_statistics": exact.get("client_released_integer_statistics"),
            "statistics": exact.get("client_released_statistics"),
            "receipt": exact_receipt,
            "call": exact_call,
            "source_sha256": exact_digest,
        },
        "uncertainty": {
            "mode": "CLEAR REFERENCE",
            "method": "trajectory_percentile_bootstrap",
            **asdict(bootstrap),
        },
        "diagnostics": {
            "per_horizon_ess": [
                float(value)
                for value in per_decision_effective_sample_size(
                    batch,
                    target,
                    weight_clip=128.0,
                )
            ],
            "support_failures": 0,
            "clipped_fraction": float(np.mean(raw_weights > 128.0)),
            "mode": "CLEAR REFERENCE / CLIENT-RELEASED AGGREGATE",
        },
    }


def _smoke_publication(run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    run_root = root / run_id
    index_path = run_root / "evidence-index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise RuntimeError("smoke evidence index is missing or not regular")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or index.get("schema_version") != (
        "unseen-loop/flagship-evidence-index-v1"
    ):
        raise RuntimeError("smoke evidence index schema is invalid")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("smoke evidence index omitted artifacts")
    analysis_entries = [
        entry
        for entry in artifacts.values()
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry["path"].startswith("analysis/")
    ]
    if len(analysis_entries) != 1:
        raise RuntimeError("smoke evidence index must identify one analysis artifact")
    analysis_entry = analysis_entries[0]
    analysis_path = run_root / analysis_entry["path"]
    if not analysis_path.is_file() or analysis_path.is_symlink():
        raise RuntimeError("smoke analysis artifact is missing or not regular")
    analysis_bytes = analysis_path.read_bytes()
    analysis_digest = hashlib.sha256(analysis_bytes).hexdigest()
    if analysis_entry.get("sha256") != analysis_digest:
        raise RuntimeError("smoke analysis artifact digest does not match its index")
    analysis = json.loads(analysis_bytes)
    if not isinstance(analysis, dict) or analysis.get("schema_version") != (
        "unseen-loop/flagship-analysis-v1"
    ):
        raise RuntimeError("smoke analysis schema is invalid")
    summary = analysis.get("evidence_summary")
    publication = analysis.get("publication")
    if not isinstance(summary, dict) or not isinstance(publication, dict):
        raise RuntimeError("smoke analysis omitted aggregate evidence")
    planned = index.get("planned_job_ids")
    status_counts = index.get("status_counts")
    if not isinstance(planned, list) or not isinstance(status_counts, dict):
        raise RuntimeError("smoke evidence index omitted denominator accounting")
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    payload = {
        "schema_version": "unseen-loop/flagship-smoke-publication-v1",
        "run_id": run_id,
        "evidence_index_sha256": index_digest,
        "analysis_sha256": analysis_digest,
        "planned_jobs": len(planned),
        "status_counts": status_counts,
        "gate_pass": publication.get("gate_pass"),
        "evidence_summary": summary,
        "claim": (
            "closed bounded execution with retained negative gates; "
            "not full-manifest release qualification"
        ),
    }
    sources = {
        f"{run_id}/evidence-index.json": index_digest,
        f"{run_id}/{analysis_entry['path']}": analysis_digest,
    }
    return payload, sources


def _positive_publication(study_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    study_root = root / "positive-pilots" / study_id
    summary_path = study_root / "summary.json"
    ledger_path = study_root / "checksums.sha256"
    if (
        not summary_path.is_file()
        or summary_path.is_symlink()
        or not ledger_path.is_file()
        or ledger_path.is_symlink()
    ):
        raise RuntimeError("positive recovery summary or checksum ledger is unavailable")
    summary_bytes = summary_path.read_bytes()
    summary_digest = hashlib.sha256(summary_bytes).hexdigest()
    ledger = ledger_path.read_text().splitlines()
    if f"{summary_digest}  summary.json" not in ledger:
        raise RuntimeError("positive recovery summary is not checksum-closed")
    summary = json.loads(summary_bytes)
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != "unseen-loop/positive-recovery-summary-v1"
        or summary.get("study_id") != study_id
        or summary.get("all_tracks_passed") is not True
        or summary.get("qualified_positive_result") is not True
    ):
        raise RuntimeError("positive recovery result did not pass every frozen gate")
    ledger_digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    return summary, {
        f"positive-pilots/{study_id}/summary.json": summary_digest,
        f"positive-pilots/{study_id}/checksums.sha256": ledger_digest,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=8_192,
    volumes={str(root): volume},
    timeout=30 * 60,
    retries=0,
)
def build_publication(
    publication_id: str,
    shield_canary_id: str,
    exact_ope_canary_id: str,
    ckks_ope_canary_id: str,
    smoke_run_id: str,
    positive_study_id: str,
) -> str:
    volume.reload()
    destination = root / "publications" / publication_id
    if destination.exists():
        raise RuntimeError("publication destination already exists")
    shield, shield_digest = _load_canary(shield_canary_id)
    exact, exact_digest = _load_canary(exact_ope_canary_id)
    ckks, ckks_digest = _load_canary(ckks_ope_canary_id)
    smoke, smoke_sources = _smoke_publication(smoke_run_id)
    positive, positive_sources = _positive_publication(positive_study_id)
    publication = {
        "schema_version": "unseen-loop/flagship-publication-v1",
        "release": {
            "release_id": publication_id,
            "label": "CipherShield-RL + private OPE qualified recovery release",
            "execution_site": "Modal",
            "source_canaries": {
                "shield": shield_canary_id,
                "exact_ope": exact_ope_canary_id,
                "ckks_ope": ckks_ope_canary_id,
            },
            "source_smoke_run": smoke_run_id,
            "source_positive_study": positive_study_id,
        },
        "shield": _shield_publication(shield, shield_digest),
        "ope": _ope_publication(exact, exact_digest, ckks, ckks_digest),
        "smoke": smoke,
        "positive_recovery": positive,
        "allowed_claims": [
            "real Concrete ciphertext execution for the declared bounded exact canaries",
            "real TenSEAL CKKS ciphertext execution under separately named "
            "approximate polynomial semantics",
            "client-side shield selection and OPE division",
            "clear recorded replay and trajectory bootstrap under their explicit trust labels",
            "three preregistered recovery tracks passed every frozen positive-result gate",
        ],
        "forbidden_claims": [
            "first predictive safety shield or first private OPE",
            "private training",
            "malicious-server integrity",
            "indefinite-horizon safety or policy optimality",
            "unbiased OPE without its support and identification assumptions",
        ],
    }
    payload = _canonical(publication)
    destination.mkdir(parents=True)
    publication_path = destination / "flagship-evidence.json"
    publication_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source_rows = {
        "flagship-evidence.json": digest,
        f"canaries/{shield_canary_id}/summary.json": shield_digest,
        f"canaries/{exact_ope_canary_id}/summary.json": exact_digest,
        f"canaries/{ckks_ope_canary_id}/summary.json": ckks_digest,
        **smoke_sources,
        **positive_sources,
    }
    (destination / "checksums.sha256").write_text(
        "".join(f"{value}  {path}\n" for path, value in sorted(source_rows.items()))
    )
    volume.commit()
    return json.dumps(
        {
            "artifact_path": str(publication_path),
            "sha256": digest,
            "bytes": len(payload),
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main(
    publication_id: str,
    shield_canary_id: str,
    exact_ope_canary_id: str,
    ckks_ope_canary_id: str,
    smoke_run_id: str,
    positive_study_id: str,
) -> str:
    return build_publication.remote(
        publication_id,
        shield_canary_id,
        exact_ope_canary_id,
        ckks_ope_canary_id,
        smoke_run_id,
        positive_study_id,
    )
