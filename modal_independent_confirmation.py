"""Durable Modal-only holdout confirmation for the positive recovery claims."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

import modal

APP_NAME = "unseen-loop-independent-confirmation"
VOLUME_NAME = "unseen-loop-flagship-evidence"
SCHEMA_VERSION = "unseen-loop/independent-confirmation-v1"
RESULT_SCHEMA = "unseen-loop/independent-confirmation-result-v1"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
volume_root = Path("/flagship-evidence")
result_root = volume_root / "confirmations"
work_root = volume_root / "confirmation-work"

core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4")
    .add_local_python_source("unseen_loop")
)
ckks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "tenseal==0.3.17")
    .add_local_python_source("unseen_loop")
)
fhe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==1.26.4",
        "concrete-python==2.10.0",
        "setuptools==75.3.0",
    )
    .add_local_python_source("unseen_loop")
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parse(config_bytes: bytes) -> dict[str, Any]:
    config = tomllib.loads(config_bytes.decode("utf-8"))
    expected = {
        "schema_version",
        "name",
        "positive_baseline",
        "positive_baseline_sha256",
        "seed_root",
        "execution_site",
        "ope",
        "ckks",
        "shield",
        "stop",
    }
    if (
        set(config) != expected
        or config.get("schema_version") != SCHEMA_VERSION
        or config.get("execution_site") != "Modal"
    ):
        raise ValueError("independent confirmation root contract is invalid")
    baseline_digest = config.get("positive_baseline_sha256")
    if not isinstance(baseline_digest, str) or len(baseline_digest) != 64:
        raise ValueError("positive baseline digest is invalid")
    return config


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"confirmation section {name} is invalid")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _seed(namespace: str, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _study_id(value: str) -> str:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in value
    ):
        raise ValueError("study_id is invalid")
    return value


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("p95 requires observations")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


@app.function(image=core_image, cpu=16.0, memory=32_768, timeout=3 * 60 * 60, retries=0)
def ope_holdout(config_bytes: bytes) -> str:
    import numpy as np

    from unseen_loop.flagship.executor_ope import _bootstrap_interval, _clear_payload

    config = _parse(config_bytes)
    spec = _section(config, "ope")
    trajectories = int(spec["trajectories"])
    horizon = int(spec["horizon"])
    overlap = float(spec["overlap"])
    weight_clip = float(spec["weight_clip"])
    batches = int(spec["independent_batches"])
    repetitions = int(spec["bootstrap_repetitions"])
    estimator = str(spec["estimator"])
    if estimator != "clipped_wpdis" or min(trajectories, horizon, batches, repetitions) < 1:
        raise ValueError("OPE holdout contract is invalid")
    errors: list[float] = []
    truths: list[float] = []
    covered = 0
    widths: list[float] = []
    minimum_ess_fraction = math.inf
    denominators_positive = True
    started = time.perf_counter_ns()
    namespace = f"{config['seed_root']}\0ope-holdout"
    for batch_index in range(batches):
        seed = _seed(namespace, batch_index)
        payload, _batch = _clear_payload(
            seed=seed,
            horizon=horizon,
            trajectories=trajectories,
            overlap=overlap,
            clip=weight_clip,
        )
        truth = _number(_mapping(payload["truth"], "truth").get("value"), "truth.value")
        estimate = _mapping(
            _mapping(payload["estimates"], "estimates").get("wpdis"),
            "estimates.wpdis",
        )
        errors.append(_number(estimate.get("value"), "estimate.value") - truth)
        truths.append(truth)
        bootstrap = _mapping(payload["_bootstrap_inputs"], "bootstrap inputs")
        interval = _bootstrap_interval(
            per_row=np.asarray(bootstrap["per_row"], dtype=np.float64),
            weights=np.asarray(bootstrap["weights"], dtype=np.float64),
            estimator=estimator,
            repetitions=repetitions,
            seed=seed ^ 0xC0FFEE,
            truth=truth,
        )
        covered += int(interval.get("covered_truth") is True)
        widths.append(
            _number(interval.get("upper"), "interval.upper")
            - _number(interval.get("lower"), "interval.lower")
        )
        diagnostics = _mapping(payload["diagnostics"], "diagnostics")
        minimum_ess_fraction = min(
            minimum_ess_fraction,
            _number(diagnostics.get("minimum_ess_fraction"), "minimum ESS fraction"),
        )
        denominators_positive &= diagnostics.get("positive_horizon_denominators") is True
    aggregate_bias = abs(float(np.mean(errors))) / max(abs(float(np.mean(truths))), 1e-12)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    coverage = covered / batches
    gates = {
        "aggregate_normalized_bias": aggregate_bias
        <= float(spec["maximum_aggregate_normalized_bias"]),
        "rmse": rmse <= float(spec["maximum_rmse"]),
        "coverage_lower": coverage >= float(spec["minimum_interval_coverage"]),
        "coverage_upper": coverage <= float(spec["maximum_interval_coverage"]),
        "minimum_ess_fraction": minimum_ess_fraction >= float(spec["minimum_ess_fraction"]),
        "positive_horizon_denominators": denominators_positive,
    }
    return _canonical(
        {
            "schema_version": "unseen-loop/independent-ope-holdout-v1",
            "execution": "Modal clear statistical holdout",
            "shape": {"trajectories": trajectories, "horizon": horizon, "batches": batches},
            "observed": {
                "aggregate_normalized_bias": aggregate_bias,
                "rmse": rmse,
                "covered_intervals": covered,
                "coverage": coverage,
                "median_interval_width": float(np.median(widths)),
                "minimum_ess_fraction": minimum_ess_fraction,
                "positive_horizon_denominators": denominators_positive,
            },
            "gates": gates,
            "all_gates_passed": all(gates.values()),
            "elapsed_ns": time.perf_counter_ns() - started,
            "private_rows_persisted": False,
        }
    ).decode()


def _confirmation_batch(spec: dict[str, Any], replica: int) -> tuple[Any, Any]:
    from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec

    trajectories = int(spec["trajectories"])
    horizon = int(spec["horizon"])
    state_dim = int(spec["state_dim"])
    action_count = int(spec["action_count"])
    behavior = float(spec["behavior_probability"])
    trajectory_spec = TrajectorySpec(
        trajectories=trajectories,
        horizon=horizon,
        state_dim=state_dim,
        action_count=action_count,
        state_min=(-1.0,) * state_dim,
        state_max=(1.0,) * state_dim,
        reward_min=-1.0,
        reward_max=1.0,
    )
    row = (1.0 / action_count,) + (0.0,) * state_dim
    policy = PolynomialPolicySpec(action_count, state_dim, 1, (row,) * action_count)
    states = tuple(
        tuple(
            tuple(
                ((trajectory * 3 + step * 5 + dimension * 7 + replica * 11) % 17 - 8) / 8.0
                for dimension in range(state_dim)
            )
            for step in range(horizon)
        )
        for trajectory in range(trajectories)
    )
    actions = tuple(
        tuple((trajectory + step + replica) % action_count for step in range(horizon))
        for trajectory in range(trajectories)
    )
    rewards = tuple(
        tuple(
            (((trajectory * 3 + step * 5 + replica * 7) % 17) - 8) / 8.0 for step in range(horizon)
        )
        for trajectory in range(trajectories)
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states,
        actions,
        rewards,
        ((behavior,) * horizon,) * trajectories,
    )
    return batch, policy


@app.function(image=ckks_image, cpu=8.0, memory=32_768, timeout=60 * 60, retries=0)
def ckks_replica(config_bytes: bytes, replica: int) -> str:
    from unseen_loop.crypto.ckks import CKKSClient, CKKSServer
    from unseen_loop.ope.ckks import (
        OPECKKSClient,
        OPECKKSServer,
        PolynomialApproxOPESpec,
        executable_ckks_parameters,
        generate_ope_contexts,
    )

    config = _parse(config_bytes)
    spec = _section(config, "ckks")
    repetitions = int(spec["independent_contexts"])
    if isinstance(replica, bool) or not isinstance(replica, int) or not 0 <= replica < repetitions:
        raise ValueError("CKKS replica index is invalid")
    batch, policy = _confirmation_batch(spec, replica)
    ope_spec = PolynomialApproxOPESpec(
        batch.spec,
        policy,
        gamma=1.0,
        weight_clip=float(spec["weight_clip"]),
        minimum_behavior_propensity=float(spec["behavior_probability"]),
    )
    parameters = executable_ckks_parameters(batch.spec.trajectories, batch.spec.horizon)
    contexts = generate_ope_contexts(ope_spec, parameters)
    client = OPECKKSClient(
        CKKSClient.from_serialized(contexts.ckks.client_context, parameters=parameters),
        ope_spec,
    )
    server = OPECKKSServer(
        CKKSServer.from_serialized(contexts.ckks.server_context, parameters=parameters),
        ope_spec,
    )
    request, encrypt_receipt = client.encrypt_batch(batch)
    response, evaluate_receipt = server.evaluate(request)
    encrypted, decrypt_receipt = client.decrypt_statistics(response, "clipped_wpdis")
    clear = ope_spec.clear_oracle(batch, "clipped_wpdis")
    if encrypted.estimate is None or clear.estimate is None:
        raise RuntimeError("CKKS confirmation estimate is undefined")
    numerator_errors = [
        abs(left - right)
        for left, right in zip(encrypted.numerators, clear.numerators, strict=True)
    ]
    denominator_errors = [
        abs(left - right)
        for left, right in zip(encrypted.denominators, clear.denominators, strict=True)
    ]
    return _canonical(
        {
            "schema_version": "unseen-loop/independent-ckks-replica-v1",
            "replica": replica,
            "execution": "Modal REAL FHE approximate arithmetic",
            "context_sha256": contexts.ckks.receipt.server_context_sha256,
            "security_level": contexts.ckks.receipt.effective_security_level,
            "server_context_is_private": contexts.ckks.receipt.server_context_is_private,
            "estimate_error": abs(encrypted.estimate - clear.estimate),
            "maximum_horizon_numerator_error": max(numerator_errors),
            "maximum_horizon_denominator_error": max(denominator_errors),
            "positive_denominators": all(value > 0 for value in encrypted.denominators),
            "transport": {
                "request_bytes": encrypt_receipt.output_bytes,
                "server_evaluate_ns": evaluate_receipt.elapsed_ns,
                "response_bytes": evaluate_receipt.output_bytes,
                "decrypt_ns": decrypt_receipt.elapsed_ns,
            },
            "private_rows_persisted": False,
        }
    ).decode()


@app.function(
    image=fhe_image,
    cpu=16.0,
    memory=32_768,
    volumes={str(volume_root): volume},
    timeout=60 * 60,
    retries=0,
)
def compile_shield(config_bytes: bytes, study_id: str) -> str:
    from unseen_loop.shield.fhe import ShieldIntegerSpec
    from unseen_loop.shield.fhe import compile_shield as compile_program

    config = _parse(config_bytes)
    _study_id(study_id)
    spec = _section(config, "shield")
    destination = work_root / study_id / "shield"
    volume.reload()
    if destination.exists():
        raise RuntimeError("confirmation shield cache already exists")
    destination.mkdir(parents=True)
    compiled = compile_program(
        ShieldIntegerSpec(),
        destination,
        global_p_error=float(spec["global_p_error"]),
    )
    volume.commit()
    return _canonical(compiled.receipt.to_dict()).decode()


@app.function(
    image=fhe_image,
    cpu=8.0,
    memory=16_384,
    volumes={str(volume_root): volume},
    max_containers=10,
    timeout=30 * 60,
    retries=0,
)
def shield_state(config_bytes: bytes, study_id: str, state_index: int) -> str:
    import numpy as np

    from unseen_loop.flagship.executor_shield_fhe import _valid_state
    from unseen_loop.shield.certificate import ErrorBuffer
    from unseen_loop.shield.fhe import (
        MARGIN_SHAPE,
        ShieldFHEClient,
        ShieldFHEServer,
        ShieldIntegerSpec,
        clear_margin_tensor,
    )
    from unseen_loop.shield.types import Action

    config = _parse(config_bytes)
    _study_id(study_id)
    spec = _section(config, "shield")
    states = int(spec["quantized_states"])
    calls = int(spec["independent_calls_per_state"])
    quorum = int(spec["minimum_completed_calls_per_state"])
    if not 0 <= state_index < states or calls != 3 or quorum != 2:
        raise ValueError("shield confirmation state contract is invalid")
    categories = ("occupancy", "extrema", "threshold", "tie", "canary")
    category = categories[state_index % len(categories)]
    offset = int(spec["state_index_offset"])
    quantized = _valid_state(category, offset + state_index)
    requested = Action(state_index % len(Action))
    integer_spec = ShieldIntegerSpec()
    clear = clear_margin_tensor(integer_spec, quantized)
    volume.reload()
    cache = work_root / study_id / "shield"
    observed_actions: list[Action] = []
    observed_certifications: list[bool] = []
    clear_action: Action | None = None
    clear_certification: bool | None = None
    completed = 0
    margin_matches = 0
    failures: dict[str, int] = {}
    for _replica in range(calls):
        try:
            client = ShieldFHEClient.from_path(cache / "shield-client-specs.bin", integer_spec)
            server = ShieldFHEServer(cache / "shield-server.zip")
            _keygen_ns, evaluation_keys = client.generate_keys()
            request = client.encrypt(quantized)
            response = server.evaluate(request, evaluation_keys)
            decrypted = client.decrypt_margin_tensor(response)
            encrypted_selection = client.select_action(
                decrypted,
                requested,
                error_buffer=ErrorBuffer(),
            )
            reference_selection = client.select_action(
                clear,
                requested,
                error_buffer=ErrorBuffer(),
            )
        except (RuntimeError, TypeError, ValueError) as error:
            code = type(error).__name__.lower()
            failures[code] = failures.get(code, 0) + 1
            continue
        completed += 1
        margin_matches += int(np.count_nonzero(decrypted == clear))
        observed_actions.append(encrypted_selection.action)
        observed_certifications.append(encrypted_selection.selected_certified)
        clear_action = reference_selection.action
        clear_certification = reference_selection.selected_certified
    consensus_action: Action | None = None
    consensus_certification: bool | None = None
    if completed >= quorum:
        consensus_action = min(
            set(observed_actions),
            key=lambda action: (-observed_actions.count(action), int(action)),
        )
        consensus_certification = sum(observed_certifications) * 2 >= completed
    state_id = hashlib.sha256(f"{config['seed_root']}\0shield\0{state_index}".encode()).hexdigest()
    return _canonical(
        {
            "schema_version": "unseen-loop/independent-shield-state-v1",
            "state_id": state_id,
            "attempted_calls": calls,
            "completed_calls": completed,
            "quorum": consensus_action is not None,
            "consensus_action_matches": consensus_action == clear_action
            if consensus_action is not None
            else False,
            "consensus_certification_matches": consensus_certification == clear_certification
            if consensus_certification is not None
            else False,
            "consensus_false_safe": bool(
                consensus_certification is True and clear_certification is False
            ),
            "individual_action_matches": sum(action == clear_action for action in observed_actions),
            "individual_certification_matches": sum(
                certification == clear_certification for certification in observed_certifications
            ),
            "exact_margin_matches": margin_matches,
            "decoded_margins": completed * int(np.prod(MARGIN_SHAPE)),
            "failures": failures,
            "private_state_persisted": False,
        }
    ).decode()


def _exception_code(value: object) -> str | None:
    return type(value).__name__.lower() if isinstance(value, BaseException) else None


@app.function(
    image=core_image,
    cpu=4.0,
    memory=16_384,
    volumes={str(volume_root): volume},
    timeout=4 * 60 * 60,
    retries=0,
)
def orchestrate(config_bytes: bytes, study_id: str) -> str:
    config = _parse(config_bytes)
    _study_id(study_id)
    destination = result_root / study_id
    volume.reload()
    if destination.exists():
        raise RuntimeError("independent confirmation destination already exists")
    ope_call = ope_holdout.spawn(config_bytes)
    compile_call = compile_shield.spawn(config_bytes, study_id)
    ckks_count = int(_section(config, "ckks")["independent_contexts"])
    ckks_raw = ckks_replica.starmap(
        [(config_bytes, index) for index in range(ckks_count)],
        return_exceptions=True,
    )
    compile_receipt = json.loads(compile_call.get())
    state_count = int(_section(config, "shield")["quantized_states"])
    shield_raw = shield_state.starmap(
        [(config_bytes, study_id, index) for index in range(state_count)],
        return_exceptions=True,
    )
    ope = json.loads(ope_call.get())

    ckks_results = [json.loads(value) for value in ckks_raw if isinstance(value, str)]
    ckks_failures = [code for value in ckks_raw if (code := _exception_code(value)) is not None]
    ckks_spec = _section(config, "ckks")
    estimate_errors = [float(row["estimate_error"]) for row in ckks_results]
    numerator_errors = [float(row["maximum_horizon_numerator_error"]) for row in ckks_results]
    contexts = {str(row["context_sha256"]) for row in ckks_results}
    ckks_success_fraction = len(ckks_results) / ckks_count
    ckks_gates = {
        "success_fraction": ckks_success_fraction >= float(ckks_spec["minimum_success_fraction"]),
        "independent_contexts": len(contexts) == len(ckks_results) == ckks_count,
        "maximum_estimate_error": bool(estimate_errors)
        and max(estimate_errors) <= float(ckks_spec["maximum_estimate_error"]),
        "p95_estimate_error": bool(estimate_errors)
        and _p95(estimate_errors) <= float(ckks_spec["maximum_p95_estimate_error"]),
        "maximum_horizon_numerator_error": bool(numerator_errors)
        and max(numerator_errors) <= float(ckks_spec["maximum_horizon_numerator_error"]),
        "security_level": all(
            row["security_level"] == ckks_spec["required_security_level"] for row in ckks_results
        ),
        "public_server_contexts": all(
            row["server_context_is_private"] is False for row in ckks_results
        ),
        "positive_denominators": all(row["positive_denominators"] is True for row in ckks_results),
    }
    ckks = {
        "schema_version": "unseen-loop/independent-ckks-confirmation-v1",
        "attempted_contexts": ckks_count,
        "successful_contexts": len(ckks_results),
        "failure_codes": ckks_failures,
        "observed": {
            "success_fraction": ckks_success_fraction,
            "maximum_estimate_error": max(estimate_errors) if estimate_errors else None,
            "median_estimate_error": sorted(estimate_errors)[len(estimate_errors) // 2]
            if estimate_errors
            else None,
            "p95_estimate_error": _p95(estimate_errors) if estimate_errors else None,
            "maximum_horizon_numerator_error": max(numerator_errors) if numerator_errors else None,
            "maximum_horizon_denominator_error": max(
                (float(row["maximum_horizon_denominator_error"]) for row in ckks_results),
                default=None,
            ),
        },
        "gates": ckks_gates,
        "all_gates_passed": all(ckks_gates.values()),
        "replicas": ckks_results,
    }

    shield_results = [json.loads(value) for value in shield_raw if isinstance(value, str)]
    shield_failures = [code for value in shield_raw if (code := _exception_code(value)) is not None]
    shield_spec = _section(config, "shield")
    attempted_calls = state_count * int(shield_spec["independent_calls_per_state"])
    completed_calls = sum(int(row["completed_calls"]) for row in shield_results)
    quorum_states = sum(int(row["quorum"] is True) for row in shield_results)
    consensus_action_matches = sum(
        int(row["consensus_action_matches"] is True) for row in shield_results
    )
    consensus_certification_matches = sum(
        int(row["consensus_certification_matches"] is True) for row in shield_results
    )
    false_safe = sum(int(row["consensus_false_safe"] is True) for row in shield_results)
    completion_fraction = completed_calls / attempted_calls
    quorum_fraction = quorum_states / state_count
    action_agreement = consensus_action_matches / quorum_states if quorum_states else 0.0
    certification_agreement = (
        consensus_certification_matches / quorum_states if quorum_states else 0.0
    )
    shield_gates = {
        "state_worker_completion": len(shield_results) == state_count,
        "call_completion_fraction": completion_fraction
        >= float(shield_spec["minimum_call_completion_fraction"]),
        "quorum_state_fraction": quorum_fraction
        >= float(shield_spec["minimum_quorum_state_fraction"]),
        "consensus_action_agreement": action_agreement
        >= float(shield_spec["minimum_consensus_action_agreement"]),
        "consensus_certification_agreement": certification_agreement
        >= float(shield_spec["minimum_consensus_certification_agreement"]),
        "consensus_false_safe": false_safe
        <= int(shield_spec["maximum_consensus_false_safe_decisions"]),
    }
    shield = {
        "schema_version": "unseen-loop/independent-shield-confirmation-v1",
        "attempted_states": state_count,
        "completed_state_workers": len(shield_results),
        "worker_failure_codes": shield_failures,
        "attempted_calls": attempted_calls,
        "completed_calls": completed_calls,
        "observed": {
            "call_completion_fraction": completion_fraction,
            "quorum_states": quorum_states,
            "quorum_state_fraction": quorum_fraction,
            "consensus_action_matches": consensus_action_matches,
            "consensus_action_agreement": action_agreement,
            "consensus_certification_matches": consensus_certification_matches,
            "consensus_certification_agreement": certification_agreement,
            "consensus_false_safe_decisions": false_safe,
            "individual_action_matches": sum(
                int(row["individual_action_matches"]) for row in shield_results
            ),
            "individual_certification_matches": sum(
                int(row["individual_certification_matches"]) for row in shield_results
            ),
            "exact_margin_matches": sum(int(row["exact_margin_matches"]) for row in shield_results),
            "decoded_margins": sum(int(row["decoded_margins"]) for row in shield_results),
        },
        "gates": shield_gates,
        "all_gates_passed": all(shield_gates.values()),
        "compile_receipt": compile_receipt,
        "states": shield_results,
    }

    all_tracks_passed = (
        ope.get("all_gates_passed") is True
        and ckks["all_gates_passed"] is True
        and shield["all_gates_passed"] is True
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "study_id": study_id,
        "positive_baseline": config["positive_baseline"],
        "positive_baseline_sha256": config["positive_baseline_sha256"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "tracks": {"ope": ope, "ckks": ckks, "shield": shield},
        "all_tracks_passed": all_tracks_passed,
        "independently_confirmed_positive_result": all_tracks_passed,
        "scale_up_allowed": all_tracks_passed
        and bool(_section(config, "stop")["launch_larger_matrix_only_if_all_gates_pass"]),
        "claim_scope": (
            "independent holdout confirmation of high-overlap OPE, approximate CKKS agreement, "
            "and three-replica shield consensus; no exact-margin or production claim"
        ),
    }
    destination.mkdir(parents=True)
    files = {
        "config.toml": config_bytes,
        "ope.json": _canonical(ope),
        "ckks.json": _canonical(ckks),
        "shield.json": _canonical(shield),
        "summary.json": _canonical(summary),
    }
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    ledger = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    )
    (destination / "checksums.sha256").write_text(ledger)
    volume.commit()
    cache_root = work_root / study_id
    if cache_root.exists() and not cache_root.is_symlink():
        shutil.rmtree(cache_root)
        volume.commit()
    return json.dumps(
        {
            "artifact_path": str(destination / "summary.json"),
            "summary_sha256": hashlib.sha256(files["summary.json"]).hexdigest(),
            "independently_confirmed_positive_result": all_tracks_passed,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def run(config: str, study_id: str) -> str:
    config_bytes = Path(config).read_bytes()
    _parse(config_bytes)
    return orchestrate.remote(config_bytes, study_id)
