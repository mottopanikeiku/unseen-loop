"""Preregistered Modal-only recovery pilot for qualified positive results."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

import modal

app = modal.App("unseen-loop-positive-pilot")
volume = modal.Volume.from_name("unseen-loop-flagship-evidence", create_if_missing=False)
root = Path("/flagship-evidence/positive-pilots")

core_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4")
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
ckks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==1.26.4", "tenseal==0.3.17")
    .add_local_python_source("unseen_loop")
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parse(config_bytes: bytes) -> dict[str, Any]:
    raw = tomllib.loads(config_bytes.decode("utf-8"))
    if raw.get("schema_version") != "unseen-loop/positive-pilot-v1":
        raise ValueError("positive pilot schema is invalid")
    expected = {
        "schema_version",
        "name",
        "baseline_run",
        "seed_root",
        "execution_site",
        "shield",
        "ope",
        "integration_ckks",
        "stop",
    }
    if set(raw) != expected or raw.get("execution_site") != "Modal":
        raise ValueError("positive pilot has missing or unknown root fields")
    return raw


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"positive pilot section {name} is invalid")
    return section


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} is not boolean")
    return value


def _seed(namespace: str, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@app.function(image=core_image, cpu=16.0, memory=32_768, timeout=3 * 60 * 60, retries=1)
def ope_calibration(config_bytes: bytes) -> str:
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
        raise ValueError("OPE pilot shape or estimator is invalid")
    errors: list[float] = []
    truths: list[float] = []
    covered = 0
    widths: list[float] = []
    minimum_ess_fraction = math.inf
    all_denominators_positive = True
    started = time.perf_counter_ns()
    for batch_index in range(batches):
        seed = _seed(str(config["seed_root"]), batch_index)
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
        error = _number(estimate.get("value"), "estimate.value") - truth
        bootstrap_inputs = _mapping(payload["_bootstrap_inputs"], "bootstrap inputs")
        interval = _bootstrap_interval(
            per_row=np.asarray(bootstrap_inputs["per_row"], dtype=np.float64),
            weights=np.asarray(bootstrap_inputs["weights"], dtype=np.float64),
            estimator=estimator,
            repetitions=repetitions,
            seed=seed ^ 0x5EED5EED,
            truth=truth,
        )
        errors.append(error)
        truths.append(truth)
        covered += int(_boolean(interval.get("covered_truth"), "covered_truth"))
        widths.append(
            _number(interval.get("upper"), "interval.upper")
            - _number(interval.get("lower"), "interval.lower")
        )
        diagnostics = _mapping(payload["diagnostics"], "diagnostics")
        minimum_ess_fraction = min(
            minimum_ess_fraction,
            _number(diagnostics.get("minimum_ess_fraction"), "minimum ESS fraction"),
        )
        all_denominators_positive &= _boolean(
            diagnostics.get("positive_horizon_denominators"),
            "positive horizon denominators",
        )
    mean_truth = float(np.mean(truths))
    aggregate_bias = abs(float(np.mean(errors))) / max(abs(mean_truth), 1e-12)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    coverage = covered / batches
    gates = {
        "aggregate_normalized_bias": aggregate_bias
        <= float(spec["maximum_aggregate_normalized_bias"]),
        "rmse": rmse <= float(spec["maximum_rmse"]),
        "interval_coverage_lower": coverage >= float(spec["minimum_interval_coverage"]),
        "interval_coverage_upper": coverage <= float(spec["maximum_interval_coverage"]),
        "positive_horizon_denominators": all_denominators_positive,
    }
    result = {
        "schema_version": "unseen-loop/positive-ope-result-v1",
        "execution": "Modal clear statistical calibration",
        "shape": {"trajectories": trajectories, "horizon": horizon, "batches": batches},
        "configuration": {
            "overlap": overlap,
            "weight_clip": weight_clip,
            "estimator": estimator,
            "bootstrap_repetitions": repetitions,
        },
        "observed": {
            "aggregate_normalized_bias": aggregate_bias,
            "rmse": rmse,
            "covered_intervals": covered,
            "coverage": coverage,
            "median_interval_width": float(np.median(widths)),
            "minimum_ess_fraction": minimum_ess_fraction,
            "all_horizon_denominators_positive": all_denominators_positive,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "elapsed_ns": time.perf_counter_ns() - started,
        "private_rows_persisted": False,
    }
    return _canonical(result).decode()


@app.function(image=ckks_image, cpu=8.0, memory=32_768, timeout=60 * 60, retries=1)
def integration_ckks(config_bytes: bytes) -> str:
    from unseen_loop.flagship.executor_integration import _ckks_receipt, _ckks_spec, _run_ckks
    from unseen_loop.ope.types import TrajectoryBatch, TrajectorySpec

    config = _parse(config_bytes)
    spec = _section(config, "integration_ckks")
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
    states = tuple(
        tuple(
            tuple(
                ((trajectory + 2 * step + dimension) % 9 - 4) / 4.0
                for dimension in range(state_dim)
            )
            for step in range(horizon)
        )
        for trajectory in range(trajectories)
    )
    actions = tuple(
        tuple((trajectory + step) % action_count for step in range(horizon))
        for trajectory in range(trajectories)
    )
    rewards = tuple(
        tuple((((trajectory * 3 + step * 5) % 17) - 8) / 8.0 for step in range(horizon))
        for trajectory in range(trajectories)
    )
    batch = TrajectoryBatch(
        trajectory_spec,
        states,
        actions,
        rewards,
        ((behavior,) * horizon,) * trajectories,
    )
    ckks_spec = _ckks_spec(batch)
    if not math.isclose(ckks_spec.weight_clip, float(spec["weight_clip"])):
        raise ValueError("integration CKKS clip differs from the frozen pilot")
    receipt = _ckks_receipt(ckks_spec)
    encrypted, transport = _run_ckks(ckks_spec, batch, receipt)
    clear = ckks_spec.clear_oracle(batch, "clipped_wpdis")
    if encrypted.estimate is None or clear.estimate is None:
        raise RuntimeError("integration CKKS pilot returned an undefined estimate")
    estimate_error = abs(encrypted.estimate - clear.estimate)
    numerator_errors = [
        abs(left - right)
        for left, right in zip(encrypted.numerators, clear.numerators, strict=True)
    ]
    denominator_errors = [
        abs(left - right)
        for left, right in zip(encrypted.denominators, clear.denominators, strict=True)
    ]
    gates = {
        "estimate_error": estimate_error <= float(spec["maximum_absolute_estimate_error"]),
        "horizon_numerator_error": max(numerator_errors)
        <= float(spec["maximum_absolute_horizon_numerator_error"]),
        "security_level": receipt.required_security_level == spec["required_security_level"],
        "positive_denominators": all(value > 0 for value in encrypted.denominators),
    }
    result = {
        "schema_version": "unseen-loop/positive-integration-ckks-result-v1",
        "execution": "Modal REAL FHE approximate arithmetic",
        "identifier": receipt.identifier,
        "shape": {
            "trajectories": trajectories,
            "horizon": horizon,
            "state_dim": state_dim,
            "action_count": action_count,
        },
        "weight_clip": ckks_spec.weight_clip,
        "observed": {
            "encrypted_estimate": encrypted.estimate,
            "clear_approximation_estimate": clear.estimate,
            "absolute_estimate_error": estimate_error,
            "maximum_horizon_numerator_error": max(numerator_errors),
            "maximum_horizon_denominator_error": max(denominator_errors),
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "receipt": json.loads(receipt.to_json()),
        "transport": transport,
        "private_rows_persisted": False,
    }
    return _canonical(result).decode()


@app.function(image=fhe_image, cpu=16.0, memory=32_768, timeout=3 * 60 * 60, retries=1)
def shield_reliability(config_bytes: bytes) -> str:
    import numpy as np

    from unseen_loop.flagship.executor_shield_fhe import _valid_state
    from unseen_loop.shield.certificate import ErrorBuffer
    from unseen_loop.shield.fhe import (
        MARGIN_SHAPE,
        ShieldFHEClient,
        ShieldFHEServer,
        ShieldIntegerSpec,
        clear_margin_tensor,
        compile_shield,
    )
    from unseen_loop.shield.types import Action

    config = _parse(config_bytes)
    spec = _section(config, "shield")
    state_count = int(spec["quantized_states"])
    calls_per_state = int(spec["independent_calls_per_state"])
    categories = ("occupancy", "extrema", "threshold", "tie", "canary")
    if state_count != len(categories) or calls_per_state != 1:
        raise ValueError("shield recovery pilot fixes five independent calls")
    completed = 0
    action_matches = 0
    selected_certification_matches = 0
    margin_matches = 0
    margin_count = 0
    failures: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="positive-shield-") as temporary:
        integer_spec = ShieldIntegerSpec()
        compiled = compile_shield(
            integer_spec,
            temporary,
            global_p_error=float(spec["global_p_error"]),
        )
        for index, category in enumerate(categories):
            quantized = _valid_state(category, 1000 + index)
            requested = Action(index)
            try:
                client = ShieldFHEClient.from_path(compiled.client_specs_path, integer_spec)
                server = ShieldFHEServer(compiled.server_path)
                _keygen_ns, evaluation_keys = client.generate_keys()
                request = client.encrypt(quantized)
                response = server.evaluate(request, evaluation_keys)
                decrypted = client.decrypt_margin_tensor(response)
                clear = clear_margin_tensor(integer_spec, quantized)
                encrypted_selection = client.select_action(
                    decrypted,
                    requested,
                    error_buffer=ErrorBuffer(),
                )
                clear_selection = client.select_action(
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
            margin_count += int(np.prod(MARGIN_SHAPE))
            action_matches += int(encrypted_selection.action == clear_selection.action)
            selected_certification_matches += int(
                encrypted_selection.selected_certified == clear_selection.selected_certified
            )
    completion_fraction = completed / (state_count * calls_per_state)
    action_agreement = action_matches / completed if completed else 0.0
    certification_agreement = selected_certification_matches / completed if completed else 0.0
    gates = {
        "completed_call_fraction": completion_fraction
        >= float(spec["minimum_completed_call_fraction"]),
        "client_action_agreement": action_agreement
        >= float(spec["minimum_client_action_agreement"]),
        "selected_certification_agreement": certification_agreement == 1.0,
    }
    result = {
        "schema_version": "unseen-loop/positive-shield-result-v1",
        "execution": "Modal REAL FHE",
        "claim": spec["claim"],
        "attempted_calls": state_count * calls_per_state,
        "completed_calls": completed,
        "observed": {
            "completed_call_fraction": completion_fraction,
            "client_action_matches": action_matches,
            "client_action_agreement": action_agreement,
            "selected_certification_matches": selected_certification_matches,
            "selected_certification_agreement": certification_agreement,
            "exact_margin_matches": margin_matches,
            "decoded_margins": margin_count,
        },
        "failures": failures,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "compile_receipt": compiled.receipt.to_dict(),
        "private_states_persisted": False,
    }
    return _canonical(result).decode()


@app.function(
    image=core_image,
    cpu=2.0,
    memory=4_096,
    volumes={str(root.parent): volume},
    timeout=10 * 60,
    retries=1,
)
def finalize_pilot(
    config_bytes: bytes,
    study_id: str,
    ope_json: str,
    ckks_json: str,
    shield_json: str,
) -> str:
    config = _parse(config_bytes)
    if not study_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in study_id
    ):
        raise ValueError("study_id is invalid")
    results = {
        "ope": json.loads(ope_json),
        "integration_ckks": json.loads(ckks_json),
        "shield": json.loads(shield_json),
    }
    all_tracks_passed = all(result.get("all_gates_passed") is True for result in results.values())
    require_all = bool(_section(config, "stop")["require_all_tracks"])
    summary = {
        "schema_version": "unseen-loop/positive-pilot-summary-v1",
        "study_id": study_id,
        "baseline_run": config["baseline_run"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "results": results,
        "all_tracks_passed": all_tracks_passed,
        "qualified_positive_result": all_tracks_passed and require_all,
        "scale_up_allowed": all_tracks_passed
        and bool(_section(config, "stop")["launch_larger_matrix_only_if_all_gates_pass"]),
        "claim_scope": (
            "post-baseline preregistered recovery pilot; positive only if every frozen gate passes"
        ),
    }
    destination = root / study_id
    volume.reload()
    if destination.exists():
        raise RuntimeError("positive pilot destination already exists")
    destination.mkdir(parents=True)
    files = {
        "config.toml": config_bytes,
        "ope.json": _canonical(results["ope"]),
        "integration-ckks.json": _canonical(results["integration_ckks"]),
        "shield.json": _canonical(results["shield"]),
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
    return json.dumps(
        {
            "artifact_path": str(destination / "summary.json"),
            "summary_sha256": hashlib.sha256(files["summary.json"]).hexdigest(),
            "qualified_positive_result": summary["qualified_positive_result"],
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def run(config: str, study_id: str) -> str:
    config_bytes = Path(config).read_bytes()
    _parse(config_bytes)
    ope_call = ope_calibration.spawn(config_bytes)
    ckks_call = integration_ckks.spawn(config_bytes)
    shield_call = shield_reliability.spawn(config_bytes)
    return finalize_pilot.remote(
        config_bytes,
        study_id,
        ope_call.get(),
        ckks_call.get(),
        shield_call.get(),
    )
