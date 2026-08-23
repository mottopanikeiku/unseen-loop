"""End-to-end training, distillation, certification, benchmarking, and artifact capture."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from unseen_loop import __version__
from unseen_loop.artifacts import ArtifactLedger, RunProvenance, dataclass_dict
from unseen_loop.certificate import BoxCertificate, certify_actions, certify_quantized_box
from unseen_loop.fhe_backend import RoundTripMeasurement, compile_policy
from unseen_loop.search import SearchConfig, SearchRecord, pareto_front, search_policies
from unseen_loop.teacher import (
    MLPTeacher,
    TeacherCheckpoint,
    collect_trajectories,
    rollout,
    train_cem_teacher,
)

Backend = Literal["clear", "simulate", "fhe"]


@dataclass(frozen=True)
class SeedPlan:
    training: int
    distillation: tuple[int, ...]
    refinement: tuple[int, ...]
    evaluation: tuple[int, ...]
    real_fhe: tuple[int, ...]
    namespace: str

    @classmethod
    def derive(cls, root: str, env_id: str, *, full: bool) -> SeedPlan:
        def seeds(purpose: str, count: int) -> tuple[int, ...]:
            result: list[int] = []
            for index in range(count):
                digest = hashlib.sha256(
                    f"unseen-loop/v1|{root}|{env_id}|{purpose}|{index}".encode()
                ).digest()
                result.append(int.from_bytes(digest[:4], "little") & 0x7FFF_FFFF)
            return tuple(result)

        return cls(
            training=seeds("training", 1)[0],
            distillation=seeds("distillation", 20 if full else 4),
            refinement=seeds("refinement", 10 if full else 2),
            evaluation=seeds("evaluation", 100 if full else 8),
            real_fhe=seeds("real-fhe", 5 if full else 2),
            namespace=f"{root}:{env_id}:{'full' if full else 'quick'}",
        )


@dataclass(frozen=True)
class ResearchPreset:
    full: bool
    teacher_iterations: int
    teacher_population: int
    episodes_per_candidate: int
    hidden_size: int
    search: SearchConfig

    @classmethod
    def quick(cls) -> ResearchPreset:
        return cls(
            full=False,
            teacher_iterations=8,
            teacher_population=32,
            episodes_per_candidate=1,
            hidden_size=12,
            search=SearchConfig(
                degrees=(1, 2),
                input_bits=(4, 5),
                coefficient_bits=(8, 10),
                ridge_values=(1e-3,),
                refinement_rounds=1,
                calibration_padding=0.75,
            ),
        )

    @classmethod
    def release(cls) -> ResearchPreset:
        return cls(
            full=True,
            teacher_iterations=40,
            teacher_population=128,
            episodes_per_candidate=3,
            hidden_size=32,
            search=SearchConfig(
                degrees=(1, 2),
                input_bits=(3, 4, 5, 6),
                coefficient_bits=(3, 4, 6, 8, 10),
                ridge_values=(1e-4, 1e-3, 1e-2),
                refinement_rounds=3,
                calibration_padding=0.5,
            ),
        )


@dataclass(frozen=True)
class ExperimentSummary:
    run_id: str
    env_id: str
    backend: str
    teacher_digest: str
    teacher_return_mean: float
    candidates: int
    frontier_candidates: int
    champion_policy_digest: str
    champion_name: str
    champion_return_mean: float
    champion_return_delta: float
    teacher_agreement: float
    certified_coverage: float
    constraint_cost: float
    estimated_output_bits: int
    encrypted_multiplications: int
    box_certificate_coverage: float | None
    box_certificate_points: int | None
    simulated_matches_integer: bool | None
    real_fhe_calls: int
    real_fhe_all_match: bool | None
    label: str
    privacy_evidence: bool
    schema_version: str = "unseen-loop/experiment-summary-v1"


def _teacher_return(teacher: MLPTeacher, seeds: tuple[int, ...]) -> tuple[float, float]:
    returns = np.asarray(
        [rollout(teacher.checkpoint.env_id, teacher, seed=seed)[0].total_return for seed in seeds]
    )
    return float(np.mean(returns)), float(np.std(returns))


def _select_champion(
    frontier: tuple[SearchRecord, ...], teacher_return_mean: float
) -> SearchRecord:
    if not frontier:
        raise RuntimeError("policy search produced no Pareto candidates")
    tolerance = max(5.0, 0.05 * abs(teacher_return_mean))
    viable = [
        record
        for record in frontier
        if record.metrics.return_mean >= teacher_return_mean - tolerance
        and record.metrics.certified_coverage >= 0.95
    ]
    pool = viable or list(frontier)
    return min(
        pool,
        key=lambda record: (
            -record.metrics.certified_coverage,
            -record.metrics.return_mean,
            record.metrics.estimated_bit_width,
            record.metrics.encrypted_multiplications,
            record.policy.spec.digest,
        ),
    )


def _calibration_with_extrema(
    record: SearchRecord, observations: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.int64]]:
    policy = record.policy
    quantized = policy.quantize(observations, reject=False)
    qmax = policy.spec.quantizer.qmax
    probes = [np.zeros(policy.spec.quantizer.n_features, dtype=np.int64)]
    for feature in range(policy.spec.quantizer.n_features):
        low = np.zeros(policy.spec.quantizer.n_features, dtype=np.int64)
        high = low.copy()
        low[feature] = -qmax
        high[feature] = qmax
        probes.extend((low, high))
    return np.unique(np.concatenate((quantized, np.asarray(probes)), axis=0), axis=0)


def _box_certificate(record: SearchRecord) -> BoxCertificate | None:
    quantizer = record.policy.spec.quantizer
    points = (2 * quantizer.qmax + 1) ** quantizer.n_features
    if points > 1_000_000:
        return None
    return certify_quantized_box(record.policy, max_points=1_000_000)


def run_experiment(
    *,
    env_id: str,
    output: str | Path,
    backend: Backend,
    preset: ResearchPreset,
    seed_root: str = "release-2026-08",
    run_id: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    teacher_checkpoint: TeacherCheckpoint | None = None,
) -> ExperimentSummary:
    """Run the complete experiment and write a self-verifying evidence bundle."""
    identifier = (
        run_id
        or hashlib.sha256(f"{seed_root}|{env_id}|{backend}|{preset.full}".encode()).hexdigest()[:16]
    )
    destination = Path(output)
    ledger = ArtifactLedger(destination)
    seeds = SeedPlan.derive(seed_root, env_id, full=preset.full)
    provenance = RunProvenance.capture(
        run_id=identifier,
        project_version=__version__,
        command=("unseen-loop", "research", "--env-id", env_id, "--backend", backend),
        mode=backend,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    ledger.write_json("provenance.json", dataclass_dict(provenance))
    ledger.write_json("seeds.json", dataclass_dict(seeds))
    ledger.write_json(
        "config.json",
        {
            "preset": dataclass_dict(preset),
            "search_candidates": preset.search.candidates,
            "teacher_source": "external-checkpoint" if teacher_checkpoint else "local-cem",
        },
    )

    if teacher_checkpoint is None:
        teacher, history = train_cem_teacher(
            env_id,
            seed=seeds.training,
            hidden_size=preset.hidden_size,
            iterations=preset.teacher_iterations,
            population=preset.teacher_population,
            episodes_per_candidate=preset.episodes_per_candidate,
        )
    else:
        if teacher_checkpoint.env_id != env_id:
            raise ValueError("external teacher checkpoint environment does not match")
        teacher = MLPTeacher(teacher_checkpoint)
        history = ()
    ledger.write_text("teacher/checkpoint.json", teacher.checkpoint.to_json() + "\n")
    ledger.write_jsonl("teacher/training.jsonl", (dataclass_dict(row) for row in history))
    teacher_return_mean, _ = _teacher_return(teacher, seeds.evaluation)

    records = search_policies(
        teacher,
        distillation_seeds=seeds.distillation,
        evaluation_seeds=seeds.evaluation,
        refinement_seeds=seeds.refinement,
        config=preset.search,
    )
    frontier = pareto_front(records)
    champion = _select_champion(frontier, teacher_return_mean)
    ledger.write_jsonl(
        "search/candidates.jsonl",
        (
            {
                "metrics": dataclass_dict(record.metrics),
                "diagnostics": dataclass_dict(record.diagnostics),
                "refinement_rounds": record.refinement_rounds,
                "train_samples": record.train_samples,
                "saturation_rate": record.saturation_rate,
                "pareto": record in frontier,
            }
            for record in records
        ),
    )
    for record in records:
        ledger.write_text(
            f"policies/{record.policy.spec.digest}.json", record.policy.spec.to_json() + "\n"
        )

    heldout = collect_trajectories(env_id, teacher, seeds.evaluation)
    quantized = champion.policy.quantize(heldout.observations, reject=False)
    certificate = certify_actions(
        champion.policy, quantized, global_p_error=preset.search.global_p_error
    )
    ledger.write_json(
        "certificates/heldout.json",
        {
            "policy_digest": champion.policy.spec.digest,
            "observations": int(quantized.shape[0]),
            "coverage": certificate.coverage,
            "mismatches": certificate.mismatches,
            "certified_mismatches": certificate.certified_mismatches,
            "global_p_error": certificate.global_p_error,
            "horizon_union_bound": min(1.0, quantized.shape[0] * certificate.global_p_error),
            "claim": (
                "integer action equals float-student action when certificate is true, "
                "conditional on correct FHE evaluation"
            ),
        },
    )
    box = _box_certificate(champion)
    if box is not None:
        ledger.write_json("certificates/box.json", dataclass_dict(box))

    simulation_match: bool | None = None
    real_measurements: list[RoundTripMeasurement] = []
    if backend in {"simulate", "fhe"}:
        calibration = _calibration_with_extrema(champion, heldout.observations)
        with tempfile.TemporaryDirectory(prefix="unseen-loop-fhe-") as temporary:
            compiled = compile_policy(
                champion.policy,
                calibration,
                temporary,
                global_p_error=preset.search.global_p_error,
            )
            simulated = np.asarray([compiled.simulate(row) for row in calibration[:32]])
            clear = champion.policy.integer_scores_from_quantized(calibration[:32])
            simulation_match = bool(np.array_equal(simulated, clear))
            ledger.write_json("circuits/receipt.json", dataclass_dict(compiled.receipt))
            ledger.write_bytes("circuits/server.zip", compiled.server_path.read_bytes())
            ledger.write_bytes("circuits/client-specs.bin", compiled.client_specs_path.read_bytes())
            if backend == "fhe":
                for row in calibration[: len(seeds.real_fhe)]:
                    real_measurements.append(compiled.real_roundtrip(row))
                ledger.write_jsonl(
                    "fhe/measurements.jsonl",
                    (dataclass_dict(measurement) for measurement in real_measurements),
                )

    real_all_match = (
        all(measurement.output_matches_clear for measurement in real_measurements)
        if real_measurements
        else None
    )
    if backend == "clear":
        label = "QUANTIZED CLEAR"
    elif backend == "simulate":
        label = "FHE SIMULATED"
    else:
        label = "REAL FHE"
    summary = ExperimentSummary(
        run_id=identifier,
        env_id=env_id,
        backend=backend,
        teacher_digest=teacher.checkpoint.digest,
        teacher_return_mean=teacher_return_mean,
        candidates=len(records),
        frontier_candidates=len(frontier),
        champion_policy_digest=champion.policy.spec.digest,
        champion_name=champion.policy.spec.name,
        champion_return_mean=champion.metrics.return_mean,
        champion_return_delta=champion.metrics.return_mean - teacher_return_mean,
        teacher_agreement=champion.metrics.teacher_agreement,
        certified_coverage=certificate.coverage,
        constraint_cost=champion.metrics.constraint_cost,
        estimated_output_bits=champion.metrics.estimated_bit_width,
        encrypted_multiplications=champion.metrics.encrypted_multiplications,
        box_certificate_coverage=box.coverage if box is not None else None,
        box_certificate_points=box.points if box is not None else None,
        simulated_matches_integer=simulation_match,
        real_fhe_calls=len(real_measurements),
        real_fhe_all_match=real_all_match,
        label=label,
        privacy_evidence=backend == "fhe" and real_all_match is True,
    )
    ledger.write_json("summary.json", dataclass_dict(summary))
    ledger.write_json(
        "claims.json",
        {
            "supported": [
                "client-side quantization is frozen and range checked",
                "certificate-guided search measures student-induced closed-loop return",
                f"execution evidence label: {label}",
            ],
            "not_supported": [
                "malicious-server computation integrity",
                "model confidentiality against adaptive clients",
                "private training",
                "traffic-flow confidentiality",
                "endpoint security",
            ],
            "privacy_evidence": summary.privacy_evidence,
        },
    )
    ledger.finalize()
    verified, failures = ledger.verify()
    if not verified:
        raise RuntimeError(f"artifact verification failed: {failures}")
    if backend == "simulate" and simulation_match is not True:
        raise RuntimeError("compiled simulation disagrees with exact integer semantics")
    if backend == "fhe" and real_all_match is not True:
        raise RuntimeError("real FHE output disagrees with exact integer semantics")
    if not math.isfinite(summary.champion_return_mean):
        raise RuntimeError("champion return is not finite")
    return summary


def verify_artifact(path: str | Path) -> tuple[bool, tuple[str, ...]]:
    return ArtifactLedger(path).verify()


def load_summary(path: str | Path) -> ExperimentSummary:
    raw = json.loads((Path(path) / "summary.json").read_text())
    return ExperimentSummary(**raw)
