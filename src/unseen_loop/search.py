"""Counterexample-guided multi-objective policy search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt

from unseen_loop.certificate import certificate_guided_weights, certify_actions
from unseen_loop.policy import FitDiagnostics, PolynomialPolicy, fit_polynomial_policy
from unseen_loop.specs import CandidateMetrics, FloatArray, QuantizerSpec
from unseen_loop.teacher import MLPTeacher, collect_trajectories, rollout


@dataclass(frozen=True)
class SearchConfig:
    degrees: tuple[int, ...] = (1, 2)
    input_bits: tuple[int, ...] = (3, 4, 5)
    coefficient_bits: tuple[int, ...] = (4, 6, 8)
    ridge_values: tuple[float, ...] = (1e-3, 1e-2)
    refinement_rounds: int = 2
    calibration_padding: float = 0.25
    global_p_error: float = 1e-6

    @property
    def candidates(self) -> int:
        return len(self.degrees) * len(self.input_bits) * len(self.coefficient_bits) * len(
            self.ridge_values
        )


@dataclass(frozen=True)
class SearchRecord:
    metrics: CandidateMetrics
    policy: PolynomialPolicy
    diagnostics: FitDiagnostics
    refinement_rounds: int
    train_samples: int
    saturation_rate: float


class IntegerStudent:
    """Gym adapter using the exact integer policy semantics."""

    def __init__(self, policy: PolynomialPolicy, *, reject: bool = False) -> None:
        self.policy = policy
        self.reject = reject
        self.saturations = 0
        self.calls = 0

    def score(self, observation: npt.ArrayLike) -> FloatArray:
        values = np.asarray(observation, dtype=np.float64)
        try:
            quantized = self.policy.quantize(values, reject=self.reject)
        except ValueError:
            raise
        unbounded = np.rint(
            (values - np.asarray(self.policy.spec.quantizer.center))
            / np.asarray(self.policy.spec.quantizer.step)
        )
        self.saturations += int(np.any(np.abs(unbounded) > self.policy.spec.quantizer.qmax))
        self.calls += 1
        return self.policy.dequantized_integer_scores(quantized)


def pareto_front(records: tuple[SearchRecord, ...]) -> tuple[SearchRecord, ...]:
    """Return candidates not dominated across utility, certificate, and circuit cost."""

    def dominates(left: SearchRecord, right: SearchRecord) -> bool:
        left_values = (
            left.metrics.return_mean,
            left.metrics.teacher_agreement,
            left.metrics.certified_coverage,
            -left.metrics.constraint_cost,
            -left.metrics.estimated_bit_width,
            -left.metrics.encrypted_multiplications,
        )
        right_values = (
            right.metrics.return_mean,
            right.metrics.teacher_agreement,
            right.metrics.certified_coverage,
            -right.metrics.constraint_cost,
            -right.metrics.estimated_bit_width,
            -right.metrics.encrypted_multiplications,
        )
        return all(a >= b for a, b in zip(left_values, right_values)) and any(
            a > b for a, b in zip(left_values, right_values)
        )

    frontier = [
        record
        for record in records
        if not any(dominates(other, record) for other in records if other is not record)
    ]
    return tuple(
        sorted(
            frontier,
            key=lambda record: (
                -record.metrics.certified_coverage,
                -record.metrics.return_mean,
                record.metrics.estimated_bit_width,
                record.policy.spec.digest,
            ),
        )
    )


def _fit_refined_candidate(
    teacher: MLPTeacher,
    observations: FloatArray,
    teacher_scores: FloatArray,
    *,
    degree: int,
    input_bits: int,
    coefficient_bits: int,
    ridge: float,
    name: str,
    config: SearchConfig,
    refinement_seeds: tuple[int, ...],
) -> tuple[PolynomialPolicy, FitDiagnostics, int, float]:
    quantizer = QuantizerSpec.calibrate(
        observations, input_bits=input_bits, padding=config.calibration_padding
    )
    train_observations = observations.copy()
    train_scores = teacher_scores.copy()
    train_weights = np.ones(train_observations.shape[0], dtype=np.float64)
    policy: PolynomialPolicy
    diagnostics: FitDiagnostics
    saturation_count = 0
    saturation_calls = 0

    for round_index in range(config.refinement_rounds + 1):
        policy, diagnostics = fit_polynomial_policy(
            train_observations,
            train_scores,
            env_id=teacher.checkpoint.env_id,
            name=name,
            degree=degree,
            input_bits=input_bits,
            coefficient_bits=coefficient_bits,
            ridge=ridge,
            sample_weights=train_weights,
            quantizer=quantizer,
        )
        quantized_train = policy.quantize(train_observations, reject=False)
        certificate = certify_actions(
            policy, quantized_train, global_p_error=config.global_p_error
        )
        train_weights = certificate_guided_weights(certificate)
        if round_index == config.refinement_rounds:
            break

        adapter = IntegerStudent(policy, reject=False)
        batch = collect_trajectories(
            teacher.checkpoint.env_id,
            adapter,
            tuple(seed + round_index * 10_000 for seed in refinement_seeds),
        )
        saturation_count += adapter.saturations
        saturation_calls += adapter.calls
        if batch.observations.size == 0:
            continue
        new_scores = np.asarray(teacher.score(batch.observations), dtype=np.float64)
        new_quantized = policy.quantize(batch.observations, reject=False)
        new_certificate = certify_actions(
            policy, new_quantized, global_p_error=config.global_p_error
        )
        new_weights = certificate_guided_weights(
            new_certificate, uncertified_gain=12.0, mismatch_gain=24.0
        )
        train_observations = np.concatenate((train_observations, batch.observations), axis=0)
        train_scores = np.concatenate((train_scores, new_scores), axis=0)
        train_weights = np.concatenate((train_weights, new_weights), axis=0)

    saturation_rate = saturation_count / saturation_calls if saturation_calls else 0.0
    return policy, diagnostics, train_observations.shape[0], saturation_rate


def search_policies(
    teacher: MLPTeacher,
    *,
    distillation_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    refinement_seeds: tuple[int, ...],
    config: SearchConfig = SearchConfig(),
) -> tuple[SearchRecord, ...]:
    """Search a fixed grid, using student-induced states as certificate counterexamples."""
    if not distillation_seeds or not evaluation_seeds or not refinement_seeds:
        raise ValueError("distillation, evaluation, and refinement seeds cannot be empty")
    if set(distillation_seeds) & set(evaluation_seeds):
        raise ValueError("distillation and evaluation seeds must be disjoint")
    teacher_batch = collect_trajectories(
        teacher.checkpoint.env_id, teacher, distillation_seeds
    )
    records: list[SearchRecord] = []

    for degree, input_bits, coefficient_bits, ridge in product(
        config.degrees, config.input_bits, config.coefficient_bits, config.ridge_values
    ):
        name = f"d{degree}-x{input_bits}-w{coefficient_bits}-r{ridge:g}"
        policy, diagnostics, train_samples, saturation_rate = _fit_refined_candidate(
            teacher,
            teacher_batch.observations,
            teacher_batch.scores,
            degree=degree,
            input_bits=input_bits,
            coefficient_bits=coefficient_bits,
            ridge=ridge,
            name=name,
            config=config,
            refinement_seeds=refinement_seeds,
        )

        heldout = collect_trajectories(teacher.checkpoint.env_id, teacher, evaluation_seeds)
        quantized_heldout = policy.quantize(heldout.observations, reject=False)
        student_actions = policy.actions_from_quantized(quantized_heldout, integer=True)
        agreement = float(np.mean(student_actions == heldout.actions))
        certificate = certify_actions(
            policy, quantized_heldout, global_p_error=config.global_p_error
        )

        adapter = IntegerStudent(policy, reject=False)
        episodes = [
            rollout(teacher.checkpoint.env_id, adapter, seed=seed)[0]
            for seed in evaluation_seeds
        ]
        returns = np.asarray([episode.total_return for episode in episodes], dtype=np.float64)
        costs = np.asarray([episode.constraint_cost for episode in episodes], dtype=np.float64)
        metrics = CandidateMetrics(
            policy_digest=policy.spec.digest,
            degree=degree,
            input_bits=input_bits,
            coefficient_bits=coefficient_bits,
            return_mean=float(np.mean(returns)),
            return_std=float(np.std(returns)),
            teacher_agreement=agreement,
            certified_coverage=certificate.coverage,
            constraint_cost=float(np.mean(costs)),
            estimated_bit_width=policy.estimated_output_bits,
            encrypted_multiplications=policy.encrypted_multiplications,
        )
        records.append(
            SearchRecord(
                metrics=metrics,
                policy=policy,
                diagnostics=diagnostics,
                refinement_rounds=config.refinement_rounds,
                train_samples=train_samples,
                saturation_rate=max(
                    saturation_rate,
                    adapter.saturations / adapter.calls if adapter.calls else 0.0,
                ),
            )
        )
    return tuple(records)
