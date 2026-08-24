"""Counterexample-guided multi-objective policy search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from unseen_loop.certificate import certificate_guided_weights, certify_actions
from unseen_loop.policy import FitDiagnostics, PolynomialPolicy, fit_polynomial_policy
from unseen_loop.specs import CandidateMetrics, FloatArray, QuantizerSpec
from unseen_loop.teacher import MLPTeacher, collect_trajectories


@dataclass(frozen=True)
class SearchConfig:
    degrees: tuple[int, ...] = (1, 2)
    input_bits: tuple[int, ...] = (3, 4, 5)
    coefficient_bits: tuple[int, ...] = (4, 6, 8)
    ridge_values: tuple[float, ...] = (1e-3, 1e-2)
    refinement_rounds: int = 2
    calibration_padding: float = 1.0
    certificate_weighting: bool = True
    student_occupancy_refinement: bool = True
    global_p_error: float = 1e-6

    @property
    def candidates(self) -> int:
        return (
            len(self.degrees)
            * len(self.input_bits)
            * len(self.coefficient_bits)
            * len(self.ridge_values)
        )


@dataclass(frozen=True)
class SelectionEpisodeMetrics:
    seed: int
    total_return: float
    constraint_cost: float
    action_digest: str
    steps: int
    teacher_agreement_count: int
    certified_count: int
    certified_mismatch_count: int
    saturation_count: int


@dataclass(frozen=True)
class SearchRecord:
    metrics: CandidateMetrics
    policy: PolynomialPolicy
    diagnostics: FitDiagnostics
    refinement_rounds: int
    train_samples: int
    saturation_rate: float
    selection_episodes: tuple[SelectionEpisodeMetrics, ...]


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
        pairs = tuple(zip(left_values, right_values, strict=True))
        return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)

    valid_records = tuple(record for record in records if record.metrics.range_valid)
    frontier = [
        record
        for record in valid_records
        if not any(dominates(other, record) for other in valid_records if other is not record)
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
        if round_index == config.refinement_rounds:
            break
        if config.certificate_weighting:
            quantized_train = policy.quantize(train_observations, reject=False)
            certificate = certify_actions(
                policy, quantized_train, global_p_error=config.global_p_error
            )
            train_weights = certificate_guided_weights(certificate)
        else:
            train_weights = np.ones(train_observations.shape[0], dtype=np.float64)
        if not config.student_occupancy_refinement and not config.certificate_weighting:
            break
        if not config.student_occupancy_refinement:
            continue
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
        if config.certificate_weighting:
            new_quantized = policy.quantize(batch.observations, reject=False)
            new_certificate = certify_actions(
                policy, new_quantized, global_p_error=config.global_p_error
            )
            new_weights = certificate_guided_weights(
                new_certificate, uncertified_gain=12.0, mismatch_gain=24.0
            )
        else:
            new_weights = np.ones(batch.observations.shape[0], dtype=np.float64)
        train_observations = np.concatenate((train_observations, batch.observations), axis=0)
        train_scores = np.concatenate((train_scores, new_scores), axis=0)
        train_weights = np.concatenate((train_weights, new_weights), axis=0)
        quantizer = QuantizerSpec.calibrate(
            train_observations,
            input_bits=input_bits,
            padding=config.calibration_padding,
        )

    saturation_rate = saturation_count / saturation_calls if saturation_calls else 0.0
    return policy, diagnostics, train_observations.shape[0], saturation_rate


def search_policies(
    teacher: MLPTeacher,
    *,
    distillation_seeds: tuple[int, ...],
    selection_seeds: tuple[int, ...],
    refinement_seeds: tuple[int, ...],
    config: SearchConfig | None = None,
) -> tuple[SearchRecord, ...]:
    """Search a fixed grid, using student-induced states as certificate counterexamples."""
    config = config or SearchConfig()
    if not distillation_seeds or not selection_seeds or not refinement_seeds:
        raise ValueError("distillation, selection, and refinement seeds cannot be empty")
    split_seeds = {
        "distillation": set(distillation_seeds),
        "selection": set(selection_seeds),
        "refinement": set(refinement_seeds),
    }
    for (left_name, left), (right_name, right) in combinations(split_seeds.items(), 2):
        if left & right:
            raise ValueError(f"{left_name} and {right_name} seeds must be disjoint")
    teacher_batch = collect_trajectories(teacher.checkpoint.env_id, teacher, distillation_seeds)
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

        adapter = IntegerStudent(policy, reject=False)
        selection = collect_trajectories(teacher.checkpoint.env_id, adapter, selection_seeds)
        quantized_selection = policy.quantize(selection.observations, reject=False)
        teacher_actions = np.argmax(teacher.score(selection.observations), axis=1)
        certificate = certify_actions(
            policy, quantized_selection, global_p_error=config.global_p_error
        )
        centers = np.asarray(policy.spec.quantizer.center, dtype=np.float64)
        quantizer_steps = np.asarray(policy.spec.quantizer.step, dtype=np.float64)
        unbounded = np.rint((selection.observations - centers) / quantizer_steps)
        saturated = np.any(np.abs(unbounded) > policy.spec.quantizer.qmax, axis=1)
        certificate_mismatches = certificate.float_actions != certificate.integer_actions
        if not np.array_equal(selection.actions, certificate.integer_actions):
            raise RuntimeError("selection actions disagree with exact integer certificate actions")
        if adapter.saturations != int(np.count_nonzero(saturated)):
            raise RuntimeError("selection saturation accounting is inconsistent")
        selection_episodes: list[SelectionEpisodeMetrics] = []
        for episode_index, episode in enumerate(selection.episodes):
            mask = selection.episode_ids == episode_index
            episode_steps = int(np.count_nonzero(mask))
            if episode_steps != episode.length:
                raise RuntimeError("selection episode step accounting is inconsistent")
            selection_episodes.append(
                SelectionEpisodeMetrics(
                    seed=episode.seed,
                    total_return=episode.total_return,
                    constraint_cost=episode.constraint_cost,
                    action_digest=episode.action_digest,
                    steps=episode_steps,
                    teacher_agreement_count=int(
                        np.count_nonzero(selection.actions[mask] == teacher_actions[mask])
                    ),
                    certified_count=int(np.count_nonzero(certificate.certified[mask])),
                    certified_mismatch_count=int(
                        np.count_nonzero(certificate.certified[mask] & certificate_mismatches[mask])
                    ),
                    saturation_count=int(np.count_nonzero(saturated[mask])),
                )
            )
        total_steps = sum(episode.steps for episode in selection_episodes)
        if total_steps < 1:
            raise RuntimeError("selection evaluation produced no occupied states")
        agreement_count = sum(episode.teacher_agreement_count for episode in selection_episodes)
        certified_count = sum(episode.certified_count for episode in selection_episodes)
        saturation_count = sum(episode.saturation_count for episode in selection_episodes)
        returns = np.asarray(selection.returns, dtype=np.float64)
        costs = np.asarray(selection.constraint_costs, dtype=np.float64)
        range_valid = saturation_count == 0
        metrics = CandidateMetrics(
            policy_digest=policy.spec.digest,
            degree=degree,
            input_bits=input_bits,
            coefficient_bits=coefficient_bits,
            return_mean=float(np.mean(returns)),
            return_std=float(np.std(returns)),
            teacher_agreement=agreement_count / total_steps,
            certified_coverage=certified_count / total_steps,
            constraint_cost=float(np.mean(costs)),
            estimated_bit_width=policy.estimated_output_bits,
            encrypted_multiplications=policy.encrypted_multiplications,
            range_valid=range_valid,
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
                selection_episodes=tuple(selection_episodes),
            )
        )
    return tuple(records)
