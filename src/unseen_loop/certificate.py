"""Sound action-invariance certificates for the frozen integer policy circuit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt

from unseen_loop.policy import PolynomialPolicy
from unseen_loop.specs import FloatArray, IntArray


@dataclass(frozen=True)
class ActionCertificate:
    """Per-state proof obligations and observed integer actions."""

    float_actions: IntArray
    integer_actions: IntArray
    margins: FloatArray
    error_bounds: FloatArray
    certified: npt.NDArray[np.bool_]
    global_p_error: float

    @property
    def coverage(self) -> float:
        return float(np.mean(self.certified)) if self.certified.size else 0.0

    @property
    def mismatches(self) -> int:
        return int(np.count_nonzero(self.float_actions != self.integer_actions))

    @property
    def certified_mismatches(self) -> int:
        return int(np.count_nonzero(self.certified & (self.float_actions != self.integer_actions)))

    def assert_sound(self) -> None:
        if self.certified_mismatches:
            raise AssertionError(
                f"certificate unsound on {self.certified_mismatches} certified observations"
            )


@dataclass(frozen=True)
class BoxCertificate:
    """Exhaustive certificate over every integer code in a bounded quantizer box."""

    lower: tuple[int, ...]
    upper: tuple[int, ...]
    points: int
    certified_points: int
    mismatches: int
    certified_mismatches: int
    minimum_margin: float
    maximum_error_bound: float
    input_digest: str
    global_p_error: float

    @property
    def coverage(self) -> float:
        return self.certified_points / self.points

    @property
    def complete(self) -> bool:
        return self.certified_points == self.points and self.certified_mismatches == 0


def _top_two_margin(scores: FloatArray) -> tuple[IntArray, FloatArray]:
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape (samples, at-least-two-actions)")
    actions = np.argmax(scores, axis=1).astype(np.int64)
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return actions, np.asarray(margins, dtype=np.float64)


def certify_actions(
    policy: PolynomialPolicy,
    quantized: npt.ArrayLike,
    *,
    global_p_error: float = 1e-6,
    use_global_bound: bool = False,
) -> ActionCertificate:
    """Certify float-student/integer-circuit argmax agreement.

    The certificate accounts for deterministic coefficient rounding. Concrete's
    whole-circuit ``global_p_error`` remains a separate probabilistic premise;
    empirical ciphertext agreement cannot replace it.
    """
    if not 0 <= global_p_error < 1:
        raise ValueError("global_p_error must lie in [0, 1)")
    values = np.asarray(quantized, dtype=np.int64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != policy.spec.quantizer.n_features:
        raise ValueError("quantized observations have the wrong shape")
    if np.any(np.abs(values) > policy.spec.quantizer.qmax):
        raise ValueError("quantized observations exceed the compiled domain")

    float_scores = policy.float_scores_from_quantized(values)
    float_actions, margins = _top_two_margin(float_scores)
    integer_actions = policy.actions_from_quantized(values, integer=True)
    if use_global_bound:
        per_action = np.broadcast_to(policy.global_coefficient_error_bound(), float_scores.shape)
    else:
        per_action = policy.coefficient_error_bound(values)
    error_bounds = np.max(per_action, axis=1)
    certified = margins > 2 * error_bounds
    certificate = ActionCertificate(
        float_actions=float_actions,
        integer_actions=integer_actions,
        margins=margins,
        error_bounds=np.asarray(error_bounds, dtype=np.float64),
        certified=np.asarray(certified, dtype=np.bool_),
        global_p_error=global_p_error,
    )
    certificate.assert_sound()
    return certificate


def certificate_guided_weights(
    certificate: ActionCertificate,
    *,
    uncertified_gain: float = 8.0,
    mismatch_gain: float = 16.0,
) -> FloatArray:
    """Prioritize occupied states at which the deployed integer action is fragile."""
    if uncertified_gain < 1 or mismatch_gain < uncertified_gain:
        raise ValueError("gains must satisfy mismatch_gain >= uncertified_gain >= 1")
    weights = np.ones(certificate.certified.shape[0], dtype=np.float64)
    weights[~certificate.certified] = uncertified_gain
    weights[certificate.float_actions != certificate.integer_actions] = mismatch_gain
    positive_margins = certificate.margins[certificate.margins > 0]
    reference = float(np.median(positive_margins)) if positive_margins.size else 1.0
    fragility = np.clip(reference / np.maximum(certificate.margins, 1e-12), 1, 4)
    return weights * fragility


def certify_quantized_box(
    policy: PolynomialPolicy,
    *,
    lower: tuple[int, ...] | None = None,
    upper: tuple[int, ...] | None = None,
    max_points: int = 1_000_000,
    batch_size: int = 16_384,
    global_p_error: float = 1e-6,
) -> BoxCertificate:
    """Exhaustively verify a low-dimensional integer region without sampling claims."""
    dimensions = policy.spec.quantizer.n_features
    qmax = policy.spec.quantizer.qmax
    low = lower or tuple(-qmax for _ in range(dimensions))
    high = upper or tuple(qmax for _ in range(dimensions))
    if len(low) != dimensions or len(high) != dimensions:
        raise ValueError("box endpoints must match the observation dimension")
    if any(
        left > right or left < -qmax or right > qmax for left, right in zip(low, high, strict=True)
    ):
        raise ValueError("box must be ordered and lie inside the quantizer domain")
    points = int(np.prod([right - left + 1 for left, right in zip(low, high, strict=True)]))
    if points > max_points:
        raise ValueError(f"box contains {points} points, above max_points={max_points}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    digest = hashlib.sha256()
    certified_points = 0
    mismatches = 0
    certified_mismatches = 0
    minimum_margin = np.inf
    maximum_error = 0.0
    pending: list[tuple[int, ...]] = []

    def consume(rows: list[tuple[int, ...]]) -> tuple[int, int, int, float, float]:
        array = np.asarray(rows, dtype=np.int64)
        digest.update(array.tobytes(order="C"))
        result = certify_actions(policy, array, global_p_error=global_p_error)
        return (
            int(np.count_nonzero(result.certified)),
            result.mismatches,
            result.certified_mismatches,
            float(np.min(result.margins)),
            float(np.max(result.error_bounds)),
        )

    axes = [range(left, right + 1) for left, right in zip(low, high, strict=True)]
    for point in product(*axes):
        pending.append(point)
        if len(pending) == batch_size:
            count, mismatch, certified_mismatch, min_margin, max_error = consume(pending)
            certified_points += count
            mismatches += mismatch
            certified_mismatches += certified_mismatch
            minimum_margin = min(minimum_margin, min_margin)
            maximum_error = max(maximum_error, max_error)
            pending.clear()
    if pending:
        count, mismatch, certified_mismatch, min_margin, max_error = consume(pending)
        certified_points += count
        mismatches += mismatch
        certified_mismatches += certified_mismatch
        minimum_margin = min(minimum_margin, min_margin)
        maximum_error = max(maximum_error, max_error)

    return BoxCertificate(
        lower=low,
        upper=high,
        points=points,
        certified_points=certified_points,
        mismatches=mismatches,
        certified_mismatches=certified_mismatches,
        minimum_margin=float(minimum_margin),
        maximum_error_bound=maximum_error,
        input_digest=digest.hexdigest(),
        global_p_error=global_p_error,
    )
