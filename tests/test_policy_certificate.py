from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from unseen_loop.certificate import certify_actions, certify_quantized_box
from unseen_loop.policy import PolynomialPolicy, fit_polynomial_policy
from unseen_loop.specs import PolicySpec, QuantizerSpec, polynomial_feature_count


def test_quantizer_rejects_outside_compiled_domain() -> None:
    quantizer = QuantizerSpec(center=(0.0, 1.0), step=(0.5, 0.25), qmax=3)
    assert np.array_equal(quantizer.quantize([1.0, 0.5]), np.array([2, -2]))
    with pytest.raises(ValueError, match="outside"):
        quantizer.quantize([2.0, 1.0])
    assert np.array_equal(quantizer.quantize([2.0, 1.0], reject=False), np.array([3, 0]))


def test_fitted_quadratic_policy_roundtrips_and_certifies() -> None:
    rng = np.random.default_rng(7)
    observations = rng.uniform(-1, 1, size=(500, 2))
    scores = np.column_stack(
        (
            3 * observations[:, 0] - observations[:, 1] ** 2,
            -3 * observations[:, 0] + observations[:, 1] ** 2,
        )
    )
    policy, diagnostics = fit_polynomial_policy(
        observations,
        scores,
        env_id="Synthetic-v0",
        name="quadratic",
        degree=2,
        input_bits=5,
        coefficient_bits=10,
    )
    restored = PolynomialPolicy(PolicySpec.from_json(policy.spec.to_json()))
    quantized = restored.quantize(observations)
    certificate = certify_actions(restored, quantized)

    assert diagnostics.weighted_mse < 0.1
    assert certificate.coverage > 0.95
    assert certificate.certified_mismatches == 0
    assert np.array_equal(
        restored.integer_scores_from_quantized(quantized),
        policy.integer_scores_from_quantized(quantized),
    )


def test_certificate_condition_is_sound_under_coefficient_rounding() -> None:
    quantizer = QuantizerSpec(center=(0.0,), step=(1.0,), qmax=2)
    spec = PolicySpec(
        name="boundary",
        env_id="Synthetic-v0",
        degree=1,
        actions=2,
        quantizer=quantizer,
        float_coefficients=((0.2, 1.0), (-0.2, -1.0)),
        integer_coefficients=((0, 2), (0, -2)),
        coefficient_scale=2.0,
    )
    policy = PolynomialPolicy(spec)
    inputs = np.arange(-2, 3, dtype=np.int64)[:, None]
    certificate = certify_actions(policy, inputs)
    certificate.assert_sound()
    assert certificate.certified_mismatches == 0


def _exactly_certifiable(policy: PolynomialPolicy, point: np.ndarray) -> bool:
    """Exact rational version of ``margin > 2 * max coefficient error``."""
    features = [int(value) for value in policy.features(point)]
    scale = Fraction(policy.spec.coefficient_scale)
    scores = []
    errors = []
    for float_row, integer_row in zip(
        policy.spec.float_coefficients, policy.spec.integer_coefficients, strict=True
    ):
        weights = [Fraction(value) for value in float_row]
        rounded = [Fraction(value) / scale for value in integer_row]
        scores.append(sum(f * w for f, w in zip(features, weights, strict=True)))
        errors.append(
            sum(abs(f) * abs(w - r) for f, w, r in zip(features, weights, rounded, strict=True))
        )
    top, second = sorted(scores)[-1], sorted(scores)[-2]
    return top - second > 2 * max(errors)


def test_certificate_rejects_exact_tie_despite_float_rounding() -> None:
    # Captured from a CartPole search on aarch64: the integer scores tie at zero, the
    # exact margin equals twice the exact bound, and float64 rounding of the bound alone
    # made the old strict comparison certify a state whose integer argmax differs.
    quantizer = QuantizerSpec(
        center=(
            0.02248256281018257,
            -0.014941088855266571,
            0.006974349729716778,
            0.02862735092639923,
        ),
        step=(
            0.05655396516833987,
            0.1684443290744509,
            0.018590354759778296,
            0.2773009325776781,
        ),
        qmax=7,
    )
    weights = (
        -0.027688453805518543,
        -0.014248486701117247,
        0.275616506771026,
        -0.07138677453573597,
        -0.2756165067665364,
    )
    spec = PolicySpec(
        name="tie",
        env_id="CartPole-v1",
        degree=1,
        actions=2,
        quantizer=quantizer,
        float_coefficients=(weights, tuple(-value for value in weights)),
        integer_coefficients=((-3, -2, 31, -8, -31), (3, 2, -31, 8, 31)),
        coefficient_scale=112.47512118624259,
    )
    policy = PolynomialPolicy(spec)
    point = np.array([[-2, 1, 4, 0]], dtype=np.int64)

    certificate = certify_actions(policy, point)

    assert policy.integer_scores_from_quantized(point).tolist() == [[0, 0]]
    assert not _exactly_certifiable(policy, point[0])
    assert not certificate.certified[0]


@pytest.mark.parametrize("degree", [1, 2])
def test_certified_states_satisfy_the_exact_rational_condition(degree: int) -> None:
    rng = np.random.default_rng(11)
    quantizer = QuantizerSpec(center=(0.0, 0.0, 0.0), step=(1.0, 1.0, 1.0), qmax=3)
    points = np.asarray(tuple(product(range(-3, 4), repeat=3)), dtype=np.int64)
    feature_count = polynomial_feature_count(quantizer.n_features, degree)
    for _ in range(20):
        scale = float(rng.uniform(5, 50))
        weights = rng.normal(size=feature_count)
        integers = tuple(int(value) for value in np.rint(weights * scale))
        # Antisymmetric scores make integer ties, the tight case, common.
        policy = PolynomialPolicy(
            PolicySpec(
                name="antisymmetric",
                env_id="Synthetic-v0",
                degree=degree,
                actions=2,
                quantizer=quantizer,
                float_coefficients=(tuple(weights), tuple(-weights)),
                integer_coefficients=(integers, tuple(-value for value in integers)),
                coefficient_scale=scale,
            )
        )
        certificate = certify_actions(policy, points)
        for point in points[certificate.certified]:
            assert _exactly_certifiable(policy, point)


def test_exhaustive_box_receipt_covers_every_integer_code() -> None:
    quantizer = QuantizerSpec(center=(0.0, 0.0), step=(1.0, 1.0), qmax=1)
    spec = PolicySpec(
        name="box",
        env_id="Synthetic-v0",
        degree=1,
        actions=2,
        quantizer=quantizer,
        float_coefficients=((5.0, 0.1, 0.1), (-5.0, -0.1, -0.1)),
        integer_coefficients=((50, 1, 1), (-50, -1, -1)),
        coefficient_scale=10.0,
    )
    policy = PolynomialPolicy(spec)
    receipt = certify_quantized_box(policy)

    assert receipt.points == len(tuple(product(range(-1, 2), repeat=2))) == 9
    assert receipt.complete
    assert receipt.input_digest


def test_integer_output_bound_contains_exhaustive_scores() -> None:
    quantizer = QuantizerSpec(center=(0.0, 0.0), step=(1.0, 1.0), qmax=2)
    spec = PolicySpec(
        name="bounds",
        env_id="Synthetic-v0",
        degree=2,
        actions=2,
        quantizer=quantizer,
        float_coefficients=((0, 1, -2, 3, -4, 5), (1, -1, 1, -1, 1, -1)),
        integer_coefficients=((0, 1, -2, 3, -4, 5), (1, -1, 1, -1, 1, -1)),
        coefficient_scale=1.0,
    )
    policy = PolynomialPolicy(spec)
    points = np.asarray(tuple(product(range(-2, 3), repeat=2)), dtype=np.int64)
    scores = np.abs(policy.integer_scores_from_quantized(points))
    assert np.all(np.max(scores, axis=0) <= policy.integer_output_bound())
