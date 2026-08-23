from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from unseen_loop.certificate import certify_actions, certify_quantized_box
from unseen_loop.policy import PolynomialPolicy, fit_polynomial_policy
from unseen_loop.specs import PolicySpec, QuantizerSpec


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
