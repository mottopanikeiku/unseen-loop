from __future__ import annotations

import numpy as np
import pytest

from unseen_loop.fhe_backend import compile_policy
from unseen_loop.policy import PolynomialPolicy
from unseen_loop.specs import PolicySpec, QuantizerSpec

pytest.importorskip("concrete.fhe")


@pytest.mark.fhe
@pytest.mark.slow
def test_real_fhe_roundtrip_uses_serialized_client_server_boundary(tmp_path) -> None:
    policy = PolynomialPolicy(
        PolicySpec(
            name="fhe-canary",
            env_id="Synthetic-v0",
            degree=1,
            actions=2,
            quantizer=QuantizerSpec(center=(0.0, 0.0), step=(1.0, 1.0), qmax=1),
            float_coefficients=((1.0, 2.0, -1.0), (-1.0, -2.0, 1.0)),
            integer_coefficients=((1, 2, -1), (-1, -2, 1)),
            coefficient_scale=1.0,
        )
    )
    inputset = np.asarray(((-1, -1), (-1, 1), (1, -1), (1, 1)), dtype=np.int64)
    compiled = compile_policy(policy, inputset, tmp_path)

    quantized = np.asarray((1, -1), dtype=np.int64)
    clear = policy.integer_scores_from_quantized(quantized)
    assert np.array_equal(compiled.simulate(quantized), clear)
    first = compiled.real_roundtrip(quantized)
    second = compiled.real_roundtrip(quantized)

    assert first.backend == "REAL FHE"
    assert first.output_matches_clear and second.output_matches_clear
    assert first.request_sha256 != second.request_sha256
    assert not first.server_secret_key_present
    assert not compiled.receipt.server_secret_key_markers
    assert compiled.receipt.security_level == 128
