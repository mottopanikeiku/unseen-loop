from __future__ import annotations

import dataclasses
import zipfile

import numpy as np
import pytest

from unseen_loop.fhe_backend import (
    RoundTripMeasurement,
    calibration_inputset,
    compile_policy,
    server_artifact_secret_markers,
)
from unseen_loop.policy import PolynomialPolicy
from unseen_loop.specs import PolicySpec, QuantizerSpec


@pytest.mark.fhe
@pytest.mark.slow
def test_real_fhe_roundtrip_uses_serialized_client_server_boundary(tmp_path) -> None:
    pytest.importorskip("concrete.fhe")
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
    assert not first.server_secret_key_marker_present
    assert not compiled.receipt.server_secret_key_markers
    assert compiled.receipt.security_level == 128


def policy(*, degree: int, qmax: int = 3, dimensions: int = 1) -> PolynomialPolicy:
    feature_count = 1 + dimensions
    if degree == 2:
        feature_count += dimensions * (dimensions + 1) // 2
    if degree == 2 and dimensions == 1:
        coefficients = ((9, -3, -1), (-9, 3, 1))
    else:
        coefficients = (
            tuple(range(1, feature_count + 1)),
            tuple(-value for value in range(1, feature_count + 1)),
        )
    return PolynomialPolicy(
        PolicySpec(
            name="calibration-test",
            env_id="Synthetic-v0",
            degree=degree,
            actions=2,
            quantizer=QuantizerSpec(
                center=(0.0,) * dimensions,
                step=(1.0,) * dimensions,
                qmax=qmax,
            ),
            float_coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
            integer_coefficients=coefficients,
            coefficient_scale=1.0,
        )
    )


def test_degree_one_calibration_uses_every_box_corner() -> None:
    linear = policy(degree=1, qmax=2, dimensions=2)
    calibration, strategy, domain_points = calibration_inputset(linear)

    assert strategy == "all signed-domain corners"
    assert domain_points == 25
    assert {tuple(row) for row in calibration} == {
        (-2, -2),
        (-2, 2),
        (2, -2),
        (2, 2),
    }


def test_quadratic_calibration_covers_interior_extremum_and_respects_cap() -> None:
    quadratic = policy(degree=2)
    calibration, strategy, domain_points = calibration_inputset(quadratic)

    assert strategy == "exhaustive signed integer domain"
    assert domain_points == 7
    assert {int(row[0]) for row in calibration} == set(range(-3, 4))
    scores = np.asarray([quadratic.integer_scores_from_quantized(row)[0] for row in calibration])
    corner_scores = np.asarray(
        [quadratic.integer_scores_from_quantized((value,))[0] for value in (-3, 3)]
    )
    assert np.max(scores) == 11
    assert np.max(scores) > np.max(corner_scores)

    with pytest.raises(ValueError, match="exhaustive calibration"):
        calibration_inputset(quadratic, max_points=6)


def test_measurements_exclude_plaintext_and_decrypted_vectors() -> None:
    measurement = RoundTripMeasurement(
        input_shape=(4,),
        output_shape=(2,),
        output_matches_clear=True,
        keygen_ns=1,
        encrypt_ns=2,
        server_evaluate_ns=3,
        decrypt_ns=4,
        end_to_end_ns=10,
        evaluation_key_bytes=5,
        request_bytes=6,
        response_bytes=7,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        server_secret_key_marker_present=False,
    )

    persisted = dataclasses.asdict(measurement)
    assert (
        not {
            "input",
            "quantized",
            "decrypted",
            "output",
            "expected",
            "clear",
        }
        & persisted.keys()
    )
    assert persisted["schema_version"] == "unseen-loop/fhe-measurement-v2"


def test_server_secret_marker_status_is_derived_from_archive(tmp_path) -> None:
    harmless = tmp_path / "server.zip"
    with zipfile.ZipFile(harmless, "w") as archive:
        archive.writestr("circuit/program.bin", b"public")
    assert not server_artifact_secret_markers(harmless)

    suspicious = tmp_path / "suspicious.zip"
    with zipfile.ZipFile(suspicious, "w") as archive:
        archive.writestr("keys/client_secret_key.bin", b"not-a-real-key")
    assert server_artifact_secret_markers(suspicious) == ("keys/client_secret_key.bin",)
