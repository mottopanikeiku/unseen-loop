from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from unseen_loop.artifacts import ArtifactLedger
from unseen_loop.challenge import (
    CHALLENGE_SCHEMA,
    LEDGERED_FILES,
    ROW_SCHEMA,
    challenge_policy_spec,
    run_fhe_challenge,
)
from unseen_loop.fhe_backend import CircuitReceipt, RoundTripMeasurement
from unseen_loop.policy import PolynomialPolicy


class FakeCompiledPolicy:
    def __init__(
        self,
        policy: PolynomialPolicy,
        calibration: np.ndarray,
        artifact_dir: Path,
        *,
        mode: str,
    ) -> None:
        self.policy = policy
        self.mode = mode
        self.simulation_calls = 0
        self.real_calls = 0
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.server_path = artifact_dir / "server.zip"
        with zipfile.ZipFile(self.server_path, "w") as archive:
            archive.writestr("circuit/program.bin", b"compiled-server")
        self.client_specs_path = artifact_dir / "client-specs.bin"
        self.client_specs_path.write_bytes(b"public-client-specs")
        server_payload = self.server_path.read_bytes()
        specs_payload = self.client_specs_path.read_bytes()
        qmax = policy.spec.quantizer.qmax
        self.receipt = CircuitReceipt(
            policy_digest=policy.spec.digest,
            concrete_python_version="test-double",
            global_p_error=1e-6,
            security_level=128,
            maximum_integer_bit_width=8,
            complexity=1.0,
            input_shape=(2,),
            calibration_strategy="exhaustive signed integer domain",
            domain_points=len(calibration),
            calibration_rows=len(calibration),
            calibration_sha256=hashlib.sha256(calibration.tobytes(order="C")).hexdigest(),
            input_min=(-qmax, -qmax),
            input_max=(qmax, qmax),
            integer_output_bound=tuple(int(value) for value in policy.integer_output_bound()),
            server_artifact_bytes=len(server_payload),
            server_artifact_sha256=hashlib.sha256(server_payload).hexdigest(),
            client_specs_bytes=len(specs_payload),
            client_specs_sha256=hashlib.sha256(specs_payload).hexdigest(),
            compile_ns=10,
            mlir_sha256="a" * 64,
            server_secret_key_markers=(),
        )

    def simulate(self, code: np.ndarray) -> np.ndarray:
        self.simulation_calls += 1
        result = self.policy.integer_scores_from_quantized(code)
        if self.mode == "simulation-mismatch":
            return np.asarray(result, dtype=np.int64) + 1
        return np.asarray(result, dtype=np.int64)

    def real_roundtrip(self, code: np.ndarray) -> RoundTripMeasurement:
        self.real_calls += 1
        request_material = f"request:{self.real_calls}:{tuple(int(v) for v in code)}".encode()
        if self.mode == "repeated-ciphertext":
            request_material = b"same-ciphertext"
        response_material = b"response:" + request_material
        return RoundTripMeasurement(
            input_shape=(2,),
            output_shape=(2,),
            output_matches_clear=self.mode != "real-mismatch",
            keygen_ns=100 + self.real_calls,
            encrypt_ns=200 + self.real_calls,
            server_evaluate_ns=300 + self.real_calls,
            decrypt_ns=400 + self.real_calls,
            end_to_end_ns=1_000 + self.real_calls,
            evaluation_key_bytes=1_024,
            request_bytes=len(request_material),
            response_bytes=len(response_material),
            request_sha256=hashlib.sha256(request_material).hexdigest(),
            response_sha256=hashlib.sha256(response_material).hexdigest(),
            server_secret_key_marker_present=False,
        )


class FakeSession:
    def __init__(self, compiled: FakeCompiledPolicy) -> None:
        self.compiled = compiled
        self.client_keygen_ns = 73
        self.evaluation_key_sha256 = "e" * 64
        self.client_context_sha256 = "c" * 64

    def run(self, code: np.ndarray) -> RoundTripMeasurement:
        return self.compiled.real_roundtrip(code)


class FakeCompiler:
    def __init__(self, mode: str = "exact") -> None:
        self.mode = mode
        self.compiled: FakeCompiledPolicy | None = None
        self.calibration: np.ndarray | None = None

    def __call__(
        self,
        policy: PolynomialPolicy,
        calibration: np.ndarray,
        artifact_dir: Path,
        *,
        global_p_error: float,
        security_level: int,
    ) -> FakeCompiledPolicy:
        assert global_p_error == 1e-6
        assert security_level == 128
        self.calibration = np.asarray(calibration, dtype=np.int64)
        self.compiled = FakeCompiledPolicy(
            policy,
            self.calibration,
            artifact_dir,
            mode=self.mode,
        )
        return self.compiled


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_call_writes_closed_exhaustive_nonlinear_bundle(tmp_path) -> None:
    compiler = FakeCompiler()
    output = tmp_path / "challenge"

    summary = run_fhe_challenge(
        output,
        qmax=2,
        canary_repetitions=2,
        compiler=compiler,
        session_factory=FakeSession,
    )

    assert summary.schema_version == CHALLENGE_SCHEMA
    assert summary.domain_points == 25
    assert summary.simulation_rows == 25
    assert summary.real_domain_rows == 25
    assert summary.canary_codes == 3
    assert summary.canary_rows == 6
    assert summary.real_fhe_rows == 31
    assert summary.quadratic_feature_products_per_inference == 3
    assert summary.simulation_all_match
    assert summary.real_fhe_all_match
    assert summary.canary_randomness_passed
    assert summary.canary_distinct_request_hashes == 6
    assert summary.single_client_key_context
    assert summary.client_keygen_ns == 73
    assert summary.evaluation_key_sha256 == "e" * 64
    assert summary.client_context_sha256 == "c" * 64
    assert all(
        distribution.samples == 31
        and distribution.p50_ns is not None
        and distribution.p95_ns is not None
        for distribution in summary.timing_distributions.values()
    )
    assert compiler.compiled is not None
    assert compiler.compiled.simulation_calls == 28
    assert compiler.compiled.real_calls == 31
    assert compiler.calibration is not None
    assert {tuple(row) for row in compiler.calibration} == {
        (left, right) for left in range(-2, 3) for right in range(-2, 3)
    }

    assert {path.name for path in output.iterdir()} == {
        *LEDGERED_FILES,
        "checksums.sha256",
    }
    valid, failures = ArtifactLedger(output).verify()
    assert valid, failures
    checksum_paths = {
        line.partition("  ")[2] for line in (output / "checksums.sha256").read_text().splitlines()
    }
    assert checksum_paths == set(LEDGERED_FILES)
    persisted_summary = json.loads((output / "summary.json").read_text())
    assert persisted_summary["schema_version"] == CHALLENGE_SCHEMA
    assert persisted_summary["backend"] == "REAL FHE"
    policy = json.loads((output / "policy.json").read_text())
    assert policy["degree"] == 2
    assert policy["actions"] == 2
    assert len(policy["quantizer"]["center"]) == 2
    assert all(value != 0 for row in policy["integer_coefficients"] for value in row[3:])


def test_raw_schema_has_exact_counts_and_no_plaintext_vectors(tmp_path) -> None:
    output = tmp_path / "challenge"
    run_fhe_challenge(
        output,
        qmax=2,
        canary_repetitions=3,
        compiler=FakeCompiler(),
        session_factory=FakeSession,
    )

    rows = read_jsonl(output / "raw.jsonl")
    phases = Counter(row["phase"] for row in rows)
    assert phases == {"exhaustive-domain": 25, "fresh-ciphertext-canary": 9}
    assert all(row["schema_version"] == ROW_SCHEMA for row in rows)
    forbidden = {
        "input",
        "quantized",
        "plaintext",
        "decrypted",
        "output",
        "expected",
        "clear",
        "scores",
        "actions",
    }
    assert all(not (forbidden & row.keys()) for row in rows)
    assert {int(row["case_index"]) for row in rows if row["phase"] == "exhaustive-domain"} == set(
        range(25)
    )
    assert all(row["simulation_matches_integer_clear"] is True for row in rows)
    assert all(row["real_fhe_matches_integer_clear"] is True for row in rows)
    assert all("request_sha256" in row and "response_sha256" in row for row in rows)
    assert all("request_bytes" in row and "server_evaluate_ns" in row for row in rows)
    assert {row["evaluation_key_sha256"] for row in rows} == {"e" * 64}
    assert {row["client_context_sha256"] for row in rows} == {"c" * 64}


def test_same_code_canaries_use_fresh_ciphertexts(tmp_path) -> None:
    output = tmp_path / "challenge"
    run_fhe_challenge(
        output,
        canary_repetitions=4,
        compiler=FakeCompiler(),
        session_factory=FakeSession,
    )
    canaries = [
        row for row in read_jsonl(output / "raw.jsonl") if row["phase"] == "fresh-ciphertext-canary"
    ]

    groups: dict[int, list[dict[str, object]]] = {}
    for row in canaries:
        groups.setdefault(int(row["case_index"]), []).append(row)
    assert len(groups) == 3
    for group in groups.values():
        assert len(group) == 4
        assert len({row["request_sha256"] for row in group}) == 4
        assert {row["repetition"] for row in group} == {0, 1, 2, 3}


def test_percentiles_are_withheld_until_sample_count_supports_them(tmp_path) -> None:
    summary = run_fhe_challenge(
        tmp_path / "small",
        qmax=1,
        canary_repetitions=2,
        compiler=FakeCompiler(),
        session_factory=FakeSession,
    )

    assert summary.real_fhe_rows == 15
    for distribution in summary.timing_distributions.values():
        assert distribution.samples == 15
        assert distribution.p50_ns is not None
        assert distribution.p95_ns is None


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("simulation-mismatch", "simulation disagrees"),
        ("real-mismatch", "REAL FHE output disagrees"),
        ("repeated-ciphertext", "repeated ciphertext hashes"),
    ],
)
def test_mismatch_seams_fail_before_artifact_publication(tmp_path, mode: str, message: str) -> None:
    output = tmp_path / mode

    with pytest.raises(RuntimeError, match=message):
        run_fhe_challenge(
            output,
            canary_repetitions=2,
            compiler=FakeCompiler(mode),
            session_factory=FakeSession,
        )

    assert not output.exists()


def test_existing_output_is_rejected_as_unledgered(tmp_path) -> None:
    output = tmp_path / "challenge"
    output.mkdir()
    (output / "foreign.txt").write_text("not part of this challenge")

    with pytest.raises(RuntimeError, match="unledgered output"):
        run_fhe_challenge(
            output,
            compiler=FakeCompiler(),
            session_factory=FakeSession,
        )


def test_policy_is_deterministic_and_rejects_larger_domain() -> None:
    first = challenge_policy_spec(qmax=2)
    second = challenge_policy_spec(qmax=2)

    assert first.to_json() == second.to_json()
    assert first.digest == second.digest
    with pytest.raises(ValueError, match="qmax"):
        challenge_policy_spec(qmax=3)
    with pytest.raises(ValueError, match="qmax"):
        challenge_policy_spec(qmax=True)
