from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from unseen_loop.flagship import executor_shield_fhe as executor


def _manifest() -> dict[str, object]:
    return {
        "shield": {
            "output_shape": [5, 2, 4],
            "fhe_challenge": {
                "valid_calls": 7,
                "occupancy_states": 1,
                "extrema_states": 1,
                "threshold_states": 1,
                "tie_states": 1,
                "canary_states": 1,
                "canary_encryptions_per_state": 2,
                "expected_decoded_margins": 280,
                "expected_action_matches": 7,
                "invalid_domain_rejections": 2,
                "security_level": 128,
                "global_p_error": 1e-6,
            },
        }
    }


def _job(
    job_id: str,
    *,
    kind: str = "valid",
    category: str = "canary",
    state: int = 0,
    encryption: int = 0,
    case: int = 0,
) -> dict[str, object]:
    coordinates: dict[str, object]
    if kind == "invalid":
        coordinates = {"kind": kind, "case": case}
    else:
        coordinates = {
            "kind": kind,
            "category": category,
            "state": state,
            "encryption": encryption,
        }
    return {
        "job_id": job_id,
        "stage": "shield_fhe_challenge",
        "seed": 17,
        "coordinates": coordinates,
    }


def _install_fake_bundle(monkeypatch: object) -> list[Path]:
    compile_destinations: list[Path] = []
    tensor = np.arange(40, dtype=np.int64).reshape(5, 2, 4)

    def fake_compile(
        spec: object,
        artifact_dir: str | Path,
        *,
        global_p_error: float,
        security_level: int,
    ) -> None:
        assert global_p_error == 1e-6
        assert security_level == 128
        destination = Path(artifact_dir)
        compile_destinations.append(destination)
        server = b"public-evaluation-server"
        client = b"public-client-specification"
        (destination / "shield-server.zip").write_bytes(server)
        (destination / "shield-client-specs.bin").write_bytes(client)
        receipt = {
            "schema_version": executor.SHIELD_SCHEMA_VERSION,
            "spec_digest": spec.spec_digest,
            "qmax": 2,
            "domain_points": 15625,
            "security_level": 128,
            "input_shape": [6],
            "output_shape": [5, 2, 4],
            "requested_global_p_error": 1e-6,
            "compiled_global_p_error": 8e-7,
            "server_secret_key_markers": [],
            "server_artifact_sha256": hashlib.sha256(server).hexdigest(),
            "client_specs_sha256": hashlib.sha256(client).hexdigest(),
            "server_artifact_bytes": len(server),
            "client_specs_bytes": len(client),
        }
        (destination / "shield-receipt.json").write_text(json.dumps(receipt))

    class FakeClient:
        encryptions = 0

        @classmethod
        def from_path(cls, path: str | Path, spec: object) -> FakeClient:
            assert Path(path).name == "shield-client-specs.bin"
            return cls()

        def generate_keys(self) -> tuple[int, bytes]:
            return 11, b"evaluation-keys"

        def encrypt(self, quantized: np.ndarray) -> bytes:
            assert quantized.shape == (6,)
            assert np.all(np.abs(quantized) <= 2)
            type(self).encryptions += 1
            return f"ciphertext-{type(self).encryptions}".encode()

        def decrypt_margin_tensor(self, response: bytes) -> np.ndarray:
            assert response.startswith(b"response:")
            return tensor.copy()

        def select_action(
            self, margins: np.ndarray, requested_action: object, *, error_buffer: object
        ) -> SimpleNamespace:
            assert margins.shape == (5, 2, 4)
            return SimpleNamespace(action=requested_action)

    class FakeServer:
        def __init__(self, path: str | Path) -> None:
            assert Path(path).name == "shield-server.zip"

        def evaluate(self, request: bytes, evaluation_keys: bytes) -> bytes:
            assert evaluation_keys == b"evaluation-keys"
            return b"response:" + request

    monkeypatch.setattr(executor, "compile_shield", fake_compile)
    monkeypatch.setattr(executor, "ShieldFHEClient", FakeClient)
    monkeypatch.setattr(executor, "ShieldFHEServer", FakeServer)
    monkeypatch.setattr(executor, "clear_margin_tensor", lambda spec, point: tensor.copy())
    return compile_destinations


def test_valid_canary_pair_reuses_compiled_bundle_and_writes_sanitized_receipts(
    tmp_path: Path, monkeypatch: object
) -> None:
    destinations = _install_fake_bundle(monkeypatch)

    first = executor.execute_flagship_job(
        _manifest(), _job("job-canary-encryption-0", encryption=0), tmp_path
    )
    second = executor.execute_flagship_job(
        _manifest(), _job("job-canary-encryption-1", encryption=1), tmp_path
    )

    assert first["status"] == second["status"] == "succeeded"
    assert len(destinations) == 1
    assert (tmp_path / "shared" / "shield-fhe" / "shield-server.zip").is_file()
    first_bytes = (tmp_path / str(first["artifact_path"])).read_bytes()
    second_bytes = (tmp_path / str(second["artifact_path"])).read_bytes()
    assert hashlib.sha256(first_bytes).hexdigest() == first["artifact_digest"]
    assert hashlib.sha256(second_bytes).hexdigest() == second["artifact_digest"]
    first_receipt = json.loads(first_bytes)
    second_receipt = json.loads(second_bytes)
    assert first_receipt["accounting"] == {
        "valid_calls": 1,
        "call_attempts": 1,
        "call_successes": 1,
        "call_failures": 0,
        "decoded_margins": 40,
        "margin_matches": 40,
        "margin_mismatches": 0,
        "action_matches": 1,
        "action_mismatches": 0,
        "invalid_domain_rejections": 0,
    }
    assert first_receipt["execution"]["mode"] == "REAL FHE"
    assert first_receipt["execution"]["privacy_evidence"] is False
    assert first_receipt["execution"]["server_selected_action"] is False
    assert first_receipt["canary"]["pair_id"] == second_receipt["canary"]["pair_id"]
    assert (
        first_receipt["canary"]["ciphertext_sha256"]
        != second_receipt["canary"]["ciphertext_sha256"]
    )
    for receipt in (first_receipt, second_receipt):
        assert "state" not in receipt
        assert "quantized" not in receipt
        assert "margin_tensor" not in receipt
        assert receipt["call"]["output_matches_clear"] is True


def test_runtime_failure_is_retained_without_replacement(
    tmp_path: Path, monkeypatch: object
) -> None:
    _install_fake_bundle(monkeypatch)

    class FailingServer:
        def __init__(self, path: str | Path) -> None:
            assert Path(path).name == "shield-server.zip"

        def evaluate(self, request: bytes, evaluation_keys: bytes) -> bytes:
            del request, evaluation_keys
            raise RuntimeError("private backend diagnostic")

    monkeypatch.setattr(executor, "ShieldFHEServer", FailingServer)
    result = executor.execute_flagship_job(
        _manifest(),
        _job("job-retained-failure", category="occupancy"),
        tmp_path,
    )

    assert result["status"] == "succeeded"
    artifact = json.loads((tmp_path / str(result["artifact_path"])).read_bytes())
    assert artifact["accounting"] == {
        "valid_calls": 1,
        "call_attempts": 1,
        "call_successes": 0,
        "call_failures": 1,
        "decoded_margins": 0,
        "margin_matches": 0,
        "margin_mismatches": 0,
        "action_matches": 0,
        "action_mismatches": 0,
        "invalid_domain_rejections": 0,
    }
    assert artifact["call"] is None
    assert artifact["failure"] == {"code": "shield-fhe.runtimeerror"}
    assert "private backend diagnostic" not in json.dumps(artifact)


def test_invalid_domain_job_rejects_before_cache_or_encryption(
    tmp_path: Path, monkeypatch: object
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid jobs must reject before FHE setup")

    monkeypatch.setattr(executor, "_ensure_compiled_cache", forbidden)
    monkeypatch.setattr(executor, "ShieldFHEClient", forbidden)

    result = executor.execute_flagship_job(
        _manifest(), _job("job-invalid-domain", kind="invalid", case=1), tmp_path
    )

    assert result == {
        "status": "rejected",
        "artifact_path": None,
        "artifact_digest": None,
        "reason_code": "shield-fhe.invalid-domain",
    }
    assert not (tmp_path / "shield_fhe_challenge").exists()
    assert not (tmp_path / "shared").exists()
