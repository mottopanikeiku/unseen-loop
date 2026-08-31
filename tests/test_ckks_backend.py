from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import pytest

from unseen_loop.crypto import ckks
from unseen_loop.crypto.ckks import (
    CKKSClient,
    CKKSOperationReceipt,
    CKKSParameters,
    CKKSServer,
    CKKSUnavailableError,
    SerializedCKKSVector,
    evaluate_clear,
    generate_contexts,
)


def test_clear_oracle_is_deterministic_for_dot_square_and_reduction() -> None:
    values = (1.0, -2.0, 3.0)

    def polynomial(vector: ckks.ClearCKKSVector) -> ckks.ClearCKKSVector:
        squared_dot = vector.square().dot((0.5, 2.0, -1.0))
        centered_energy = (vector - (1.0, 1.0, 1.0)).square().reduce_sum()
        return squared_dot.square() + centered_energy * 0.25

    first = evaluate_clear(values, polynomial)
    second = evaluate_clear(values, polynomial)

    assert evaluate_clear(values, lambda vector: vector.sum_slots()) == (2.0,)
    assert first == second == (3.5,)


def test_serialized_vector_preserves_honest_logical_slot_count() -> None:
    original = SerializedCKKSVector(ciphertext=b"opaque-tenseal-ciphertext", slots=17)

    restored = SerializedCKKSVector.from_bytes(original.to_bytes())

    assert restored == original
    assert restored.slots == 17
    assert restored.sha256 == original.sha256
    with pytest.raises(ValueError, match="format marker"):
        SerializedCKKSVector.from_bytes(b"BADMAG!" + original.to_bytes()[7:])
    with pytest.raises(ValueError, match="truncated"):
        SerializedCKKSVector.from_bytes(b"")


def test_server_rejects_any_context_reporting_a_secret_key() -> None:
    class PrivateContext:
        @staticmethod
        def is_private() -> bool:
            return True

    with pytest.raises(ValueError, match="rejects contexts containing a secret key"):
        CKKSServer(object(), PrivateContext(), CKKSParameters())


def test_parameters_reject_coeff_modulus_above_tc128_budget() -> None:
    with pytest.raises(ValueError, match="tc128 limit of 109 bits"):
        CKKSParameters(
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=(40, 30, 40),
            global_scale=float(2**30),
        )


def test_client_rejects_tenseal_multi_ciphertext_chunking() -> None:
    class PrivateContext:
        @staticmethod
        def is_private() -> bool:
            return True

    parameters = CKKSParameters(
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=(40, 20, 40),
        global_scale=float(2**20),
    )
    client = CKKSClient(object(), PrivateContext(), parameters)

    with pytest.raises(ValueError, match="context capacity is 2048"):
        client.encrypt(np.zeros(2049, dtype=np.float64))


def test_missing_tenseal_raises_explicit_unavailable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def import_without_tenseal(name: str, package: str | None = None) -> object:
        if name == "tenseal":
            raise ImportError("deliberately unavailable")
        return real_import(name, package)

    monkeypatch.setattr(ckks.importlib, "import_module", import_without_tenseal)

    with pytest.raises(CKKSUnavailableError, match="no clear backend will be substituted"):
        generate_contexts()


def test_operation_receipt_persists_no_plaintext_or_decrypted_vector() -> None:
    receipt = CKKSOperationReceipt(
        operation="evaluate",
        elapsed_ns=10,
        input_bytes=100,
        output_bytes=80,
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        input_slots=32,
        output_slots=4,
    )

    persisted = dataclasses.asdict(receipt)

    assert not {"plaintext", "clear", "decrypted", "values", "secret_key"} & persisted.keys()
    assert persisted["schema_version"] == "unseen-loop/ckks-operation-receipt-v1"
    assert "server observes" in persisted["trust_scope"]


@pytest.mark.fhe
@pytest.mark.slow
def test_real_ckks_roundtrip_uses_public_serialized_server_context() -> None:
    tenseal = pytest.importorskip("tenseal")
    parameters = CKKSParameters(
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=(40, 20, 40),
        global_scale=float(2**20),
    )
    artifacts = generate_contexts(parameters)

    assert tenseal.context_from(artifacts.client_context).is_private()
    assert not tenseal.context_from(artifacts.server_context).is_private()
    assert not artifacts.receipt.server_context_is_private
    assert artifacts.receipt.server_context_sha256 != artifacts.receipt.client_context_sha256

    client = CKKSClient.from_serialized(artifacts.client_context, parameters=parameters)
    server = CKKSServer.from_serialized(artifacts.server_context, parameters=parameters)
    encrypted, encrypt_receipt = client.encrypt((1.0, -2.0, 3.0))
    evaluated, evaluation_receipt = server.evaluate(
        encrypted,
        lambda vector: vector.square().dot((0.5, 2.0, -1.0)) + 1.25,
    )
    decrypted, decrypt_receipt = client.decrypt(evaluated)

    assert decrypted == pytest.approx((0.75,), abs=2e-3)
    assert encrypt_receipt.input_sha256 is None
    assert evaluation_receipt.input_sha256 == encrypted.sha256
    assert evaluation_receipt.output_sha256 == evaluated.sha256
    assert decrypt_receipt.output_sha256 is None
    assert np.isfinite(decrypted).all()
