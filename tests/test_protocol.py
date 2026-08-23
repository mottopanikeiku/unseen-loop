from __future__ import annotations

import dataclasses

import pytest

from unseen_loop.protocol import (
    FixedShapeGuard,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    SignedEnvelope,
    TranscriptAuthenticator,
)


def request(payload: bytes = b"ciphertext") -> RequestEnvelope:
    return RequestEnvelope.create(
        payload,
        policy_digest="policy",
        circuit_digest="circuit",
        client_context_digest="context",
        evaluation_key_digest="evaluation-key",
        observation_shape=(4,),
    )


def guard(request_bytes: int = 10, response_bytes: int = 8) -> FixedShapeGuard:
    return FixedShapeGuard(
        policy_digest="policy",
        circuit_digest="circuit",
        client_context_digest="context",
        observation_shape=(4,),
        output_shape=(2,),
        request_bytes=request_bytes,
        response_bytes=response_bytes,
    )


def test_request_and_response_are_bound_and_fixed_shape() -> None:
    incoming = request()
    boundary = guard()
    assert boundary.validate_request(incoming) == b"ciphertext"
    response = ResponseEnvelope.create(incoming, b"response", output_shape=(2,))
    assert boundary.validate_response(incoming, response) == b"response"


def test_replay_is_rejected() -> None:
    incoming = request()
    boundary = guard()
    boundary.validate_request(incoming)
    with pytest.raises(ProtocolError, match="replay"):
        boundary.validate_request(incoming)


def test_policy_downgrade_and_response_swap_are_rejected() -> None:
    incoming = request()
    downgraded = dataclasses.replace(incoming, policy_digest="old-policy")
    with pytest.raises(ProtocolError, match="downgrade"):
        guard().validate_request(downgraded)

    other = request()
    response = ResponseEnvelope.create(other, b"response", output_shape=(2,))
    with pytest.raises(ProtocolError, match="another request"):
        guard().validate_response(incoming, response)


def test_hmac_authentication_detects_mutation() -> None:
    authenticator = TranscriptAuthenticator(b"a" * 32)
    signed = authenticator.sign(request())
    verified = authenticator.verify(signed, RequestEnvelope)
    assert isinstance(verified, RequestEnvelope)

    modified_payload = dict(signed.payload)
    modified_payload["policy_digest"] = "attacker"
    modified = SignedEnvelope(
        payload=modified_payload,
        authentication_tag=signed.authentication_tag,
    )
    with pytest.raises(ProtocolError, match="authentication"):
        authenticator.verify(modified, RequestEnvelope)


def test_length_mismatch_fails_before_payload_use() -> None:
    incoming = request(b"short")
    with pytest.raises(ProtocolError, match="fixed transcript"):
        guard().validate_request(incoming)
