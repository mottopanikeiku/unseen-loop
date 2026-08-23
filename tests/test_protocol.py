from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json

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
        evaluation_key_digest="evaluation-key",
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
    roundtripped = SignedEnvelope.from_json(signed.to_json())
    json_verified = authenticator.verify(roundtripped, RequestEnvelope)
    assert json_verified.observation_shape == (4,)
    assert roundtripped.to_json() == signed.to_json()

    response = ResponseEnvelope.create(request(), b"response", output_shape=(2,))
    signed_response = authenticator.sign(response)
    response_roundtrip = SignedEnvelope.from_json(signed_response.to_json())
    verified_response = authenticator.verify(response_roundtrip, ResponseEnvelope)
    assert verified_response == response
    assert response_roundtrip.to_json() == signed_response.to_json()

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


def test_context_version_and_duplicate_fields_are_rejected() -> None:
    with pytest.raises(ProtocolError, match="evaluation key"):
        guard().validate_request(dataclasses.replace(request(), evaluation_key_digest="wrong-key"))
    with pytest.raises(ProtocolError, match="schema version"):
        guard().validate_request(
            dataclasses.replace(request(), schema_version="unseen-loop/request-v0")
        )
    duplicate = (
        '{"algorithm":"HMAC-SHA256","algorithm":"none","authentication_tag":"x","payload":{}}'
    )
    with pytest.raises(ProtocolError, match="duplicate"):
        SignedEnvelope.from_json(duplicate)


def test_strict_json_rejects_wrong_wrapper_types_and_non_finite_values() -> None:
    wrong_algorithm_type = (
        '{"algorithm":1,"authentication_tag":"x","payload":{"observation_shape":[4]}}'
    )
    with pytest.raises(ProtocolError, match="algorithm must be a string"):
        SignedEnvelope.from_json(wrong_algorithm_type)

    with pytest.raises(ProtocolError, match="non-finite"):
        SignedEnvelope.from_json(
            '{"algorithm":"HMAC-SHA256","authentication_tag":"x","payload":{"value":NaN}}'
        )


def test_response_context_and_version_are_bound_to_request() -> None:
    incoming = request()
    response = ResponseEnvelope.create(incoming, b"response", output_shape=(2,))
    boundary = guard()

    with pytest.raises(ProtocolError, match="nonce"):
        boundary.validate_response(incoming, dataclasses.replace(response, nonce="other"))
    with pytest.raises(ProtocolError, match="circuit"):
        boundary.validate_response(
            incoming,
            dataclasses.replace(response, circuit_digest="other-circuit"),
        )
    with pytest.raises(ProtocolError, match="client context"):
        wrong_context = dataclasses.replace(incoming, client_context_digest="other-context")
        boundary.validate_response(
            wrong_context,
            ResponseEnvelope.create(wrong_context, b"response", output_shape=(2,)),
        )
    with pytest.raises(ProtocolError, match="evaluation key"):
        wrong_key = dataclasses.replace(incoming, evaluation_key_digest="other-key")
        boundary.validate_response(
            wrong_key,
            ResponseEnvelope.create(wrong_key, b"response", output_shape=(2,)),
        )
    with pytest.raises(ProtocolError, match="schema version"):
        boundary.validate_response(
            incoming,
            dataclasses.replace(response, schema_version="unseen-loop/response-v0"),
        )


def test_authenticated_payload_schema_is_exact() -> None:
    key = b"a" * 32
    authenticator = TranscriptAuthenticator(key)
    payload = dict(authenticator.sign(request()).payload)
    payload["unknown"] = "field"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signed = SignedEnvelope(
        payload=payload,
        authentication_tag=hmac.new(key, canonical, hashlib.sha256).hexdigest(),
    )
    with pytest.raises(ProtocolError, match="schema"):
        authenticator.verify(signed, RequestEnvelope)
