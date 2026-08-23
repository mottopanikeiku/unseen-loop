"""Fixed-shape authenticated transcript envelopes for encrypted policy requests.

Authentication detects replay, substitution, and transport corruption. It does not
prove that a malicious evaluator ran the committed FHE circuit.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass, fields
from typing import Any


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RequestEnvelope:
    request_id: str
    nonce: str
    created_unix_ns: int
    policy_digest: str
    circuit_digest: str
    client_context_digest: str
    evaluation_key_digest: str
    observation_shape: tuple[int, ...]
    ciphertext: str
    ciphertext_bytes: int
    schema_version: str = "unseen-loop/request-v1"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def ciphertext_raw(self) -> bytes:
        return _decode_ciphertext(self.ciphertext, self.ciphertext_bytes, "request")

    @classmethod
    def create(
        cls,
        ciphertext: bytes,
        *,
        policy_digest: str,
        circuit_digest: str,
        client_context_digest: str,
        evaluation_key_digest: str,
        observation_shape: tuple[int, ...],
    ) -> RequestEnvelope:
        return cls(
            request_id=secrets.token_hex(16),
            nonce=secrets.token_hex(24),
            created_unix_ns=time.time_ns(),
            policy_digest=policy_digest,
            circuit_digest=circuit_digest,
            client_context_digest=client_context_digest,
            evaluation_key_digest=evaluation_key_digest,
            observation_shape=observation_shape,
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            ciphertext_bytes=len(ciphertext),
        )


@dataclass(frozen=True)
class ResponseEnvelope:
    request_digest: str
    request_id: str
    nonce: str
    policy_digest: str
    circuit_digest: str
    output_shape: tuple[int, ...]
    ciphertext: str
    ciphertext_bytes: int
    status: str
    completed_unix_ns: int
    schema_version: str = "unseen-loop/response-v1"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def ciphertext_raw(self) -> bytes:
        return _decode_ciphertext(self.ciphertext, self.ciphertext_bytes, "response")

    @classmethod
    def create(
        cls,
        request: RequestEnvelope,
        ciphertext: bytes,
        *,
        output_shape: tuple[int, ...],
    ) -> ResponseEnvelope:
        return cls(
            request_digest=request.digest,
            request_id=request.request_id,
            nonce=request.nonce,
            policy_digest=request.policy_digest,
            circuit_digest=request.circuit_digest,
            output_shape=output_shape,
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            ciphertext_bytes=len(ciphertext),
            status="ok",
            completed_unix_ns=time.time_ns(),
        )


@dataclass(frozen=True)
class SignedEnvelope:
    payload: dict[str, Any]
    authentication_tag: str
    algorithm: str = "HMAC-SHA256"

    def to_json(self) -> str:
        return _canonical_json(asdict(self)).decode("utf-8")

    @classmethod
    def from_json(cls, value: str) -> SignedEnvelope:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ProtocolError(f"duplicate envelope field: {key}")
                result[key] = item
            return result

        def reject_constant(value: str) -> None:
            raise ProtocolError(f"non-finite JSON constant is forbidden: {value}")

        if not isinstance(value, str):
            raise ProtocolError("signed envelope must be encoded as JSON text")
        try:
            raw = json.loads(
                value,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, TypeError) as error:
            raise ProtocolError("signed envelope is not valid JSON") from error
        if not isinstance(raw, dict) or set(raw) != {
            "payload",
            "authentication_tag",
            "algorithm",
        }:
            raise ProtocolError("signed envelope wrapper schema is invalid")
        if not isinstance(raw["payload"], dict):
            raise ProtocolError("signed envelope payload must be an object")
        if not isinstance(raw["authentication_tag"], str):
            raise ProtocolError("envelope authentication tag must be a string")
        if not isinstance(raw["algorithm"], str):
            raise ProtocolError("envelope authentication algorithm must be a string")
        return cls(
            payload=raw["payload"],
            authentication_tag=raw["authentication_tag"],
            algorithm=raw["algorithm"],
        )


class TranscriptAuthenticator:
    """Operational transcript authentication; deliberately not an evaluation proof."""

    def __init__(self, authentication_key: bytes) -> None:
        if len(authentication_key) < 32:
            raise ValueError("authentication key must contain at least 256 bits")
        self._key = bytes(authentication_key)

    def sign(self, envelope: RequestEnvelope | ResponseEnvelope) -> SignedEnvelope:
        if type(envelope) not in (RequestEnvelope, ResponseEnvelope):
            raise TypeError("only request and response envelopes can be authenticated")
        payload = asdict(envelope)
        tag = hmac.new(self._key, _canonical_json(payload), hashlib.sha256).hexdigest()
        return SignedEnvelope(payload=payload, authentication_tag=tag)

    def verify(
        self, signed: SignedEnvelope, expected_type: type[RequestEnvelope] | type[ResponseEnvelope]
    ) -> RequestEnvelope | ResponseEnvelope:
        if expected_type not in (RequestEnvelope, ResponseEnvelope):
            raise TypeError("expected_type must be RequestEnvelope or ResponseEnvelope")
        if signed.algorithm != "HMAC-SHA256":
            raise ProtocolError("unsupported envelope authentication algorithm")
        if not isinstance(signed.authentication_tag, str):
            raise ProtocolError("envelope authentication tag must be a string")
        expected_tag = hmac.new(
            self._key,
            _canonical_json(signed.payload),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signed.authentication_tag, expected_tag):
            raise ProtocolError("envelope authentication failed")
        if not isinstance(signed.payload, dict):
            raise ProtocolError("signed envelope payload must be an object")

        payload = dict(signed.payload)
        expected_fields = {field.name for field in fields(expected_type)}
        if set(payload) != expected_fields:
            raise ProtocolError("envelope schema is invalid")
        shape_field = "observation_shape" if expected_type is RequestEnvelope else "output_shape"
        shape = payload.get(shape_field)
        if (
            not isinstance(shape, (list, tuple))
            or not shape
            or not all(type(value) is int and value > 0 for value in shape)
        ):
            raise ProtocolError("envelope shape is invalid")
        payload[shape_field] = tuple(shape)
        _validate_payload_types(payload, expected_type, shape_field)
        try:
            envelope = expected_type(**payload)
        except (TypeError, ValueError) as error:
            raise ProtocolError("envelope schema is invalid") from error
        _ = envelope.ciphertext_raw
        return envelope


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError("envelope contains a non-JSON value") from error


def _decode_ciphertext(value: str, expected_bytes: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} ciphertext must be a base64 string")
    if type(expected_bytes) is not int or expected_bytes < 1:
        raise ProtocolError(f"{label} ciphertext length is invalid")
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProtocolError(f"{label} ciphertext is not canonical base64") from error
    if base64.b64encode(payload).decode("ascii") != value:
        raise ProtocolError(f"{label} ciphertext is not canonical base64")
    if len(payload) != expected_bytes:
        raise ProtocolError(f"{label} ciphertext length does not match envelope")
    return payload


def _validate_payload_types(
    payload: dict[str, Any],
    expected_type: type[RequestEnvelope] | type[ResponseEnvelope],
    shape_field: str,
) -> None:
    integer_fields = (
        {"created_unix_ns", "ciphertext_bytes"}
        if expected_type is RequestEnvelope
        else {"ciphertext_bytes", "completed_unix_ns"}
    )
    for name, value in payload.items():
        if name == shape_field:
            continue
        if name in integer_fields:
            if type(value) is not int or value < 1:
                raise ProtocolError(f"envelope field {name} must be a positive integer")
        elif not isinstance(value, str):
            raise ProtocolError(f"envelope field {name} must be a string")


class FixedShapeGuard:
    """Fail-closed validation for context, tensor shape, size, freshness, and replay."""

    def __init__(
        self,
        *,
        policy_digest: str,
        circuit_digest: str,
        client_context_digest: str,
        evaluation_key_digest: str,
        observation_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        request_bytes: int,
        response_bytes: int,
        max_clock_skew_ns: int = 300_000_000_000,
    ) -> None:
        if request_bytes < 1 or response_bytes < 1:
            raise ValueError("fixed transcript byte lengths must be positive")
        self.policy_digest = policy_digest
        self.circuit_digest = circuit_digest
        self.client_context_digest = client_context_digest
        self.evaluation_key_digest = evaluation_key_digest
        self.observation_shape = observation_shape
        self.output_shape = output_shape
        self.request_bytes = request_bytes
        self.response_bytes = response_bytes
        self.max_clock_skew_ns = max_clock_skew_ns
        self._seen_nonces: set[str] = set()

    def validate_request(self, request: RequestEnvelope, *, now_ns: int | None = None) -> bytes:
        now = now_ns if now_ns is not None else time.time_ns()
        if request.schema_version != "unseen-loop/request-v1":
            raise ProtocolError("unsupported request schema version")
        if request.policy_digest != self.policy_digest:
            raise ProtocolError("policy downgrade or substitution")
        if request.circuit_digest != self.circuit_digest:
            raise ProtocolError("circuit substitution")
        if request.client_context_digest != self.client_context_digest:
            raise ProtocolError("client context mismatch")
        if request.evaluation_key_digest != self.evaluation_key_digest:
            raise ProtocolError("evaluation key context mismatch")
        if request.observation_shape != self.observation_shape:
            raise ProtocolError("observation shape mismatch")
        if request.ciphertext_bytes != self.request_bytes:
            raise ProtocolError("request violates the fixed transcript length")
        if abs(now - request.created_unix_ns) > self.max_clock_skew_ns:
            raise ProtocolError("request is stale or has an invalid clock")
        if request.nonce in self._seen_nonces:
            raise ProtocolError("request replay detected")
        ciphertext = request.ciphertext_raw
        self._seen_nonces.add(request.nonce)
        return ciphertext

    def validate_response(
        self,
        request: RequestEnvelope,
        response: ResponseEnvelope,
    ) -> bytes:
        if request.schema_version != "unseen-loop/request-v1":
            raise ProtocolError("unsupported request schema version")
        if request.client_context_digest != self.client_context_digest:
            raise ProtocolError("response request client context mismatch")
        if request.evaluation_key_digest != self.evaluation_key_digest:
            raise ProtocolError("response request evaluation key context mismatch")
        if request.observation_shape != self.observation_shape:
            raise ProtocolError("response request observation shape mismatch")
        if response.schema_version != "unseen-loop/response-v1":
            raise ProtocolError("unsupported response schema version")
        if response.status != "ok":
            raise ProtocolError("server returned a non-success status")
        if response.request_digest != request.digest or response.request_id != request.request_id:
            raise ProtocolError("response is bound to another request")
        if response.nonce != request.nonce:
            raise ProtocolError("response nonce mismatch")
        if response.policy_digest != self.policy_digest:
            raise ProtocolError("response policy mismatch")
        if response.circuit_digest != self.circuit_digest:
            raise ProtocolError("response circuit mismatch")
        if response.output_shape != self.output_shape:
            raise ProtocolError("response output shape mismatch")
        if response.ciphertext_bytes != self.response_bytes:
            raise ProtocolError("response violates the fixed transcript length")
        return response.ciphertext_raw
