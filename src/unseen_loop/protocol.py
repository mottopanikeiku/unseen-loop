"""Fixed-shape authenticated transcript envelopes for encrypted policy requests.

Authentication detects replay, substitution, and transport corruption. It does not
prove that a malicious evaluator ran the committed FHE circuit.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass
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
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @property
    def ciphertext_raw(self) -> bytes:
        try:
            payload = base64.b64decode(self.ciphertext, validate=True)
        except ValueError as error:
            raise ProtocolError("request ciphertext is not canonical base64") from error
        if len(payload) != self.ciphertext_bytes:
            raise ProtocolError("request ciphertext length does not match envelope")
        return payload

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
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @property
    def ciphertext_raw(self) -> bytes:
        try:
            payload = base64.b64decode(self.ciphertext, validate=True)
        except ValueError as error:
            raise ProtocolError("response ciphertext is not canonical base64") from error
        if len(payload) != self.ciphertext_bytes:
            raise ProtocolError("response ciphertext length does not match envelope")
        return payload

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
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class TranscriptAuthenticator:
    """Operational transcript authentication; deliberately not an evaluation proof."""

    def __init__(self, authentication_key: bytes) -> None:
        if len(authentication_key) < 32:
            raise ValueError("authentication key must contain at least 256 bits")
        self._key = bytes(authentication_key)

    def sign(self, envelope: RequestEnvelope | ResponseEnvelope) -> SignedEnvelope:
        payload = asdict(envelope)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        tag = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        return SignedEnvelope(payload=payload, authentication_tag=tag)

    def verify(self, signed: SignedEnvelope, expected_type: type[RequestEnvelope] | type[ResponseEnvelope]) -> RequestEnvelope | ResponseEnvelope:
        if signed.algorithm != "HMAC-SHA256":
            raise ProtocolError("unsupported envelope authentication algorithm")
        canonical = json.dumps(signed.payload, sort_keys=True, separators=(",", ":")).encode()
        expected_tag = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signed.authentication_tag, expected_tag):
            raise ProtocolError("envelope authentication failed")
        try:
            envelope = expected_type(**signed.payload)
        except (TypeError, ValueError) as error:
            raise ProtocolError("envelope schema is invalid") from error
        return envelope


class FixedShapeGuard:
    """Fail-closed validation for context, tensor shape, size, freshness, and replay."""

    def __init__(
        self,
        *,
        policy_digest: str,
        circuit_digest: str,
        client_context_digest: str,
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
        self.observation_shape = observation_shape
        self.output_shape = output_shape
        self.request_bytes = request_bytes
        self.response_bytes = response_bytes
        self.max_clock_skew_ns = max_clock_skew_ns
        self._seen_nonces: set[str] = set()

    def validate_request(self, request: RequestEnvelope, *, now_ns: int | None = None) -> bytes:
        now = now_ns if now_ns is not None else time.time_ns()
        if request.policy_digest != self.policy_digest:
            raise ProtocolError("policy downgrade or substitution")
        if request.circuit_digest != self.circuit_digest:
            raise ProtocolError("circuit substitution")
        if request.client_context_digest != self.client_context_digest:
            raise ProtocolError("client context mismatch")
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
