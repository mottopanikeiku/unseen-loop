"""Optional TenSEAL CKKS primitives with an explicit client/server boundary.

The server API accepts only a serialized public context.  CKKS is approximate.
``slots`` is the logical number of packed real values; vectors larger than the
single-ciphertext capacity are rejected rather than silently chunked.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt


class CKKSUnavailableError(RuntimeError):
    """Raised when encrypted execution is requested without TenSEAL installed."""


@dataclass(frozen=True)
class CKKSParameters:
    """TenSEAL parameters for approximate packed-real arithmetic."""

    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60)
    global_scale: float = float(2**40)

    def __post_init__(self) -> None:
        if self.poly_modulus_degree < 4096 or self.poly_modulus_degree & (
            self.poly_modulus_degree - 1
        ):
            raise ValueError("poly_modulus_degree must be a power of two of at least 4096")
        if len(self.coeff_mod_bit_sizes) < 3 or any(bits <= 0 for bits in self.coeff_mod_bit_sizes):
            raise ValueError("coeff_mod_bit_sizes must contain at least three positive values")
        if not math.isfinite(self.global_scale) or self.global_scale <= 1.0:
            raise ValueError("global_scale must be finite and greater than one")

    @property
    def slot_capacity(self) -> int:
        """Maximum complex CKKS slots; this backend uses their real components."""

        return self.poly_modulus_degree // 2


@dataclass(frozen=True)
class CKKSContextReceipt:
    """Closed metadata for generated client and server context artifacts."""

    parameters: CKKSParameters
    tenseal_version: str
    keygen_ns: int
    client_context_bytes: int
    server_context_bytes: int
    client_context_sha256: str
    server_context_sha256: str
    server_context_is_private: bool
    backend: str = "TenSEAL CKKS"
    mode: str = "REAL FHE (approximate arithmetic)"
    trust_scope: str = (
        "client context contains the secret key; server context contains only public, "
        "relinearization, and Galois key material"
    )
    security_claim: str = (
        "cryptographic parameters are recorded; this module asserts no numeric security level"
    )
    packing_scope: str = (
        "one logical real vector per ciphertext; inputs above degree/2 slots are rejected"
    )
    schema_version: str = "unseen-loop/ckks-context-receipt-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


@dataclass(frozen=True)
class CKKSOperationReceipt:
    """Timing and transport metadata without plaintext or decrypted values."""

    operation: str
    elapsed_ns: int
    input_bytes: int
    output_bytes: int
    input_sha256: str | None
    output_sha256: str | None
    input_slots: int
    output_slots: int
    backend: str = "TenSEAL CKKS"
    mode: str = "REAL FHE (approximate arithmetic)"
    trust_scope: str = (
        "server observes the public context, circuit, ciphertext sizes, and timing; "
        "plaintext values and the secret key remain client-side"
    )
    schema_version: str = "unseen-loop/ckks-operation-receipt-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


@dataclass(frozen=True)
class SerializedCKKSVector:
    """A TenSEAL ciphertext plus its honest logical packed-vector length."""

    ciphertext: bytes = field(repr=False)
    slots: int

    _MAGIC = b"ULCKKS1"
    _HEADER = struct.Struct(">7sQ")

    def __post_init__(self) -> None:
        if not isinstance(self.ciphertext, bytes) or not self.ciphertext:
            raise ValueError("ciphertext must be non-empty bytes")
        if self.slots < 1:
            raise ValueError("slots must be positive")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        """Serialize a self-describing transport envelope."""

        return self._HEADER.pack(self._MAGIC, self.slots) + self.ciphertext

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        """Deserialize an envelope without loading or decrypting its ciphertext."""

        if not isinstance(payload, bytes) or len(payload) <= cls._HEADER.size:
            raise ValueError("serialized CKKS vector is truncated")
        magic, slots = cls._HEADER.unpack_from(payload)
        if magic != cls._MAGIC:
            raise ValueError("serialized CKKS vector has an invalid format marker")
        return cls(ciphertext=payload[cls._HEADER.size :], slots=slots)


@dataclass(frozen=True)
class CKKSContextArtifacts:
    """Serialized split contexts; only ``client_context`` contains a secret key."""

    client_context: bytes = field(repr=False)
    server_context: bytes = field(repr=False)
    receipt: CKKSContextReceipt


class ClearCKKSVector:
    """Deterministic clear oracle mirroring the supported packed operations."""

    __slots__ = ("_values",)

    def __init__(self, values: npt.ArrayLike) -> None:
        array = _real_vector(values, name="values")
        self._values = tuple(float(value) for value in array)

    @property
    def slots(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    def dot(self, weights: npt.ArrayLike) -> ClearCKKSVector:
        public = _real_vector(weights, name="weights", slots=self.slots)
        products = (x * float(y) for x, y in zip(self._values, public, strict=True))
        return ClearCKKSVector((math.fsum(products),))

    def square(self) -> ClearCKKSVector:
        return ClearCKKSVector(tuple(value * value for value in self._values))

    def reduce_sum(self) -> ClearCKKSVector:
        return ClearCKKSVector((math.fsum(self._values),))

    def sum_slots(self) -> ClearCKKSVector:
        return self.reduce_sum()

    def __add__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return ClearCKKSVector(_binary_clear(self, other, lambda left, right: left + right))

    def __radd__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return self + other

    def __sub__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return ClearCKKSVector(_binary_clear(self, other, lambda left, right: left - right))

    def __rsub__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return ClearCKKSVector(_binary_clear(self, other, lambda left, right: right - left))

    def __mul__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return ClearCKKSVector(_binary_clear(self, other, lambda left, right: left * right))

    def __rmul__(self, other: float | npt.ArrayLike | ClearCKKSVector) -> ClearCKKSVector:
        return self * other


class CKKSEncryptedVector:
    """Server-side packed ciphertext supporting only polynomial operations."""

    __slots__ = ("_owner", "_slots", "_vector")

    def __init__(self, vector: Any, slots: int, owner: object) -> None:
        self._vector = vector
        self._slots = slots
        self._owner = owner

    @property
    def slots(self) -> int:
        return self._slots

    def dot(self, weights: npt.ArrayLike) -> CKKSEncryptedVector:
        public = _real_vector(weights, name="weights", slots=self.slots)
        return self._new(self._vector.dot(public.tolist()), slots=1)

    def square(self) -> CKKSEncryptedVector:
        return self._new(self._vector.square(), slots=self.slots)

    def reduce_sum(self) -> CKKSEncryptedVector:
        return self._new(self._vector.sum(), slots=1)

    def sum_slots(self) -> CKKSEncryptedVector:
        return self.reduce_sum()

    def __add__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        operand = self._operand(other)
        return self._new(self._vector + operand, slots=self.slots)

    def __radd__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        return self + other

    def __sub__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        operand = self._operand(other)
        return self._new(self._vector - operand, slots=self.slots)

    def __rsub__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        if isinstance(other, CKKSEncryptedVector):
            other._require_same_owner_and_slots(self)
            return self._new(other._vector - self._vector, slots=self.slots)
        operand = _public_operand(other, self.slots)
        return self._new((self._vector * -1.0) + operand, slots=self.slots)

    def __mul__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        operand = self._operand(other)
        return self._new(self._vector * operand, slots=self.slots)

    def __rmul__(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> CKKSEncryptedVector:
        return self * other

    def _operand(self, other: float | npt.ArrayLike | CKKSEncryptedVector) -> Any:
        if isinstance(other, CKKSEncryptedVector):
            self._require_same_owner_and_slots(other)
            return other._vector
        return _public_operand(other, self.slots)

    def _require_same_owner_and_slots(self, other: CKKSEncryptedVector) -> None:
        if self._owner is not other._owner:
            raise ValueError("cannot combine ciphertexts from different server contexts")
        if self.slots != other.slots:
            raise ValueError("packed ciphertext operands must have the same logical slots")

    def _new(self, vector: Any, *, slots: int) -> CKKSEncryptedVector:
        return CKKSEncryptedVector(vector, slots, self._owner)


class CKKSClient:
    """Client holding the private TenSEAL context for encryption and decryption."""

    def __init__(self, tenseal: Any, context: Any, parameters: CKKSParameters) -> None:
        if not _context_is_private(context):
            raise ValueError("CKKSClient requires a context containing the secret key")
        self._tenseal = tenseal
        self._context = context
        self.parameters = parameters

    @classmethod
    def from_serialized(cls, payload: bytes, *, parameters: CKKSParameters) -> CKKSClient:
        tenseal = _import_tenseal()
        context = tenseal.context_from(payload)
        return cls(tenseal, context, parameters)

    def encrypt(self, values: npt.ArrayLike) -> tuple[SerializedCKKSVector, CKKSOperationReceipt]:
        clear = _real_vector(values, name="values")
        _require_single_ciphertext(int(clear.size), self.parameters)
        started = time.perf_counter_ns()
        vector = self._tenseal.ckks_vector(self._context, clear.tolist())
        result = SerializedCKKSVector(vector.serialize(), int(clear.size))
        elapsed = time.perf_counter_ns() - started
        encoded = result.to_bytes()
        receipt = CKKSOperationReceipt(
            operation="encrypt",
            elapsed_ns=elapsed,
            input_bytes=int(clear.nbytes),
            output_bytes=len(encoded),
            input_sha256=None,
            output_sha256=hashlib.sha256(encoded).hexdigest(),
            input_slots=int(clear.size),
            output_slots=result.slots,
            trust_scope="plaintext input and secret key remain client-side",
        )
        return result, receipt

    def decrypt(
        self, value: SerializedCKKSVector
    ) -> tuple[npt.NDArray[np.float64], CKKSOperationReceipt]:
        _require_single_ciphertext(value.slots, self.parameters)
        encoded = value.to_bytes()
        started = time.perf_counter_ns()
        vector = self._tenseal.ckks_vector_from(self._context, value.ciphertext)
        _require_loaded_slots(vector, value.slots)
        clear = np.asarray(vector.decrypt(), dtype=np.float64)
        elapsed = time.perf_counter_ns() - started
        if clear.shape != (value.slots,):
            raise ValueError("decrypted vector length does not match its declared logical slots")
        receipt = CKKSOperationReceipt(
            operation="decrypt",
            elapsed_ns=elapsed,
            input_bytes=len(encoded),
            output_bytes=int(clear.nbytes),
            input_sha256=hashlib.sha256(encoded).hexdigest(),
            output_sha256=None,
            input_slots=value.slots,
            output_slots=int(clear.size),
            trust_scope="ciphertext is decrypted only inside the secret-key client boundary",
        )
        return clear, receipt


class CKKSServer:
    """Evaluator constructed only from a public TenSEAL context."""

    def __init__(self, tenseal: Any, context: Any, parameters: CKKSParameters) -> None:
        if _context_is_private(context):
            raise ValueError("CKKSServer rejects contexts containing a secret key")
        self._tenseal = tenseal
        self._context = context
        self.parameters = parameters
        self._owner = object()

    @classmethod
    def from_serialized(cls, payload: bytes, *, parameters: CKKSParameters) -> CKKSServer:
        tenseal = _import_tenseal()
        context = tenseal.context_from(payload)
        return cls(tenseal, context, parameters)

    def evaluate(
        self,
        value: SerializedCKKSVector,
        evaluator: Callable[[CKKSEncryptedVector], CKKSEncryptedVector],
    ) -> tuple[SerializedCKKSVector, CKKSOperationReceipt]:
        """Deserialize, evaluate a polynomial callback, and serialize the result."""
        _require_single_ciphertext(value.slots, self.parameters)

        encoded_input = value.to_bytes()
        started = time.perf_counter_ns()
        vector = self._tenseal.ckks_vector_from(self._context, value.ciphertext)
        _require_loaded_slots(vector, value.slots)
        encrypted = CKKSEncryptedVector(vector, value.slots, self._owner)
        evaluated = evaluator(encrypted)
        if not isinstance(evaluated, CKKSEncryptedVector):
            raise TypeError("evaluator must return CKKSEncryptedVector")
        if evaluated._owner is not self._owner:
            raise ValueError("evaluator returned a ciphertext from another server context")
        result = SerializedCKKSVector(evaluated._vector.serialize(), evaluated.slots)
        elapsed = time.perf_counter_ns() - started
        encoded_output = result.to_bytes()
        receipt = CKKSOperationReceipt(
            operation="evaluate",
            elapsed_ns=elapsed,
            input_bytes=len(encoded_input),
            output_bytes=len(encoded_output),
            input_sha256=hashlib.sha256(encoded_input).hexdigest(),
            output_sha256=hashlib.sha256(encoded_output).hexdigest(),
            input_slots=value.slots,
            output_slots=result.slots,
        )
        return result, receipt


def generate_contexts(parameters: CKKSParameters | None = None) -> CKKSContextArtifacts:
    """Generate split serialized contexts and prove the server artifact is public on reload."""
    if parameters is None:
        parameters = CKKSParameters()

    tenseal = _import_tenseal()
    started = time.perf_counter_ns()
    context = tenseal.context(
        tenseal.SCHEME_TYPE.CKKS,
        poly_modulus_degree=parameters.poly_modulus_degree,
        coeff_mod_bit_sizes=list(parameters.coeff_mod_bit_sizes),
    )
    context.global_scale = parameters.global_scale
    context.auto_relin = True
    context.auto_rescale = True
    context.auto_mod_switch = True
    context.generate_relin_keys()
    context.generate_galois_keys()
    keygen_ns = time.perf_counter_ns() - started
    client_context = _serialize_context(context, save_secret_key=True)

    public_context = tenseal.context_from(client_context)
    public_context.make_context_public()
    server_context = _serialize_context(public_context, save_secret_key=False)
    reloaded_server = tenseal.context_from(server_context)
    server_is_private = _context_is_private(reloaded_server)
    if server_is_private:
        raise RuntimeError("serialized CKKS server context unexpectedly contains a secret key")
    receipt = CKKSContextReceipt(
        tenseal_version=str(getattr(tenseal, "__version__", "unknown")),
        parameters=parameters,
        keygen_ns=keygen_ns,
        client_context_bytes=len(client_context),
        server_context_bytes=len(server_context),
        client_context_sha256=hashlib.sha256(client_context).hexdigest(),
        server_context_sha256=hashlib.sha256(server_context).hexdigest(),
        server_context_is_private=server_is_private,
    )
    return CKKSContextArtifacts(client_context, server_context, receipt)


def evaluate_clear(
    values: npt.ArrayLike,
    evaluator: Callable[[ClearCKKSVector], ClearCKKSVector],
) -> tuple[float, ...]:
    """Run the same supported polynomial shape as a deterministic clear oracle."""

    result = evaluator(ClearCKKSVector(values))
    if not isinstance(result, ClearCKKSVector):
        raise TypeError("clear evaluator must return ClearCKKSVector")
    return result.values


def _import_tenseal() -> Any:
    try:
        return importlib.import_module("tenseal")
    except ImportError as error:
        raise CKKSUnavailableError(
            "TenSEAL is not installed; CKKS execution is unavailable and no clear backend "
            "will be substituted. Install a compatible TenSEAL build explicitly."
        ) from error


def _serialize_context(context: Any, *, save_secret_key: bool) -> bytes:
    return bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=save_secret_key,
            save_galois_keys=True,
            save_relin_keys=True,
        )
    )


def _context_is_private(context: Any) -> bool:
    status = context.is_private()
    if not isinstance(status, (bool, np.bool_)):
        raise TypeError("TenSEAL context returned a non-boolean private-key status")
    return bool(status)


def _require_single_ciphertext(slots: int, parameters: CKKSParameters) -> None:
    if slots > parameters.slot_capacity:
        raise ValueError(
            f"logical vector has {slots} slots but context capacity is {parameters.slot_capacity}"
        )


def _require_loaded_slots(vector: Any, declared_slots: int) -> None:
    size_member = vector.size
    loaded_slots = size_member() if callable(size_member) else size_member
    if int(loaded_slots) != declared_slots:
        raise ValueError("loaded ciphertext length does not match its declared logical slots")


def _real_vector(
    values: npt.ArrayLike,
    *,
    name: str,
    slots: int | None = None,
) -> npt.NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values, not complex values")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only real numeric values") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional real vector")
    if slots is not None and array.size != slots:
        raise ValueError(f"{name} must contain exactly {slots} logical slots")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _real_scalar(value: Any, *, name: str) -> float:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real, not complex")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric value") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _public_operand(value: float | npt.ArrayLike, slots: int) -> float | list[float]:
    if np.isscalar(value):
        return _real_scalar(value, name="public scalar operand")
    return [float(item) for item in _real_vector(value, name="public operand", slots=slots)]


def _binary_clear(
    left: ClearCKKSVector,
    right: float | npt.ArrayLike | ClearCKKSVector,
    operation: Callable[[float, float], float],
) -> Sequence[float]:
    if isinstance(right, ClearCKKSVector):
        if left.slots != right.slots:
            raise ValueError("packed clear operands must have the same logical slots")
        operands: Sequence[float] = right.values
    elif np.isscalar(right):
        scalar = _real_scalar(right, name="clear scalar operand")
        operands = (scalar,) * left.slots
    else:
        operands = _real_vector(right, name="clear operand", slots=left.slots).tolist()
    return tuple(operation(x, float(y)) for x, y in zip(left.values, operands, strict=True))
