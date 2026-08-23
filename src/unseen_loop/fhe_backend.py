"""Concrete-Python compilation, simulation, and serialized client/server execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from unseen_loop.policy import PolynomialPolicy
from unseen_loop.specs import IntArray


class FHEUnavailableError(RuntimeError):
    """Raised rather than silently falling back to clear execution."""


@dataclass(frozen=True)
class CircuitReceipt:
    policy_digest: str
    concrete_python_version: str
    global_p_error: float
    security_level: int
    maximum_integer_bit_width: int
    complexity: float
    input_shape: tuple[int, ...]
    calibration_rows: int
    input_min: tuple[int, ...]
    input_max: tuple[int, ...]
    integer_output_bound: tuple[int, ...]
    server_artifact_bytes: int
    server_artifact_sha256: str
    client_specs_bytes: int
    client_specs_sha256: str
    compile_ns: int
    mlir_sha256: str
    server_secret_key_markers: tuple[str, ...]
    backend: str = "Concrete-Python TFHE"
    mode: str = "COMPILED"
    schema_version: str = "unseen-loop/circuit-receipt-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


@dataclass(frozen=True)
class RoundTripMeasurement:
    integer_input: tuple[int, ...]
    integer_output: tuple[int, ...]
    clear_output: tuple[int, ...]
    output_matches_clear: bool
    keygen_ns: int
    encrypt_ns: int
    server_evaluate_ns: int
    decrypt_ns: int
    end_to_end_ns: int
    evaluation_key_bytes: int
    request_bytes: int
    response_bytes: int
    request_sha256: str
    response_sha256: str
    server_secret_key_present: bool
    backend: str = "REAL FHE"
    schema_version: str = "unseen-loop/fhe-measurement-v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


@dataclass
class CompiledPolicy:
    policy: PolynomialPolicy
    circuit: Any
    server_path: Path
    client_specs_path: Path
    receipt: CircuitReceipt

    def simulate(self, quantized: npt.ArrayLike) -> IntArray:
        values = np.asarray(quantized, dtype=np.int64)
        result = self.circuit.simulate(values)
        return np.asarray(result, dtype=np.int64)

    def real_roundtrip(self, quantized: npt.ArrayLike) -> RoundTripMeasurement:
        """Exercise serialized artifacts; the server object never receives the client key."""
        fhe = _import_fhe()
        values = np.asarray(quantized, dtype=np.int64)
        if values.shape != (self.policy.spec.quantizer.n_features,):
            raise ValueError("real FHE roundtrip accepts exactly one observation")
        if np.any(np.abs(values) > self.policy.spec.quantizer.qmax):
            raise ValueError("input is outside the compiled domain")

        started = time.perf_counter_ns()
        client_specs_payload = self.client_specs_path.read_bytes()
        client_specs = fhe.ClientSpecs.deserialize(client_specs_payload)
        client = fhe.Client(client_specs)
        phase = time.perf_counter_ns()
        client.keys.generate()
        keygen_ns = time.perf_counter_ns() - phase

        evaluation_keys = client.evaluation_keys.serialize()
        phase = time.perf_counter_ns()
        encrypted_input = client.encrypt(values)
        serialized_input = encrypted_input.serialize()
        encrypt_ns = time.perf_counter_ns() - phase

        server = fhe.Server.load(str(self.server_path))
        phase = time.perf_counter_ns()
        server_input = fhe.Value.deserialize(serialized_input)
        server_evaluation_keys = fhe.EvaluationKeys.deserialize(evaluation_keys)
        encrypted_output = server.run(
            server_input,
            evaluation_keys=server_evaluation_keys,
        )
        serialized_output = encrypted_output.serialize()
        server_evaluate_ns = time.perf_counter_ns() - phase

        phase = time.perf_counter_ns()
        client_output = fhe.Value.deserialize(serialized_output)
        decrypted = np.asarray(client.decrypt(client_output), dtype=np.int64)
        decrypt_ns = time.perf_counter_ns() - phase
        end_to_end_ns = time.perf_counter_ns() - started
        clear = np.atleast_1d(self.policy.integer_scores_from_quantized(values))
        decrypted = np.atleast_1d(decrypted)
        return RoundTripMeasurement(
            integer_input=tuple(int(value) for value in values),
            integer_output=tuple(int(value) for value in decrypted),
            clear_output=tuple(int(value) for value in clear),
            output_matches_clear=bool(np.array_equal(decrypted, clear)),
            keygen_ns=keygen_ns,
            encrypt_ns=encrypt_ns,
            server_evaluate_ns=server_evaluate_ns,
            decrypt_ns=decrypt_ns,
            end_to_end_ns=end_to_end_ns,
            evaluation_key_bytes=len(evaluation_keys),
            request_bytes=len(serialized_input),
            response_bytes=len(serialized_output),
            request_sha256=hashlib.sha256(serialized_input).hexdigest(),
            response_sha256=hashlib.sha256(serialized_output).hexdigest(),
            server_secret_key_present=bool(self.receipt.server_secret_key_markers),
        )


def _import_fhe() -> Any:
    try:
        fhe = importlib.import_module("concrete.fhe")
    except ImportError as error:
        raise FHEUnavailableError(
            "Concrete-Python is not installed. Run `uv sync --extra fhe`; "
            "Unseen Loop will not substitute a clear backend."
        ) from error
    return fhe


def _policy_kernel(policy: PolynomialPolicy) -> Any:
    fhe = _import_fhe()
    weights = policy.spec.integer_array
    n_features = policy.spec.quantizer.n_features
    degree = policy.spec.degree

    def kernel(x: Any) -> Any:
        result = weights[:, 0] + weights[:, 1 : 1 + n_features] @ x
        if degree == 2:
            coefficient = 1 + n_features
            for left in range(n_features):
                for right in range(left, n_features):
                    result = result + weights[:, coefficient] * (x[left] * x[right])
                    coefficient += 1
        return result

    return fhe.compiler({"x": "encrypted"})(kernel)


def _zip_secret_markers(path: Path) -> tuple[str, ...]:
    markers: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            lowered = name.lower()
            if "secret" in lowered or "private_key" in lowered or "client_key" in lowered:
                markers.append(name)
    return tuple(sorted(markers))


def compile_policy(
    policy: PolynomialPolicy,
    calibration_quantized: npt.ArrayLike,
    artifact_dir: str | Path,
    *,
    global_p_error: float = 1e-6,
    security_level: int = 128,
) -> CompiledPolicy:
    """Compile and package one architecture-specific Concrete server artifact."""
    if security_level != 128:
        raise ValueError("release backend currently requires the Concrete 128-bit security default")
    if not 0 < global_p_error <= 1e-3:
        raise ValueError("global_p_error must lie in (0, 1e-3]")
    calibration = np.asarray(calibration_quantized, dtype=np.int64)
    expected_shape = (policy.spec.quantizer.n_features,)
    if calibration.ndim != 2 or calibration.shape[1:] != expected_shape:
        raise ValueError("calibration_quantized must have shape (samples, observation_features)")
    if calibration.shape[0] < 2:
        raise ValueError("at least two calibration rows are required")
    if np.any(np.abs(calibration) > policy.spec.quantizer.qmax):
        raise ValueError("calibration input exceeds the quantizer domain")

    fhe = _import_fhe()
    compiler = _policy_kernel(policy)
    configuration = fhe.Configuration(
        enable_unsafe_features=False,
        use_insecure_key_cache=False,
        show_progress=False,
    )
    started = time.perf_counter_ns()
    circuit = compiler.compile(
        calibration,
        configuration=configuration,
        global_p_error=global_p_error,
    )
    compile_ns = time.perf_counter_ns() - started

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    server_path = destination / "server.zip"
    client_specs_path = destination / "client-specs.bin"
    circuit.server.save(str(server_path))
    serialized_client_specs = circuit.server.client_specs.serialize()
    specs_bytes = (
        serialized_client_specs.encode()
        if isinstance(serialized_client_specs, str)
        else bytes(serialized_client_specs)
    )
    client_specs_path.write_bytes(specs_bytes)

    server_bytes = server_path.read_bytes()
    mlir = str(getattr(circuit, "mlir", circuit))
    receipt = CircuitReceipt(
        policy_digest=policy.spec.digest,
        concrete_python_version=importlib.metadata.version("concrete-python"),
        global_p_error=global_p_error,
        security_level=security_level,
        maximum_integer_bit_width=int(circuit.graph.maximum_integer_bit_width()),
        complexity=float(circuit.complexity),
        input_shape=expected_shape,
        calibration_rows=calibration.shape[0],
        input_min=tuple(int(value) for value in np.min(calibration, axis=0)),
        input_max=tuple(int(value) for value in np.max(calibration, axis=0)),
        integer_output_bound=tuple(int(value) for value in policy.integer_output_bound()),
        server_artifact_bytes=len(server_bytes),
        server_artifact_sha256=hashlib.sha256(server_bytes).hexdigest(),
        client_specs_bytes=len(specs_bytes),
        client_specs_sha256=hashlib.sha256(specs_bytes).hexdigest(),
        compile_ns=compile_ns,
        mlir_sha256=hashlib.sha256(mlir.encode()).hexdigest(),
        server_secret_key_markers=_zip_secret_markers(server_path),
    )
    (destination / "receipt.json").write_text(receipt.to_json() + "\n")
    return CompiledPolicy(
        policy=policy,
        circuit=circuit,
        server_path=server_path,
        client_specs_path=client_specs_path,
        receipt=receipt,
    )
