"""Approximate-polynomial CKKS off-policy evaluation.

``POLYNOMIAL_APPROX_OPE_V1`` is deliberately distinct from the exact integer
OPE circuit.  A client places the trajectory axis in CKKS lanes, one
ciphertext per field and time, and sends no secret key.  The server evaluates
its target propensity polynomial, cumulative importance products, and a
frozen polynomial soft clip.  It returns exactly three encrypted scalars per
horizon step; only the client decrypts, aggregates, and divides.

The frozen soft clip is ``C * (x - x**2 / 4)``, where ``x = weight / C`` and
``0 <= x <= 2``.  It is a polynomial approximation, not an exact/hard clip.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from unseen_loop.crypto.ckks import (
    CKKSClient,
    CKKSContextArtifacts,
    CKKSEncryptedVector,
    CKKSParameters,
    CKKSServer,
    SerializedCKKSVector,
    generate_contexts,
)
from unseen_loop.ope.types import (
    FailureRow,
    OPEValidationError,
    PolynomialPolicySpec,
    SufficientStatistics,
    TrajectoryBatch,
    TrajectorySpec,
)

POLYNOMIAL_APPROX_OPE_V1 = "POLYNOMIAL_APPROX_OPE_V1"
EstimatorName = Literal["clipped_pdis", "clipped_wpdis"]
SOFT_CLIP_COEFFICIENTS = (0.0, 1.0, -0.25)
SOFT_CLIP_NORMALIZED_DOMAIN = (0.0, 2.0)


def executable_ckks_parameters(trajectories: int, horizon: int) -> CKKSParameters:
    """Choose the smallest supported tc128 chain for the frozen OPE depth."""

    if (
        isinstance(trajectories, bool)
        or not isinstance(trajectories, int)
        or trajectories < 1
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or horizon > 64
    ):
        raise ValueError("CKKS OPE trajectories and horizon are outside the supported shape")
    required_depth = horizon + 6
    scale_bits = 24
    modulus_bits = 80 + required_depth * scale_bits
    for degree, limit in ((8192, 218), (16384, 438), (32768, 881)):
        if trajectories <= degree // 2 and modulus_bits <= limit:
            return CKKSParameters(
                poly_modulus_degree=degree,
                coeff_mod_bit_sizes=(40, *((scale_bits,) * required_depth), 40),
                global_scale=float(2**scale_bits),
            )
    raise ValueError("CKKS OPE depth or packed trajectory count exceeds the tc128 frontier")


@dataclass(frozen=True)
class OPECKKSChunk:
    """A contiguous trajectory interval that fits one CKKS ciphertext."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("chunk must be a non-empty non-negative interval")

    @property
    def slots(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class OPECKKSChunkPlan:
    """Deterministic lane plan; no vector may exceed ``slot_capacity``."""

    trajectories: int
    slot_capacity: int
    chunks: tuple[OPECKKSChunk, ...]
    identifier: str = POLYNOMIAL_APPROX_OPE_V1

    def __post_init__(self) -> None:
        if self.trajectories < 1 or self.slot_capacity < 1 or not self.chunks:
            raise ValueError("chunk plan dimensions must be positive")
        cursor = 0
        for chunk in self.chunks:
            if chunk.start != cursor or chunk.slots > self.slot_capacity:
                raise ValueError("chunks must exactly and contiguously cover the trajectory axis")
            cursor = chunk.stop
        if cursor != self.trajectories:
            raise ValueError("chunks do not cover the declared trajectory count")

    @property
    def is_chunked(self) -> bool:
        return len(self.chunks) > 1


def plan_chunks(trajectories: int, parameters: CKKSParameters | None = None) -> OPECKKSChunkPlan:
    """Plan one lane per trajectory, chunking only at ciphertext capacity."""

    if isinstance(trajectories, bool) or not isinstance(trajectories, (int, np.integer)):
        raise TypeError("trajectories must be an integer")
    if trajectories < 1:
        raise ValueError("trajectories must be positive")
    parameters = parameters or CKKSParameters()
    capacity = parameters.slot_capacity
    chunks = tuple(
        OPECKKSChunk(start, min(start + capacity, int(trajectories)))
        for start in range(0, int(trajectories), capacity)
    )
    return OPECKKSChunkPlan(int(trajectories), capacity, chunks)


@dataclass(frozen=True)
class PolynomialApproxOPESpec:
    """Frozen server program and closed approximation domain."""

    trajectories: TrajectorySpec
    target_policy: PolynomialPolicySpec = field(repr=False)
    gamma: float = 1.0
    weight_clip: float = 20.0
    minimum_behavior_propensity: float = 1e-3
    identifier: str = POLYNOMIAL_APPROX_OPE_V1
    _target_probability_upper_bound: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.identifier != POLYNOMIAL_APPROX_OPE_V1:
            raise ValueError(f"identifier must be {POLYNOMIAL_APPROX_OPE_V1}")
        if self.trajectories.horizon > 64:
            raise ValueError("POLYNOMIAL_APPROX_OPE_V1 supports horizon at most 64")
        if self.target_policy.state_dim != self.trajectories.state_dim:
            raise ValueError("trajectory and target-policy state dimensions differ")
        if self.target_policy.action_count != self.trajectories.action_count:
            raise ValueError("trajectory and target-policy action counts differ")
        if not self.trajectories.state_min:
            raise ValueError("closed state bounds are required for the CKKS range receipt")
        if self.trajectories.reward_min is None or self.trajectories.reward_max is None:
            raise ValueError("closed reward bounds are required for the CKKS range receipt")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.weight_clip) or self.weight_clip <= 0:
            raise ValueError("weight_clip must be finite and positive")
        if (
            not math.isfinite(self.minimum_behavior_propensity)
            or not 0 < self.minimum_behavior_propensity <= 1
        ):
            raise ValueError("minimum_behavior_propensity must be in (0, 1]")
        self._prove_policy_range()
        if self.raw_weight_bounds[-1] > self.weight_clip * SOFT_CLIP_NORMALIZED_DOMAIN[1]:
            raise ValueError(
                "closed cumulative-weight range exceeds the frozen soft-clip domain; "
                "increase weight_clip or minimum_behavior_propensity"
            )

    @property
    def raw_weight_bounds(self) -> tuple[float, ...]:
        reciprocal = self._target_probability_upper_bound / self.minimum_behavior_propensity
        bounds: list[float] = []
        current = 1.0
        for _ in range(self.trajectories.horizon):
            current *= reciprocal
            if not math.isfinite(current):
                raise ValueError("closed cumulative-weight range is not finite")
            bounds.append(current)
        return tuple(bounds)

    @staticmethod
    def soft_clip(weight: npt.ArrayLike, clip: float) -> npt.NDArray[np.float64] | float:
        """Evaluate the frozen quadratic; this function never hard-clips."""

        values = np.asarray(weight, dtype=np.float64)
        normalized = values / clip
        result = clip * (normalized - 0.25 * normalized * normalized)
        return float(result) if result.ndim == 0 else result

    def _prove_policy_range(self) -> None:
        """Interval-prove that every server score stays in the probability simplex."""

        tolerance = self.target_policy.probability_tolerance
        coefficient_sums = np.sum(np.asarray(self.target_policy.coefficients), axis=0)
        if abs(float(coefficient_sums[0]) - 1.0) > tolerance or np.any(
            np.abs(coefficient_sums[1:]) > tolerance
        ):
            raise ValueError("target polynomial is not proved to sum to one on the state box")
        lows = self.trajectories.state_min
        highs = self.trajectories.state_max
        intervals: list[tuple[float, float]] = [(1.0, 1.0)]
        intervals.extend(zip(lows, highs, strict=True))
        if self.target_policy.degree == 2:
            for left in range(self.trajectories.state_dim):
                for right in range(left, self.trajectories.state_dim):
                    products = (
                        lows[left] * lows[right],
                        lows[left] * highs[right],
                        highs[left] * lows[right],
                        highs[left] * highs[right],
                    )
                    intervals.append((min(products), max(products)))
        upper_bounds: list[float] = []
        for action, coefficients in enumerate(self.target_policy.coefficients):
            lower = 0.0
            upper = 0.0
            for coefficient, (feature_low, feature_high) in zip(
                coefficients, intervals, strict=True
            ):
                endpoints = coefficient * feature_low, coefficient * feature_high
                lower += min(endpoints)
                upper += max(endpoints)
            if lower < -tolerance or upper > 1.0 + tolerance:
                raise ValueError(
                    f"action-{action} target polynomial is not proved inside [0, 1] "
                    "on the closed state box"
                )
            upper_bounds.append(min(1.0, upper))
        object.__setattr__(self, "_target_probability_upper_bound", max(upper_bounds))

    def validate_batch(self, batch: TrajectoryBatch) -> None:
        if batch.spec != self.trajectories:
            raise ValueError("batch spec differs from the frozen CKKS OPE spec")
        failures: list[FailureRow] = []
        for trajectory, step in np.argwhere(
            batch.behavior_array < self.minimum_behavior_propensity
        ):
            failures.append(
                FailureRow(
                    "unsupported_behavior",
                    "behavior_propensities",
                    "propensity is below the frozen reciprocal domain",
                    int(trajectory),
                    int(step),
                    float(batch.behavior_array[trajectory, step]),
                )
            )
        if failures:
            raise OPEValidationError(tuple(failures))
        # This also checks the supplied points despite the stronger interval proof.
        self.target_policy.logged_action_probabilities(batch)

    def _logged_target_polynomial(self, batch: TrajectoryBatch) -> npt.NDArray[np.float64]:
        features = self.target_policy.polynomial_features(batch.state_array)
        probabilities = features @ np.asarray(self.target_policy.coefficients, dtype=np.float64).T
        return np.take_along_axis(probabilities, batch.action_array[..., None], axis=-1)[..., 0]

    def clear_oracle(
        self, batch: TrajectoryBatch, estimator: EstimatorName
    ) -> SufficientStatistics:
        """Run exactly the polynomial server semantics without encryption."""

        self.validate_batch(batch)
        target = self._logged_target_polynomial(batch)
        ratios = target / batch.behavior_array
        raw_weights = np.cumprod(ratios, axis=1)
        normalized = raw_weights / self.weight_clip
        if np.any(normalized < 0) or np.any(normalized > SOFT_CLIP_NORMALIZED_DOMAIN[1] + 1e-12):
            raise ValueError("observed cumulative weight is outside the frozen soft-clip domain")
        weights = np.asarray(self.soft_clip(raw_weights, self.weight_clip), dtype=np.float64)
        discounts = self.gamma ** np.arange(self.trajectories.horizon, dtype=np.float64)
        numerators = tuple(
            float(discounts[step] * np.sum(weights[:, step] * batch.reward_array[:, step]))
            for step in range(self.trajectories.horizon)
        )
        weighted_denominators = tuple(
            float(np.sum(weights[:, step])) for step in range(self.trajectories.horizon)
        )
        counts = (self.trajectories.trajectories,) * self.trajectories.horizon
        if estimator == "clipped_pdis":
            denominators = tuple(float(value) for value in counts)
        elif estimator == "clipped_wpdis":
            denominators = weighted_denominators
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        failures = tuple(
            FailureRow(
                "zero_weight_denominator",
                "denominators",
                "polynomial-WPDIS is undefined at a horizon with zero total soft weight",
                step=step,
                value=0.0,
            )
            for step, value in enumerate(denominators)
            if estimator == "clipped_wpdis" and value == 0
        )
        return SufficientStatistics(
            estimator=estimator,
            numerators=numerators,
            denominators=denominators,
            counts=counts,
            failures=failures,
        )

    def receipt(self, parameters: CKKSParameters | None = None) -> PolynomialApproxOPEReceipt:
        parameters = parameters or CKKSParameters()
        # Includes rescaled plaintext coefficient products and the two-level
        # w - w^2/(4C) schedule. A nontrivial server-side discount adds one
        # level when a post-initial horizon exists; none of these levels are free.
        discount_depth = int(self.trajectories.horizon > 1 and self.gamma != 1.0)
        depth = self.trajectories.horizon + self.target_policy.degree + 4 + discount_depth
        available = len(parameters.coeff_mod_bit_sizes) - 2
        logarithmic_scale = math.log2(parameters.global_scale)
        scale_bits = round(logarithmic_scale)
        if not math.isclose(logarithmic_scale, scale_bits, abs_tol=1e-12):
            raise ValueError("global_scale must be an exact power of two for the scale receipt")
        if self.trajectories.reward_min is None or self.trajectories.reward_max is None:
            raise ValueError("CKKS OPE requires finite reward_min and reward_max")
        reward_bound = max(
            abs(float(self.trajectories.reward_min)), abs(float(self.trajectories.reward_max))
        )
        numerator_bounds = tuple(
            self.trajectories.trajectories * self.weight_clip * reward_bound * self.gamma**step
            for step in range(self.trajectories.horizon)
        )
        denominator_bounds = (self.trajectories.trajectories * self.weight_clip,) * (
            self.trajectories.horizon
        )
        return PolynomialApproxOPEReceipt(
            parameters=parameters,
            chunk_plan=plan_chunks(self.trajectories.trajectories, parameters),
            scale_bits=scale_bits,
            required_multiplicative_depth=depth,
            available_multiplicative_depth=available,
            configured_modulus_bits=sum(parameters.coeff_mod_bit_sizes),
            estimated_required_modulus_bits=(depth + 2) * scale_bits,
            raw_weight_bounds=self.raw_weight_bounds,
            normalized_soft_clip_domain=SOFT_CLIP_NORMALIZED_DOMAIN,
            soft_clip_coefficients=SOFT_CLIP_COEFFICIENTS,
            soft_clip_absolute_error_bound=self.weight_clip / 4.0,
            numerator_abs_bounds=numerator_bounds,
            denominator_bounds=denominator_bounds,
            target_policy_sha256=hashlib.sha256(
                self.target_policy.to_json().encode("utf-8")
            ).hexdigest(),
        )


@dataclass(frozen=True)
class PolynomialApproxOPEReceipt:
    """Scale, depth, modulus, range, approximation, and transport plan evidence."""

    parameters: CKKSParameters
    chunk_plan: OPECKKSChunkPlan
    scale_bits: int
    required_multiplicative_depth: int
    available_multiplicative_depth: int
    configured_modulus_bits: int
    estimated_required_modulus_bits: int
    raw_weight_bounds: tuple[float, ...]
    normalized_soft_clip_domain: tuple[float, float]
    soft_clip_coefficients: tuple[float, ...]
    soft_clip_absolute_error_bound: float
    numerator_abs_bounds: tuple[float, ...]
    denominator_bounds: tuple[float, ...]
    target_policy_sha256: str
    identifier: str = POLYNOMIAL_APPROX_OPE_V1
    required_security_level: str = "tc128"
    output_ciphertexts: int = field(init=False)
    schema_version: str = "unseen-loop/polynomial-approx-ope-ckks-receipt-v1"
    trust_scope: str = (
        "TenSEAL CKKS approximate polynomial arithmetic; target model remains server-held; "
        "secret key, decryption, and division remain client-side"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_ciphertexts", 3 * len(self.raw_weight_bounds))

    @property
    def depth_supported(self) -> bool:
        return self.available_multiplicative_depth >= self.required_multiplicative_depth

    @property
    def modulus_supported(self) -> bool:
        interior = self.parameters.coeff_mod_bit_sizes[1:-1]
        return self.configured_modulus_bits >= self.estimated_required_modulus_bits and all(
            bits >= self.scale_bits for bits in interior
        )

    def require_executable(self) -> None:
        if not self.depth_supported:
            raise ValueError(
                "CKKS modulus chain provides "
                f"{self.available_multiplicative_depth} multiplication levels but "
                f"POLYNOMIAL_APPROX_OPE_V1 requires {self.required_multiplicative_depth}"
            )
        if not self.modulus_supported:
            raise ValueError(
                "CKKS modulus chain does not meet the recorded scale/modulus-bit requirement"
            )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class EncryptedOPEChunk:
    """N-lane ciphertexts for one planned trajectory interval."""

    interval: OPECKKSChunk
    states: tuple[tuple[SerializedCKKSVector, ...], ...]
    action_masks: tuple[tuple[SerializedCKKSVector, ...], ...]
    rewards: tuple[SerializedCKKSVector, ...]
    behavior_reciprocals: tuple[SerializedCKKSVector, ...]
    counts: tuple[SerializedCKKSVector, ...]


@dataclass(frozen=True)
class EncryptedOPERequest:
    """Ciphertext-only request; the target policy is intentionally absent."""

    chunks: tuple[EncryptedOPEChunk, ...]
    chunk_plan: OPECKKSChunkPlan
    identifier: str = POLYNOMIAL_APPROX_OPE_V1


@dataclass(frozen=True)
class EncryptedOPEResponse:
    """Exactly 3H encrypted scalar sufficient statistics."""

    numerators: tuple[SerializedCKKSVector, ...]
    denominators: tuple[SerializedCKKSVector, ...]
    counts: tuple[SerializedCKKSVector, ...]
    identifier: str = POLYNOMIAL_APPROX_OPE_V1

    def __post_init__(self) -> None:
        if not self.numerators or not (
            len(self.numerators) == len(self.denominators) == len(self.counts)
        ):
            raise ValueError("response vectors must have the same positive horizon")
        if any(
            value.slots != 1
            for vector in (self.numerators, self.denominators, self.counts)
            for value in vector
        ):
            raise ValueError("every encrypted sufficient statistic must be a scalar ciphertext")


@dataclass(frozen=True)
class OPECKKSTransportReceipt:
    """Ciphertext hashes, sizes, and timing; never plaintext private logs."""

    operation: str
    elapsed_ns: int
    input_ciphertexts: int
    output_ciphertexts: int
    input_bytes: int
    output_bytes: int
    input_sha256: str | None
    output_sha256: str | None
    identifier: str = POLYNOMIAL_APPROX_OPE_V1
    schema_version: str = "unseen-loop/polynomial-approx-ope-ckks-operation-v1"
    trust_scope: str = (
        "receipt contains ciphertext transport metadata only; no plaintext trajectories, "
        "propensities, rewards, decrypted values, or secret keys"
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class PolynomialApproxOPEContexts:
    """Split CKKS artifacts plus the closed computation receipt."""

    ckks: CKKSContextArtifacts
    computation: PolynomialApproxOPEReceipt


def generate_ope_contexts(
    spec: PolynomialApproxOPESpec, parameters: CKKSParameters | None = None
) -> PolynomialApproxOPEContexts:
    """Generate tc128-enforced split contexts after checking circuit depth."""

    parameters = parameters or CKKSParameters()
    receipt = spec.receipt(parameters)
    receipt.require_executable()
    artifacts = generate_contexts(parameters)
    if (
        not artifacts.receipt.security_enforced
        or artifacts.receipt.effective_security_level != "tc128"
        or artifacts.receipt.server_context_is_private
    ):
        raise RuntimeError("CKKS backend did not provide public-server tc128 evidence")
    return PolynomialApproxOPEContexts(artifacts, receipt)


class OPECKKSClient:
    """Secret-key boundary for trajectory encryption, decryption, and division."""

    def __init__(self, client: CKKSClient, spec: PolynomialApproxOPESpec) -> None:
        self._client = client
        self.spec = spec
        self.receipt = spec.receipt(client.parameters)
        self.receipt.require_executable()

    @classmethod
    def from_serialized(
        cls, payload: bytes, *, parameters: CKKSParameters, spec: PolynomialApproxOPESpec
    ) -> OPECKKSClient:
        return cls(CKKSClient.from_serialized(payload, parameters=parameters), spec)

    def encrypt_batch(
        self, batch: TrajectoryBatch
    ) -> tuple[EncryptedOPERequest, OPECKKSTransportReceipt]:
        self.spec.validate_batch(batch)
        started = time.perf_counter_ns()
        encrypted_chunks: list[EncryptedOPEChunk] = []
        payloads: list[bytes] = []
        states = batch.state_array
        actions = batch.action_array
        rewards = batch.reward_array
        reciprocals = 1.0 / batch.behavior_array
        horizon = batch.spec.horizon
        for interval in self.receipt.chunk_plan.chunks:
            section = slice(interval.start, interval.stop)

            def encrypt(values: npt.NDArray[np.float64]) -> SerializedCKKSVector:
                result, _ = self._client.encrypt(values)
                payloads.append(result.to_bytes())
                return result

            encrypted_states = tuple(
                tuple(encrypt(states[section, step, dim]) for dim in range(batch.spec.state_dim))
                for step in range(horizon)
            )
            encrypted_masks = tuple(
                tuple(
                    encrypt((actions[section, step] == action).astype(np.float64))
                    for action in range(batch.spec.action_count)
                )
                for step in range(horizon)
            )
            encrypted_rewards = tuple(encrypt(rewards[section, step]) for step in range(horizon))
            encrypted_reciprocals = tuple(
                encrypt(reciprocals[section, step]) for step in range(horizon)
            )
            encrypted_counts = tuple(
                encrypt(np.ones(interval.slots, dtype=np.float64)) for _ in range(horizon)
            )
            encrypted_chunks.append(
                EncryptedOPEChunk(
                    interval,
                    encrypted_states,
                    encrypted_masks,
                    encrypted_rewards,
                    encrypted_reciprocals,
                    encrypted_counts,
                )
            )
        elapsed = time.perf_counter_ns() - started
        request = EncryptedOPERequest(tuple(encrypted_chunks), self.receipt.chunk_plan)
        receipt = OPECKKSTransportReceipt(
            operation="encrypt_ope_request",
            elapsed_ns=elapsed,
            input_ciphertexts=0,
            output_ciphertexts=len(payloads),
            input_bytes=int(states.nbytes + actions.nbytes + rewards.nbytes + reciprocals.nbytes),
            output_bytes=sum(map(len, payloads)),
            input_sha256=None,
            output_sha256=_payload_digest(payloads),
        )
        return request, receipt

    def decrypt_statistics(
        self, response: EncryptedOPEResponse, estimator: EstimatorName
    ) -> tuple[SufficientStatistics, OPECKKSTransportReceipt]:
        horizon = self.spec.trajectories.horizon
        if response.identifier != POLYNOMIAL_APPROX_OPE_V1 or len(response.numerators) != horizon:
            raise ValueError("encrypted response does not match the frozen CKKS OPE spec")
        started = time.perf_counter_ns()
        payloads = [
            value.to_bytes()
            for vector in (response.numerators, response.denominators, response.counts)
            for value in vector
        ]

        def decrypt(vector: tuple[SerializedCKKSVector, ...]) -> tuple[float, ...]:
            return tuple(float(self._client.decrypt(value)[0][0]) for value in vector)

        numerators = decrypt(response.numerators)
        weighted_denominators = decrypt(response.denominators)
        approximate_counts = decrypt(response.counts)
        counts = tuple(round(value) for value in approximate_counts)
        if any(
            abs(value - rounded) > 0.25
            for value, rounded in zip(approximate_counts, counts, strict=True)
        ):
            raise ValueError("decrypted CKKS count is outside the accepted precision tolerance")
        if estimator == "clipped_pdis":
            denominators = tuple(float(value) for value in counts)
        elif estimator == "clipped_wpdis":
            denominators = weighted_denominators
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        failures = tuple(
            FailureRow(
                "zero_weight_denominator",
                "denominators",
                "polynomial-WPDIS denominator is numerically zero",
                step=step,
                value=0.0,
            )
            for step, value in enumerate(denominators)
            if estimator == "clipped_wpdis" and abs(value) <= 1e-12
        )
        statistics = SufficientStatistics(estimator, numerators, denominators, counts, failures)
        elapsed = time.perf_counter_ns() - started
        receipt = OPECKKSTransportReceipt(
            operation="decrypt_ope_response",
            elapsed_ns=elapsed,
            input_ciphertexts=len(payloads),
            output_ciphertexts=0,
            input_bytes=sum(map(len, payloads)),
            output_bytes=8 * 3 * horizon,
            input_sha256=_payload_digest(payloads),
            output_sha256=None,
        )
        return statistics, receipt


class OPECKKSServer:
    """Public-context evaluator holding the target propensity model."""

    def __init__(self, server: CKKSServer, spec: PolynomialApproxOPESpec) -> None:
        self._server = server
        self.spec = spec
        self.receipt = spec.receipt(server.parameters)
        self.receipt.require_executable()

    @classmethod
    def from_serialized(
        cls, payload: bytes, *, parameters: CKKSParameters, spec: PolynomialApproxOPESpec
    ) -> OPECKKSServer:
        return cls(CKKSServer.from_serialized(payload, parameters=parameters), spec)

    def evaluate(
        self, request: EncryptedOPERequest
    ) -> tuple[EncryptedOPEResponse, OPECKKSTransportReceipt]:
        if request.identifier != POLYNOMIAL_APPROX_OPE_V1:
            raise ValueError("request has an incompatible OPE identifier")
        if request.chunk_plan != self.receipt.chunk_plan:
            raise ValueError("request chunk plan differs from the frozen server plan")
        if len(request.chunks) != len(request.chunk_plan.chunks):
            raise ValueError("request chunk count differs from its declared plan")
        input_payloads: list[bytes] = []
        output_payloads: list[bytes] = []
        started = time.perf_counter_ns()
        horizon = self.spec.trajectories.horizon
        total_numerators: list[CKKSEncryptedVector | None] = [None] * horizon
        total_denominators: list[CKKSEncryptedVector | None] = [None] * horizon
        total_counts: list[CKKSEncryptedVector | None] = [None] * horizon
        coefficients = self.spec.target_policy.coefficients

        def load(value: SerializedCKKSVector, slots: int) -> CKKSEncryptedVector:
            if value.slots != slots:
                raise ValueError("ciphertext slots differ from the declared chunk interval")
            input_payloads.append(value.to_bytes())
            raw = self._server._tenseal.ckks_vector_from(
                self._server._context,
                value.ciphertext,
            )
            size = raw.size() if callable(raw.size) else raw.size
            if int(size) != slots:
                raise ValueError("loaded ciphertext length differs from its declared slots")
            return CKKSEncryptedVector(raw, slots, self._server._owner)

        for expected, chunk in zip(request.chunk_plan.chunks, request.chunks, strict=True):
            if chunk.interval != expected:
                raise ValueError("encrypted chunk interval differs from the declared plan")
            slots = expected.slots
            self._validate_chunk_shape(chunk, horizon)
            raw_weight: CKKSEncryptedVector | None = None
            for step in range(horizon):
                state = [load(value, slots) for value in chunk.states[step]]
                masks = [load(value, slots) for value in chunk.action_masks[step]]
                reward = load(chunk.rewards[step], slots)
                reciprocal = load(chunk.behavior_reciprocals[step], slots)
                count = load(chunk.counts[step], slots)
                scores: list[CKKSEncryptedVector] = []
                for row in coefficients:
                    score = state[0] * float(row[1]) + float(row[0])
                    offset = 1 + self.spec.trajectories.state_dim
                    for dim in range(1, self.spec.trajectories.state_dim):
                        score = score + state[dim] * float(row[1 + dim])
                    if self.spec.target_policy.degree == 2:
                        index = offset
                        for left in range(self.spec.trajectories.state_dim):
                            for right in range(left, self.spec.trajectories.state_dim):
                                score = score + (state[left] * state[right]) * float(row[index])
                                index += 1
                    scores.append(score)
                probability = scores[0] * masks[0]
                for action in range(1, self.spec.trajectories.action_count):
                    probability = probability + scores[action] * masks[action]
                ratio = probability * reciprocal
                raw_weight = ratio if raw_weight is None else raw_weight * ratio
                # Algebraically identical to C*(x-x^2/4), x=w/C, with two
                # multiplication levels rather than a normalized Horner chain.
                soft_weight = raw_weight - raw_weight.square() * (0.25 / self.spec.weight_clip)
                denominator = soft_weight.reduce_sum()
                numerator = (soft_weight * reward).reduce_sum()
                discount = self.spec.gamma**step
                if discount != 1.0:
                    numerator = numerator * discount
                count_sum = count.reduce_sum()
                total_numerators[step] = _accumulate(total_numerators[step], numerator)
                total_denominators[step] = _accumulate(total_denominators[step], denominator)
                total_counts[step] = _accumulate(total_counts[step], count_sum)

        def serialize(values: list[CKKSEncryptedVector | None]) -> tuple[SerializedCKKSVector, ...]:
            serialized: list[SerializedCKKSVector] = []
            for value in values:
                if value is None:
                    raise RuntimeError("server evaluation produced an incomplete horizon")
                item = SerializedCKKSVector(value._vector.serialize(), value.slots)
                output_payloads.append(item.to_bytes())
                serialized.append(item)
            return tuple(serialized)

        response = EncryptedOPEResponse(
            serialize(total_numerators), serialize(total_denominators), serialize(total_counts)
        )
        elapsed = time.perf_counter_ns() - started
        receipt = OPECKKSTransportReceipt(
            operation="evaluate_ope",
            elapsed_ns=elapsed,
            input_ciphertexts=len(input_payloads),
            output_ciphertexts=len(output_payloads),
            input_bytes=sum(map(len, input_payloads)),
            output_bytes=sum(map(len, output_payloads)),
            input_sha256=_payload_digest(input_payloads),
            output_sha256=_payload_digest(output_payloads),
        )
        return response, receipt

    def _validate_chunk_shape(self, chunk: EncryptedOPEChunk, horizon: int) -> None:
        if not (
            len(chunk.states)
            == len(chunk.action_masks)
            == len(chunk.rewards)
            == len(chunk.behavior_reciprocals)
            == len(chunk.counts)
            == horizon
        ):
            raise ValueError("encrypted chunk does not have the frozen horizon")
        if any(len(row) != self.spec.trajectories.state_dim for row in chunk.states):
            raise ValueError("encrypted state tensor does not have the frozen state dimension")
        if any(len(row) != self.spec.trajectories.action_count for row in chunk.action_masks):
            raise ValueError("encrypted action tensor does not have the frozen action count")


def clear_oracle(
    spec: PolynomialApproxOPESpec, batch: TrajectoryBatch, estimator: EstimatorName
) -> SufficientStatistics:
    """Functional clear-oracle entry point."""

    return spec.clear_oracle(batch, estimator)


def _accumulate(
    current: CKKSEncryptedVector | None, value: CKKSEncryptedVector
) -> CKKSEncryptedVector:
    return value if current is None else current + value


def _payload_digest(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
