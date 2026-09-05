"""Policy-independent ratio lifting and unclipped, mean-statistic CKKS WPDIS.

The prefix network is a standard divide-and-conquer inclusive scan. Range
receipts establish a sufficient plaintext-magnitude frontier, not a CKKS error
certificate or malicious-server integrity. No private values occur in receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Self, TypeVar

import numpy as np

from unseen_loop.crypto.ckks import (
    CKKSClient,
    CKKSEncryptedVector,
    CKKSParameters,
    CKKSServer,
    SerializedCKKSVector,
    _require_tc128_security,
    context_modulus_primes,
)
from unseen_loop.ope.ckks import (
    OPECKKSChunk,
    OPECKKSChunkPlan,
    OPECKKSTransportReceipt,
    plan_chunks,
)
from unseen_loop.ope.types import (
    PolynomialPolicySpec,
    SufficientStatistics,
    TrajectoryBatch,
    TrajectorySpec,
)

UNCLIPPED_RATIO_LIFT_WPDIS_V1 = "UNCLIPPED_RATIO_LIFT_WPDIS_V1"
_OPERATION_SCHEMA = "unseen-loop/ratio-lift-wpdis-ckks-operation-v1"


class _Multiplicative(Protocol):
    def __mul__(self, other: Self, /) -> Self: ...


T = TypeVar("T", bound=_Multiplicative)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _policy_digest(policy: PolynomialPolicySpec) -> str:
    return hashlib.sha256(policy.to_json().encode()).hexdigest()


def _inclusive_prefix(values: Sequence[T]) -> tuple[T, ...]:
    """Preserve reused deeper right operands under TenSEAL's auto-mod-switch."""
    if not values:
        raise ValueError("prefix requires a nonempty sequence")
    if len(values) == 1:
        return (values[0],)
    split = 1 << ((len(values) - 1).bit_length() - 1)
    left = _inclusive_prefix(values[:split])
    right = _inclusive_prefix(values[split:])
    return left + tuple(value * left[-1] for value in right)


def _reduction_term_counts(size: int) -> tuple[int, ...]:
    """Term counts in TenSEAL 0.3.17's remainder-first sum_vector DAG."""
    power = 1 << (size.bit_length() - 1)
    counts = {1 << k for k in range(power.bit_length())}
    if size != power:
        counts.update(_reduction_term_counts(size - power))
        counts.add(size)
    return tuple(sorted(counts))


@dataclass(frozen=True)
class IntermediateBound:
    name: str
    policy_sha256: str | None
    magnitude: float
    level: int
    scale_bits: int
    modulus_lower_bound_bits: int
    strict_magnitude_limit: float


class _RangeProof:
    """Audit each actual DAG node at its level; never reason after cancellation."""

    def __init__(self, parameters: CKKSParameters, actual_primes: tuple[int, ...] | None = None):
        self.parameters = parameters
        self.scale_bits = round(math.log2(parameters.global_scale))
        if parameters.global_scale != 2.0**self.scale_bits:
            raise ValueError("domain.range_bound: scale is not a power of two")
        self.actual_primes = actual_primes
        if actual_primes is not None and (
            len(actual_primes) != len(parameters.coeff_mod_bit_sizes)
            or any(
                type(q) is not int or q.bit_length() != bits
                for q, bits in zip(actual_primes, parameters.coeff_mod_bit_sizes, strict=True)
            )
        ):
            raise ValueError("domain.range_bound: actual modulus differs from parameter plan")
        self.rows: list[IntermediateBound] = []

    def check(
        self,
        name: str,
        magnitude: float,
        level: int,
        scale_bits: int | None = None,
        policy_sha256: str | None = None,
    ) -> None:
        scale_bits = self.scale_bits if scale_bits is None else scale_bits
        remaining = len(self.parameters.coeff_mod_bit_sizes) - 1 - level
        if remaining < 1 or level < 0:
            raise ValueError("domain.range_bound: exhausted data modulus")
        bits = sum(q - 1 for q in self.parameters.coeff_mod_bit_sizes[:remaining])
        if self.actual_primes is None:
            limit = 2.0 ** (bits - scale_bits - 2)
        else:
            modulus = math.prod(self.actual_primes[:remaining])
            # Round downward so conversion to binary64 never enlarges the frontier.
            limit = math.nextafter(float(modulus / (4 * 2**scale_bits)), 0.0)
        if not math.isfinite(magnitude) or magnitude < 0 or not magnitude < limit:
            raise ValueError(f"domain.range_bound: {name} exceeds the strict modulus/scale reserve")
        self.rows.append(
            IntermediateBound(name, policy_sha256, float(magnitude), level, scale_bits, bits, limit)
        )

    def product(
        self,
        name: str,
        magnitude: float,
        left_level: int,
        right_level: int,
        policy_sha256: str | None = None,
    ) -> int:
        level = max(left_level, right_level)
        self.check(name + ":pre_rescale", magnitude, level, 2 * self.scale_bits, policy_sha256)
        self.check(name + ":post_rescale", magnitude, level + 1, self.scale_bits, policy_sha256)
        return level + 1


@dataclass(frozen=True)
class RatioLiftComputationReceipt:
    parameters: CKKSParameters
    chunk_plan: OPECKKSChunkPlan
    target_policy_sha256: tuple[str, ...]
    required_multiplicative_depth: int
    intermediate_bounds: tuple[IntermediateBound, ...]
    output_ciphertexts: int
    actual_coeff_modulus_primes: tuple[int, ...] | None = None
    counts_source: str = "public_fixed_shape"
    scale_bits: int = 40
    required_security_level: str = "tc128"
    identifier: str = UNCLIPPED_RATIO_LIFT_WPDIS_V1
    schema_version: str = "unseen-loop/ratio-lift-wpdis-computation-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()

    def require_executable(self) -> None:
        # Construction has already audited every intermediate and the fixed chain.
        if self.required_multiplicative_depth != len(self.parameters.coeff_mod_bit_sizes) - 2:
            raise ValueError("domain.range_bound: unsupported depth")


@dataclass(frozen=True)
class RatioLiftWPDISSpec:
    trajectories: TrajectorySpec
    target_policies: tuple[PolynomialPolicySpec, ...]
    gamma: float
    minimum_behavior_propensity: float
    maximum_importance_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.target_policies, tuple) or not self.target_policies:
            raise ValueError("target policies must be a nonempty ordered tuple")
        if self.trajectories.horizon > 64:
            raise ValueError("ratio lift supports H<=64")
        if (
            not self.trajectories.state_min
            or self.trajectories.reward_min is None
            or self.trajectories.reward_max is None
        ):
            raise ValueError("ratio lift requires closed finite state and reward domains")
        for name, value in (
            ("gamma", self.gamma),
            ("minimum_behavior_propensity", self.minimum_behavior_propensity),
            ("maximum_importance_ratio", self.maximum_importance_ratio),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if (
            not 0 <= self.gamma <= 1
            or not 0 < self.minimum_behavior_propensity <= 1
            or self.maximum_importance_ratio <= 0
        ):
            raise ValueError("invalid discount, propensity floor or ratio bound")
        identities = tuple(_policy_digest(policy) for policy in self.target_policies)
        if len(set(identities)) != len(identities):
            raise ValueError("target policies must be digest-unique")
        for policy in self.target_policies:
            if policy.degree != self.target_policies[0].degree:
                raise ValueError("all target policies must share their feature degree")
            policy.probability_bounds(self.trajectories)
        self.computation_receipt()

    def validate_batch(self, batch: TrajectoryBatch) -> None:
        if batch.spec != self.trajectories:
            raise ValueError(
                "domain.invalid_input: batch differs from frozen trajectory specification"
            )
        behavior = batch.behavior_array
        if np.any(behavior < self.minimum_behavior_propensity):
            raise ValueError("domain.ratio_bound: behavior below frozen propensity floor")
        # Use the exact polynomial, not action_probabilities' tolerance clipping.
        features = self.target_policies[0].polynomial_features(batch.state_array)
        for policy in self.target_policies:
            policy.logged_action_probabilities(batch)
            probabilities = features @ np.asarray(policy.coefficients).T
            logged = np.take_along_axis(probabilities, batch.action_array[..., None], axis=-1)[
                ..., 0
            ]
            ratios = logged / behavior
            # A one-ULP comparison allowance handles binary64 endpoint arithmetic
            # (e.g. (.8-.5)/.25); it neither clips ratios nor admits a new domain.
            if (
                np.any(~np.isfinite(ratios))
                or np.any(ratios < 0)
                or np.any(ratios > math.nextafter(self.maximum_importance_ratio, math.inf))
            ):
                raise ValueError(
                    "domain.ratio_bound: frozen target logged-action ratio outside domain"
                )

    def computation_receipt(
        self, parameters: CKKSParameters | None = None
    ) -> RatioLiftComputationReceipt:
        return self._computation_receipt(parameters, None)

    def _computation_receipt(
        self, parameters: CKKSParameters | None, actual_primes: tuple[int, ...] | None
    ) -> RatioLiftComputationReceipt:
        parameters = parameters or lifted_ckks_parameters(self)
        if parameters != lifted_ckks_parameters(self):
            raise ValueError(
                "domain.range_bound: ratio lift requires the frozen 16384/40-bit chain"
            )
        proof = _RangeProof(parameters, actual_primes)
        n, horizon = self.trajectories.trajectories, self.trajectories.horizon
        plan = _chunk_plan(n, parameters)
        reduction_counts = tuple(_reduction_term_counts(chunk.slots) for chunk in plan.chunks)
        lows, highs = self.trajectories.state_min, self.trajectories.state_max
        intervals = [(1.0, 1.0), *zip(lows, highs, strict=True)]
        if self.target_policies[0].degree == 2:
            for a in range(self.trajectories.state_dim):
                for b in range(a, self.trajectories.state_dim):
                    products = (
                        lows[a] * lows[b],
                        lows[a] * highs[b],
                        highs[a] * lows[b],
                        highs[a] * highs[b],
                    )
                    intervals.append(
                        (
                            math.nextafter(min(products), -math.inf),
                            math.nextafter(max(products), math.inf),
                        )
                    )
        lift_intervals = []
        for j, (low, high) in enumerate(intervals):
            proof.check(f"monomial/{j}", max(abs(low), abs(high)), 0)
            low = min(0.0, math.nextafter(low / self.minimum_behavior_propensity, -math.inf))
            high = max(0.0, math.nextafter(high / self.minimum_behavior_propensity, math.inf))
            lift_intervals.append((low, high))
            proof.check(f"lifted_feature/{j}", max(abs(low), abs(high)), 0)
        reward_min, reward_max = self.trajectories.reward_min, self.trajectories.reward_max
        assert reward_min is not None and reward_max is not None
        reward = math.nextafter(max(abs(reward_min), abs(reward_max)) / n, math.inf)
        proof.check("normalized_reward", reward, 0)
        # Include the declared binary64 endpoint comparison allowance and round
        # every range operation outward; these are bounds, never point estimates.
        ratio_bound = math.nextafter(self.maximum_importance_ratio, math.inf)
        weight_bounds = [1.0]
        for _ in range(horizon):
            bound = math.nextafter(weight_bounds[-1] * ratio_bound, math.inf)
            if not math.isfinite(bound):
                raise ValueError("domain.range_bound: cumulative ratio is not finite")
            weight_bounds.append(bound)
        for policy in self.target_policies:
            identity = _policy_digest(policy)
            low = high = 0.0
            for a, row in enumerate(policy.coefficients):
                for j, coefficient in enumerate(row):
                    if coefficient == 0:
                        continue
                    # Public coefficient encoding and every term before any cancellation.
                    proof.check(f"coefficient/{a}/{j}", abs(coefficient), 0, policy_sha256=identity)
                    endpoints = tuple(coefficient * v for v in lift_intervals[j])
                    term_low = math.nextafter(min(endpoints), -math.inf)
                    term_high = math.nextafter(max(endpoints), math.inf)
                    proof.product(
                        f"coefficient_product/{a}/{j}",
                        max(abs(term_low), abs(term_high)),
                        0,
                        0,
                        identity,
                    )
                    low = math.nextafter(low + term_low, -math.inf)
                    high = math.nextafter(high + term_high, math.inf)
                    proof.check(
                        f"partial_ratio_sum/{a}/{j}",
                        max(abs(low), abs(high)),
                        1,
                        policy_sha256=identity,
                    )
            # Only after the full accumulation may the honest client's all-policy
            # support validation tighten the final ratio to the declared bound.
            proof.check("validated_ratio", ratio_bound, 1, policy_sha256=identity)

            def prefix_nodes(
                start: int,
                length: int,
                policy_sha256: str = identity,
            ) -> tuple[tuple[int, int], ...]:
                if length == 1:
                    return ((1, 1),)
                split = 1 << ((length - 1).bit_length() - 1)
                left = prefix_nodes(start, split)
                right = prefix_nodes(start + split, length - split)
                result = list(left)
                for offset, (count, level) in enumerate(right):
                    count += split
                    level = proof.product(
                        f"prefix/{start}/{length}/{offset}",
                        weight_bounds[count],
                        level,
                        left[-1][1],
                        policy_sha256,
                    )
                    result.append((count, level))
                return tuple(result)

            nodes = prefix_nodes(0, horizon)
            for t, (count, level) in enumerate(nodes):
                weight = weight_bounds[count]
                for label, lane_bound in (
                    ("mean_weighted_reward", math.nextafter(weight * reward, math.inf)),
                    ("mean_weight", math.nextafter(weight / n, math.inf)),
                ):
                    output_level = proof.product(
                        f"{label}/{t}",
                        lane_bound,
                        0 if label == "mean_weighted_reward" else level,
                        level,
                        identity,
                    )
                    merged = 0.0
                    for index, interval in enumerate(plan.chunks):
                        # The backend splits off a non-power-of-two remainder,
                        # then doubles partial sums and adds the remainder.
                        for width in reduction_counts[index]:
                            proof.check(
                                f"{label}/{t}/chunk/{index}/sum/{width}",
                                math.nextafter(lane_bound * width, math.inf),
                                output_level,
                                policy_sha256=identity,
                            )
                        merged = math.nextafter(
                            merged + math.nextafter(lane_bound * interval.slots, math.inf), math.inf
                        )
                        proof.check(
                            f"{label}/{t}/merge/{index}",
                            merged,
                            output_level,
                            policy_sha256=identity,
                        )
        return RatioLiftComputationReceipt(
            parameters,
            plan,
            tuple(_policy_digest(p) for p in self.target_policies),
            (horizon - 1).bit_length() + 2,
            tuple(proof.rows),
            2 * horizon,
            actual_primes,
        )


def lifted_ckks_parameters(spec: RatioLiftWPDISSpec) -> CKKSParameters:
    horizon = spec.trajectories.horizon
    if not 1 <= horizon <= 64:
        raise ValueError("ratio lift supports 1<=H<=64")
    depth = (horizon - 1).bit_length() + 2
    return CKKSParameters(16384, (60, *((40,) * depth), 58), float(2**40))


def _chunk_plan(n: int, parameters: CKKSParameters) -> OPECKKSChunkPlan:
    plan = plan_chunks(n, parameters)
    return OPECKKSChunkPlan(
        plan.trajectories, plan.slot_capacity, plan.chunks, UNCLIPPED_RATIO_LIFT_WPDIS_V1
    )


def _integer(value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer in ratio-lift envelope")
    return value


def _keys(value: Any, names: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("unknown or missing ratio-lift envelope fields")
    return value


def _hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid ratio-lift digest")
    return value


def _frame(metadata: dict[str, Any], vectors: Sequence[SerializedCKKSVector]) -> bytes:
    header = _canonical(metadata)
    parts = [len(header).to_bytes(8, "big"), header]
    for vector in vectors:
        payload = vector.to_bytes()
        parts.extend((len(payload).to_bytes(8, "big"), payload))
    return b"".join(parts)


def _unframe(data: bytes) -> tuple[dict[str, Any], tuple[SerializedCKKSVector, ...]]:
    if not isinstance(data, bytes):
        raise ValueError("ratio-lift wire input must be bytes")
    cursor = 0

    def take() -> bytes:
        nonlocal cursor
        if cursor + 8 > len(data):
            raise ValueError("truncated ratio-lift frame")
        size = int.from_bytes(data[cursor : cursor + 8], "big")
        cursor += 8
        if size == 0 or cursor + size > len(data):
            raise ValueError("truncated ratio-lift payload")
        result = data[cursor : cursor + size]
        cursor += size
        return result

    header = take()
    try:
        metadata = json.loads(header)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid ratio-lift JSON") from error
    if not isinstance(metadata, dict) or _canonical(metadata) != header:
        raise ValueError("ratio-lift metadata must be canonical JSON")
    vectors = []
    while cursor < len(data):
        vectors.append(SerializedCKKSVector.from_bytes(take()))
    return metadata, tuple(vectors)


@dataclass(frozen=True)
class RatioLiftChunk:
    interval: OPECKKSChunk
    lifted_features: tuple[tuple[tuple[SerializedCKKSVector, ...], ...], ...]
    normalized_rewards: tuple[SerializedCKKSVector, ...]


@dataclass(frozen=True)
class RatioLiftRequest:
    identifier: str
    chunk_plan: OPECKKSChunkPlan
    chunks: tuple[RatioLiftChunk, ...]
    feature_degree: int
    state_dim: int
    action_count: int
    _wire_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            self.identifier != UNCLIPPED_RATIO_LIFT_WPDIS_V1
            or self.chunk_plan.identifier != self.identifier
        ):
            raise ValueError("ckks.request_binding: incompatible identifier")
        if _integer(self.feature_degree) not in (1, 2):
            raise ValueError("unsupported feature degree")
        _integer(self.state_dim)
        _integer(self.action_count, 2)
        _integer(self.chunk_plan.trajectories)
        if not isinstance(self.chunks, tuple) or not isinstance(self.chunk_plan.chunks, tuple):
            raise ValueError("ratio-lift chunks must be immutable tuples")
        for interval in self.chunk_plan.chunks:
            _integer(interval.start, 0)
            _integer(interval.stop)
        if self.chunk_plan.slot_capacity != 8192:
            raise ValueError("invalid ratio-lift slot capacity")
        expected = _chunk_plan(
            self.chunk_plan.trajectories, CKKSParameters(16384, (60, 40, 40, 58), float(2**40))
        )
        if self.chunk_plan != expected or len(self.chunks) != len(expected.chunks):
            raise ValueError("ckks.request_binding: invalid exact chunk cover")
        features = (
            1
            + self.state_dim
            + (self.state_dim * (self.state_dim + 1) // 2 if self.feature_degree == 2 else 0)
        )
        horizon = len(self.chunks[0].normalized_rewards)
        if not 1 <= horizon <= 64:
            raise ValueError("invalid ratio-lift horizon")
        for chunk in self.chunks:
            if (
                not isinstance(chunk.lifted_features, tuple)
                or not isinstance(chunk.normalized_rewards, tuple)
                or any(
                    not isinstance(step, tuple) or any(not isinstance(row, tuple) for row in step)
                    for step in chunk.lifted_features
                )
            ):
                raise ValueError("ratio-lift ciphertext tensors must be immutable tuples")
        for interval, chunk in zip(expected.chunks, self.chunks, strict=True):
            if (
                chunk.interval != interval
                or len(chunk.normalized_rewards) != horizon
                or len(chunk.lifted_features) != horizon
            ):
                raise ValueError("ckks.request_binding: chunk shape mismatch")
            for step in chunk.lifted_features:
                if len(step) != self.action_count or any(len(row) != features for row in step):
                    raise ValueError("ckks.request_binding: feature tensor shape mismatch")
            if any(
                type(value.slots) is not int or value.slots != interval.slots
                for value in _chunk_vectors(chunk)
            ):
                raise ValueError("ckks.request_binding: incorrect ciphertext slots")

    def _metadata(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "chunk_plan": asdict(self.chunk_plan),
            "feature_degree": self.feature_degree,
            "state_dim": self.state_dim,
            "action_count": self.action_count,
            "horizon": len(self.chunks[0].normalized_rewards),
        }

    def to_bytes(self) -> bytes:
        self.validate()
        payload = self._wire_bytes
        if payload is None:
            payload = _frame(
                self._metadata(), tuple(v for chunk in self.chunks for v in _chunk_vectors(chunk))
            )
            object.__setattr__(self, "_wire_bytes", payload)
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> RatioLiftRequest:
        metadata, vectors = _unframe(data)
        _keys(
            metadata,
            {"identifier", "chunk_plan", "feature_degree", "state_dim", "action_count", "horizon"},
        )
        raw = _keys(
            metadata["chunk_plan"], {"identifier", "trajectories", "slot_capacity", "chunks"}
        )
        if not isinstance(raw["chunks"], list):
            raise ValueError("invalid chunk list")
        intervals = []
        for item in raw["chunks"]:
            _keys(item, {"start", "stop"})
            intervals.append(OPECKKSChunk(_integer(item["start"], 0), _integer(item["stop"])))
        plan = OPECKKSChunkPlan(
            _integer(raw["trajectories"]),
            _integer(raw["slot_capacity"]),
            tuple(intervals),
            raw["identifier"],
        )
        degree, dim, actions, horizon = (
            _integer(metadata[key])
            for key in ("feature_degree", "state_dim", "action_count", "horizon")
        )
        if degree not in (1, 2) or actions < 2 or horizon > 64:
            raise ValueError("invalid feature shape")
        count = 1 + dim + (dim * (dim + 1) // 2 if degree == 2 else 0)
        if len(vectors) != len(intervals) * horizon * (actions * count + 1):
            raise ValueError("ckks.request_binding: wrong ciphertext count")
        source = iter(vectors)
        chunks = tuple(
            RatioLiftChunk(
                interval,
                tuple(
                    tuple(tuple(next(source) for _ in range(count)) for _ in range(actions))
                    for _ in range(horizon)
                ),
                tuple(next(source) for _ in range(horizon)),
            )
            for interval in intervals
        )
        request = cls(metadata["identifier"], plan, chunks, degree, dim, actions)
        object.__setattr__(request, "_wire_bytes", data)
        return request


def _chunk_vectors(chunk: RatioLiftChunk) -> tuple[SerializedCKKSVector, ...]:
    return (
        tuple(v for step in chunk.lifted_features for row in step for v in row)
        + chunk.normalized_rewards
    )


@dataclass(frozen=True)
class RatioLiftResponse:
    identifier: str
    policy_sha256: str
    request_sha256: str
    mean_weighted_rewards: tuple[SerializedCKKSVector, ...]
    mean_weights: tuple[SerializedCKKSVector, ...]
    _wire_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.identifier != UNCLIPPED_RATIO_LIFT_WPDIS_V1:
            raise ValueError("ckks.request_binding: incompatible response identifier")
        _hash(self.policy_sha256)
        _hash(self.request_sha256)
        if not isinstance(self.mean_weights, tuple) or not isinstance(
            self.mean_weighted_rewards, tuple
        ):
            raise ValueError("ratio-lift response vectors must be immutable tuples")
        if not 1 <= len(self.mean_weights) <= 64 or len(self.mean_weighted_rewards) != len(
            self.mean_weights
        ):
            raise ValueError("ckks.request_binding: response must contain exactly 2H statistics")
        if any(
            type(v.slots) is not int or v.slots != 1
            for v in self.mean_weighted_rewards + self.mean_weights
        ):
            raise ValueError("ckks.request_binding: response ciphertexts must be scalar")

    def to_bytes(self) -> bytes:
        self.validate()
        payload = self._wire_bytes
        if payload is None:
            payload = _frame(
                {
                    "identifier": self.identifier,
                    "policy_sha256": self.policy_sha256,
                    "request_sha256": self.request_sha256,
                    "horizon": len(self.mean_weights),
                },
                self.mean_weighted_rewards + self.mean_weights,
            )
            object.__setattr__(self, "_wire_bytes", payload)
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> RatioLiftResponse:
        metadata, vectors = _unframe(data)
        _keys(metadata, {"identifier", "policy_sha256", "request_sha256", "horizon"})
        horizon = _integer(metadata["horizon"])
        if horizon > 64 or len(vectors) != 2 * horizon:
            raise ValueError("ckks.request_binding: wrong response ciphertext count")
        response = cls(
            metadata["identifier"],
            metadata["policy_sha256"],
            metadata["request_sha256"],
            vectors[:horizon],
            vectors[horizon:],
        )
        object.__setattr__(response, "_wire_bytes", data)
        return response


def _transport(
    operation: str,
    started: int,
    inputs: bytes | None,
    outputs: bytes | None,
    input_count: int,
    output_count: int,
    *,
    identifier: str = UNCLIPPED_RATIO_LIFT_WPDIS_V1,
    schema_version: str = _OPERATION_SCHEMA,
) -> OPECKKSTransportReceipt:
    return OPECKKSTransportReceipt(
        operation,
        time.perf_counter_ns() - started,
        input_count,
        output_count,
        len(inputs) if inputs else 0,
        len(outputs) if outputs else 0,
        hashlib.sha256(inputs).hexdigest() if inputs else None,
        hashlib.sha256(outputs).hexdigest() if outputs else None,
        identifier=identifier,
        schema_version=schema_version,
    )


def _validate_request(
    spec: RatioLiftWPDISSpec, request: RatioLiftRequest, parameters: CKKSParameters
) -> None:
    request.validate()
    policy = spec.target_policies[0]
    if (
        request.chunk_plan != _chunk_plan(spec.trajectories.trajectories, parameters)
        or (request.feature_degree, request.state_dim, request.action_count)
        != (policy.degree, policy.state_dim, policy.action_count)
        or len(request.chunks[0].normalized_rewards) != spec.trajectories.horizon
    ):
        raise ValueError("ckks.request_binding: request differs from frozen program")


def _runtime_computation_receipt(
    spec: RatioLiftWPDISSpec, backend: CKKSClient | CKKSServer
) -> RatioLiftComputationReceipt:
    context = backend._context
    _require_tc128_security(context, backend._tenseal, backend.parameters)
    wrapped = context.seal_context()
    seal = getattr(wrapped, "data", wrapped)
    first = seal.first_context_data()
    if (
        int(first.parms().poly_modulus_degree()) != backend.parameters.poly_modulus_degree
        or int(first.chain_index()) != len(backend.parameters.coeff_mod_bit_sizes) - 2
        or float(context.global_scale) != backend.parameters.global_scale
    ):
        raise ValueError("ckks.context_failure: loaded context differs from declared parameters")
    return spec._computation_receipt(backend.parameters, context_modulus_primes(context))


class RatioLiftWPDISClient:
    def __init__(self, client: CKKSClient, spec: RatioLiftWPDISSpec):
        self._client, self.spec = client, spec
        self.receipt = _runtime_computation_receipt(spec, client)
        self.last_preprocessing_ns = 0

    def encrypt_batch(
        self, batch: TrajectoryBatch
    ) -> tuple[RatioLiftRequest, OPECKKSTransportReceipt]:
        started = time.perf_counter_ns()
        self.last_preprocessing_ns = 0
        self.spec.validate_batch(batch)
        policy = self.spec.target_policies[0]
        features = policy.polynomial_features(batch.state_array)
        actions, rewards, behavior = batch.action_array, batch.reward_array, batch.behavior_array
        self.last_preprocessing_ns = time.perf_counter_ns() - started
        chunks = []
        for interval in self.receipt.chunk_plan.chunks:
            preparation = time.perf_counter_ns()
            section = slice(interval.start, interval.stop)
            reciprocal_features = features[section] / behavior[section, :, None]
            self.last_preprocessing_ns += time.perf_counter_ns() - preparation
            lifted_steps = []
            for t in range(batch.spec.horizon):
                action_rows = []
                for a in range(policy.action_count):
                    preparation = time.perf_counter_ns()
                    mask = actions[section, t] == a
                    self.last_preprocessing_ns += time.perf_counter_ns() - preparation
                    row = []
                    for j in range(policy.feature_count):
                        preparation = time.perf_counter_ns()
                        values = mask * reciprocal_features[:, t, j]
                        self.last_preprocessing_ns += time.perf_counter_ns() - preparation
                        row.append(self._client.encrypt(values)[0])
                    action_rows.append(tuple(row))
                lifted_steps.append(tuple(action_rows))
            normalized = []
            for t in range(batch.spec.horizon):
                preparation = time.perf_counter_ns()
                values = rewards[section, t] / batch.spec.trajectories
                self.last_preprocessing_ns += time.perf_counter_ns() - preparation
                normalized.append(self._client.encrypt(values)[0])
            chunks.append(RatioLiftChunk(interval, tuple(lifted_steps), tuple(normalized)))
        request = RatioLiftRequest(
            UNCLIPPED_RATIO_LIFT_WPDIS_V1,
            self.receipt.chunk_plan,
            tuple(chunks),
            policy.degree,
            policy.state_dim,
            policy.action_count,
        )
        payload = request.to_bytes()
        return request, _transport(
            "encrypt_ratio_lift_request",
            started,
            None,
            payload,
            0,
            sum(len(_chunk_vectors(c)) for c in chunks),
        )

    def decrypt_statistics(
        self, request: RatioLiftRequest, response: RatioLiftResponse
    ) -> tuple[SufficientStatistics, OPECKKSTransportReceipt]:
        started = time.perf_counter_ns()
        _validate_request(self.spec, request, self._client.parameters)
        response.validate()
        if (
            response.request_sha256 != request.digest
            or response.policy_sha256 not in self.receipt.target_policy_sha256
            or len(response.mean_weights) != self.spec.trajectories.horizon
        ):
            raise ValueError("ckks.request_binding: response is not bound to this request/program")
        payload = response.to_bytes()
        numerator = tuple(
            float(self._client.decrypt(v)[0][0]) for v in response.mean_weighted_rewards
        )
        denominator = tuple(float(self._client.decrypt(v)[0][0]) for v in response.mean_weights)
        if any(not math.isfinite(v) for v in numerator + denominator):
            raise ValueError("ckks.nonfinite: rejected mean statistics")
        if any(v <= 0 for v in denominator):
            raise ValueError("ckks.nonpositive_denominator: rejected mean weights")
        statistics = SufficientStatistics(
            "wpdis",
            tuple(self.spec.gamma**t * v for t, v in enumerate(numerator)),
            denominator,
            (request.chunk_plan.trajectories,) * len(denominator),
        )
        return statistics, _transport(
            "decrypt_ratio_lift_statistics", started, payload, None, 2 * len(denominator), 0
        )


class RatioLiftWPDISServer:
    def __init__(self, server: CKKSServer, spec: RatioLiftWPDISSpec):
        self._server, self.spec = server, spec
        self.receipt = _runtime_computation_receipt(spec, server)

    def evaluate(
        self, request: RatioLiftRequest, policy_sha256: str
    ) -> tuple[RatioLiftResponse, OPECKKSTransportReceipt]:
        return self._evaluate_with_prefix_builder(request, policy_sha256, _inclusive_prefix)

    def _evaluate_with_prefix_builder(
        self,
        request: RatioLiftRequest,
        policy_sha256: str,
        prefix_builder: Callable[[Sequence[CKKSEncryptedVector]], Sequence[CKKSEncryptedVector]],
    ) -> tuple[RatioLiftResponse, OPECKKSTransportReceipt]:
        started = time.perf_counter_ns()
        _validate_request(self.spec, request, self._server.parameters)
        if policy_sha256 not in self.receipt.target_policy_sha256:
            raise ValueError("ckks.request_binding: policy is not frozen")
        coefficients = self.spec.target_policies[
            self.receipt.target_policy_sha256.index(policy_sha256)
        ].coefficients
        request_bytes = request.to_bytes()
        horizon = self.spec.trajectories.horizon
        numerators: list[CKKSEncryptedVector | None] = [None] * horizon
        denominators: list[CKKSEncryptedVector | None] = [None] * horizon

        def load(value: SerializedCKKSVector) -> CKKSEncryptedVector:
            raw = self._server._tenseal.ckks_vector_from(self._server._context, value.ciphertext)
            size = raw.size() if callable(raw.size) else raw.size
            if int(size) != value.slots:
                raise ValueError("ckks.request_binding: loaded ciphertext slots differ")
            return CKKSEncryptedVector(raw, value.slots, self._server._owner)

        for chunk in request.chunks:
            ratios = []
            for step in chunk.lifted_features:
                ratio = None
                for row, values in zip(coefficients, step, strict=True):
                    for coefficient, value in zip(row, values, strict=True):
                        if coefficient == 0:
                            continue
                        term = load(value) * float(coefficient)
                        ratio = term if ratio is None else ratio + term
                if ratio is None:
                    raise ValueError("domain.invalid_input: no nonzero public polynomial term")
                ratios.append(ratio)
            prefixes = prefix_builder(ratios)
            del ratios, ratio, term
            if len(prefixes) != horizon:
                raise ValueError("ckks.request_binding: incomplete prefix result")
            for t, weight in enumerate(prefixes):
                # Shallower disposable left; deeper reused right. Multiplication
                # must not modulus-switch a saved prefix or request operand.
                numerator = (load(chunk.normalized_rewards[t]) * weight).reduce_sum()
                denominator = (weight * (1.0 / self.spec.trajectories.trajectories)).reduce_sum()
                previous_numerator, previous_denominator = numerators[t], denominators[t]
                numerators[t] = (
                    numerator if previous_numerator is None else previous_numerator + numerator
                )
                denominators[t] = (
                    denominator
                    if previous_denominator is None
                    else previous_denominator + denominator
                )
            del prefixes, weight

        def serialize(
            values: Sequence[CKKSEncryptedVector | None],
        ) -> tuple[SerializedCKKSVector, ...]:
            if any(value is None for value in values):
                raise ValueError("ckks.request_binding: incomplete statistics")
            return tuple(
                SerializedCKKSVector(value._vector.serialize(), value.slots)
                for value in values
                if value is not None
            )

        response = RatioLiftResponse(
            UNCLIPPED_RATIO_LIFT_WPDIS_V1,
            policy_sha256,
            hashlib.sha256(request_bytes).hexdigest(),
            serialize(numerators),
            serialize(denominators),
        )
        payload = response.to_bytes()
        return response, _transport(
            "evaluate_ratio_lift_wpdis",
            started,
            request_bytes,
            payload,
            sum(len(_chunk_vectors(c)) for c in request.chunks),
            2 * horizon,
        )
