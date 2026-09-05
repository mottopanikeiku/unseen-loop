"""Source-bound private OPE execution and deterministic fixed-denominator analysis.

No registry, transport publication, claims, or Modal dispatch occurs here.  The
coordinator is the sole mutation authority.  Secret contexts and trajectory rows
exist only in worker memory; returned records contain released aggregates.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import resource
import struct
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from _pytest.reports import TestReport
    from _typeshed import DataclassInstance

    from unseen_loop.ope.lifted import RatioLiftWPDISSpec

from unseen_loop.crypto.ckks import (
    CKKSClient,
    CKKSContextArtifacts,
    CKKSContextReceipt,
    CKKSEncryptedVector,
    CKKSParameters,
    CKKSServer,
    SerializedCKKSVector,
    generate_contexts,
)
from unseen_loop.flagship.manifest import PlannedJob, PrivateOPEManifest, derive_seed
from unseen_loop.ope.ckks import OPECKKSTransportReceipt, plan_chunks
from unseen_loop.ope.study import PairedWPDISBootstrap
from unseen_loop.ope.types import (
    PolynomialPolicySpec,
    SufficientStatistics,
    TrajectoryBatch,
    TrajectorySpec,
)

BaselineID = Literal[
    "is", "pdis", "wpdis", "clipped_wpdis_2", "clipped_wpdis_10", "dm", "dr", "wdr", "mis"
]
CountsSource = Literal["public_fixed_shape", "legacy_encrypted", "diagnostic_sum", "not-applicable"]

BASELINE_IDS: tuple[BaselineID, ...] = (
    "is",
    "pdis",
    "wpdis",
    "clipped_wpdis_2",
    "clipped_wpdis_10",
    "dm",
    "dr",
    "wdr",
    "mis",
)
FAILURE_CODES = frozenset(
    (
        "probe.deliberate_exception",
        "ckks.count_precision",
        "ckks.nonpositive_denominator",
        "ckks.nonfinite",
        "ckks.context_failure",
        "ckks.request_binding",
        "ckks.backend_error",
        "domain.ratio_bound",
        "domain.range_bound",
        "domain.invalid_input",
        "statistics.invalid_support",
        "verification.failed",
        "runtime.timeout",
        "runtime.interrupted",
        "runtime.reentry",
        "runtime.dispatch_unknown",
        "runtime.not_dispatched",
        "evidence.source_mismatch",
        "evidence.invalid_artifact",
        "analysis.failed",
    )
)
GATE_IDS = (
    "evidence_closure",
    "verification",
    "diagnostic_precision",
    "probe_accounting",
    "clear_gap",
    "clear_coverage",
    "clear_choice",
    "interval_width",
    "left_ess_median",
    "left_ess_p05",
    "right_ess_median",
    "right_ess_p05",
    "required_context_completion",
    "cipher_numerics",
    "resource_bounds",
    "timing_pair_completion",
    "timing_median",
    "timing_lower",
    "confirmation_coverage",
    "confirmation_choice",
    "confirmation_bias",
    "confirmation_rmse",
)
MODERN_KINDS = frozenset(
    ("paired_context", "ablation_context", "statistical_context", "timing_context")
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_RUN = re.compile(r"private-ope-(diagnostic|pilot|confirmation)-[0-9a-f]{24}\Z")
_SEED = re.compile(r"(?:0|[1-9][0-9]*)\Z")
TOLERANCE = 1e-12


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _plain(value: object) -> object:
    if isinstance(value, PlannedJob):
        return job_to_dict(value)
    if isinstance(value, SufficientStatistics):
        return _plain(value.to_dict())
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _keys(data: object, names: set[str]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != names:
        raise ValueError("evidence fields do not match schema")
    return cast(dict[str, Any], data)


def _coordinate_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("incorrect job coordinate type")
    return value


def _coordinate_str(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("incorrect job coordinate type")
    return value


def _seed(value: object) -> int:
    if not isinstance(value, str) or _SEED.fullmatch(value) is None:
        raise ValueError("seed must be canonical decimal string")
    result = int(value)
    _require(result < 2**128, "seed exceeds 128 bits")
    return result


def _parse(annotation: Any, value: object) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        raise TypeError("open evidence types are prohibited")
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _parse(option, value)
            except (ValueError, TypeError, KeyError):
                pass
        raise ValueError("value does not match any permitted evidence variant")
    if origin is Literal:
        _require(any(type(value) is type(x) and value == x for x in args), "invalid evidence tag")
        return value
    if annotation is type(None):
        _require(value is None, "expected null")
        return None
    if annotation in (str, bool, int, float):
        if annotation is float:
            if (
                type(value) not in (int, float)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("expected finite number")
            return float(value)
        _require(type(value) is annotation, "incorrect primitive type")
        if annotation is int:
            _require(isinstance(value, int) and value >= 0, "negative evidence count")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("expected evidence array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_parse(args[0], x) for x in value)
        _require(len(value) == len(args), "incorrect evidence array length")
        return tuple(_parse(t, x) for t, x in zip(args, value, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("expected evidence map")
        return {_parse(args[0], k): _parse(args[1], v) for k, v in value.items()}
    if annotation is PlannedJob:
        return job_from_dict(value)
    if annotation is SufficientStatistics:
        names = {"estimator", "numerators", "denominators", "counts", "failures", "estimate"}
        row = _keys(value, names)
        _parse(str, row["estimator"])
        for name in ("numerators", "denominators"):
            _parse(tuple[float, ...], row[name])
        _parse(tuple[int, ...], row["counts"])
        _require(
            row["failures"] == [] or row["failures"] == (),
            "failed historical statistics are not admissible aggregates",
        )
        result = SufficientStatistics.from_dict(row)
        _parse(float | None, row["estimate"])
        _require(row["estimate"] == result.estimate, "historical estimate is inconsistent")
        return result
    if annotation is PairedWPDISBootstrap:
        return annotation.from_dict(_keys(value, {f.name for f in dataclasses.fields(annotation)}))
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if issubclass(annotation, EvidenceRecord):
            return annotation.from_dict(value)
        names = {f.name for f in dataclasses.fields(annotation)}
        row = _keys(value, names)
        hints = get_type_hints(annotation)
        record = cast(Callable[..., "DataclassInstance"], annotation)(
            **{k: _parse(hints[k], v) for k, v in row.items()}
        )
        _validate_common(record)
        if isinstance(record, CKKSContextReceipt):
            _require(
                record.schema_version == "unseen-loop/ckks-context-receipt-v2",
                "wrong context schema",
            )
        if isinstance(record, OPECKKSTransportReceipt):
            expected = {
                "POLYNOMIAL_APPROX_OPE_V1": "unseen-loop/polynomial-approx-ope-ckks-operation-v1",
                "UNCLIPPED_RATIO_LIFT_WPDIS_V1": "unseen-loop/ratio-lift-wpdis-ckks-operation-v1",
                "RAW_PREFIX_WPDIS_BENCHMARK_V1": "unseen-loop/raw-prefix-wpdis-ckks-operation-v1",
                "PRIVATE_OPE_COUNT_DIAGNOSTIC_V1": (
                    "unseen-loop/private-ope-count-ckks-operation-v1"
                ),
            }
            _require(
                expected.get(record.identifier) == record.schema_version,
                "operation schema/identifier mismatch",
            )
        return record
    raise TypeError(f"unsupported closed evidence type {annotation}")


def _validate_common(record: DataclassInstance) -> None:
    for f in dataclasses.fields(record):
        value = getattr(record, f.name)
        if (
            value is not None
            and (f.name.endswith("sha256") or f.name == "config_sha256")
            and isinstance(value, str)
        ):
            _require(_DIGEST.fullmatch(value) is not None, "invalid digest")
        if f.name == "failure_code" and value is not None:
            _require(value in FAILURE_CODES, "unknown failure code")


class EvidenceRecord:
    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _plain(self))

    @classmethod
    def from_dict(cls, data: object) -> Self:
        if not bool(dataclasses.is_dataclass(cls)):
            raise TypeError("evidence records must be dataclasses")
        record_type = cast("type[DataclassInstance]", cls)
        row = _keys(data, {f.name for f in dataclasses.fields(record_type)})
        hints = get_type_hints(cls)
        result = cast(
            Self,
            cast(Callable[..., object], cls)(
                **{name: _parse(hints[name], value) for name, value in row.items()}
            ),
        )
        _validate_common(cast("DataclassInstance", result))
        result._validate()
        return result

    def _validate(self) -> None:
        pass


_JOB_COORDS = {
    "kind",
    "cohort",
    "case_index",
    "trajectories",
    "horizon",
    "behavior",
    "arm",
    "wave_index",
    "case_id",
    "data_seed",
    "bootstrap_seed",
}


def job_to_dict(job: PlannedJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "seed": str(job.seed),
        "coordinates": dict(job.coordinates),
    }


def job_from_dict(data: object) -> PlannedJob:
    row = _keys(data, {"job_id", "stage", "seed", "coordinates"})
    _require(
        isinstance(row["job_id"], str) and _ID.fullmatch(row["job_id"]) is not None,
        "invalid job identity",
    )
    _require(
        row["stage"] in ("private_ope_diagnostic", "private_ope_pilot", "private_ope_confirmation"),
        "invalid job stage",
    )
    coords = _keys(row["coordinates"], _JOB_COORDS)
    for key, value in coords.items():
        if key in {"case_index", "trajectories", "horizon", "wave_index"}:
            _parse(int, value)
        else:
            _parse(str, value)
    _require(
        coords["kind"]
        in MODERN_KINDS
        | {
            "clear_batch",
            "count_precision",
            "historical_context",
            "protocol_verification",
            "smoke_error",
            "smoke_timeout",
            "analysis",
        },
        "unknown job kind",
    )
    stochastic = coords["kind"] in MODERN_KINDS | {"clear_batch", "historical_context"}
    for key in ("data_seed", "bootstrap_seed"):
        if stochastic:
            _seed(coords[key])
        else:
            _require(coords[key] == "not-applicable", "unexpected data seed")
    _require(bool(coords["trajectories"]) == bool(coords["horizon"]), "inconsistent job shape")
    if stochastic:
        _require(
            coords["trajectories"] > 0
            and coords["horizon"] > 0
            and coords["case_id"] != "not-applicable",
            "missing stochastic case",
        )
    return PlannedJob(
        row["job_id"], row["stage"], _seed(row["seed"]), tuple(sorted(coords.items()))
    )


@dataclass(frozen=True)
class PrivateOPEJobPayload(EvidenceRecord):
    schema_version: Literal["unseen-loop/private-ope-job-v1"]
    run_id: str
    config_sha256: str
    job: PlannedJob
    dispatch_intent_sha256: str
    deadline_utc: str

    def _validate(self) -> None:
        _require(_RUN.fullmatch(self.run_id) is not None, "invalid run identity")
        deadline = dt.datetime.fromisoformat(self.deadline_utc.replace("Z", "+00:00"))
        _require(deadline.utcoffset() == dt.timedelta(0), "deadline must be absolute UTC")


@dataclass(frozen=True)
class PrivateOPETransportResult(EvidenceRecord):
    schema_version: Literal["unseen-loop/private-ope-transport-v1"]
    run_id: str
    job_id: str
    function_call_id: str
    input_id: str
    entry_path: str
    result_path: str | None
    result_sha256: str | None
    delivery: Literal["result", "reentry"]

    def _validate(self) -> None:
        _require(_RUN.fullmatch(self.run_id) is not None, "invalid run identity")
        for identity in (self.job_id, self.function_call_id, self.input_id):
            _require(_ID.fullmatch(identity) is not None, "unsafe invocation identity")
        root = PurePosixPath("private-ope-transport", self.run_id, self.job_id, self.input_id)
        for value in (self.entry_path, self.result_path):
            if value is not None:
                path = PurePosixPath(value)
                _require(
                    str(path) == value
                    and path.parent == root
                    and path.name not in (".", "..")
                    and ".." not in path.parts,
                    "invalid transport namespace",
                )
        _require(
            (self.result_path is None) == (self.result_sha256 is None),
            "result path and digest must be paired",
        )
        _require(
            self.delivery != "result" or self.result_path is not None,
            "result delivery lacks result",
        )


@dataclass(frozen=True)
class RuntimeReceipt(EvidenceRecord):
    image_id: str
    image_spec_sha256: str
    code_commit: str
    candidate_code_sha256: str
    baseline_code_sha256: str
    domain_code_sha256: str
    analysis_code_sha256: str
    lockfile_sha256: str
    python_version: str
    numpy_version: str
    scipy_version: str
    tenseal_version: str
    seal_version: str
    modal_version: str
    source_match: bool
    execution_site: Literal["Modal"]

    def _validate(self) -> None:
        _require(re.fullmatch(r"[0-9a-f]{40}", self.code_commit) is not None, "invalid code commit")
        _require(
            bool(self.image_id)
            and all(
                getattr(self, k)
                for k in (
                    "python_version",
                    "numpy_version",
                    "scipy_version",
                    "tenseal_version",
                    "seal_version",
                    "modal_version",
                )
            ),
            "missing runtime identity",
        )


@dataclass(frozen=True)
class AttemptReceipts(EvidenceRecord):
    runtime: RuntimeReceipt | None
    context: CKKSContextReceipt | None
    computation_sha256: str | None
    batch_sha256: str | None
    request_sha256: str | None
    response_sha256: tuple[str | None, ...]
    operations: tuple[OPECKKSTransportReceipt, ...]
    public_context_sha256: str | None
    public_context_bytes: int | None
    client_context_bytes: int | None
    request_bytes: int | None
    response_bytes: tuple[int | None, ...]
    counts_source: Literal[
        "public_fixed_shape", "legacy_encrypted", "diagnostic_sum", "not-applicable"
    ]

    def _validate(self) -> None:
        _require(
            len(self.response_sha256) == len(self.response_bytes), "response receipt lengths differ"
        )
        for digest, size in zip(self.response_sha256, self.response_bytes, strict=True):
            _require((digest is None) == (size is None), "partial response receipt")
            if digest is not None:
                _require(_DIGEST.fullmatch(digest) is not None, "invalid response digest")
        if self.context is not None:
            _require(
                self.public_context_sha256 == self.context.server_context_sha256
                and self.public_context_bytes == self.context.server_context_bytes
                and self.client_context_bytes == self.context.client_context_bytes,
                "context receipt mismatch",
            )


@dataclass(frozen=True)
class PolicyReference(EvidenceRecord):
    id: Literal["A", "B"]
    policy_sha256: str
    mean_weighted_rewards: tuple[float, ...]
    mean_weights: tuple[float, ...]
    raw_ess: tuple[float, ...]
    counts: tuple[int, ...]
    raw_value: float
    normalized_value: float
    minimum_logged_propensity: float
    maximum_logged_ratio: float
    maximum_cumulative_weight: float
    support_failures: int

    def _validate(self) -> None:
        _vectors(self.mean_weighted_rewards, self.mean_weights, self.counts)
        _require(
            len(self.raw_ess) == len(self.counts)
            and all(
                0 <= e <= n + TOLERANCE for e, n in zip(self.raw_ess, self.counts, strict=True)
            ),
            "invalid ESS",
        )
        _require(
            0 < self.minimum_logged_propensity <= 1
            and self.maximum_logged_ratio >= 0
            and self.maximum_cumulative_weight >= 0,
            "invalid support bounds",
        )


def _vectors(
    numerators: tuple[float, ...], denominators: tuple[float, ...], counts: tuple[int, ...]
) -> None:
    _require(
        0 < len(counts) <= 64 and len(numerators) == len(denominators) == len(counts),
        "invalid horizon arrays",
    )
    _require(len(set(counts)) == 1 and counts[0] > 0, "counts are not exact fixed shape")
    _require(all(x > 0 for x in denominators), "nonpositive denominator")


@dataclass(frozen=True)
class BaselinePair(EvidenceRecord):
    estimator_id: Literal[
        "is", "pdis", "wpdis", "clipped_wpdis_2", "clipped_wpdis_10", "dm", "dr", "wdr", "mis"
    ]
    left_raw: float | None
    right_raw: float | None
    contrast_normalized: float | None
    failure_code: str | None

    def _validate(self) -> None:
        complete = all(
            x is not None for x in (self.left_raw, self.right_raw, self.contrast_normalized)
        )
        _require(complete == (self.failure_code is None), "baseline completion/failure mismatch")


@dataclass(frozen=True)
class BatchReference(EvidenceRecord):
    kind: Literal["batch_reference"]
    batch_sha256: str
    kernel_sha256: str
    policies_sha256: str
    normalization: float
    truth_left_raw: float
    truth_right_raw: float
    truth_contrast_normalized: float
    policy_rows: tuple[PolicyReference, PolicyReference]
    bootstrap: PairedWPDISBootstrap
    baseline_rows: tuple[BaselinePair, ...]

    def _validate(self) -> None:
        _require(self.normalization > 0, "invalid normalization")
        _require(tuple(r.id for r in self.policy_rows) == ("A", "B"), "wrong policy order")
        _require(
            self.policy_rows[0].policy_sha256 != self.policy_rows[1].policy_sha256,
            "duplicate policies",
        )
        _require(self.policy_rows[0].counts == self.policy_rows[1].counts, "policy shape mismatch")
        _require(
            tuple(r.estimator_id for r in self.baseline_rows) == BASELINE_IDS,
            "incomplete baseline order",
        )
        _near(
            self.truth_contrast_normalized,
            (self.truth_right_raw - self.truth_left_raw) / self.normalization,
        )
        _near(self.bootstrap.normalization, self.normalization)
        for row, raw in zip(
            self.policy_rows, (self.bootstrap.raw_left, self.bootstrap.raw_right), strict=True
        ):
            _near(row.normalized_value, row.raw_value / self.normalization)
            _near(row.raw_value, raw)
        for baseline in self.baseline_rows:
            if baseline.failure_code is None:
                if baseline.right_raw is None or baseline.left_raw is None:
                    raise ValueError("baseline completion/failure mismatch")
                _near(
                    baseline.contrast_normalized,
                    (baseline.right_raw - baseline.left_raw) / self.normalization,
                )


@dataclass(frozen=True)
class CipherValue(EvidenceRecord):
    policy_sha256: str
    mean_weighted_rewards: tuple[float, ...]
    mean_weights: tuple[float, ...]
    counts: tuple[int, ...]
    raw_value: float
    normalized_value: float

    def _validate(self) -> None:
        _vectors(self.mean_weighted_rewards, self.mean_weights, self.counts)


@dataclass(frozen=True)
class CipherErrors(EvidenceRecord):
    normalized_value_errors: tuple[float, float]
    normalized_contrast_error: float
    mean_numerator_abs_errors: tuple[tuple[float, ...], tuple[float, ...]]
    mean_denominator_abs_errors: tuple[tuple[float, ...], tuple[float, ...]]
    mean_denominator_relative_errors: tuple[tuple[float, ...], tuple[float, ...]]
    normalized_ratio_perturbation_bounds: tuple[float | None, float | None]

    def _validate(self) -> None:
        lengths = {
            len(v)
            for vectors in (
                self.mean_numerator_abs_errors,
                self.mean_denominator_abs_errors,
                self.mean_denominator_relative_errors,
            )
            for v in vectors
        }
        _require(len(lengths) == 1 and next(iter(lengths)) in range(1, 65), "error arrays differ")
        values = (
            *self.normalized_value_errors,
            self.normalized_contrast_error,
            *(
                v
                for arrays in (
                    self.mean_numerator_abs_errors,
                    self.mean_denominator_abs_errors,
                    self.mean_denominator_relative_errors,
                )
                for row in arrays
                for v in row
            ),
            *(v for v in self.normalized_ratio_perturbation_bounds if v is not None),
        )
        _require(all(v >= 0 for v in values), "negative absolute error")


_TIMER_NAMES = frozenset(
    (
        "keygen/context serialization",
        "client preprocessing",
        "encryption/request serialization",
        "server context/request loading",
        "A evaluation/response serialization",
        "B evaluation/response serialization",
        "A decrypt/validation",
        "B decrypt/validation",
    )
)


@dataclass(frozen=True)
class TimerRecord(EvidenceRecord):
    name: str
    elapsed_ns: int
    parent: str | None

    def _validate(self) -> None:
        _require(
            self.name in _TIMER_NAMES
            and (self.parent is None or self.parent in _TIMER_NAMES)
            and self.parent != self.name,
            "invalid timer name or parent",
        )


@dataclass(frozen=True)
class TimingReceipt(EvidenceRecord):
    total_ns: int | None
    job_elapsed_ns: int
    clear_reference_ns: int | None
    peak_rss_bytes: int
    operations: tuple[TimerRecord, ...]

    def _validate(self) -> None:
        _require(
            self.total_ns is None or 0 < self.total_ns <= self.job_elapsed_ns, "invalid total timer"
        )
        _require(len({r.name for r in self.operations}) == len(self.operations), "duplicate timer")
        lookup = {r.name: r for r in self.operations}
        for row in self.operations:
            if row.parent is not None:
                _require(
                    row.parent in lookup
                    and row.elapsed_ns <= lookup[row.parent].elapsed_ns
                    and lookup[row.parent].parent is None,
                    "invalid nested timer",
                )


@dataclass(frozen=True)
class ClearBatchMetrics(EvidenceRecord):
    kind: Literal["clear_batch"]
    reference: BatchReference


@dataclass(frozen=True)
class CipherBatchMetrics(EvidenceRecord):
    kind: Literal["cipher_batch"]
    reference: BatchReference | None
    left: CipherValue | None
    right: CipherValue | None
    errors: CipherErrors | None
    cipher_contrast_normalized: float | None
    cipher_interval_lower: float | None
    cipher_interval_upper: float | None
    cipher_decision: Literal["positive", "negative", "abstain", "unavailable"]
    timing: TimingReceipt

    def _validate(self) -> None:
        _require(self.right is None or self.left is not None, "B cannot precede A")
        complete = self.reference is not None and self.left is not None and self.right is not None
        fields = (
            self.errors,
            self.cipher_contrast_normalized,
            self.cipher_interval_lower,
            self.cipher_interval_upper,
        )
        _require(
            all(x is not None for x in fields) if complete else all(x is None for x in fields),
            "inconsistent partial ciphertext metrics",
        )
        _require(
            (self.cipher_decision != "unavailable") == complete, "decision without complete values"
        )
        if self.reference is not None:
            for value, ref in zip((self.left, self.right), self.reference.policy_rows, strict=True):
                if value is not None:
                    _require(
                        value.policy_sha256 == ref.policy_sha256 and value.counts == ref.counts,
                        "cipher policy/shape mismatch",
                    )
                    _near(value.normalized_value, value.raw_value / self.reference.normalization)
        if self.reference is not None and self.left is not None and self.right is not None:
            from unseen_loop.ope.study import translate_cipher_interval

            contrast = self.right.normalized_value - self.left.normalized_value
            _near(self.cipher_contrast_normalized, contrast)
            lower, upper, decision = translate_cipher_interval(self.reference.bootstrap, contrast)
            _near(self.cipher_interval_lower, lower)
            _near(self.cipher_interval_upper, upper)
            _require(self.cipher_decision == decision, "incorrect interval decision")


@dataclass(frozen=True)
class CountPrecisionMetrics(EvidenceRecord):
    kind: Literal["count_precision"]
    expected_sum: int
    raw_decrypted_sum: float | None
    absolute_error: float | None
    timing: TimingReceipt

    def _validate(self) -> None:
        _require(
            self.expected_sum > 0
            and (self.raw_decrypted_sum is None) == (self.absolute_error is None),
            "invalid count diagnostic",
        )
        if self.raw_decrypted_sum is not None:
            _near(self.absolute_error, abs(self.raw_decrypted_sum - self.expected_sum))


@dataclass(frozen=True)
class HistoricalMetrics(EvidenceRecord):
    kind: Literal["historical"]
    identifier: Literal["POLYNOMIAL_APPROX_OPE_V1"]
    target_policy_sha256: str
    clear_statistics: SufficientStatistics
    decrypted_statistics: SufficientStatistics | None
    raw_decrypted_counts: tuple[float, ...] | None
    maximum_numerator_error: float | None
    maximum_denominator_error: float | None
    estimate_error: float | None
    timing: TimingReceipt

    def _validate(self) -> None:
        if self.raw_decrypted_counts is not None:
            _require(
                len(self.raw_decrypted_counts) == len(self.clear_statistics.counts),
                "historical count horizon mismatch",
            )
        if self.decrypted_statistics is None:
            _require(
                all(
                    v is None
                    for v in (
                        self.maximum_numerator_error,
                        self.maximum_denominator_error,
                        self.estimate_error,
                    )
                ),
                "historical errors without result",
            )
        else:
            _require(
                len(self.decrypted_statistics.counts) == len(self.clear_statistics.counts),
                "historical result shape mismatch",
            )
            _near(
                self.maximum_numerator_error,
                max(
                    abs(a - b)
                    for a, b in zip(
                        self.clear_statistics.numerators,
                        self.decrypted_statistics.numerators,
                        strict=True,
                    )
                ),
            )
            _near(
                self.maximum_denominator_error,
                max(
                    abs(a - b)
                    for a, b in zip(
                        self.clear_statistics.denominators,
                        self.decrypted_statistics.denominators,
                        strict=True,
                    )
                ),
            )
            clear_estimate = self.clear_statistics.estimate
            decrypted_estimate = self.decrypted_statistics.estimate
            if clear_estimate is None or decrypted_estimate is None:
                raise ValueError("historical estimate is unavailable")
            _near(
                self.estimate_error,
                abs(clear_estimate - decrypted_estimate),
            )


@dataclass(frozen=True)
class VerificationOutcome(EvidenceRecord):
    node_id: str
    outcome: Literal["passed", "failed", "skipped", "error"]

    def _validate(self) -> None:
        _require(
            re.fullmatch(r"tests/test_[A-Za-z0-9_]+\.py::[A-Za-z0-9_:\[\].-]+", self.node_id)
            is not None,
            "unsafe verification node ID",
        )


@dataclass(frozen=True)
class VerificationMetrics(EvidenceRecord):
    kind: Literal["verification"]
    test_source_sha256: str
    collected_node_ids: tuple[str, ...]
    outcomes: tuple[VerificationOutcome, ...]
    exit_code: int
    required_cases_passed: bool
    elapsed_ns: int

    def _validate(self) -> None:
        _require(
            len(set(self.collected_node_ids)) == len(self.collected_node_ids),
            "duplicate collected test",
        )
        _require(
            tuple(o.node_id for o in self.outcomes) == self.collected_node_ids,
            "verification outcome denominator mismatch",
        )
        _require(
            self.required_cases_passed
            == (
                self.exit_code == 0
                and bool(self.outcomes)
                and all(o.outcome == "passed" for o in self.outcomes)
            ),
            "verification status mismatch",
        )


@dataclass(frozen=True)
class ProbeMetrics(EvidenceRecord):
    kind: Literal["probe"]
    expected_failure_code: str
    observed_failure_code: str | None
    elapsed_ns: int

    def _validate(self) -> None:
        _require(
            self.expected_failure_code in ("probe.deliberate_exception", "runtime.timeout")
            and self.observed_failure_code in FAILURE_CODES | {None},
            "invalid probe category",
        )


@dataclass(frozen=True)
class CountSummary(EvidenceRecord):
    planned: int
    attempted_true: int
    attempted_false: int
    attempted_unknown: int
    completed: int

    def _validate(self) -> None:
        _require(
            self.planned == self.attempted_true + self.attempted_false + self.attempted_unknown
            and self.completed <= self.attempted_true,
            "count denominator mismatch",
        )


@dataclass(frozen=True)
class AggregateComparison(EvidenceRecord):
    planned_batches: int
    complete_batches: int
    truth_contrast_normalized: float | None
    bias: float | None
    bias_lower: float | None
    bias_upper: float | None
    rmse: float | None
    rmse_lower: float | None
    rmse_upper: float | None
    coverage_successes: int
    coverage_lower: float | None
    correct_choice_successes: int
    correct_choice_lower: float | None
    abstentions: int
    median_width: float | None
    left_median_ess_fraction: float | None
    left_p05_ess_fraction: float | None
    right_median_ess_fraction: float | None
    right_p05_ess_fraction: float | None
    clear_cipher_decision_disagreements: int

    def _validate(self) -> None:
        _require(self.complete_batches <= self.planned_batches, "aggregate completion exceeds plan")
        _require(
            all(
                v <= self.complete_batches
                for v in (
                    self.coverage_successes,
                    self.correct_choice_successes,
                    self.abstentions,
                    self.clear_cipher_decision_disagreements,
                )
            ),
            "aggregate successes exceed completion",
        )


@dataclass(frozen=True)
class BaselineAggregate(EvidenceRecord):
    cohort: Literal["primary", "stress"]
    estimator_id: Literal[
        "is", "pdis", "wpdis", "clipped_wpdis_2", "clipped_wpdis_10", "dm", "dr", "wdr", "mis"
    ]
    planned_batches: int
    complete_batches: int
    conditional_on_completion: bool
    left_bias: float | None
    left_rmse: float | None
    right_bias: float | None
    right_rmse: float | None
    contrast_bias: float | None
    contrast_bias_lower: float | None
    contrast_bias_upper: float | None
    contrast_rmse: float | None
    contrast_rmse_lower: float | None
    contrast_rmse_upper: float | None

    def _validate(self) -> None:
        _require(
            self.complete_batches <= self.planned_batches
            and self.conditional_on_completion == (self.complete_batches != self.planned_batches),
            "baseline denominator mismatch",
        )


@dataclass(frozen=True)
class TimingPair(EvidenceRecord):
    case_id: str
    candidate_job_id: str
    baseline_job_id: str
    reference_digest_matches: bool
    valid: bool
    speedup: float | None

    def _validate(self) -> None:
        _require(
            (self.speedup is not None) == self.valid
            and (
                not self.valid
                or (self.reference_digest_matches and self.speedup is not None and self.speedup > 0)
            ),
            "invalid timing pair",
        )


@dataclass(frozen=True)
class AggregateTiming(EvidenceRecord):
    planned_pairs: int
    valid_pairs: int
    pairs: tuple[TimingPair, ...]
    median_speedup: float | None
    speedup_lower: float | None

    def _validate(self) -> None:
        _require(
            len(self.pairs) == self.planned_pairs
            and sum(p.valid for p in self.pairs) == self.valid_pairs,
            "timing pair denominator mismatch",
        )
        _require(
            (self.median_speedup is not None and self.speedup_lower is not None)
            == (self.valid_pairs == self.planned_pairs and self.planned_pairs > 0),
            "partial timing summary forbidden",
        )


@dataclass(frozen=True)
class DiagnosticSummary(EvidenceRecord):
    planned_count_contexts: Literal[12]
    new_profile_passes: int
    old_profile_passes: int
    verification_passed: bool
    expected_probe_failures: int

    def _validate(self) -> None:
        _require(
            self.new_profile_passes <= 6
            and self.old_profile_passes <= 6
            and self.expected_probe_failures <= 2,
            "diagnostic denominator mismatch",
        )


@dataclass(frozen=True)
class GateResult(EvidenceRecord):
    id: str
    passed: bool
    numerator: int | None
    denominator: int | None
    observed: float | None
    lower: float | None
    upper: float | None
    comparison: str
    threshold: float | None
    reason_code: str | None

    def _validate(self) -> None:
        _require(self.id in GATE_IDS, "unknown gate")
        if self.numerator is not None and self.denominator is not None:
            _require(self.numerator <= self.denominator, "gate denominator mismatch")


@dataclass(frozen=True)
class AnalysisMetrics(EvidenceRecord):
    kind: Literal["analysis"]
    phase: Literal["diagnostic", "pilot", "confirmation"]
    planned_job_ids: tuple[str, ...]
    attempt_row_sha256: tuple[str, ...]
    counts_by_kind: dict[str, CountSummary]
    primary: AggregateComparison | None
    stress: AggregateComparison | None
    baselines: tuple[BaselineAggregate, ...]
    timing: AggregateTiming | None
    diagnostic: DiagnosticSummary | None
    gates: tuple[GateResult, ...]
    promotion_allowed: bool
    status: Literal["passed", "failed"]

    def _validate(self) -> None:
        _require(
            len(self.planned_job_ids)
            == len(set(self.planned_job_ids))
            == len(self.attempt_row_sha256),
            "analysis row denominator mismatch",
        )
        _require(
            all(_DIGEST.fullmatch(d) is not None for d in self.attempt_row_sha256),
            "invalid row digest",
        )
        _require(
            sum(c.planned for c in self.counts_by_kind.values()) == len(self.planned_job_ids),
            "analysis kind denominator mismatch",
        )
        _require(len(set(g.id for g in self.gates)) == len(self.gates), "duplicate gates")
        _require(
            self.promotion_allowed == all(g.passed for g in self.gates)
            and bool(self.gates)
            and (self.status == "passed") == self.promotion_allowed,
            "analysis gate status mismatch",
        )
        diagnostic_ids = {
            "evidence_closure",
            "verification",
            "diagnostic_precision",
            "probe_accounting",
        }
        shared_ids = {
            "evidence_closure",
            "interval_width",
            "left_ess_median",
            "left_ess_p05",
            "right_ess_median",
            "right_ess_p05",
            "required_context_completion",
            "cipher_numerics",
            "resource_bounds",
            "timing_pair_completion",
            "timing_median",
            "timing_lower",
        }
        expected_ids = (
            diagnostic_ids
            if self.phase == "diagnostic"
            else shared_ids
            | (
                {"clear_gap", "clear_coverage", "clear_choice"}
                if self.phase == "pilot"
                else {
                    "confirmation_coverage",
                    "confirmation_choice",
                    "confirmation_bias",
                    "confirmation_rmse",
                }
            )
        )
        _require(
            tuple(g.id for g in self.gates) == tuple(k for k in GATE_IDS if k in expected_ids),
            "incomplete or reordered phase gates",
        )
        expected_counts = {
            "diagnostic": {
                "protocol_verification": 1,
                "count_precision": 12,
                "smoke_error": 1,
                "smoke_timeout": 1,
            },
            "pilot": {
                "clear_batch": 128,
                "paired_context": 20,
                "ablation_context": 6,
                "historical_context": 1,
            },
            "confirmation": {"statistical_context": 200, "timing_context": 40},
        }[self.phase]
        _require(
            {k: v.planned for k, v in self.counts_by_kind.items()} == expected_counts,
            "phase denominator mismatch",
        )
        if self.phase == "diagnostic":
            _require(
                self.primary is None
                and self.stress is None
                and self.timing is None
                and self.diagnostic is not None
                and not self.baselines,
                "invalid diagnostic variants",
            )
        else:
            if self.primary is None or self.timing is None or self.diagnostic is not None:
                raise ValueError("missing scientific aggregate")
            _require(
                self.primary.planned_batches == (64 if self.phase == "pilot" else 200)
                and self.timing.planned_pairs == (10 if self.phase == "pilot" else 20),
                "scientific denominator mismatch",
            )
            _require((self.stress is not None) == (self.phase == "pilot"), "stress phase mismatch")
            if self.stress is not None:
                _require(self.stress.planned_batches == 64, "stress denominator mismatch")
            cohorts = ("primary", "stress") if self.phase == "pilot" else ("primary",)
            _require(
                tuple((r.cohort, r.estimator_id) for r in self.baselines)
                == tuple((c, e) for c in cohorts for e in BASELINE_IDS),
                "incomplete baseline aggregates",
            )


Metrics = (
    ClearBatchMetrics
    | CipherBatchMetrics
    | CountPrecisionMetrics
    | HistoricalMetrics
    | VerificationMetrics
    | ProbeMetrics
    | AnalysisMetrics
)


@dataclass(frozen=True)
class PrivateOPEAttempt(EvidenceRecord):
    schema_version: Literal["unseen-loop/private-ope-attempt-v1"]
    run_id: str
    config_sha256: str
    provenance_sha256: str
    job: PlannedJob
    function_call_id: str | None
    input_id: str | None
    attempted: bool | None
    completed: bool
    failure_code: str | None
    worker_result_sha256: str | None
    metrics: Metrics | None
    receipts: AttemptReceipts
    private_rows_persisted: Literal[False]
    secret_material_persisted: Literal[False]

    def _validate(self) -> None:
        _require(_RUN.fullmatch(self.run_id) is not None, "invalid attempt run")
        _require(
            (self.function_call_id is None) == (self.input_id is None),
            "incomplete invocation identity",
        )
        if self.input_id is not None:
            _require(
                _ID.fullmatch(self.input_id) is not None
                and self.function_call_id is not None
                and _ID.fullmatch(self.function_call_id) is not None
                and self.attempted is True,
                "invalid worker entry identity",
            )
        _require(
            (
                self.completed
                and self.attempted is True
                and self.failure_code is None
                and self.metrics is not None
            )
            or (not self.completed and self.failure_code is not None),
            "attempt completion/failure mismatch",
        )
        coords = dict(self.job.coordinates)
        kind = coords["kind"]
        if not isinstance(kind, str):
            raise ValueError("unknown job kind")
        expected = {
            "clear_batch": ClearBatchMetrics,
            "count_precision": CountPrecisionMetrics,
            "historical_context": HistoricalMetrics,
            "protocol_verification": VerificationMetrics,
            "smoke_error": ProbeMetrics,
            "smoke_timeout": ProbeMetrics,
            "analysis": AnalysisMetrics,
        }
        if self.metrics is not None:
            _require(
                isinstance(
                    self.metrics, CipherBatchMetrics if kind in MODERN_KINDS else expected[kind]
                ),
                "metrics tag disagrees with job",
            )
        slots = 2 if kind in MODERN_KINDS else 1 if kind == "historical_context" else 0
        expected_counts_source = (
            "public_fixed_shape"
            if kind in MODERN_KINDS
            else "legacy_encrypted"
            if kind == "historical_context"
            else "diagnostic_sum"
            if kind == "count_precision"
            else "not-applicable"
        )
        _require(
            self.receipts.counts_source == expected_counts_source,
            "counts source disagrees with job",
        )
        _require(
            self.run_id.startswith(self.job.stage.replace("private_ope_", "private-ope-") + "-"),
            "attempt phase mismatch",
        )
        _require(len(self.receipts.response_sha256) == slots, "wrong A/B receipt arity")
        if isinstance(self.metrics, (ClearBatchMetrics, CipherBatchMetrics)):
            ref = self.metrics.reference
            if ref is not None:
                _require(
                    ref.policy_rows[0].counts
                    == (coords["trajectories"],) * _coordinate_int(coords["horizon"]),
                    "reference/job shape mismatch",
                )
                _require(self.receipts.batch_sha256 == ref.batch_sha256, "batch receipt mismatch")
        if isinstance(self.metrics, CountPrecisionMetrics):
            _require(
                self.metrics.expected_sum == coords["trajectories"], "diagnostic/job count mismatch"
            )
            _require(
                not self.completed or self.metrics.raw_decrypted_sum is not None,
                "completed count has no raw aggregate",
            )
        if isinstance(self.metrics, HistoricalMetrics):
            _require(
                len(self.metrics.clear_statistics.counts) == coords["horizon"],
                "historical/job horizon mismatch",
            )
            _require(
                not self.completed or self.metrics.decrypted_statistics is not None,
                "completed historical context lacks result",
            )
        if isinstance(self.metrics, VerificationMetrics):
            _require(
                not self.completed or self.metrics.required_cases_passed,
                "failed verification marked complete",
            )
        if isinstance(self.metrics, ProbeMetrics):
            _require(
                not self.completed and self.metrics.observed_failure_code == self.failure_code,
                "invalid probe completion",
            )
        if self.completed:
            _require(
                self.receipts.runtime is not None and self.receipts.runtime.source_match,
                "completion lacks verified runtime source",
            )
            if kind in MODERN_KINDS | {"historical_context", "count_precision"}:
                _require(
                    self.receipts.context is not None
                    and self.receipts.computation_sha256 is not None
                    and self.receipts.request_sha256 is not None,
                    "completion lacks cryptographic receipts",
                )
                _require(
                    all(v is not None for v in self.receipts.response_sha256),
                    "completion lacks policy response receipts",
                )
        if isinstance(self.metrics, CipherBatchMetrics) and self.completed:
            _require(
                self.metrics.right is not None and self.metrics.timing.total_ns is not None,
                "completed context missing B result",
            )
        if self.attempted is not True:
            _require(
                not self.completed and self.metrics is None,
                "unattempted job cannot have measurements",
            )


def _near(actual: float | None, expected: float, tolerance: float = TOLERANCE) -> None:
    _require(
        actual is not None and math.isfinite(actual) and abs(actual - expected) <= tolerance,
        "stored numerical evidence differs from replay",
    )


def validate_job_payload(
    config_bytes: bytes, payload: dict[str, object] | PrivateOPEJobPayload, run_root: str
) -> PrivateOPEJobPayload:
    from unseen_loop.flagship.manifest import (
        iter_private_ope_jobs,
        parse_private_ope_manifest_bytes,
    )

    manifest = parse_private_ope_manifest_bytes(config_bytes)
    parsed = PrivateOPEJobPayload.from_dict(
        payload.to_dict() if isinstance(payload, PrivateOPEJobPayload) else payload
    )
    digest = _sha(config_bytes)
    run_id = f"private-ope-{manifest.phase}-{digest[:24]}"
    _require(
        parsed.config_sha256 == digest and parsed.run_id == run_id,
        "job config/run identity mismatch",
    )
    root = Path(run_root)
    _require(
        root.name == run_id and root.parent.name == "private-ope" and ".." not in root.parts,
        "unsafe canonical run root",
    )
    jobs = {job.job_id: job for job in iter_private_ope_jobs(manifest)}
    _require(
        parsed.job.job_id in jobs
        and job_to_dict(parsed.job) == job_to_dict(jobs[parsed.job.job_id]),
        "job does not exactly match reserved expansion",
    )
    return parsed


def _independent_products(values: Sequence[CKKSEncryptedVector]) -> tuple[CKKSEncryptedVector, ...]:
    """Benchmark-only recomputation: a separate balanced tree per prefix."""

    def reduce_tree(items: Sequence[CKKSEncryptedVector]) -> CKKSEncryptedVector:
        if len(items) == 1:
            value = items[0]
            return value._new(value._vector.copy(), slots=value.slots)
        split = 1 << ((len(items) - 1).bit_length() - 1)
        left, right = reduce_tree(items[:split]), reduce_tree(items[split:])
        return right * left

    return tuple(reduce_tree(values[:stop]) for stop in range(1, len(values) + 1))


RAW_IDENTIFIER = "RAW_PREFIX_WPDIS_BENCHMARK_V1"


@dataclass(frozen=True)
class RawPrefixSpec:
    """Independent raw-input, linear-policy, unclipped benchmark program.

    Unlike V1 this program has no soft clipping and never encrypts public
    counts.  Its extra action-selection/reciprocal levels are paid explicitly.
    """

    trajectories: TrajectorySpec
    target_policies: tuple[PolynomialPolicySpec, ...]
    gamma: float
    minimum_behavior_propensity: float
    maximum_importance_ratio: float

    def __post_init__(self) -> None:
        _require(1 <= self.trajectories.horizon <= 64, "domain.invalid_input")
        _require(0 <= self.gamma <= 1 and math.isfinite(self.gamma), "domain.invalid_input")
        _require(
            0 < self.minimum_behavior_propensity <= 1
            and self.maximum_importance_ratio > 0
            and math.isfinite(self.maximum_importance_ratio),
            "domain.invalid_input",
        )
        _require(
            bool(self.trajectories.state_min)
            and self.trajectories.reward_min is not None
            and self.trajectories.reward_max is not None,
            "domain.range_bound",
        )
        _require(
            bool(self.target_policies)
            and len({_sha(canonical_bytes(p.to_dict())) for p in self.target_policies})
            == len(self.target_policies),
            "domain.invalid_input",
        )
        for policy in self.target_policies:
            _require(
                policy.degree == 1
                and policy.state_dim == self.trajectories.state_dim
                and policy.action_count == self.trajectories.action_count,
                "domain.invalid_input",
            )
            policy.probability_bounds(self.trajectories)
            _require(
                all(any(c != 0 for c in row[1:]) for row in policy.coefficients),
                "raw benchmark requires a nonconstant term per action",
            )
        self.computation_receipt()

    @property
    def parameters(self) -> CKKSParameters:
        depth = (self.trajectories.horizon - 1).bit_length() + 4
        return CKKSParameters(16384, (60, *([32] * depth), 58), float(2**32))

    def computation_receipt(
        self, actual_primes: tuple[int, ...] | None = None
    ) -> dict[str, object]:
        from unseen_loop.ope.lifted import _RangeProof, _reduction_term_counts

        proof = _RangeProof(self.parameters, actual_primes)
        n, horizon = self.trajectories.trajectories, self.trajectories.horizon
        magnitude = tuple(
            max(abs(a), abs(b))
            for a, b in zip(self.trajectories.state_min, self.trajectories.state_max, strict=True)
        )
        if self.trajectories.reward_min is None or self.trajectories.reward_max is None:
            raise ValueError("domain.range_bound")
        reward = max(abs(self.trajectories.reward_min), abs(self.trajectories.reward_max))

        def upward(value: float) -> float:
            return math.nextafter(value, math.inf)

        ratio_bound = upward(self.maximum_importance_ratio)
        reciprocal_bound = upward(1 / self.minimum_behavior_propensity)
        normalized_reward = upward(reward / n)
        for j, bound in enumerate(magnitude):
            proof.check(f"raw_state_{j}", bound, 0, scale_bits=32)
        proof.check("action_mask", 1.0, 0, scale_bits=32)
        proof.check("behavior_reciprocal", reciprocal_bound, 0, scale_bits=32)
        proof.check("normalized_reward", normalized_reward, 0, scale_bits=32)
        for policy in self.target_policies:
            digest = _sha(canonical_bytes(policy.to_dict()))
            action_bounds = []
            for a, row in enumerate(policy.coefficients):
                running = 0.0
                for j, coefficient in enumerate(row[1:]):
                    if coefficient == 0:
                        continue
                    proof.check(
                        f"public_coefficient_{a}_{j}",
                        abs(coefficient),
                        0,
                        scale_bits=32,
                        policy_sha256=digest,
                    )
                    term = upward(abs(coefficient) * magnitude[j])
                    proof.product(f"coefficient_{a}_{j}", term, 0, 0, policy_sha256=digest)
                    running = upward(running + term)
                    proof.check(
                        f"score_partial_{a}_{j}", running, 1, scale_bits=32, policy_sha256=digest
                    )
                proof.check(
                    f"public_intercept_{a}", abs(row[0]), 1, scale_bits=32, policy_sha256=digest
                )
                running = upward(running + abs(row[0]))
                proof.check(f"score_intercept_{a}", running, 1, scale_bits=32, policy_sha256=digest)
                proof.product(f"action_selection_{a}", running, 0, 1, policy_sha256=digest)
                action_bounds.append(running)
            running = 0.0
            for a, bound in enumerate(action_bounds):
                running = upward(running + bound)
                proof.check(
                    f"selected_partial_{a}", running, 2, scale_bits=32, policy_sha256=digest
                )
            proof.product(
                "selected_reciprocal",
                upward(running * reciprocal_bound),
                0,
                2,
                policy_sha256=digest,
            )
            # Honest-client logged-ratio validation establishes the tighter
            # semantic ratio bound only after the complete public polynomial.
            proof.check("validated_ratio", ratio_bound, 3, scale_bits=32, policy_sha256=digest)

            def prefix_nodes(
                length: int, offset: int = 0, policy_sha256: str = digest
            ) -> list[tuple[float, int]]:
                if length == 1:
                    return [(ratio_bound, 3)]
                split = 1 << ((length - 1).bit_length() - 1)
                left = prefix_nodes(split, offset)
                right = prefix_nodes(length - split, offset + split)
                merged = []
                for k, (bound, level) in enumerate(right):
                    product = upward(bound * left[-1][0])
                    result_level = proof.product(
                        f"prefix_{offset}_{length}_{k}",
                        product,
                        level,
                        left[-1][1],
                        policy_sha256=policy_sha256,
                    )
                    merged.append((product, result_level))
                return left + merged

            for t, (bound, level) in enumerate(prefix_nodes(horizon)):
                numerator_lane = upward(normalized_reward * bound)
                denominator_lane = upward(bound * upward(1 / n))
                final = proof.product(
                    f"numerator_lane_{t}", numerator_lane, 0, level, policy_sha256=digest
                )
                proof.product(
                    f"denominator_lane_{t}", denominator_lane, level, 0, policy_sha256=digest
                )
                merged_a = merged_b = 0.0
                for chunk in plan_chunks(n, self.parameters).chunks:
                    for index, terms in enumerate(_reduction_term_counts(chunk.slots)):
                        proof.check(
                            f"numerator_reduce_{t}_{chunk.start}_{index}",
                            upward(numerator_lane * terms),
                            final,
                            scale_bits=32,
                            policy_sha256=digest,
                        )
                        proof.check(
                            f"denominator_reduce_{t}_{chunk.start}_{index}",
                            upward(denominator_lane * terms),
                            final,
                            scale_bits=32,
                            policy_sha256=digest,
                        )
                    merged_a = upward(merged_a + upward(numerator_lane * chunk.slots))
                    merged_b = upward(merged_b + upward(denominator_lane * chunk.slots))
                    proof.check(
                        f"numerator_merged_{t}_{chunk.start}",
                        merged_a,
                        final,
                        scale_bits=32,
                        policy_sha256=digest,
                    )
                    proof.check(
                        f"denominator_merged_{t}_{chunk.start}",
                        merged_b,
                        final,
                        scale_bits=32,
                        policy_sha256=digest,
                    )
        return {
            "schema_version": "unseen-loop/raw-prefix-wpdis-computation-v1",
            "identifier": RAW_IDENTIFIER,
            "counts_source": "public_fixed_shape",
            "parameters": dataclasses.asdict(self.parameters),
            "actual_coeff_modulus_primes": actual_primes,
            "trajectories": self.trajectories.to_dict(),
            "policy_sha256": [_sha(canonical_bytes(p.to_dict())) for p in self.target_policies],
            "gamma": self.gamma,
            "minimum_behavior_propensity": self.minimum_behavior_propensity,
            "maximum_importance_ratio": self.maximum_importance_ratio,
            "intermediate_bounds": [dataclasses.asdict(row) for row in proof.rows],
        }

    def validate_batch(self, batch: TrajectoryBatch) -> None:
        _require(batch.spec == self.trajectories, "domain.invalid_input")
        _require(not batch.validation_failures(), "domain.invalid_input")
        _require(
            bool(np.all(batch.behavior_array >= self.minimum_behavior_propensity)),
            "domain.ratio_bound",
        )
        for policy in self.target_policies:
            ratios = policy.logged_action_probabilities(batch) / batch.behavior_array
            _require(
                bool(
                    np.all(np.isfinite(ratios))
                    and np.all(ratios >= 0)
                    and np.all(ratios <= math.nextafter(self.maximum_importance_ratio, math.inf))
                ),
                "domain.ratio_bound",
            )


@dataclass(frozen=True)
class _RawRequest:
    metadata: Mapping[str, object]
    vectors: tuple[SerializedCKKSVector, ...]
    _wire_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        row = _keys(
            dict(self.metadata),
            {
                "identifier",
                "trajectories",
                "horizon",
                "state_dim",
                "action_count",
                "feature_degree",
                "chunks",
            },
        )
        chunks = row["chunks"]
        if not isinstance(chunks, (tuple, list)) or any(
            not isinstance(chunk, (tuple, list)) or len(chunk) != 2 for chunk in chunks
        ):
            raise ValueError("ckks.request_binding")
        row["chunks"] = tuple(tuple(chunk) for chunk in chunks)
        object.__setattr__(self, "metadata", types.MappingProxyType(row))
        self.validate()

    def validate(self) -> None:
        if (
            self.metadata["identifier"] != RAW_IDENTIFIER
            or _coordinate_int(self.metadata["feature_degree"]) != 1
        ):
            raise ValueError("ckks.request_binding")
        n, horizon, dimensions, actions = (
            _coordinate_int(self.metadata[name])
            for name in ("trajectories", "horizon", "state_dim", "action_count")
        )
        if n < 1 or not 1 <= horizon <= 64 or dimensions < 1 or actions < 2:
            raise ValueError("ckks.request_binding")
        chunks = self.metadata["chunks"]
        if not isinstance(chunks, tuple) or not isinstance(self.vectors, tuple):
            raise ValueError("ckks.request_binding")
        per_chunk = horizon * (dimensions + actions + 2)
        if len(self.vectors) != len(chunks) * per_chunk:
            raise ValueError("ckks.request_binding")
        previous = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise ValueError("ckks.request_binding")
            start, stop = (_coordinate_int(value) for value in chunk)
            if start != previous or not start < stop <= n:
                raise ValueError("ckks.request_binding")
            for vector in self.vectors[index * per_chunk : (index + 1) * per_chunk]:
                if type(vector.slots) is not int or vector.slots != stop - start:
                    raise ValueError("ckks.request_binding")
            previous = stop
        if previous != n:
            raise ValueError("ckks.request_binding")

    def to_bytes(self) -> bytes:
        from unseen_loop.ope.lifted import _frame

        self.validate()
        wire = self._wire_bytes
        if wire is None:
            wire = _frame(dict(self.metadata), self.vectors)
            object.__setattr__(self, "_wire_bytes", wire)
        return wire

    @property
    def digest(self) -> str:
        return _sha(self.to_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes, spec: RawPrefixSpec) -> Self:
        from unseen_loop.ope.lifted import _unframe

        metadata, vectors = _unframe(payload)
        request = cls(metadata, vectors)
        _validate_raw_request(request, spec)
        object.__setattr__(request, "_wire_bytes", payload)
        return request


def _raw_metadata(spec: RawPrefixSpec) -> dict[str, object]:
    shape = spec.trajectories
    return {
        "identifier": RAW_IDENTIFIER,
        "trajectories": shape.trajectories,
        "horizon": shape.horizon,
        "state_dim": shape.state_dim,
        "action_count": shape.action_count,
        "feature_degree": 1,
        "chunks": tuple(
            (c.start, c.stop) for c in plan_chunks(shape.trajectories, spec.parameters).chunks
        ),
    }


def _validate_raw_request(request: _RawRequest, spec: RawPrefixSpec) -> None:
    request.validate()
    _require(request.metadata == _raw_metadata(spec), "ckks.request_binding")


@dataclass(frozen=True)
class _RawResponse:
    policy_sha256: str
    request_sha256: str
    mean_weighted_rewards: tuple[SerializedCKKSVector, ...]
    mean_weights: tuple[SerializedCKKSVector, ...]
    _wire_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            _DIGEST.fullmatch(self.policy_sha256) is None
            or _DIGEST.fullmatch(self.request_sha256) is None
        ):
            raise ValueError("ckks.request_binding")
        if not isinstance(self.mean_weighted_rewards, tuple) or not isinstance(
            self.mean_weights, tuple
        ):
            raise ValueError("ckks.request_binding")
        if not 1 <= len(self.mean_weights) <= 64 or len(self.mean_weighted_rewards) != len(
            self.mean_weights
        ):
            raise ValueError("ckks.request_binding")
        if any(
            type(v.slots) is not int or v.slots != 1
            for v in self.mean_weighted_rewards + self.mean_weights
        ):
            raise ValueError("ckks.request_binding")

    def validate_binding(self, request: _RawRequest, spec: RawPrefixSpec) -> None:
        self.validate()
        _require(
            self.request_sha256 == request.digest
            and self.policy_sha256
            in {_sha(canonical_bytes(p.to_dict())) for p in spec.target_policies}
            and len(self.mean_weights) == spec.trajectories.horizon,
            "ckks.request_binding",
        )

    def to_bytes(self) -> bytes:
        from unseen_loop.ope.lifted import _frame

        self.validate()
        wire = self._wire_bytes
        if wire is None:
            wire = _frame(
                {
                    "identifier": RAW_IDENTIFIER,
                    "policy_sha256": self.policy_sha256,
                    "request_sha256": self.request_sha256,
                    "horizon": len(self.mean_weights),
                },
                self.mean_weighted_rewards + self.mean_weights,
            )
            object.__setattr__(self, "_wire_bytes", wire)
        return wire

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        from unseen_loop.ope.lifted import _unframe

        metadata, vectors = _unframe(payload)
        row = _keys(metadata, {"identifier", "policy_sha256", "request_sha256", "horizon"})
        horizon = _coordinate_int(row["horizon"])
        if (
            row["identifier"] != RAW_IDENTIFIER
            or not 1 <= horizon <= 64
            or len(vectors) != 2 * horizon
        ):
            raise ValueError("ckks.request_binding")
        response = cls(
            row["policy_sha256"], row["request_sha256"], vectors[:horizon], vectors[horizon:]
        )
        object.__setattr__(response, "_wire_bytes", payload)
        return response


def _operation(
    identifier: str,
    operation: str,
    started: int,
    inputs: Sequence[SerializedCKKSVector] = (),
    outputs: Sequence[SerializedCKKSVector] = (),
) -> OPECKKSTransportReceipt:
    incoming, outgoing = [v.to_bytes() for v in inputs], [v.to_bytes() for v in outputs]
    schema = {
        RAW_IDENTIFIER: "unseen-loop/raw-prefix-wpdis-ckks-operation-v1",
        "PRIVATE_OPE_COUNT_DIAGNOSTIC_V1": "unseen-loop/private-ope-count-ckks-operation-v1",
    }[identifier]
    return OPECKKSTransportReceipt(
        operation,
        time.perf_counter_ns() - started,
        len(incoming),
        len(outgoing),
        sum(map(len, incoming)),
        sum(map(len, outgoing)),
        _sha(b"".join(struct.pack(">Q", len(v)) + v for v in incoming)) if incoming else None,
        _sha(b"".join(struct.pack(">Q", len(v)) + v for v in outgoing)) if outgoing else None,
        identifier=identifier,
        schema_version=schema,
    )


def _raw_encrypt(
    client: CKKSClient, spec: RawPrefixSpec, batch: TrajectoryBatch
) -> tuple[_RawRequest, OPECKKSTransportReceipt, int]:
    from unseen_loop.ope.lifted import _transport

    started = time.perf_counter_ns()
    spec.validate_batch(batch)
    states, actions = batch.state_array, batch.action_array
    reciprocals, normalized = 1 / batch.behavior_array, batch.reward_array / batch.spec.trajectories
    preprocessing = time.perf_counter_ns() - started
    vectors: list[SerializedCKKSVector] = []
    for chunk in plan_chunks(batch.spec.trajectories, spec.parameters).chunks:
        section = slice(chunk.start, chunk.stop)
        for t in range(batch.spec.horizon):
            step_started = time.perf_counter_ns()
            arrays: list[npt.NDArray[np.float64]] = [
                states[section, t, j] for j in range(batch.spec.state_dim)
            ]
            arrays += [
                (actions[section, t] == a).astype(float) for a in range(batch.spec.action_count)
            ]
            arrays += [reciprocals[section, t], normalized[section, t]]
            preprocessing += time.perf_counter_ns() - step_started
            vectors.extend(client.encrypt(v)[0] for v in arrays)
    request = _RawRequest(_raw_metadata(spec), tuple(vectors))
    return (
        request,
        _transport(
            "encrypt_raw_request",
            started,
            None,
            request.to_bytes(),
            0,
            len(vectors),
            identifier=RAW_IDENTIFIER,
            schema_version="unseen-loop/raw-prefix-wpdis-ckks-operation-v1",
        ),
        preprocessing,
    )


def _load(server: CKKSServer, vector: SerializedCKKSVector) -> CKKSEncryptedVector:
    raw = server._tenseal.ckks_vector_from(server._context, vector.ciphertext)
    size = raw.size() if callable(raw.size) else raw.size
    _require(int(size) == vector.slots, "ckks.request_binding")
    return CKKSEncryptedVector(raw, vector.slots, server._owner)


def _raw_evaluate(
    server: CKKSServer, spec: RawPrefixSpec, request: _RawRequest, policy_sha256: str
) -> tuple[_RawResponse, OPECKKSTransportReceipt]:
    from unseen_loop.ope.lifted import _inclusive_prefix, _transport

    started = time.perf_counter_ns()
    _validate_raw_request(request, spec)
    policy = next(
        (p for p in spec.target_policies if _sha(canonical_bytes(p.to_dict())) == policy_sha256),
        None,
    )
    if policy is None:
        raise ValueError("ckks.request_binding")
    horizon, n = spec.trajectories.horizon, spec.trajectories.trajectories
    numerator: list[CKKSEncryptedVector | None] = [None] * horizon
    denominator: list[CKKSEncryptedVector | None] = [None] * horizon
    source = iter(request.vectors)
    for _chunk in plan_chunks(n, spec.parameters).chunks:
        ratios, rewards = [], []
        for _ in range(horizon):
            states = [_load(server, next(source)) for _ in range(spec.trajectories.state_dim)]
            masks = [_load(server, next(source)) for _ in range(spec.trajectories.action_count)]
            reciprocal, reward = _load(server, next(source)), _load(server, next(source))
            selected = None
            for mask, coefficients in zip(masks, policy.coefficients, strict=True):
                score = None
                for j, coefficient in enumerate(coefficients[1:]):
                    if coefficient != 0:
                        term = states[j] * coefficient
                        score = term if score is None else score + term
                if score is None:
                    raise ValueError("domain.invalid_input")
                score = score + coefficients[0]
                term = mask * score
                selected = term if selected is None else selected + term
            if selected is None:
                raise ValueError("domain.invalid_input")
            ratios.append(reciprocal * selected)
            rewards.append(reward)
        prefixes = _inclusive_prefix(tuple(ratios))
        for t, weight in enumerate(prefixes):
            a = (rewards[t] * weight).reduce_sum()
            b = (weight * (1 / n)).reduce_sum()
            old_a, old_b = numerator[t], denominator[t]
            numerator[t] = a if old_a is None else old_a + a
            denominator[t] = b if old_b is None else old_b + b
    _require(next(source, None) is None, "ckks.request_binding")
    arrays = []
    for values in (numerator, denominator):
        serialized = []
        for value in values:
            if value is None:
                raise ValueError("ckks.request_binding")
            serialized.append(SerializedCKKSVector(value._vector.serialize(), 1))
        arrays.append(tuple(serialized))
    response = _RawResponse(policy_sha256, request.digest, arrays[0], arrays[1])
    return response, _transport(
        "evaluate_raw_request",
        started,
        request.to_bytes(),
        response.to_bytes(),
        len(request.vectors),
        len(response.mean_weighted_rewards) + len(response.mean_weights),
        identifier=RAW_IDENTIFIER,
        schema_version="unseen-loop/raw-prefix-wpdis-ckks-operation-v1",
    )


def _raw_decrypt(
    client: CKKSClient, spec: RawPrefixSpec, request: _RawRequest, response: _RawResponse
) -> tuple[SufficientStatistics, OPECKKSTransportReceipt]:
    from unseen_loop.ope.lifted import _transport

    started = time.perf_counter_ns()
    _validate_raw_request(request, spec)
    response.validate_binding(request, spec)
    means = tuple(float(client.decrypt(v)[0][0]) for v in response.mean_weighted_rewards)
    weights = tuple(float(client.decrypt(v)[0][0]) for v in response.mean_weights)
    _require(all(math.isfinite(v) for v in means + weights), "ckks.nonfinite")
    _require(all(v > 0 for v in weights), "ckks.nonpositive_denominator")
    stats = SufficientStatistics(
        "wpdis",
        tuple(v * spec.gamma**t for t, v in enumerate(means)),
        weights,
        (spec.trajectories.trajectories,) * spec.trajectories.horizon,
    )
    return stats, _transport(
        "decrypt_raw_response",
        started,
        response.to_bytes(),
        None,
        len(response.mean_weighted_rewards) + len(response.mean_weights),
        0,
        identifier=RAW_IDENTIFIER,
        schema_version="unseen-loop/raw-prefix-wpdis-ckks-operation-v1",
    )


def recompute_cipher_errors(
    reference: BatchReference, left: CipherValue, right: CipherValue, gamma: float
) -> CipherErrors:
    numerator_errors, denominator_errors, relative_errors, bounds = [], [], [], []
    value_errors = []
    for value, clear in zip((left, right), reference.policy_rows, strict=True):
        _require(
            value.policy_sha256 == clear.policy_sha256 and value.counts == clear.counts,
            "ckks.request_binding",
        )
        raw = sum(
            gamma**t * a / b
            for t, (a, b) in enumerate(
                zip(value.mean_weighted_rewards, value.mean_weights, strict=True)
            )
        )
        _near(value.raw_value, raw)
        _near(value.normalized_value, raw / reference.normalization)
        ea = tuple(
            abs(a - b)
            for a, b in zip(value.mean_weighted_rewards, clear.mean_weighted_rewards, strict=True)
        )
        eb = tuple(abs(a - b) for a, b in zip(value.mean_weights, clear.mean_weights, strict=True))
        er = tuple(e / b for e, b in zip(eb, clear.mean_weights, strict=True))
        numerator_errors.append(ea)
        denominator_errors.append(eb)
        relative_errors.append(er)
        if all(e < b for e, b in zip(eb, clear.mean_weights, strict=True)):
            bound = (
                sum(
                    gamma**t * (aerr + abs(a / b) * berr) / (b - berr)
                    for t, (a, b, aerr, berr) in enumerate(
                        zip(clear.mean_weighted_rewards, clear.mean_weights, ea, eb, strict=True)
                    )
                )
                / reference.normalization
            )
        else:
            bound = None
        bounds.append(bound)
        value_errors.append(abs(value.normalized_value - clear.normalized_value))
    contrast = right.normalized_value - left.normalized_value
    return CipherErrors(
        (value_errors[0], value_errors[1]),
        abs(contrast - reference.bootstrap.normalized_contrast),
        (numerator_errors[0], numerator_errors[1]),
        (denominator_errors[0], denominator_errors[1]),
        (relative_errors[0], relative_errors[1]),
        (bounds[0], bounds[1]),
    )


def _assert_numeric_copy(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("numerical evidence schema mismatch")
        for k, value in expected.items():
            _assert_numeric_copy(actual[k], value)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError("numerical evidence shape mismatch")
        for a, b in zip(actual, expected, strict=True):
            _assert_numeric_copy(a, b)
    elif isinstance(expected, float):
        if actual is not None and not isinstance(actual, (int, float)):
            raise ValueError("stored numerical evidence differs from replay")
        _near(actual, expected)
    else:
        _require(actual == expected, "numerical evidence mismatch")


def _modern_spec(
    manifest: PrivateOPEManifest, job: PlannedJob
) -> RawPrefixSpec | RatioLiftWPDISSpec:
    from unseen_loop.ope.lifted import RatioLiftWPDISSpec
    from unseen_loop.ope.study import queue_policies

    coordinates = dict(job.coordinates)
    arm = coordinates["arm"]
    if arm not in ("raw_prefix", "lifted_prefix", "lifted_independent_products"):
        raise ValueError("domain.invalid_input")
    shape = TrajectorySpec(
        _coordinate_int(coordinates["trajectories"]),
        _coordinate_int(coordinates["horizon"]),
        1,
        2,
        (0.0,),
        (1.0,),
        -1.0,
        0.0,
    )
    spec_type = RawPrefixSpec if arm == "raw_prefix" else RatioLiftWPDISSpec
    return spec_type(
        shape,
        queue_policies(),
        manifest.domain.gamma,
        manifest.policies.primary_propensity_floor,
        manifest.policies.primary_ratio_bound,
    )


def _reconstructed_computation(
    manifest: PrivateOPEManifest, job: PlannedJob, context: CKKSContextReceipt
) -> dict[str, object]:
    """Replay the complete public range proof, including the recorded actual moduli."""
    if context.parameters != _expected_parameters(job):
        raise ValueError("ckks.context_failure")
    spec = _modern_spec(manifest, job)
    if isinstance(spec, RawPrefixSpec):
        return spec.computation_receipt(context.actual_coeff_modulus_primes)
    return spec._computation_receipt(
        context.parameters, context.actual_coeff_modulus_primes
    ).to_dict()


def _numeric_pass(attempt: PrivateOPEAttempt, manifest: PrivateOPEManifest) -> bool:
    if not attempt.completed or not isinstance(attempt.metrics, CipherBatchMetrics):
        return False
    metrics = attempt.metrics
    if (
        metrics.reference is None
        or metrics.left is None
        or metrics.right is None
        or metrics.errors is None
    ):
        return False
    context = attempt.receipts.context
    if context is None:
        return False
    try:
        errors = recompute_cipher_errors(
            metrics.reference, metrics.left, metrics.right, manifest.domain.gamma
        )
        _assert_numeric_copy(metrics.errors.to_dict(), errors.to_dict())
        metrics._validate()
        computation = _reconstructed_computation(manifest, attempt.job, context)
        if attempt.receipts.computation_sha256 != _sha(canonical_bytes(computation)):
            return False
    except (ValueError, TypeError, ArithmeticError):
        return False
    g, boot = manifest.gates, metrics.reference.bootstrap
    # A zero SE admits only an exactly zero observed numerical error.
    for error, se in zip(
        (*errors.normalized_value_errors, errors.normalized_contrast_error),
        (boot.normalized_left_se, boot.normalized_right_se, boot.normalized_contrast_se),
        strict=True,
    ):
        if error > min(g.maximum_normalized_cipher_error, g.maximum_cipher_error_se_fraction * se):
            return False
    if any(
        v > g.maximum_mean_statistic_abs_error
        for arrays in (errors.mean_numerator_abs_errors, errors.mean_denominator_abs_errors)
        for row in arrays
        for v in row
    ):
        return False
    if any(
        v > g.maximum_denominator_relative_error
        for row in errors.mean_denominator_relative_errors
        for v in row
    ):
        return False
    receipts = attempt.receipts
    context = receipts.context
    return bool(
        context is not None
        and context.parameters == _expected_parameters(attempt.job)
        and context.security_enforced
        and context.effective_security_level == "tc128"
        and not context.server_context_is_private
        and receipts.counts_source == "public_fixed_shape"
        and context.data_chain_length == len(context.parameters.coeff_mod_bit_sizes) - 1
        and receipts.request_sha256
        and all(receipts.response_sha256)
        and receipts.runtime is not None
        and receipts.runtime.source_match
        and all(row.support_failures == 0 for row in metrics.reference.policy_rows)
    )


def _expected_parameters(job: PlannedJob) -> CKKSParameters:
    c = dict(job.coordinates)
    if c["arm"] == "old24":
        return CKKSParameters(16384, (40, *([24] * 14), 40), float(2**24))
    if c["arm"] == "candidate40":
        return CKKSParameters(16384, (60, *([40] * 8), 58), float(2**40))
    raw = c["arm"] == "raw_prefix"
    scale = 32 if raw else 40
    depth = (_coordinate_int(c["horizon"]) - 1).bit_length() + (4 if raw else 2)
    return CKKSParameters(16384, (60, *([scale] * depth), 58), float(2**scale))


def _resource_pass(attempt: PrivateOPEAttempt, manifest: PrivateOPEManifest) -> bool:
    if not attempt.completed or not isinstance(attempt.metrics, CipherBatchMetrics):
        return False
    timing, receipts, g = attempt.metrics.timing, attempt.receipts, manifest.gates
    return bool(
        timing.peak_rss_bytes <= g.maximum_peak_rss_gib * 2**30
        and timing.job_elapsed_ns <= manifest.execution.crypto_timeout_s * 10**9
        and receipts.client_context_bytes is not None
        and receipts.public_context_bytes is not None
        and receipts.client_context_bytes < g.maximum_context_gib_exclusive * 2**30
        and receipts.public_context_bytes < g.maximum_context_gib_exclusive * 2**30
    )


def _decision(lower: float, upper: float) -> Literal["positive", "negative", "abstain"]:
    return "positive" if lower > 0 else "negative" if upper < 0 else "abstain"


def _aggregate_comparison(
    rows: Sequence[PrivateOPEAttempt],
    manifest: PrivateOPEManifest,
    cohort: Literal["primary", "stress"],
) -> AggregateComparison:
    from unseen_loop.ope.study import (
        batch_rmse_interval,
        clopper_pearson_lower,
        student_t_bias_interval,
    )

    refs: list[BatchReference] = []
    estimates: list[float] = []
    intervals: list[tuple[float, float]] = []
    decisions: list[str] = []
    disagreements = 0
    encrypted = manifest.phase == "confirmation"
    for row in rows:
        metrics = row.metrics
        if (
            not row.completed
            or not isinstance(metrics, (ClearBatchMetrics, CipherBatchMetrics))
            or metrics.reference is None
        ):
            continue
        ref = metrics.reference
        if encrypted:
            if not isinstance(metrics, CipherBatchMetrics) or metrics.right is None:
                continue
            estimate = metrics.cipher_contrast_normalized
            lower, upper, decision = (
                metrics.cipher_interval_lower,
                metrics.cipher_interval_upper,
                metrics.cipher_decision,
            )
            if estimate is None or lower is None or upper is None:
                raise ValueError("completed ciphertext metrics lack interval")
            disagreements += decision != _decision(
                ref.bootstrap.normalized_lower, ref.bootstrap.normalized_upper
            )
        else:
            estimate = ref.bootstrap.normalized_contrast
            lower, upper = ref.bootstrap.normalized_lower, ref.bootstrap.normalized_upper
            decision = _decision(lower, upper)
        refs.append(ref)
        estimates.append(estimate)
        intervals.append((lower, upper))
        decisions.append(decision)
    n, complete = len(rows), len(refs)
    truth = refs[0].truth_contrast_normalized if refs else None
    for ref in refs:
        _near(ref.truth_contrast_normalized, refs[0].truth_contrast_normalized)
        _require(
            (ref.kernel_sha256, ref.policies_sha256)
            == (refs[0].kernel_sha256, refs[0].policies_sha256),
            "cohort source mismatch",
        )
    coverage = (
        sum(lower <= truth <= upper for lower, upper in intervals) if truth is not None else 0
    )
    correct = (
        sum(
            d == ("positive" if truth > 0 else "negative" if truth < 0 else "unavailable")
            for d in decisions
        )
        if truth is not None
        else 0
    )
    abstentions = decisions.count("abstain")
    bias = blo = bhi = rmse = rlo = rhi = width = None
    ess: list[float | None] = [None] * 4
    # No complete-case statistical primary interval can pass a fixed denominator.
    if complete == n and n:
        if truth is None:
            raise ValueError("complete cohort has no truth")
        errors = np.asarray(estimates) - truth
        bias, blo, bhi = student_t_bias_interval(errors)
        rmse, rlo, rhi = batch_rmse_interval(
            errors,
            seed=derive_seed(manifest.seed_root, f"analysis:rmse:{cohort}"),
            repetitions=manifest.statistics.rmse_bootstrap_repetitions,
        )
        width = float(np.median([b - a for a, b in intervals]))
        for i in range(2):
            fractions = [
                ref.policy_rows[i].raw_ess[-1] / ref.policy_rows[i].counts[-1] for ref in refs
            ]
            ess[2 * i] = float(np.quantile(fractions, 0.5, method="linear"))
            ess[2 * i + 1] = float(np.quantile(fractions, 0.05, method="linear"))
    return AggregateComparison(
        n,
        complete,
        truth,
        bias,
        blo,
        bhi,
        rmse,
        rlo,
        rhi,
        coverage,
        clopper_pearson_lower(coverage, n) if n else None,
        correct,
        clopper_pearson_lower(correct, n) if n else None,
        abstentions,
        width,
        ess[0],
        ess[1],
        ess[2],
        ess[3],
        disagreements,
    )


def _aggregate_baselines(
    rows: Sequence[PrivateOPEAttempt],
    manifest: PrivateOPEManifest,
    cohort: Literal["primary", "stress"],
) -> tuple[BaselineAggregate, ...]:
    from unseen_loop.ope.study import batch_rmse_interval, student_t_bias_interval

    aggregates = []
    for estimator_id in BASELINE_IDS:
        left, right, contrasts = [], [], []
        for attempt in rows:
            metrics = attempt.metrics
            # References produced before a failed FHE operation remain valid
            # clear comparator evidence; they are not ciphertext successes.
            if (
                not isinstance(metrics, (ClearBatchMetrics, CipherBatchMetrics))
                or metrics.reference is None
            ):
                continue
            ref = metrics.reference
            pair = next(b for b in ref.baseline_rows if b.estimator_id == estimator_id)
            if pair.failure_code is None:
                if (
                    pair.left_raw is None
                    or pair.right_raw is None
                    or pair.contrast_normalized is None
                ):
                    raise ValueError("baseline completion/failure mismatch")
                left.append((pair.left_raw - ref.truth_left_raw) / ref.normalization)
                right.append((pair.right_raw - ref.truth_right_raw) / ref.normalization)
                contrasts.append(pair.contrast_normalized - ref.truth_contrast_normalized)
        count = len(contrasts)
        values: list[float | None] = [None] * 10
        if count:
            values[0], values[1] = float(np.mean(left)), float(np.sqrt(np.mean(np.square(left))))
            values[2], values[3] = float(np.mean(right)), float(np.sqrt(np.mean(np.square(right))))
            values[4:7] = student_t_bias_interval(contrasts)
            values[7:10] = batch_rmse_interval(
                contrasts,
                seed=derive_seed(
                    manifest.seed_root, f"analysis:baseline:{cohort}:{estimator_id}:rmse"
                ),
                repetitions=manifest.statistics.rmse_bootstrap_repetitions,
            )
        aggregates.append(
            BaselineAggregate(
                cohort,
                estimator_id,
                len(rows),
                count,
                count != len(rows),
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
            )
        )
    return tuple(aggregates)


def _aggregate_timing(
    rows: Sequence[PrivateOPEAttempt], manifest: PrivateOPEManifest
) -> AggregateTiming:
    from unseen_loop.ope.study import timing_speedup_interval

    cases: dict[str, dict[str, PrivateOPEAttempt]] = {}
    for row in rows:
        c = dict(row.job.coordinates)
        cases.setdefault(_coordinate_str(c["case_id"]), {})[_coordinate_str(c["arm"])] = row
    pairs = []
    for case_id, arms in cases.items():
        candidate, baseline = arms.get("lifted_prefix"), arms.get("raw_prefix")
        if candidate is None or baseline is None:
            raise ValueError("timing plan lacks paired arm")
        cm, bm = candidate.metrics, baseline.metrics
        match = bool(
            isinstance(cm, CipherBatchMetrics)
            and isinstance(bm, CipherBatchMetrics)
            and cm.reference is not None
            and bm.reference is not None
            and canonical_bytes(cm.reference) == canonical_bytes(bm.reference)
        )
        valid = bool(
            match
            and isinstance(cm, CipherBatchMetrics)
            and isinstance(bm, CipherBatchMetrics)
            and _numeric_pass(candidate, manifest)
            and _numeric_pass(baseline, manifest)
            and _resource_pass(candidate, manifest)
            and _resource_pass(baseline, manifest)
            and cm.timing.total_ns is not None
            and bm.timing.total_ns is not None
            and cm.timing.total_ns > 0
            and bm.timing.total_ns > 0
        )
        speedup = None
        if valid:
            if (
                not isinstance(cm, CipherBatchMetrics)
                or not isinstance(bm, CipherBatchMetrics)
                or cm.timing.total_ns is None
                or bm.timing.total_ns is None
            ):
                raise ValueError("valid timing pair lacks totals")
            speedup = bm.timing.total_ns / cm.timing.total_ns
        pairs.append(
            TimingPair(
                case_id,
                candidate.job.job_id,
                baseline.job.job_id,
                match,
                valid,
                speedup,
            )
        )
    complete = sum(p.valid for p in pairs)
    median = lower = None
    if pairs and complete == len(pairs):
        speedups = []
        for pair in pairs:
            if pair.speedup is None:
                raise ValueError("valid timing pair lacks speedup")
            speedups.append(pair.speedup)
        median, lower = timing_speedup_interval(
            speedups,
            seed=derive_seed(manifest.seed_root, "analysis:timing"),
            repetitions=manifest.statistics.timing_bootstrap_repetitions,
        )
    return AggregateTiming(len(pairs), complete, tuple(pairs), median, lower)


def _gate(
    name: str,
    passed: bool,
    *,
    numerator: int | None = None,
    denominator: int | None = None,
    observed: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    comparison: str = "all",
    threshold: float | None = None,
    reason: str | None = None,
) -> GateResult:
    return GateResult(
        name,
        bool(passed),
        numerator,
        denominator,
        observed,
        lower,
        upper,
        comparison,
        threshold,
        None if passed else reason or "gate.not_met",
    )


def _threshold_gate(
    name: str, observed: float | None, comparison: str, threshold: float, *, available: bool = True
) -> GateResult:
    comparisons: dict[str, Callable[[float, float], bool]] = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
    }
    passed = available and observed is not None and comparisons[comparison](observed, threshold)
    return _gate(
        name,
        passed,
        observed=observed,
        comparison=comparison,
        threshold=threshold,
        reason="gate.unavailable" if not available or observed is None else None,
    )


def analyze_private_ope(
    manifest: PrivateOPEManifest, attempts: tuple[PrivateOPEAttempt, ...] | list[PrivateOPEAttempt]
) -> AnalysisMetrics:
    """Replay every reserved non-analysis row; never revise or persist evidence.

    Missing/duplicate/untrusted rows are an invalid analysis input, not an
    opportunity to synthesize artifact hashes. Coordinators must first close
    every absent computation with explicit failed terminal evidence.
    """
    from unseen_loop.flagship.manifest import iter_private_ope_jobs

    planned = tuple(
        j for j in iter_private_ope_jobs(manifest) if dict(j.coordinates)["kind"] != "analysis"
    )
    rows = tuple(
        PrivateOPEAttempt.from_dict(a.to_dict() if isinstance(a, PrivateOPEAttempt) else a)
        for a in attempts
    )
    lookup = {a.job.job_id: a for a in rows}
    _require(
        len(lookup) == len(rows) and set(lookup) == {j.job_id for j in planned},
        "analysis requires every fixed terminal row exactly once",
    )
    rows = tuple(lookup[j.job_id] for j in planned)
    expected_run = f"private-ope-{manifest.phase}-{manifest.digest[:24]}"
    provenance = {r.provenance_sha256 for r in rows}
    _require(len(provenance) == 1, "attempt provenance mismatch")
    by_kind: dict[str, list[PrivateOPEAttempt]] = {}
    source_inputs = {}
    for job, row in zip(planned, rows, strict=True):
        _require(
            job_to_dict(row.job) == job_to_dict(job)
            and row.config_sha256 == manifest.digest
            and row.run_id == expected_run,
            "attempt source/job identity mismatch",
        )
        runtime = row.receipts.runtime
        if runtime is not None:
            for field in (
                "image_spec_sha256",
                "code_commit",
                "candidate_code_sha256",
                "baseline_code_sha256",
                "domain_code_sha256",
                "analysis_code_sha256",
                "lockfile_sha256",
            ):
                _require(
                    getattr(runtime, field) == getattr(manifest.execution, field),
                    "attempt runtime source mismatch",
                )
        if (
            isinstance(row.metrics, (ClearBatchMetrics, CipherBatchMetrics))
            and row.metrics.reference is not None
        ):
            from unseen_loop.ope.study import queue_inputs, queue_policies

            horizon = _coordinate_int(dict(job.coordinates)["horizon"])
            if horizon not in source_inputs:
                source_inputs[horizon] = queue_inputs(horizon, manifest.domain.gamma)
            inputs, ref = source_inputs[horizon], row.metrics.reference
            _require(
                ref.kernel_sha256 == inputs["kernel_sha256"]
                and ref.policies_sha256 == inputs["policies_sha256"],
                "reference domain source mismatch",
            )
            _require(
                tuple(r.policy_sha256 for r in ref.policy_rows)
                == tuple(_sha(canonical_bytes(p.to_dict())) for p in queue_policies()),
                "reference frozen policy mismatch",
            )
            _near(ref.normalization, inputs["truth"]["normalization"])
            _near(ref.truth_left_raw, inputs["truth"]["values_raw"]["A"])
            _near(ref.truth_right_raw, inputs["truth"]["values_raw"]["B"])
            for policy in ref.policy_rows:
                _near(
                    policy.raw_value,
                    sum(
                        manifest.domain.gamma**t * a / b
                        for t, (a, b) in enumerate(
                            zip(policy.mean_weighted_rewards, policy.mean_weights, strict=True)
                        )
                    ),
                )
        if isinstance(row.metrics, CipherBatchMetrics) and row.metrics.errors is not None:
            if (
                row.metrics.reference is None
                or row.metrics.left is None
                or row.metrics.right is None
            ):
                raise ValueError("cipher errors lack complete values")
            expected_errors = recompute_cipher_errors(
                row.metrics.reference, row.metrics.left, row.metrics.right, manifest.domain.gamma
            )
            _assert_numeric_copy(row.metrics.errors.to_dict(), expected_errors.to_dict())
        by_kind.setdefault(_coordinate_str(dict(job.coordinates)["kind"]), []).append(row)
    counts = {
        kind: CountSummary(
            len(group),
            sum(r.attempted is True for r in group),
            sum(r.attempted is False for r in group),
            sum(r.attempted is None for r in group),
            sum(r.completed for r in group),
        )
        for kind, group in by_kind.items()
    }
    gates = [_gate("evidence_closure", True, numerator=len(rows), denominator=len(planned))]
    primary: AggregateComparison | None = None
    stress: AggregateComparison | None = None
    timing: AggregateTiming | None = None
    diagnostic: DiagnosticSummary | None = None
    baselines: tuple[BaselineAggregate, ...] = ()
    g = manifest.gates
    if manifest.phase == "diagnostic":
        verification = by_kind["protocol_verification"][0]
        verified = bool(
            verification.completed
            and isinstance(verification.metrics, VerificationMetrics)
            and verification.metrics.required_cases_passed
        )
        new_passes = old_passes = 0
        for row in by_kind["count_precision"]:
            metrics, context = row.metrics, row.receipts.context
            passed = bool(
                row.completed
                and isinstance(metrics, CountPrecisionMetrics)
                and metrics.expected_sum == dict(row.job.coordinates)["trajectories"]
                and metrics.absolute_error is not None
                and metrics.absolute_error <= g.diagnostic_sum_abs_error
                and context is not None
                and context.parameters == _expected_parameters(row.job)
                and context.security_enforced
                and context.effective_security_level == "tc128"
                and not context.server_context_is_private
                and row.receipts.runtime is not None
                and row.receipts.runtime.source_match
            )
            if dict(row.job.coordinates)["arm"] == "candidate40":
                new_passes += passed
            else:
                old_passes += passed
        probes = sum(
            row.attempted is True and not row.completed and row.failure_code == expected
            for kind, expected in (
                ("smoke_error", "probe.deliberate_exception"),
                ("smoke_timeout", "runtime.timeout"),
            )
            for row in by_kind[kind]
        )
        diagnostic = DiagnosticSummary(12, new_passes, old_passes, verified, probes)
        gates += [
            _gate("verification", verified, numerator=int(verified), denominator=1),
            _gate(
                "diagnostic_precision",
                new_passes == 6,
                numerator=new_passes,
                denominator=6,
                comparison="all absolute errors <=",
                threshold=g.diagnostic_sum_abs_error,
            ),
            _gate("probe_accounting", probes == 2, numerator=probes, denominator=2),
        ]
    else:
        if manifest.phase == "pilot":
            primary_rows = [
                r
                for r in by_kind["clear_batch"]
                if dict(r.job.coordinates)["cohort"] == "screen_primary"
            ]
            stress_rows = [
                r
                for r in by_kind["clear_batch"]
                if dict(r.job.coordinates)["cohort"] == "screen_stress"
            ]
            stress = _aggregate_comparison(stress_rows, manifest, "stress")
        else:
            primary_rows, stress_rows = by_kind["statistical_context"], []
        primary = _aggregate_comparison(primary_rows, manifest, "primary")
        baselines = _aggregate_baselines(primary_rows, manifest, "primary") + (
            _aggregate_baselines(stress_rows, manifest, "stress") if stress_rows else ()
        )
        full = primary.complete_batches == primary.planned_batches
        gap = (
            abs(primary.truth_contrast_normalized)
            if primary.truth_contrast_normalized is not None
            else None
        )
        if manifest.phase == "pilot":
            gates += [
                _threshold_gate("clear_gap", gap, ">=", g.minimum_normalized_gap, available=full),
                _gate(
                    "clear_coverage",
                    full and primary.coverage_successes >= g.pilot_coverage_successes,
                    numerator=primary.coverage_successes,
                    denominator=primary.planned_batches,
                    comparison=">=",
                    threshold=float(g.pilot_coverage_successes),
                ),
                _gate(
                    "clear_choice",
                    full and primary.correct_choice_successes >= g.pilot_choice_successes,
                    numerator=primary.correct_choice_successes,
                    denominator=primary.planned_batches,
                    comparison=">=",
                    threshold=float(g.pilot_choice_successes),
                ),
            ]
        gates.append(
            _threshold_gate(
                "interval_width",
                primary.median_width,
                "<",
                gap if gap is not None else 0.0,
                available=full and gap is not None,
            )
        )
        for side in ("left", "right"):
            gates.append(
                _threshold_gate(
                    f"{side}_ess_median",
                    getattr(primary, f"{side}_median_ess_fraction"),
                    ">=",
                    g.median_ess_fraction,
                    available=full,
                )
            )
            gates.append(
                _threshold_gate(
                    f"{side}_ess_p05",
                    getattr(primary, f"{side}_p05_ess_fraction"),
                    ">=",
                    g.p05_ess_fraction,
                    available=full,
                )
            )
        required = [
            r
            for r in rows
            if dict(r.job.coordinates)["kind"] in MODERN_KINDS
            and dict(r.job.coordinates)["arm"] in ("lifted_prefix", "raw_prefix")
        ]
        complete_contexts = sum(r.completed for r in required)
        numerical = sum(_numeric_pass(r, manifest) for r in required)
        resources = sum(_resource_pass(r, manifest) for r in required)
        gates += [
            _gate(
                "required_context_completion",
                complete_contexts == len(required),
                numerator=complete_contexts,
                denominator=len(required),
            ),
            _gate(
                "cipher_numerics",
                numerical == len(required),
                numerator=numerical,
                denominator=len(required),
            ),
            _gate(
                "resource_bounds",
                resources == len(required),
                numerator=resources,
                denominator=len(required),
            ),
        ]
        timing_rows = [r for r in rows if dict(r.job.coordinates)["cohort"] == "timing"]
        timing = _aggregate_timing(timing_rows, manifest)
        gates += [
            _gate(
                "timing_pair_completion",
                timing.valid_pairs == timing.planned_pairs,
                numerator=timing.valid_pairs,
                denominator=timing.planned_pairs,
            ),
            _threshold_gate("timing_median", timing.median_speedup, ">=", g.minimum_median_speedup),
            _threshold_gate(
                "timing_lower",
                timing.speedup_lower,
                ">" if manifest.phase == "pilot" else ">=",
                g.pilot_speedup_lower_exclusive
                if manifest.phase == "pilot"
                else g.confirmation_speedup_lower_inclusive,
            ),
        ]
        if manifest.phase == "confirmation":
            gates += [
                _gate(
                    "confirmation_coverage",
                    full
                    and primary.coverage_lower is not None
                    and primary.coverage_lower >= g.confirmation_coverage_lower,
                    numerator=primary.coverage_successes,
                    denominator=primary.planned_batches,
                    lower=primary.coverage_lower,
                    comparison="lower >=",
                    threshold=g.confirmation_coverage_lower,
                ),
                _gate(
                    "confirmation_choice",
                    full
                    and primary.correct_choice_lower is not None
                    and primary.correct_choice_lower >= g.confirmation_choice_lower,
                    numerator=primary.correct_choice_successes,
                    denominator=primary.planned_batches,
                    lower=primary.correct_choice_lower,
                    comparison="lower >=",
                    threshold=g.confirmation_choice_lower,
                ),
                _gate(
                    "confirmation_bias",
                    full
                    and primary.bias_lower is not None
                    and primary.bias_upper is not None
                    and primary.bias_lower >= -g.confirmation_bias_equivalence_margin
                    and primary.bias_upper <= g.confirmation_bias_equivalence_margin,
                    observed=primary.bias,
                    lower=primary.bias_lower,
                    upper=primary.bias_upper,
                    comparison="interval contained in symmetric closed margin",
                    threshold=g.confirmation_bias_equivalence_margin,
                ),
                _threshold_gate(
                    "confirmation_rmse",
                    primary.rmse,
                    "<=",
                    g.confirmation_maximum_rmse,
                    available=full,
                ),
            ]
    order = {name: i for i, name in enumerate(GATE_IDS)}
    gates.sort(key=lambda gate: order[gate.id])
    allowed = all(gate.passed for gate in gates)
    phase = manifest.phase
    if phase not in ("diagnostic", "pilot", "confirmation"):
        raise ValueError("invalid analysis phase")
    result = AnalysisMetrics(
        "analysis",
        cast(Literal["diagnostic", "pilot", "confirmation"], phase),
        tuple(j.job_id for j in planned),
        tuple(_sha(canonical_bytes(row)) for row in rows),
        counts,
        primary,
        stress,
        baselines,
        timing,
        diagnostic,
        tuple(gates),
        allowed,
        "passed" if allowed else "failed",
    )
    return AnalysisMetrics.from_dict(result.to_dict())


def _failure_code(error: Exception, stage: str) -> str:
    declared = getattr(error, "failure_code", None)
    if isinstance(declared, str) and declared in FAILURE_CODES:
        return declared
    # Only enum codes are released, never exception messages or tracebacks.
    prefix = str(error).split(":", 1)[0]
    if prefix in FAILURE_CODES:
        return prefix
    text = str(error).lower()
    if "count is outside" in text:
        return "ckks.count_precision"
    if stage == "context":
        return "ckks.context_failure"
    if stage == "verification":
        return "verification.failed"
    if stage == "analysis":
        return "analysis.failed"
    if stage == "clear":
        return (
            "statistics.invalid_support"
            if "support" in text or "denominator" in text
            else "domain.invalid_input"
        )
    if stage == "source":
        return "evidence.source_mismatch"
    if stage == "binding":
        return "ckks.request_binding"
    if "nonfinite" in text or "must be finite" in text:
        return "ckks.nonfinite"
    if "denominator" in text and ("nonpositive" in text or "positive" in text):
        return "ckks.nonpositive_denominator"
    if "range" in text or "modulus" in text:
        return "domain.range_bound"
    if "ratio" in text or "propensity" in text:
        return "domain.ratio_bound"
    return "ckks.backend_error"


def _legacy_batch(replica: int) -> tuple[TrajectoryBatch, PolynomialPolicySpec]:
    """The retained confirmation001 fixture, unchanged N64/H8/D6/A5."""
    n, h, dimensions, actions = 64, 8, 6, 5
    spec = TrajectorySpec(
        n, h, dimensions, actions, (-1.0,) * dimensions, (1.0,) * dimensions, -1.0, 1.0
    )
    coefficients = (1 / actions,) + (0.0,) * dimensions
    policy = PolynomialPolicySpec(actions, dimensions, 1, (coefficients,) * actions)
    states = tuple(
        tuple(
            tuple(((i * 3 + t * 5 + j * 7 + replica * 11) % 17 - 8) / 8 for j in range(dimensions))
            for t in range(h)
        )
        for i in range(n)
    )
    selected = tuple(tuple((i + t + replica) % actions for t in range(h)) for i in range(n))
    rewards = tuple(
        tuple(((i * 3 + t * 5 + replica * 7) % 17 - 8) / 8 for t in range(h)) for i in range(n)
    )
    return TrajectoryBatch(spec, states, selected, rewards, ((0.2,) * h,) * n), policy


def _legacy_wire(value: object) -> object:
    if isinstance(value, SerializedCKKSVector):
        return base64.b64encode(value.to_bytes()).decode("ascii")
    if dataclasses.is_dataclass(value):
        return {f.name: _legacy_wire(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_legacy_wire(v) for v in value]
    return value


def _legacy_unwire(annotation: Any, value: object) -> Any:
    if annotation is SerializedCKKSVector:
        if not isinstance(value, (str, bytes)):
            raise ValueError("ckks.request_binding")
        return SerializedCKKSVector.from_bytes(base64.b64decode(value, validate=True))
    if get_origin(annotation) is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("ckks.request_binding")
        return tuple(_legacy_unwire(get_args(annotation)[0], v) for v in value)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        row = _keys(value, {f.name for f in dataclasses.fields(annotation)})
        hints = get_type_hints(annotation)
        return cast(Callable[..., "DataclassInstance"], annotation)(
            **{k: _legacy_unwire(hints[k], v) for k, v in row.items()}
        )
    return _parse(annotation, value)


def _verify_protocol() -> VerificationMetrics:
    import contextlib
    import os

    import pytest

    started = time.perf_counter_ns()
    root = Path(__file__).resolve().parents[3]
    paths = (
        "tests/test_ckks_backend.py",
        "tests/test_ope_ckks.py",
        "tests/test_ratio_lift_wpdis.py",
    )
    source = _sha(canonical_bytes({p: _sha((root / p).read_bytes()) for p in paths}))

    class Results:
        def __init__(self) -> None:
            self.ids: list[str] = []
            self.results: dict[str, Literal["passed", "failed", "skipped", "error"]] = {}

        @staticmethod
        def safe(nodeid: str) -> str:
            position = nodeid.find("tests/")
            nodeid = nodeid[position:] if position >= 0 else nodeid
            return re.sub(r"[^A-Za-z0-9_:/\[\].-]", "_", nodeid)

        def pytest_collection_finish(self, session: pytest.Session) -> None:
            self.ids = [self.safe(item.nodeid) for item in session.items]

        def pytest_runtest_logreport(self, report: TestReport) -> None:
            node = self.safe(report.nodeid)
            if report.failed:
                self.results[node] = "failed" if report.when == "call" else "error"
            elif report.skipped and self.results.get(node) not in ("failed", "error"):
                self.results[node] = "skipped"
            elif report.when == "call" and report.passed:
                self.results.setdefault(node, "passed")

    plugin = Results()
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        code = int(
            pytest.main(
                [
                    *(str(root / p) for p in paths),
                    "--rootdir",
                    str(root),
                    "-q",
                    "-o",
                    "addopts=",
                    "--tb=no",
                    "-p",
                    "no:cacheprovider",
                ],
                plugins=[plugin],
            )
        )
    outcomes = tuple(
        VerificationOutcome(node, plugin.results.get(node, "error")) for node in plugin.ids
    )
    return VerificationMetrics(
        "verification",
        source,
        tuple(plugin.ids),
        outcomes,
        code,
        code == 0 and bool(outcomes) and all(o.outcome == "passed" for o in outcomes),
        time.perf_counter_ns() - started,
    )


class _Execution:
    """Mutable worker-local accumulation; only frozen release records escape."""

    def __init__(self, runtime: RuntimeReceipt, kind: str) -> None:
        self.started = time.perf_counter_ns()
        self.timed_started: int | None = None
        self.total_ns: int | None = None
        self.clear_ns: int | None = None
        self.timers: list[TimerRecord] = []
        self.metrics: Metrics | None = None
        self.stage = "source"
        self.context: CKKSContextReceipt | None = None
        self.computation: str | None = None
        self.batch: str | None = None
        self.request: str | None = None
        self.request_bytes: int | None = None
        size = 2 if kind in MODERN_KINDS else 1 if kind == "historical_context" else 0
        self.responses: list[str | None] = [None] * size
        self.response_bytes: list[int | None] = [None] * size
        self.operations: list[OPECKKSTransportReceipt] = []
        self.runtime = runtime
        self.counts_source: CountsSource = (
            "public_fixed_shape"
            if kind in MODERN_KINDS
            else "legacy_encrypted"
            if kind == "historical_context"
            else "diagnostic_sum"
            if kind == "count_precision"
            else "not-applicable"
        )

    def timing(self) -> TimingReceipt:
        return TimingReceipt(
            self.total_ns,
            time.perf_counter_ns() - self.started,
            self.clear_ns,
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            tuple(self.timers),
        )

    def timer(self, name: str, started: int, parent: str | None = None) -> None:
        self.timers.append(TimerRecord(name, time.perf_counter_ns() - started, parent))

    def receipts(self) -> AttemptReceipts:
        context = self.context
        return AttemptReceipts(
            self.runtime,
            context,
            self.computation,
            self.batch,
            self.request,
            tuple(self.responses),
            tuple(self.operations),
            context.server_context_sha256 if context else None,
            context.server_context_bytes if context else None,
            context.client_context_bytes if context else None,
            self.request_bytes,
            tuple(self.response_bytes),
            self.counts_source,
        )

    def contexts(self, parameters: CKKSParameters) -> tuple[CKKSContextArtifacts, CKKSClient]:
        self.stage = "context"
        self.timed_started = time.perf_counter_ns()
        artifacts = generate_contexts(parameters)
        self.context = artifacts.receipt
        if artifacts.receipt.seal_version != "not-exposed":
            self.runtime = dataclasses.replace(
                self.runtime, seal_version=artifacts.receipt.seal_version
            )
        client = CKKSClient.from_serialized(artifacts.client_context, parameters=parameters)
        self.timer("keygen/context serialization", self.timed_started)
        return artifacts, client


def _cipher_value(
    statistics: SufficientStatistics, policy_sha256: str, gamma: float, normalization: float
) -> CipherValue:
    if statistics.estimate is None:
        raise ValueError("ckks.nonpositive_denominator")
    means = tuple(v / gamma**t for t, v in enumerate(statistics.numerators))
    return CipherValue(
        policy_sha256,
        means,
        statistics.denominators,
        statistics.counts,
        statistics.estimate,
        statistics.estimate / normalization,
    )


def _modern_context(
    state: _Execution,
    manifest: PrivateOPEManifest,
    job: PlannedJob,
    batch: TrajectoryBatch,
    reference: BatchReference,
) -> None:
    from unseen_loop.crypto.ckks import context_modulus_primes
    from unseen_loop.ope.lifted import (
        RatioLiftRequest,
        RatioLiftResponse,
        RatioLiftWPDISClient,
        RatioLiftWPDISServer,
        lifted_ckks_parameters,
    )
    from unseen_loop.ope.study import translate_cipher_interval

    c = dict(job.coordinates)
    state.metrics = CipherBatchMetrics(
        "cipher_batch", reference, None, None, None, None, None, None, "unavailable", state.timing()
    )
    spec = _modern_spec(manifest, job)
    _require(batch.spec == spec.trajectories, "domain.invalid_input")
    if isinstance(spec, RawPrefixSpec):
        parameters = spec.parameters
        state.computation = _sha(canonical_bytes(spec.computation_receipt()))
    else:
        parameters = lifted_ckks_parameters(spec)
        state.computation = _sha(canonical_bytes(spec.computation_receipt().to_dict()))
    values: list[CipherValue | None] = [None, None]
    state.metrics = CipherBatchMetrics(
        "cipher_batch", reference, None, None, None, None, None, None, "unavailable", state.timing()
    )
    try:
        artifacts, client = state.contexts(parameters)
        state.stage = "crypto"
        boundary_client: RatioLiftWPDISClient | None = None
        if isinstance(spec, RawPrefixSpec):
            raw_computation = spec.computation_receipt(context_modulus_primes(client._context))
            state.computation = _sha(canonical_bytes(raw_computation))
            boundary_client = None
        else:
            boundary_client = RatioLiftWPDISClient(client, spec)
            state.computation = _sha(canonical_bytes(boundary_client.receipt.to_dict()))
        started = time.perf_counter_ns()
        request: _RawRequest | RatioLiftRequest
        if isinstance(spec, RawPrefixSpec):
            request, operation, preprocess = _raw_encrypt(client, spec, batch)
        else:
            if boundary_client is None:
                raise ValueError("ckks.request_binding")
            request, operation = boundary_client.encrypt_batch(batch)
            preprocess = boundary_client.last_preprocessing_ns
        request_bytes = request.to_bytes()
        state.timer("encryption/request serialization", started)
        state.timers.append(
            TimerRecord("client preprocessing", preprocess, "encryption/request serialization")
        )
        state.operations.append(operation)
        state.request, state.request_bytes = _sha(request_bytes), len(request_bytes)
        started = time.perf_counter_ns()
        server = CKKSServer.from_serialized(artifacts.server_context, parameters=parameters)
        state.stage = "binding"
        server_request: _RawRequest | RatioLiftRequest
        boundary_server: RatioLiftWPDISServer | None = None
        if isinstance(spec, RawPrefixSpec):
            server_request = _RawRequest.from_bytes(request_bytes, spec)
        else:
            server_request = RatioLiftRequest.from_bytes(request_bytes)
            boundary_server = RatioLiftWPDISServer(server, spec)
        state.timer("server context/request loading", started)
        for i, policy_ref in enumerate(reference.policy_rows):
            label = ("A", "B")[i]
            state.stage = "crypto"
            started = time.perf_counter_ns()
            response: _RawResponse | RatioLiftResponse
            if isinstance(spec, RawPrefixSpec):
                if not isinstance(server_request, _RawRequest):
                    raise ValueError("ckks.request_binding")
                response, operation = _raw_evaluate(
                    server, spec, server_request, policy_ref.policy_sha256
                )
            else:
                if boundary_server is None or not isinstance(server_request, RatioLiftRequest):
                    raise ValueError("ckks.request_binding")
                if c["arm"] == "lifted_independent_products":
                    response, operation = boundary_server._evaluate_with_prefix_builder(
                        server_request, policy_ref.policy_sha256, _independent_products
                    )
                else:
                    response, operation = boundary_server.evaluate(
                        server_request, policy_ref.policy_sha256
                    )
            response_bytes = response.to_bytes()
            state.timer(f"{label} evaluation/response serialization", started)
            state.operations.append(operation)
            state.responses[i], state.response_bytes[i] = _sha(response_bytes), len(response_bytes)
            started = time.perf_counter_ns()
            state.stage = "binding"
            if isinstance(spec, RawPrefixSpec):
                if not isinstance(request, _RawRequest):
                    raise ValueError("ckks.request_binding")
                raw_received = _RawResponse.from_bytes(response_bytes)
                _require(
                    raw_received.policy_sha256 == policy_ref.policy_sha256, "ckks.request_binding"
                )
                state.stage = "crypto"
                statistics, operation = _raw_decrypt(client, spec, request, raw_received)
            else:
                if boundary_client is None or not isinstance(request, RatioLiftRequest):
                    raise ValueError("ckks.request_binding")
                lifted_received = RatioLiftResponse.from_bytes(response_bytes)
                _require(
                    lifted_received.policy_sha256 == policy_ref.policy_sha256,
                    "ckks.request_binding",
                )
                state.stage = "crypto"
                statistics, operation = boundary_client.decrypt_statistics(request, lifted_received)
            value = _cipher_value(
                statistics, policy_ref.policy_sha256, spec.gamma, reference.normalization
            )
            value._validate()
            _require(
                request.digest == state.request
                and request.to_bytes() == request_bytes
                and server_request.to_bytes() == request_bytes,
                "ckks.request_binding",
            )
            values[i] = value
            state.timer(f"{label} decrypt/validation", started)
            state.operations.append(operation)
        if state.timed_started is None:
            raise ValueError("context timer not started")
        state.total_ns = time.perf_counter_ns() - state.timed_started
        left, right = values
        if left is None or right is None:
            raise ValueError("ckks.request_binding")
        errors = recompute_cipher_errors(reference, left, right, spec.gamma)
        contrast = right.normalized_value - left.normalized_value
        lower, upper, decision = translate_cipher_interval(reference.bootstrap, contrast)
        state.metrics = CipherBatchMetrics(
            "cipher_batch",
            reference,
            left,
            right,
            errors,
            contrast,
            lower,
            upper,
            decision,
            state.timing(),
        )
    except Exception:
        # A result whose final binding/error validation failed is not a valid B
        # outcome; preserve the earlier strictly validated A result only.
        values[1] = None
        state.total_ns = None
        state.metrics = CipherBatchMetrics(
            "cipher_batch",
            reference,
            values[0],
            values[1],
            None,
            None,
            None,
            None,
            "unavailable",
            state.timing(),
        )
        raise


def _count_context(state: _Execution, manifest: PrivateOPEManifest, job: PlannedJob) -> None:
    c = dict(job.coordinates)
    n = _coordinate_int(c["trajectories"])
    parameters = _expected_parameters(job)
    state.computation = _sha(
        canonical_bytes(
            {
                "identifier": "PRIVATE_OPE_COUNT_DIAGNOSTIC_V1",
                "parameters": dataclasses.asdict(parameters),
                "expected_sum": n,
                "counts_source": "diagnostic_sum",
            }
        )
    )
    state.metrics = CountPrecisionMetrics("count_precision", n, None, None, state.timing())
    artifacts, client = state.contexts(parameters)
    state.stage = "crypto"
    started = time.perf_counter_ns()
    vector, _ = client.encrypt(np.ones(n, dtype=np.float64))
    payload = vector.to_bytes()
    state.request, state.request_bytes = _sha(payload), len(payload)
    state.operations.append(
        _operation("PRIVATE_OPE_COUNT_DIAGNOSTIC_V1", "encrypt_count", started, outputs=(vector,))
    )
    started = time.perf_counter_ns()
    server = CKKSServer.from_serialized(artifacts.server_context, parameters=parameters)
    result, _ = server.evaluate(SerializedCKKSVector.from_bytes(payload), lambda v: v.reduce_sum())
    state.operations.append(
        _operation("PRIVATE_OPE_COUNT_DIAGNOSTIC_V1", "reduce_count", started, (vector,), (result,))
    )
    started = time.perf_counter_ns()
    raw = float(client.decrypt(SerializedCKKSVector.from_bytes(result.to_bytes()))[0][0])
    state.operations.append(
        _operation("PRIVATE_OPE_COUNT_DIAGNOSTIC_V1", "decrypt_count", started, (result,))
    )
    if state.timed_started is None:
        raise ValueError("context timer not started")
    state.total_ns = time.perf_counter_ns() - state.timed_started
    _require(math.isfinite(raw), "ckks.nonfinite")
    state.metrics = CountPrecisionMetrics("count_precision", n, raw, abs(raw - n), state.timing())
    # Raw value is retained before tolerance; a numerical rejection is not a
    # replacement context and remains part of the fixed twelve-slot diagnostic.
    _require(abs(raw - n) <= manifest.gates.diagnostic_sum_abs_error, "ckks.count_precision")


def _historical_context(state: _Execution, manifest: PrivateOPEManifest, job: PlannedJob) -> None:
    from unseen_loop.ope.ckks import (
        EncryptedOPERequest,
        EncryptedOPEResponse,
        OPECKKSClient,
        OPECKKSServer,
        PolynomialApproxOPESpec,
        executable_ckks_parameters,
    )

    started = time.perf_counter_ns()
    batch, policy = _legacy_batch(_coordinate_int(dict(job.coordinates)["case_index"]))
    spec = PolynomialApproxOPESpec(
        batch.spec, policy, gamma=1.0, weight_clip=2.0, minimum_behavior_propensity=0.2
    )
    clear = spec.clear_oracle(batch, "clipped_wpdis")
    state.clear_ns = time.perf_counter_ns() - started
    state.batch = _sha(canonical_bytes(batch.to_dict()))
    parameters = executable_ckks_parameters(64, 8)
    state.computation = _sha(canonical_bytes(dataclasses.asdict(spec.receipt(parameters))))
    digest = _sha(canonical_bytes(policy.to_dict()))
    raw_counts: tuple[float, ...] | None = None
    decrypted: SufficientStatistics | None = None
    state.metrics = HistoricalMetrics(
        "historical",
        "POLYNOMIAL_APPROX_OPE_V1",
        digest,
        clear,
        None,
        None,
        None,
        None,
        None,
        state.timing(),
    )
    try:
        artifacts, client = state.contexts(parameters)
        state.stage = "crypto"
        boundary_client = OPECKKSClient(client, spec)
        started = time.perf_counter_ns()
        request, operation = boundary_client.encrypt_batch(batch)
        wire = canonical_bytes(_legacy_wire(request))
        state.request, state.request_bytes = _sha(wire), len(wire)
        state.operations.append(operation)
        state.timer("encryption/request serialization", started)
        started = time.perf_counter_ns()
        server = OPECKKSServer(
            CKKSServer.from_serialized(artifacts.server_context, parameters=parameters), spec
        )
        received = _legacy_unwire(EncryptedOPERequest, json.loads(wire))
        state.timer("server context/request loading", started)
        started = time.perf_counter_ns()
        response, operation = server.evaluate(received)
        wire = canonical_bytes(_legacy_wire(response))
        state.responses[0], state.response_bytes[0] = _sha(wire), len(wire)
        state.operations.append(operation)
        state.timer("A evaluation/response serialization", started)
        started = time.perf_counter_ns()
        response = _legacy_unwire(EncryptedOPEResponse, json.loads(wire))
        # Preserve the original count path; inspect its raw aggregate before
        # legacy nearest-integer validation can reject the output.
        counts = tuple(float(client.decrypt(v)[0][0]) for v in response.counts)
        if all(math.isfinite(v) for v in counts):
            raw_counts = counts
        decrypted, operation = boundary_client.decrypt_statistics(response, "clipped_wpdis")
        _require(decrypted.estimate is not None, "ckks.nonpositive_denominator")
        state.operations.append(operation)
        state.timer("A decrypt/validation", started)
        if state.timed_started is None:
            raise ValueError("context timer not started")
        state.total_ns = time.perf_counter_ns() - state.timed_started
    finally:
        state.metrics = HistoricalMetrics(
            "historical",
            "POLYNOMIAL_APPROX_OPE_V1",
            digest,
            clear,
            decrypted,
            raw_counts,
            max(abs(a - b) for a, b in zip(clear.numerators, decrypted.numerators, strict=True))
            if decrypted
            else None,
            max(abs(a - b) for a, b in zip(clear.denominators, decrypted.denominators, strict=True))
            if decrypted
            else None,
            abs(clear.estimate - decrypted.estimate)
            if decrypted and decrypted.estimate is not None and clear.estimate is not None
            else None,
            state.timing(),
        )


def execute_private_ope_job(
    config_bytes: bytes,
    payload: dict[str, object] | PrivateOPEJobPayload,
    run_root: str,
    runtime_receipt: RuntimeReceipt | dict[str, object],
) -> PrivateOPEAttempt:
    """Execute one already-claimed logical attempt in a verified Modal worker.

    Main's wrappers bind invocation IDs, persist disjoint result envelopes and
    handle timeout sleeping.  No retry, fallback, private-row write, or dispatch
    is performed here.
    """
    from unseen_loop.flagship.manifest import (
        iter_private_ope_jobs,
        parse_private_ope_manifest_bytes,
    )
    from unseen_loop.ope.study import evaluate_queue_batch, queue_batch

    parsed = validate_job_payload(config_bytes, payload, run_root)
    manifest = parse_private_ope_manifest_bytes(config_bytes)
    runtime = RuntimeReceipt.from_dict(
        runtime_receipt.to_dict()
        if isinstance(runtime_receipt, RuntimeReceipt)
        else runtime_receipt
    )
    root = Path(run_root)
    provenance = _sha((root / "provenance.json").read_bytes())
    c = dict(parsed.job.coordinates)
    state = _Execution(runtime, _coordinate_str(c["kind"]))
    failure = None
    try:
        _require(runtime.source_match, "evidence.source_mismatch")
        deadline = dt.datetime.fromisoformat(parsed.deadline_utc.replace("Z", "+00:00"))
        _require(dt.datetime.now(dt.UTC) <= deadline, "runtime.timeout")
        for field in (
            "image_spec_sha256",
            "code_commit",
            "candidate_code_sha256",
            "baseline_code_sha256",
            "domain_code_sha256",
            "analysis_code_sha256",
            "lockfile_sha256",
        ):
            _require(
                getattr(runtime, field) == getattr(manifest.execution, field),
                "evidence.source_mismatch",
            )
        if c["kind"] in MODERN_KINDS | {"clear_batch"}:
            state.stage = "clear"
            started = time.perf_counter_ns()
            batch, next_states = queue_batch(
                _seed(c["data_seed"]),
                _coordinate_int(c["trajectories"]),
                _coordinate_int(c["horizon"]),
                _coordinate_str(c["behavior"]),
            )
            reference = BatchReference.from_dict(
                evaluate_queue_batch(
                    batch,
                    next_states,
                    gamma=manifest.domain.gamma,
                    repetitions=manifest.statistics.bootstrap_repetitions,
                    bootstrap_seed=_seed(c["bootstrap_seed"]),
                )
            )
            state.clear_ns = time.perf_counter_ns() - started
            state.batch = reference.batch_sha256
            if c["kind"] == "clear_batch":
                state.metrics = ClearBatchMetrics("clear_batch", reference)
            else:
                _modern_context(state, manifest, parsed.job, batch, reference)
        elif c["kind"] == "count_precision":
            _count_context(state, manifest, parsed.job)
        elif c["kind"] == "historical_context":
            _historical_context(state, manifest, parsed.job)
        elif c["kind"] == "protocol_verification":
            state.stage = "verification"
            state.metrics = _verify_protocol()
            _require(state.metrics.required_cases_passed, "verification.failed")
        elif c["kind"] == "smoke_error":
            state.metrics = ProbeMetrics(
                "probe",
                "probe.deliberate_exception",
                "probe.deliberate_exception",
                time.perf_counter_ns() - state.started,
            )
            raise RuntimeError("probe.deliberate_exception")
        elif c["kind"] == "smoke_timeout":
            # The two-second Modal Function is the actual timeout surface.
            # Invoking this executor branch is an orchestration contract error.
            raise ValueError("runtime.interrupted")
        elif c["kind"] == "analysis":
            state.stage = "analysis"
            rows = [
                PrivateOPEAttempt.from_dict(
                    json.loads((root / "attempts" / f"{job.job_id}.json").read_bytes())
                )
                for job in iter_private_ope_jobs(manifest)
                if dict(job.coordinates)["kind"] != "analysis"
            ]
            state.metrics = analyze_private_ope(manifest, rows)
    except Exception as error:
        failure = _failure_code(error, state.stage)
    if isinstance(state.metrics, (CipherBatchMetrics, CountPrecisionMetrics, HistoricalMetrics)):
        state.metrics = dataclasses.replace(state.metrics, timing=state.timing())
    attempt = PrivateOPEAttempt(
        "unseen-loop/private-ope-attempt-v1",
        parsed.run_id,
        parsed.config_sha256,
        provenance,
        parsed.job,
        None,
        None,
        True,
        failure is None,
        failure,
        None,
        state.metrics,
        state.receipts(),
        False,
        False,
    )
    return PrivateOPEAttempt.from_dict(attempt.to_dict())
