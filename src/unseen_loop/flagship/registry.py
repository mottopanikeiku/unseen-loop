"""Append-only execution registry and closed flagship evidence index."""

from __future__ import annotations

import enum
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .manifest import PlannedJob, canonical_json, content_digest

REGISTRY_SCHEMA_VERSION = "unseen-loop/flagship-registry-v1"
EVIDENCE_SCHEMA_VERSION = "unseen-loop/flagship-evidence-index-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REASON = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


class RegistryError(RuntimeError):
    """Raised when registry history or a requested transition is invalid."""


class JobStatus(enum.StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed-out"
    REJECTED = "rejected"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.TIMED_OUT, JobStatus.REJECTED}
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Digests required to bind evidence to source, configuration, and images."""

    source_digest: str
    config_digest: str
    image_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.config_digest, "config_digest")
        if not self.image_digests or len({name for name, _ in self.image_digests}) != len(
            self.image_digests
        ):
            raise RegistryError("image_digests must contain uniquely named images")
        if tuple(sorted(self.image_digests)) != self.image_digests:
            raise RegistryError("image_digests must be sorted by image name")
        for name, digest in self.image_digests:
            if not name or not _REASON.fullmatch(name):
                raise RegistryError(f"invalid image name {name!r}")
            _require_digest(digest, f"image digest {name}")

    @classmethod
    def from_mapping(
        cls, *, source_digest: str, config_digest: str, image_digests: Mapping[str, str]
    ) -> Provenance:
        return cls(source_digest, config_digest, tuple(sorted(image_digests.items())))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "image_digests": dict(self.image_digests),
        }


@dataclass(frozen=True, slots=True)
class PlanEntry:
    job_id: str
    stage: str
    expected_terminal: JobStatus


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    stage: str
    expected_terminal: JobStatus
    status: JobStatus | None
    artifact_path: str | None = None
    artifact_digest: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    registry_id: str
    provenance: Provenance
    plan: tuple[PlanEntry, ...]
    records: tuple[JobRecord, ...]
    event_count: int
    tail_hash: str


@dataclass(frozen=True, slots=True)
class Transition:
    job_id: str
    status: JobStatus
    artifact_path: str | None = None
    artifact_digest: str | None = None
    reason_code: str | None = None


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RegistryError(f"{name} must be a lowercase SHA-256 digest")


def _expected_terminal(job: PlannedJob) -> JobStatus:
    coordinates = job.coordinate_dict()
    invalid = coordinates.get("kind") in {"invalid", "fhe_invalid"}
    return JobStatus.REJECTED if invalid else JobStatus.SUCCEEDED


def _plan_entries(jobs: Iterable[PlannedJob]) -> tuple[PlanEntry, ...]:
    entries = tuple(PlanEntry(job.job_id, job.stage, _expected_terminal(job)) for job in jobs)
    if not entries:
        raise RegistryError("registry plan must not be empty")
    if len({entry.job_id for entry in entries}) != len(entries):
        raise RegistryError("registry plan contains duplicate job IDs")
    return tuple(sorted(entries, key=lambda entry: entry.job_id))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _line(payload: Mapping[str, object]) -> bytes:
    return canonical_json(payload) + b"\n"


def _relative_artifact(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "." in candidate.parts
        or str(candidate) != path
    ):
        raise RegistryError("artifact_path must be a normalized relative POSIX path")
    return path


class AppendOnlyRegistry:
    """A hash-chained JSONL ledger with one irreversible attempt per planned job."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file() or self.path.is_symlink():
            raise RegistryError(f"registry does not exist or is not a regular file: {self.path}")
        self._cache_size = -1
        self._cache_mtime_ns = -1
        self._cache_event_count = 0
        self._cache_tail_hash = "0" * 64
        self._cache_records: dict[str, JobRecord] = {}
        self.snapshot()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        jobs: Iterable[PlannedJob],
        provenance: Provenance,
    ) -> AppendOnlyRegistry:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = _plan_entries(jobs)
        plan_payload = [
            {
                "job_id": entry.job_id,
                "stage": entry.stage,
                "expected_terminal": entry.expected_terminal.value,
            }
            for entry in entries
        ]
        registry_id = (
            "registry-"
            + content_digest({"provenance": provenance.to_dict(), "plan": plan_payload})[:24]
        )
        header = {
            "type": "header",
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "registry_id": registry_id,
            "created_at": _timestamp(),
            "provenance": provenance.to_dict(),
            "plan": plan_payload,
        }
        try:
            with destination.open("xb") as handle:
                handle.write(_line(header))
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise RegistryError("refusing to replace an existing registry") from exc
        return cls(destination)

    def snapshot(self) -> RegistrySnapshot:
        with self.path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                snapshot = self._read_locked(handle)
                self._install_cache(snapshot, os.fstat(handle.fileno()))
                return snapshot
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _install_cache(self, snapshot: RegistrySnapshot, stat: os.stat_result) -> None:
        self._cache_size = stat.st_size
        self._cache_mtime_ns = stat.st_mtime_ns
        self._cache_event_count = snapshot.event_count
        self._cache_tail_hash = snapshot.tail_hash
        self._cache_records = {record.job_id: record for record in snapshot.records}

    def _read_locked(self, handle: BinaryIO) -> RegistrySnapshot:
        handle.seek(0)
        lines = handle.readlines()
        if not lines or any(not line.endswith(b"\n") for line in lines):
            raise RegistryError("registry is empty or has a partial trailing record")
        try:
            payloads = [json.loads(line) for line in lines]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError("registry contains invalid JSONL") from exc
        header = payloads[0]
        if (
            not isinstance(header, dict)
            or header.get("type") != "header"
            or header.get("schema_version") != REGISTRY_SCHEMA_VERSION
        ):
            raise RegistryError("registry header is missing or has the wrong schema")
        if set(header) != {
            "type",
            "schema_version",
            "registry_id",
            "created_at",
            "provenance",
            "plan",
        }:
            raise RegistryError("registry header contains unknown or missing fields")
        provenance_payload = header["provenance"]
        if not isinstance(provenance_payload, dict) or set(provenance_payload) != {
            "source_digest",
            "config_digest",
            "image_digests",
        }:
            raise RegistryError("invalid registry provenance")
        images = provenance_payload["image_digests"]
        if not isinstance(images, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in images.items()
        ):
            raise RegistryError("invalid image digest mapping")
        provenance = Provenance.from_mapping(
            source_digest=provenance_payload["source_digest"],
            config_digest=provenance_payload["config_digest"],
            image_digests=images,
        )
        plan_payload = header["plan"]
        if not isinstance(plan_payload, list):
            raise RegistryError("registry plan must be an array")
        plan: list[PlanEntry] = []
        for raw in plan_payload:
            if not isinstance(raw, dict) or set(raw) != {"job_id", "stage", "expected_terminal"}:
                raise RegistryError("invalid registry plan entry")
            try:
                terminal = JobStatus(raw["expected_terminal"])
            except (TypeError, ValueError) as exc:
                raise RegistryError("invalid expected terminal status") from exc
            if terminal not in {JobStatus.SUCCEEDED, JobStatus.REJECTED}:
                raise RegistryError("planned terminal must be succeeded or rejected")
            if not isinstance(raw["job_id"], str) or not isinstance(raw["stage"], str):
                raise RegistryError("plan IDs and stages must be strings")
            plan.append(PlanEntry(raw["job_id"], raw["stage"], terminal))
        if (
            not plan
            or plan != sorted(plan, key=lambda item: item.job_id)
            or len({item.job_id for item in plan}) != len(plan)
        ):
            raise RegistryError("registry plan is empty, unsorted, or duplicated")
        expected_registry_id = (
            "registry-"
            + content_digest({"provenance": provenance.to_dict(), "plan": plan_payload})[:24]
        )
        if header["registry_id"] != expected_registry_id:
            raise RegistryError("registry header content does not match registry_id")
        plan_by_id = {entry.job_id: entry for entry in plan}
        states: dict[str, JobRecord] = {
            entry.job_id: JobRecord(entry.job_id, entry.stage, entry.expected_terminal, None)
            for entry in plan
        }
        tail_hash = "0" * 64
        expected_sequence = 1
        for raw in payloads[1:]:
            if not isinstance(raw, dict) or set(raw) != {
                "type",
                "sequence",
                "timestamp",
                "job_id",
                "stage",
                "status",
                "artifact_path",
                "artifact_digest",
                "reason_code",
                "previous_hash",
                "event_hash",
            }:
                raise RegistryError("registry event contains unknown or missing fields")
            event_hash = raw["event_hash"]
            unsigned = dict(raw)
            del unsigned["event_hash"]
            if raw["type"] != "event" or raw["sequence"] != expected_sequence:
                raise RegistryError("registry event sequence is invalid")
            if raw["previous_hash"] != tail_hash or content_digest(unsigned) != event_hash:
                raise RegistryError("registry hash chain is invalid")
            entry = plan_by_id.get(raw["job_id"])
            if entry is None or raw["stage"] != entry.stage:
                raise RegistryError("registry event references an extra job or wrong stage")
            try:
                status = JobStatus(raw["status"])
            except (TypeError, ValueError) as exc:
                raise RegistryError("registry event has invalid status") from exc
            previous = states[entry.job_id]
            if previous.status is None and status != JobStatus.STARTED:
                raise RegistryError("job terminal event occurred before started")
            if previous.status == JobStatus.STARTED and status not in TERMINAL_STATUSES:
                raise RegistryError("started job must transition exactly once to a terminal state")
            if previous.status in TERMINAL_STATUSES:
                raise RegistryError("terminal job has a replacement/retry event")
            artifact_path = raw["artifact_path"]
            artifact_digest = raw["artifact_digest"]
            reason_code = raw["reason_code"]
            if status == JobStatus.SUCCEEDED:
                if not isinstance(artifact_path, str) or not isinstance(artifact_digest, str):
                    raise RegistryError("succeeded event requires artifact path and digest")
                _relative_artifact(artifact_path)
                _require_digest(artifact_digest, "artifact_digest")
                if reason_code is not None:
                    raise RegistryError("succeeded event cannot have a reason code")
            elif status == JobStatus.STARTED:
                if any(
                    value is not None for value in (artifact_path, artifact_digest, reason_code)
                ):
                    raise RegistryError("started event cannot contain terminal fields")
            else:
                if artifact_path is not None or artifact_digest is not None:
                    raise RegistryError("non-success terminal events cannot claim artifacts")
                if not isinstance(reason_code, str) or _REASON.fullmatch(reason_code) is None:
                    raise RegistryError("non-success terminal events require a bounded reason code")
            states[entry.job_id] = JobRecord(
                entry.job_id,
                entry.stage,
                entry.expected_terminal,
                status,
                artifact_path,
                artifact_digest,
                reason_code,
            )
            tail_hash = event_hash
            expected_sequence += 1
        return RegistrySnapshot(
            header["registry_id"],
            provenance,
            tuple(plan),
            tuple(states[entry.job_id] for entry in plan),
            len(payloads) - 1,
            tail_hash,
        )

    def apply(self, transitions: Iterable[Transition]) -> None:
        """Validate and durably append a batch with one lock and one fsync."""
        requested = tuple(transitions)
        if not requested:
            return
        with self.path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                observed_stat = os.fstat(handle.fileno())
                if (
                    observed_stat.st_size != self._cache_size
                    or observed_stat.st_mtime_ns != self._cache_mtime_ns
                ):
                    snapshot = self._read_locked(handle)
                    self._install_cache(snapshot, observed_stat)
                working = dict(self._cache_records)
                sequence = self._cache_event_count
                tail_hash = self._cache_tail_hash
                output = bytearray()
                for transition in requested:
                    job_id = transition.job_id
                    status = transition.status
                    artifact_path = transition.artifact_path
                    artifact_digest = transition.artifact_digest
                    reason_code = transition.reason_code
                    record = working.get(job_id)
                    if record is None:
                        raise RegistryError(f"job is not in immutable plan: {job_id}")
                    if status == JobStatus.STARTED:
                        if record.status is not None:
                            raise RegistryError(
                                "job already has an attempt; retries cannot replace failures"
                            )
                    elif record.status != JobStatus.STARTED:
                        raise RegistryError(
                            "terminal transition requires exactly one started event"
                        )
                    if status == JobStatus.SUCCEEDED:
                        if (
                            artifact_path is None
                            or artifact_digest is None
                            or reason_code is not None
                        ):
                            raise RegistryError(
                                "success requires only artifact_path and artifact_digest"
                            )
                        artifact_path = _relative_artifact(artifact_path)
                        _require_digest(artifact_digest, "artifact_digest")
                    elif status == JobStatus.STARTED:
                        if any(
                            value is not None
                            for value in (artifact_path, artifact_digest, reason_code)
                        ):
                            raise RegistryError("started event cannot have terminal fields")
                    else:
                        if artifact_path is not None or artifact_digest is not None:
                            raise RegistryError(
                                "failed, timed-out, and rejected jobs cannot claim artifacts"
                            )
                        if reason_code is None or _REASON.fullmatch(reason_code) is None:
                            raise RegistryError("terminal reason_code is missing or invalid")
                    sequence += 1
                    unsigned: dict[str, object] = {
                        "type": "event",
                        "sequence": sequence,
                        "timestamp": _timestamp(),
                        "job_id": job_id,
                        "stage": record.stage,
                        "status": status.value,
                        "artifact_path": artifact_path,
                        "artifact_digest": artifact_digest,
                        "reason_code": reason_code,
                        "previous_hash": tail_hash,
                    }
                    tail_hash = content_digest(unsigned)
                    output.extend(_line({**unsigned, "event_hash": tail_hash}))
                    working[job_id] = JobRecord(
                        record.job_id,
                        record.stage,
                        record.expected_terminal,
                        status,
                        artifact_path,
                        artifact_digest,
                        reason_code,
                    )
                handle.seek(0, os.SEEK_END)
                handle.write(output)
                handle.flush()
                os.fsync(handle.fileno())
                self._cache_event_count = sequence
                self._cache_tail_hash = tail_hash
                current_stat = os.fstat(handle.fileno())
                self._cache_size = current_stat.st_size
                self._cache_mtime_ns = current_stat.st_mtime_ns
                self._cache_records = working
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append(
        self,
        job_id: str,
        status: JobStatus,
        *,
        artifact_path: str | None = None,
        artifact_digest: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        self.apply((Transition(job_id, status, artifact_path, artifact_digest, reason_code),))

    def started(self, job_id: str) -> None:
        self._append(job_id, JobStatus.STARTED)

    def succeeded(self, job_id: str, *, artifact_path: str, artifact_digest: str) -> None:
        self._append(
            job_id,
            JobStatus.SUCCEEDED,
            artifact_path=artifact_path,
            artifact_digest=artifact_digest,
        )

    def failed(self, job_id: str, *, reason_code: str) -> None:
        self._append(job_id, JobStatus.FAILED, reason_code=reason_code)

    def timed_out(self, job_id: str, *, reason_code: str = "worker.timeout") -> None:
        self._append(job_id, JobStatus.TIMED_OUT, reason_code=reason_code)

    def rejected(self, job_id: str, *, reason_code: str) -> None:
        self._append(job_id, JobStatus.REJECTED, reason_code=reason_code)


def finalize_evidence(
    registry: AppendOnlyRegistry,
    *,
    evidence_root: str | Path,
    index_name: str = "evidence-index.json",
    reject_extra_files: bool = True,
    supporting_paths: Sequence[str] = (),
) -> Path:
    """Close a complete plan into the sole root index without replacing prior evidence."""

    if index_name != "evidence-index.json":
        raise RegistryError("the root evidence index name is fixed")
    root = Path(evidence_root)
    if not root.is_dir() or root.is_symlink():
        raise RegistryError("evidence_root must be an existing regular directory")
    snapshot = registry.snapshot()
    records = {record.job_id: record for record in snapshot.records}
    planned = {entry.job_id for entry in snapshot.plan}
    if set(records) != planned:
        raise RegistryError("finalization rejected missing or extra jobs")
    incomplete = sorted(
        job_id for job_id, record in records.items() if record.status not in TERMINAL_STATUSES
    )
    if incomplete:
        raise RegistryError(f"finalization rejected {len(incomplete)} incomplete jobs")
    wrong = sorted(
        job_id for job_id, record in records.items() if record.status != record.expected_terminal
    )
    if wrong:
        raise RegistryError(
            f"finalization rejected {len(wrong)} failed, timed-out, or unexpectedly rejected jobs"
        )
    artifacts: dict[str, dict[str, str]] = {}
    referenced_paths: set[str] = set()
    for job_id, record in records.items():
        if record.status != JobStatus.SUCCEEDED:
            continue
        assert record.artifact_path is not None and record.artifact_digest is not None
        if record.artifact_path in referenced_paths:
            raise RegistryError("multiple jobs reference the same artifact path")
        referenced_paths.add(record.artifact_path)
        artifact = root / record.artifact_path
        if not artifact.is_file() or artifact.is_symlink():
            raise RegistryError(f"missing regular artifact for {job_id}")
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != record.artifact_digest:
            raise RegistryError(f"artifact digest mismatch for {job_id}")
        artifacts[job_id] = {"path": record.artifact_path, "sha256": observed}
    supporting_artifacts: dict[str, str] = {}
    for raw in supporting_paths:
        relative_path = PurePosixPath(raw)
        if (
            not raw
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != raw
            or raw in referenced_paths
            or raw in supporting_artifacts
        ):
            raise RegistryError("supporting artifact path is invalid or duplicated")
        artifact = root / raw
        if not artifact.is_file() or artifact.is_symlink():
            raise RegistryError(f"missing regular supporting artifact: {raw}")
        supporting_artifacts[raw] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    index_path = root / index_name
    registry_relative: str | None
    try:
        registry_relative = registry.path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        registry_relative = None
    if index_name in supporting_artifacts or (
        registry_relative is not None and registry_relative in supporting_artifacts
    ):
        raise RegistryError("supporting artifacts cannot replace the registry or evidence index")
    if index_name in referenced_paths or (
        registry_relative is not None and registry_relative in referenced_paths
    ):
        raise RegistryError("job artifacts cannot replace the registry or evidence index")
    if "checksums.sha256" in referenced_paths:
        raise RegistryError("the checksum ledger must be a supporting artifact")
    if "checksums.sha256" in supporting_artifacts:
        expected_ledger = {value["path"]: value["sha256"] for value in artifacts.values()}
        expected_ledger.update(
            {
                path: digest
                for path, digest in supporting_artifacts.items()
                if path != "checksums.sha256"
            }
        )
        if registry_relative is None:
            raise RegistryError("checksum closure requires an in-tree registry")
        expected_ledger[registry_relative] = hashlib.sha256(registry.path.read_bytes()).hexdigest()
        try:
            ledger_text = (root / "checksums.sha256").read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise RegistryError("checksum ledger must be UTF-8") from exc
        observed_ledger: dict[str, str] = {}
        for row in ledger_text.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", row)
            if match is None or match[2] in observed_ledger:
                raise RegistryError("checksum ledger has malformed or duplicate rows")
            observed_ledger[_relative_artifact(match[2])] = match[1]
        if not ledger_text.endswith("\n") or observed_ledger != expected_ledger:
            raise RegistryError(
                "checksum ledger must cover every canonical file except itself and the index"
            )
    if reject_extra_files:
        allowed = set(referenced_paths)
        allowed.update(supporting_artifacts)
        if registry_relative is not None:
            allowed.add(registry_relative)
        actual: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RegistryError(f"evidence tree contains symlink: {path}")
            if path.is_file() and path != index_path:
                actual.add(path.relative_to(root).as_posix())
        if actual != allowed:
            raise RegistryError(
                "finalization rejected extra/missing evidence files: "
                f"extra={sorted(actual - allowed)}, "
                f"missing={sorted(allowed - actual)}"
            )
    status_counts = Counter(record.status.value for record in records.values() if record.status)
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "registry_id": snapshot.registry_id,
        "registry_sha256": hashlib.sha256(registry.path.read_bytes()).hexdigest(),
        "registry_tail_hash": snapshot.tail_hash,
        "provenance": snapshot.provenance.to_dict(),
        "planned_job_ids": sorted(planned),
        "status_counts": dict(sorted(status_counts.items())),
        "supporting_artifacts": dict(sorted(supporting_artifacts.items())),
        "artifacts": artifacts,
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    if index_path.is_symlink():
        raise RegistryError("root evidence index must not be a symlink")
    if index_path.exists():
        if not index_path.is_file() or index_path.read_bytes() != encoded:
            raise RegistryError("refusing to replace a conflicting root evidence index")
        return index_path
    # Publish only a complete, fsynced file. A crash before publication cannot leave
    # a partially written closure marker that would be mistaken for closed evidence.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=".evidence-index-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # The coordinator serializes finalization. Volume v1 does not provide
        # the v2 hard-link publication primitive; rename keeps the marker atomic.
        if index_path.exists() or index_path.is_symlink():
            if (
                index_path.is_symlink()
                or not index_path.is_file()
                or index_path.read_bytes() != encoded
            ):
                raise RegistryError("refusing to replace a conflicting root evidence index")
        else:
            os.rename(temporary, index_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return index_path
