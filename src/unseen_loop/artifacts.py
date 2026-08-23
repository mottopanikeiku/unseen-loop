"""Content-addressed, secret-excluding research artifact ledger."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    created_utc: str
    project_version: str
    python_version: str
    platform: str
    machine: str
    processor: str
    git_commit: str | None
    git_dirty: bool | None
    command: tuple[str, ...]
    mode: str
    schema_version: str = "unseen-loop/run-provenance-v1"

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        project_version: str,
        command: Iterable[str],
        mode: str,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
    ) -> RunProvenance:
        return cls(
            run_id=run_id,
            created_utc=datetime.now(UTC).isoformat(),
            project_version=project_version,
            python_version=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            command=tuple(command),
            mode=mode,
        )


class ArtifactLedger:
    """Atomic writer whose final checksum file is the completion marker."""

    _FORBIDDEN_NAMES = (
        "secret-key",
        "secret_key",
        "private-key",
        "private_key",
        "client-key",
        "client_key",
        "keyset",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: set[Path] = set()

    def _validate_relative(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must stay inside the ledger")
        lowered = "/".join(path.parts).lower()
        if any(marker in lowered for marker in self._FORBIDDEN_NAMES):
            raise ValueError("client secret-key material cannot be written to the artifact ledger")
        return path

    def write_bytes(self, relative: str | Path, payload: bytes) -> Path:
        path = self._validate_relative(relative)
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        self._paths.add(path)
        return destination

    def write_text(self, relative: str | Path, payload: str) -> Path:
        return self.write_bytes(relative, payload.encode())

    def write_json(self, relative: str | Path, value: Any) -> Path:
        return self.write_text(relative, json.dumps(value, sort_keys=True, indent=2) + "\n")

    def write_jsonl(self, relative: str | Path, rows: Iterable[Any]) -> Path:
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        return self.write_text(relative, payload)

    def finalize(self) -> Path:
        checksums: list[str] = []
        for relative in sorted(self._paths):
            payload = (self.root / relative).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            checksums.append(f"{digest}  {relative.as_posix()}")
        return self.write_text("checksums.sha256", "\n".join(checksums) + "\n")

    def verify(self) -> tuple[bool, tuple[str, ...]]:
        checksum_path = self.root / "checksums.sha256"
        if not checksum_path.exists():
            return False, ("checksums.sha256 is missing",)
        failures: list[str] = []
        for line in checksum_path.read_text().splitlines():
            expected, separator, relative = line.partition("  ")
            if not separator:
                failures.append(f"malformed checksum row: {line}")
                continue
            path = self._validate_relative(relative)
            source = self.root / path
            if not source.exists():
                failures.append(f"missing artifact: {relative}")
                continue
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(f"checksum mismatch: {relative}")
        return not failures, tuple(failures)


def dataclass_dict(value: Any) -> dict[str, Any]:
    raw = asdict(value)
    cleaned = _json_safe(raw)
    if not isinstance(cleaned, dict):
        raise TypeError("dataclass serialization did not produce an object")
    return cleaned


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
