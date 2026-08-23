from __future__ import annotations

import pytest

from unseen_loop.artifacts import ArtifactLedger


def test_ledger_finalizes_and_detects_tampering(tmp_path) -> None:
    ledger = ArtifactLedger(tmp_path)
    ledger.write_json("summary.json", {"mode": "QUANTIZED CLEAR", "value": 3})
    ledger.write_text("raw/rows.jsonl", '{"row":1}\n')
    ledger.finalize()

    assert ledger.verify() == (True, ())
    (tmp_path / "summary.json").write_text("tampered")
    valid, failures = ledger.verify()
    assert not valid
    assert failures == ("checksum mismatch: summary.json",)


def test_ledger_rejects_unledgered_files(tmp_path) -> None:
    ledger = ArtifactLedger(tmp_path)
    ledger.write_json("summary.json", {"complete": True})
    ledger.finalize()
    (tmp_path / "stale.json").write_text("{}")

    valid, failures = ledger.verify()
    assert not valid
    assert failures == ("unledgered artifact: stale.json",)


def test_ledger_refuses_secret_key_paths(tmp_path) -> None:
    ledger = ArtifactLedger(tmp_path)
    with pytest.raises(ValueError, match="secret-key"):
        ledger.write_bytes("client-keys/secret-key.bin", b"never")
    assert not tuple(tmp_path.rglob("*"))


def test_ledger_refuses_path_traversal(tmp_path) -> None:
    ledger = ArtifactLedger(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        ledger.write_text("../escape", "no")
