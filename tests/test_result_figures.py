from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
_module_spec = importlib.util.spec_from_file_location(
    "result_figures", ROOT / "tools" / "result_figures.py"
)
assert _module_spec is not None and _module_spec.loader is not None
fig = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(fig)


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    for path in (
        fig.ENVIRONMENTS,
        fig.ABLATIONS,
        f"{fig.NONLINEAR}/summary.json",
        f"{fig.NONLINEAR}/raw.jsonl",
        f"{fig.NONLINEAR}/policy.json",
        fig.REFERENCE,
        fig.REFERENCE_POLICY,
    ):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    return tmp_path


def test_clear_figure_preserves_committed_intervals_and_comparison_scopes() -> None:
    image = fig.clear_returns()
    assert "-29.480 [-70.988, 0.072]" in image
    assert "+0.796 [-1.620, 3.502]" in image
    assert "-231.896 [-387.814, -76.253]" in image
    assert "+83.619 [26.971, 147.058]" in image
    assert "-108.461 [-288.061, 68.172]" in image
    assert "Teacher -94.260 → student -326.156" in image
    assert "500 episode pairs, 5 checkpoints" in image
    assert "CLEAR rollouts only" in image
    assert "different comparisons" in image
    assert "do not establish equivalence" in image
    assert "Reward scales differ" in image


def test_encrypted_figure_does_not_merge_synthetic_and_affine_evidence() -> None:
    image = fig.encrypted_scope()
    assert "SYNTHETIC quadratic circuit" in image
    assert "40/40 encrypted calls" in image
    assert "Server p50 362.091 ms · p95 374.514 ms" in image
    assert "472,645,952 bytes" in image
    assert "2 inputs · 2 outputs · degree 2 · 12 coefficients · qmax=2" in image
    assert "EARLIER affine CartPole trace" in image
    assert "25/25 encrypted control steps" in image
    assert "4 inputs · 2 outputs · degree 1 · 10 coefficients · qmax=15" in image
    assert "64 bytes (this affine circuit only)" in image
    assert "not a full-episode comparison" in image
    assert "NOT SHOWN" in image


def test_missing_bytes_and_changed_equality_are_not_borrowed_from_summary(
    artifact_root: Path,
) -> None:
    path = artifact_root / fig.NONLINEAR / "raw.jsonl"
    rows = fig.load_jsonl(artifact_root, f"{fig.NONLINEAR}/raw.jsonl")
    rows[0].pop("evaluation_key_bytes")
    rows[0]["real_fhe_matches_integer_clear"] = False
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    image = fig.encrypted_scope(artifact_root)
    assert "39/40 encrypted calls" in image
    assert "not recorded for some/all calls" in image
    assert "472,645,952" not in image


def test_clear_figure_reads_intervals_instead_of_recomputing_them(artifact_root: Path) -> None:
    rows = fig.load_jsonl(artifact_root, fig.ENVIRONMENTS)
    rows[0]["paired_return_delta"]["bootstrap"]["ci95_high"] = 12.345
    (artifact_root / fig.ENVIRONMENTS).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    assert "-29.480 [-70.988, 12.345]" in fig.clear_returns(artifact_root)


def test_unrecorded_latency_is_explicit_and_zero_is_not_missing(artifact_root: Path) -> None:
    path = artifact_root / fig.NONLINEAR / "summary.json"
    summary = fig.load_json(artifact_root, f"{fig.NONLINEAR}/summary.json")
    summary["challenge_summary"]["timing_distributions"]["server_evaluate_ns"].pop("p95_ns")
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert "p95 not recorded" in fig.encrypted_scope(artifact_root)
    assert fig.milliseconds(0) == "0.000 ms"
    assert fig.material([{"evaluation_key_bytes": 0}]) == "0 bytes"


def test_absent_source_does_not_make_an_empty_figure(artifact_root: Path) -> None:
    (artifact_root / fig.ENVIRONMENTS).unlink()
    with pytest.raises(FileNotFoundError):
        fig.clear_returns(artifact_root)


def test_checked_in_svgs_are_deterministic_accessible_and_source_linked() -> None:
    expected = fig.figures()
    assert fig.figures() == expected
    for name, image in expected.items():
        path = ROOT / "docs" / "figures" / name
        assert path.read_text(encoding="utf-8") == image
        tree = ET.fromstring(image)
        assert tree.attrib["role"] == "img"
        assert tree.find(f"{{{fig.SVG}}}title") is not None
        assert tree.find(f"{{{fig.SVG}}}desc") is not None
        links = tree.findall(f".//{{{fig.SVG}}}a")
        assert links
        for link in links:
            assert (path.parent / link.attrib["href"]).is_file()
