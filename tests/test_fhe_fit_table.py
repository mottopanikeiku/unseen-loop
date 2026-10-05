from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_module_spec = importlib.util.spec_from_file_location(
    "fhe_fit_table", ROOT / "tools" / "fhe_fit_table.py"
)
assert _module_spec is not None and _module_spec.loader is not None
fit = importlib.util.module_from_spec(_module_spec)
sys.modules[_module_spec.name] = fit
_module_spec.loader.exec_module(fit)


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    paths = (
        f"{fit.NONLINEAR}/policy.json",
        f"{fit.NONLINEAR}/summary.json",
        f"{fit.NONLINEAR}/raw.jsonl",
        f"{fit.TIMING}/summary.json",
        f"{fit.TIMING}/context.json",
        f"{fit.TIMING}/raw.jsonl",
        fit.PUBLICATION,
        fit.SMOKE,
        fit.REFERENCE,
        fit.REFERENCE_POLICY,
    )
    for path in paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    return tmp_path


def test_committed_extraction_keeps_measurement_populations_separate() -> None:
    rows = fit.extract_rows(ROOT)
    assert len(rows) == 8
    nonlinear, repeated, shield, ope, ckks, smoke, systems, reference = rows
    assert "features=2; actions=2; degree=2; coefficients=12; qmax=2" == nonlinear.shape
    assert "p50 0.362091 s; p95 0.374514 s; n=40" in nonlinear.latency
    assert "40/40 match" in nonlinear.correctness
    assert "domain: 25 calls; fresh-ciphertext canary: 15 calls" in nonlinear.correctness
    assert "472,645,952 B" in nonlinear.material
    assert "p50 0.544536 s; p95 0.830709 s; n=64, 4" in repeated.latency
    assert "64/64" in repeated.correctness
    assert "excluded warmups: 12" in repeated.correctness
    assert "output equality not recorded" in repeated.correctness
    assert "(5,2,4)" in shield.shape
    assert shield.latency == "73.916227 s; one colocated call, not a percentile"
    assert "758,473,160 B" in shield.material
    assert "1.826500 s" in ope.latency
    assert "(N,H,D)=(1,2,1)" in ope.shape
    assert "589,084,056 B" in ope.material
    assert ckks.latency == "not recorded in retained publication excerpt"
    assert "public server context: 813,969,437 B" in ckks.material
    assert "evaluation keys" not in ckks.material
    assert "request: not recorded" in ckks.material
    assert "8.399916065832734" in ckks.correctness
    assert "185/200" in smoke.correctness and "5/5 actions" in smoke.correctness
    assert "exact-clear conformance=False" in smoke.correctness
    assert "2/5 measured successes" in systems.correctness
    assert "not labeled server-only" in systems.latency
    assert "not a colocated measurement" in reference.latency
    assert "coefficients=10" in reference.shape
    assert "evaluation keys: 64 B" in reference.material
    assert "2/2 match" in reference.correctness and "25/25 match" in reference.correctness


def test_extraction_reads_raw_failures_and_does_not_borrow_missing_bytes(
    artifact_root: Path,
) -> None:
    path = artifact_root / fit.NONLINEAR / "raw.jsonl"
    raw = fit.load_jsonl(artifact_root, f"{fit.NONLINEAR}/raw.jsonl")
    raw[0]["real_fhe_matches_integer_clear"] = False
    raw[0].pop("evaluation_key_bytes")
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n", encoding="utf-8")
    row = fit.extract_rows(artifact_root)[0]
    assert "39/40 match" in row.correctness
    assert "evaluation keys: not recorded for some/all calls" in row.material
    assert "472,645,952" not in row.material


def test_timing_shape_requires_matching_policy_digest(artifact_root: Path) -> None:
    path = artifact_root / fit.TIMING / "context.json"
    context = fit.load_json(artifact_root, f"{fit.TIMING}/context.json")
    context["container_contexts"][0]["circuit_receipt"]["policy_digest"] = "different-policy"
    path.write_text(json.dumps(context), encoding="utf-8")
    row = fit.extract_rows(artifact_root)[1]
    assert row.shape == "policy size not linked to nonlinear artifact"
    assert "coefficients=12" not in row.shape


def test_absent_canary_measurements_are_explicit(artifact_root: Path) -> None:
    path = artifact_root / fit.PUBLICATION
    publication = fit.load_json(artifact_root, fit.PUBLICATION)
    call = publication["shield"]["canary"]["real_call"]["call"]
    call.pop("server_evaluate_ns")
    call.pop("evaluation_key_bytes")
    path.write_text(json.dumps(publication), encoding="utf-8")
    row = fit.extract_rows(artifact_root)[2]
    assert row.latency.startswith("not recorded;")
    assert row.material.startswith("evaluation keys: not recorded")
    assert "758,473,160" not in row.material


def test_missing_source_is_an_error_not_an_empty_table(artifact_root: Path) -> None:
    (artifact_root / fit.PUBLICATION).unlink()
    with pytest.raises(FileNotFoundError):
        fit.extract_rows(artifact_root)


def test_formatting_does_not_treat_zero_as_missing() -> None:
    assert fit.seconds(0) == "0.000000 s"
    assert fit.seconds(None) == "not recorded"
    assert fit.byte_values([{"request_bytes": 0}], "request_bytes") == "0 B"
    assert fit.byte_values([{"request_bytes": 2}, {"request_bytes": 1}], "request_bytes") == "1/2 B"
    assert fit.matches([{"equal": False}, {"equal": True}, {}], "equal") == (
        "1/3 match (1 unrecorded)"
    )


def test_markdown_escapes_cells_and_links_sources() -> None:
    row = fit.FitRow("label|part", "shape\nsize", "latency", "bytes", "counts", (fit.SMOKE,))
    markdown = fit.render([row])
    assert "label\\|part | shape size" in markdown
    assert f"[{fit.SMOKE}](../{fit.SMOKE})" in markdown
    assert "different experiments, not pooled samples" in markdown
    assert "does not demonstrate" in markdown
    assert markdown.endswith("\n")


def test_generated_document_is_deterministic_and_current() -> None:
    expected = fit.render(fit.extract_rows(ROOT))
    assert fit.render(fit.extract_rows(ROOT)) == expected
    assert (ROOT / "docs" / "fhe-fit.md").read_text(encoding="utf-8") == expected
