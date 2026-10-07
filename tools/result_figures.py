"""Draw source-linked SVGs from committed results; never run training or encryption.

Run from the repository root: python tools/result_figures.py
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = "artifacts/studies/unseen-loop-release-analysis-004"
ENVIRONMENTS = f"{ANALYSIS}/expanded-environments.jsonl"
ABLATIONS = f"{ANALYSIS}/ablation-effects.jsonl"
NONLINEAR = "artifacts/studies/modal-nonlinear-qmax2-002"
REFERENCE = "artifacts/reference/modal-smoke-001.json"
REFERENCE_POLICY = "artifacts/reference/modal-smoke-001/modal/policy.json"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def load_json(root: Path, path: str) -> Any:
    return json.loads((root / path).read_text(encoding="utf-8"))


def load_jsonl(root: Path, path: str) -> list[dict[str, Any]]:
    with (root / path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def element(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k: str(v) for k, v in attrs.items()})


def text(parent: ET.Element, x: float, y: float, value: str, **attrs: Any) -> None:
    element(parent, "text", x=x, y=y, **attrs).text = value


def source(parent: ET.Element, path: str) -> ET.Element:
    # Figures are committed in docs/figures/; links work when opened as SVG files.
    return element(parent, "a", href=f"../../{path}")


def canvas(title: str, description: str, height: int) -> ET.Element:
    image = ET.Element(
        f"{{{SVG}}}svg",
        {
            "viewBox": f"0 0 1080 {height}",
            "width": "1080",
            "height": str(height),
            "role": "img",
            "aria-labelledby": "title description",
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": "20",
            "fill": "#172b3a",
        },
    )
    element(image, "title", id="title").text = title
    element(image, "desc", id="description").text = description
    element(image, "rect", width=1080, height=height, fill="#ffffff", rx=12)
    text(image, 32, 44, title, **{"font-size": 29, "font-weight": 700})
    return image


def serialize(image: ET.Element) -> str:
    ET.indent(image, space="  ")
    return ET.tostring(image, encoding="unicode") + "\n"


def position(value: float) -> str:
    # Shared return-unit axis; task reward scales are not normalized.
    return f"{270 + (value + 420) / 600 * 355:.3f}"


def interval(parent: ET.Element, y: int, metric: dict[str, Any]) -> None:
    low, high, estimate = (metric[key] for key in ("ci95_low", "ci95_high", "estimate"))
    color = "#ad3333" if high < 0 else "#17624d" if low > 0 else "#385c7c"
    element(
        parent,
        "line",
        x1=position(low),
        x2=position(high),
        y1=y,
        y2=y,
        stroke=color,
        **{"stroke-width": 4},
    )
    for bound in (low, high):
        element(
            parent,
            "line",
            x1=position(bound),
            x2=position(bound),
            y1=y - 7,
            y2=y + 7,
            stroke=color,
            **{"stroke-width": 2},
        )
    element(parent, "circle", cx=position(estimate), cy=y, r=6, fill=color)
    text(parent, 655, y + 7, f"{estimate:+.3f} [{low:.3f}, {high:.3f}]", fill=color)


def clear_returns(root: Path = ROOT) -> str:
    environments = load_jsonl(root, ENVIRONMENTS)
    ablations = {row["effect"]: row for row in load_jsonl(root, ABLATIONS)}
    image = canvas(
        "Distillation in the clear: Acrobot loses return",
        "Paired clear integer-student minus teacher returns, followed by matched clear "
        "CartPole ablations. Dots and whiskers use the committed bootstrap estimates "
        "and 95% intervals. No encrypted RL rollouts are plotted.",
        750,
    )
    text(
        image,
        32,
        78,
        "CLEAR rollouts only. Dots = estimates; whiskers = recorded 95% bootstrap CIs.",
    )
    text(image, 270, 114, "Return difference", **{"font-weight": 700})
    text(image, 655, 114, "Estimate [95% CI]", **{"font-weight": 700})
    for tick in (-400, -200, 0, 150):
        x = position(tick)
        element(
            image,
            "line",
            x1=x,
            x2=x,
            y1=143,
            y2=630,
            stroke="#647889" if tick == 0 else "#e0e6eb",
            **{"stroke-dasharray": "5 5" if tick == 0 else "none"},
        )
        text(image, float(x), 139, str(tick), **{"text-anchor": "middle", "font-size": 17})
    for index, row in enumerate(environments):
        y = 180 + index * 88
        group = source(image, ENVIRONMENTS)
        text(group, 32, y + 7, row["environment"], **{"font-weight": 700})
        interval(group, y, row["paired_return_delta"]["bootstrap"])
        text(
            group,
            32,
            y + 37,
            f"Teacher {row['teacher_return']['mean']:.3f} → student "
            f"{row['student_return']['mean']:.3f}; {row['evaluation_pairs']} episode pairs, "
            f"{row['checkpoints']} checkpoints",
            **{"font-size": 18},
        )
    text(
        image,
        32,
        422,
        "Matched clear CartPole ablations (different comparisons)",
        **{"font-weight": 700},
    )
    for index, (key, label, definition) in enumerate(
        (
            ("occupancy_refinement_bundle_main_effect", "Refinement", "Refined minus unrefined"),
            ("weighting_main_effect", "Weighting", "Weighted minus unweighted"),
        )
    ):
        y = 475 + index * 88
        group = source(image, ABLATIONS)
        text(group, 32, y + 7, label, **{"font-weight": 700})
        interval(group, y, ablations[key]["paired_return_delta"])
        text(group, 32, y + 37, definition, **{"font-size": 18})
    text(
        image,
        32,
        652,
        "Zero = no mean difference. Intervals crossing zero do not establish equivalence.",
        **{"font-size": 18},
    )
    text(
        image,
        32,
        681,
        "Reward scales differ across tasks; compare each student with its own teacher.",
        **{"font-size": 18},
    )
    for y, path, label in (
        (717, ENVIRONMENTS, "Source: paired environment summaries"),
        (717, ABLATIONS, "Source: matched ablation effects"),
    ):
        text(
            source(image, path),
            32 if path == ENVIRONMENTS else 590,
            y,
            label,
            fill="#235c91",
            **{"font-size": 18, "text-decoration": "underline"},
        )
    return serialize(image)


def material(rows: list[dict[str, Any]]) -> str:
    if not rows or any(row.get("evaluation_key_bytes") is None for row in rows):
        return "not recorded for some/all calls"
    values = sorted({row["evaluation_key_bytes"] for row in rows})
    return "/".join(f"{value:,}" for value in values) + " bytes"


def milliseconds(value: Any) -> str:
    if value is None:
        return "not recorded"
    return f"{Decimal(str(value)) / Decimal(1_000_000):.3f} ms"


def policy_shape(policy: dict[str, Any]) -> str:
    return (
        f"{len(policy['quantizer']['center'])} inputs · {policy['actions']} outputs · "
        f"degree {policy['degree']} · "
        f"{sum(len(row) for row in policy['integer_coefficients'])} coefficients · "
        f"qmax={policy['quantizer']['qmax']}"
    )


def encrypted_scope(root: Path = ROOT) -> str:
    summary = load_json(root, f"{NONLINEAR}/summary.json")["challenge_summary"]
    timings = summary["timing_distributions"]["server_evaluate_ns"]
    raw = load_jsonl(root, f"{NONLINEAR}/raw.jsonl")
    policy = load_json(root, f"{NONLINEAR}/policy.json")
    reference = load_json(root, REFERENCE)
    affine = load_json(root, REFERENCE_POLICY)
    trace = reference["closed_loop_real_fhe"]
    image = canvas(
        "Encrypted evidence: two small, different experiments",
        "A synthetic quadratic circuit was checked on a colocated client/server worker. "
        "An earlier affine CartPole policy had a short local-client/remote-server trace. "
        "Neither demonstrates encrypted reproduction of the expanded RL study.",
        650,
    )
    text(image, 32, 78, "Existing Concrete measurements, not new runs or a head-to-head benchmark.")
    for y, color in ((110, "#edf4fa"), (352, "#f1f5ef")):
        element(image, "rect", x=24, y=y, width=1032, height=226, rx=10, fill=color)
    text(
        image,
        44,
        148,
        "SYNTHETIC quadratic circuit · client/server on one worker",
        **{"font-weight": 700, "font-size": 23},
    )
    text(source(image, f"{NONLINEAR}/policy.json"), 44, 183, policy_shape(policy))
    matches = sum(row.get("real_fhe_matches_integer_clear") is True for row in raw)
    text(
        source(image, f"{NONLINEAR}/raw.jsonl"),
        44,
        219,
        f"{matches}/{len(raw)} encrypted calls matched the clear integer circuit",
    )
    text(
        source(image, f"{NONLINEAR}/summary.json"),
        44,
        254,
        f"Server p50 {milliseconds(timings.get('p50_ns'))}"
        f" · p95 {milliseconds(timings.get('p95_ns'))}",
    )
    text(source(image, f"{NONLINEAR}/raw.jsonl"), 44, 289, f"Evaluation keys: {material(raw)}")
    text(
        source(image, f"{NONLINEAR}/summary.json"),
        44,
        319,
        "Sources: summary + raw calls + policy (linked above)",
        fill="#235c91",
        **{"font-size": 17},
    )
    text(
        image,
        44,
        390,
        "EARLIER affine CartPole trace · local client / remote server",
        **{"font-weight": 700, "font-size": 23},
    )
    text(source(image, REFERENCE_POLICY), 44, 425, policy_shape(affine))
    text(
        source(image, REFERENCE),
        44,
        461,
        f"{trace['exact_matches']}/{trace['completed_steps']} encrypted control steps "
        "matched clear scores",
    )
    text(
        source(image, REFERENCE),
        44,
        496,
        f"Trace terminated: {str(trace['terminated']).lower()} · not a full-episode comparison",
    )
    text(
        source(image, REFERENCE),
        44,
        531,
        f"Evaluation keys: {material([reference['client']])} (this affine circuit only)",
    )
    text(
        source(image, REFERENCE),
        44,
        561,
        "Source: earlier remote reference (linked above)",
        fill="#235c91",
        **{"font-size": 17},
    )
    text(
        image,
        32,
        610,
        "NOT SHOWN: encrypted expanded-study champions, nonlinear remote RL, or task safety.",
        **{"font-size": 19, "font-weight": 700},
    )
    return serialize(image)


def figures(root: Path = ROOT) -> dict[str, str]:
    return {"clear-returns.svg": clear_returns(root), "encrypted-scope.svg": encrypted_scope(root)}


def main() -> None:
    destination = ROOT / "docs" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for name, image in figures().items():
        (destination / name).write_text(image, encoding="utf-8")


if __name__ == "__main__":
    main()
