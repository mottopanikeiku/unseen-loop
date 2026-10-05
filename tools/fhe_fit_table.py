"""Render committed FHE measurements without importing or executing a crypto backend."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NONLINEAR = "artifacts/studies/modal-nonlinear-qmax2-002"
TIMING = "artifacts/studies/modal-fhe-timing-003"
PUBLICATION = "site/data/flagship-evidence.json"
SMOKE = "artifacts/flagship-smoke-20260902-023/analysis.json"
REFERENCE = "artifacts/reference/modal-smoke-001.json"
REFERENCE_POLICY = "artifacts/reference/modal-smoke-001/modal/policy.json"
MISSING = "not recorded"


@dataclass(frozen=True)
class FitRow:
    name: str
    shape: str
    latency: str
    material: str
    correctness: str
    sources: tuple[str, ...]


def load_json(root: Path, path: str) -> Any:
    return json.loads((root / path).read_text(encoding="utf-8"))


def load_jsonl(root: Path, path: str) -> list[dict[str, Any]]:
    with (root / path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def number(value: Any) -> str:
    return MISSING if value is None else f"{value:,}"


def seconds(value: Any) -> str:
    if value is None:
        return MISSING
    return f"{Decimal(str(value)) / Decimal(1_000_000_000):.6f} s"


def shape(value: Any) -> str:
    return MISSING if value is None else "(" + ",".join(map(str, value)) + ")"


def policy_shape(policy: dict[str, Any]) -> str:
    center = policy.get("quantizer", {}).get("center")
    coefficients = policy.get("integer_coefficients")
    count = None if coefficients is None else sum(len(row) for row in coefficients)
    return (
        f"features={number(None if center is None else len(center))}; "
        f"actions={number(policy.get('actions'))}; degree={number(policy.get('degree'))}; "
        f"coefficients={number(count)}; qmax={number(policy.get('quantizer', {}).get('qmax'))}"
    )


def byte_values(rows: list[dict[str, Any]], field: str) -> str:
    values = sorted({row[field] for row in rows if row.get(field) is not None})
    if not rows or len([row for row in rows if row.get(field) is not None]) != len(rows):
        return MISSING + " for some/all calls"
    return "/".join(number(value) for value in values) + " B"


def transport(rows: list[dict[str, Any]], *, ckks: bool = False) -> str:
    label = "public server context" if ckks else "evaluation keys"
    field = "server_context_bytes" if ckks else "evaluation_key_bytes"
    return (
        f"{label}: {byte_values(rows, field)}; "
        f"request: {byte_values(rows, 'request_bytes')}; "
        f"response: {byte_values(rows, 'response_bytes')}"
    )


def matches(rows: list[dict[str, Any]], field: str) -> str:
    matched = sum(row.get(field) is True for row in rows)
    recorded = sum(isinstance(row.get(field), bool) for row in rows)
    return f"{matched}/{len(rows)} match ({len(rows) - recorded} unrecorded)"


def distribution(metric: dict[str, Any], *, suffix: str = "") -> str:
    # Use the committed study's own estimator; do not pool or recompute quantiles.
    p50 = metric.get("p50_ns", metric.get("p50"))
    p95 = metric.get("p95_ns", metric.get("p95"))
    return f"p50 {seconds(p50)}; p95 {seconds(p95)}" + suffix


def extract_rows(root: Path = ROOT) -> list[FitRow]:
    policy = load_json(root, f"{NONLINEAR}/policy.json")
    nonlinear = load_json(root, f"{NONLINEAR}/summary.json")["challenge_summary"]
    raw = load_jsonl(root, f"{NONLINEAR}/raw.jsonl")
    domain = [row for row in raw if row.get("phase") == "exhaustive-domain"]
    fresh = [row for row in raw if row.get("phase") == "fresh-ciphertext-canary"]
    rows = [
        FitRow(
            "Synthetic polynomial / Concrete",
            policy_shape(policy),
            distribution(
                nonlinear["timing_distributions"]["server_evaluate_ns"],
                suffix=f"; n={len(raw)}, one colocated worker",
            ),
            transport(raw),
            f"REAL: {matches(raw, 'real_fhe_matches_integer_clear')}; "
            f"domain: {len(domain)} calls; fresh-ciphertext canary: {len(fresh)} calls; "
            f"simulation: {number(nonlinear.get('simulation_rows'))} rows, "
            f"all-match={nonlinear.get('simulation_all_match', MISSING)}",
            (f"{NONLINEAR}/summary.json", f"{NONLINEAR}/raw.jsonl", f"{NONLINEAR}/policy.json"),
        )
    ]

    timing = load_json(root, f"{TIMING}/summary.json")["timing_summary"]
    context = load_json(root, f"{TIMING}/context.json")
    timing_raw = load_jsonl(root, f"{TIMING}/raw.jsonl")
    measured = [row for row in timing_raw if row.get("is_warmup") is False]
    successful = [row for row in measured if row.get("success") is True]
    receipts = [item["circuit_receipt"] for item in context["container_contexts"]]
    # A shared family name alone is not enough to borrow the policy size.
    same_policy = bool(receipts) and all(
        receipt.get("policy_digest") == nonlinear["policy_digest"] for receipt in receipts
    )
    rows.append(
        FitRow(
            "Synthetic polynomial / repeated Concrete timing",
            policy_shape(policy) if same_policy else "policy size not linked to nonlinear artifact",
            distribution(
                timing["timing_ns"]["server_evaluate_ns"],
                suffix=f"; n={len(successful)}, "
                f"{len({row['container_id'] for row in measured})} colocated contexts",
            ),
            transport([row.get("byte_metrics", {}) for row in successful]),
            f"measured successes: {len(successful)}/{len(measured)}; "
            f"excluded warmups: {sum(row.get('is_warmup') is True for row in timing_raw)}; "
            "per-call output equality not recorded in sanitized raw rows",
            (f"{TIMING}/summary.json", f"{TIMING}/raw.jsonl", f"{TIMING}/context.json"),
        )
    )

    publication = load_json(root, PUBLICATION)
    shield = publication["shield"]["canary"]
    shield_call = shield["real_call"]["call"]
    rows.append(
        FitRow(
            "Exact shield canary / Concrete",
            f"input={shape(shield_call.get('input_shape'))}; margins="
            f"{shape(shield_call.get('output_shape'))} (actions,horizon,families); "
            f"qmax={number(shield['receipt'].get('qmax'))}; "
            "policy degree/coefficients: not applicable",
            seconds(shield_call.get("server_evaluate_ns"))
            + "; one colocated call, not a percentile",
            transport([shield_call]),
            f"one REAL tensor equality flag={shield_call.get('output_matches_clear', MISSING)}; "
            f"complete-domain simulation flag={shield.get('exact_complete_domain', MISSING)} "
            f"over {number(shield.get('domain_points'))} points; no per-point raw rows retained",
            (PUBLICATION,),
        )
    )

    exact = publication["ope"]["exact_canary"]
    call = exact["call"]
    spec = exact["shape"]
    rows.append(
        FitRow(
            "Exact hard-clipped OPE canary / Concrete",
            f"(N,H,D)=({spec['trajectories']},{spec['horizon']},{spec['state_dim']}); "
            f"input={shape(call.get('input_shape'))}; output={shape(call.get('output_shape'))} "
            "(statistics,horizon); policy degree/actions/coefficients not recorded in excerpt",
            seconds(call.get("server_evaluate_ns")) + "; one colocated call, not a percentile",
            transport([call]),
            "one REAL/reference equality flag="
            f"{call.get('output_matches_integer_reference', MISSING)}; "
            f"simulation/REAL equality flag={exact.get('simulation_matches_real', MISSING)}; "
            f"released counts={shape(exact['integer_statistics'].get('counts'))}",
            (PUBLICATION,),
        )
    )

    ope = publication["ope"]
    ckks = ope["variant"]
    ckks_spec = ope["batch"]["trajectory_spec"]
    computation = ckks["receipt"]["computation"]
    target = ope["target_policy"]
    rows.append(
        FitRow(
            "Approximate polynomial OPE / TenSEAL CKKS",
            f"(N,H,D)=({ckks_spec['trajectories']},{ckks_spec['horizon']},"
            f"{ckks_spec['state_dim']}); "
            f"actions={number(target.get('action_count'))}; "
            f"policy degree={number(target.get('degree'))}; "
            f"policy coefficient count={MISSING}; soft-clip coefficients="
            f"{number(len(computation['soft_clip_coefficients']))}; output ciphertexts="
            f"{number(computation.get('output_ciphertexts'))}",
            "not recorded in retained publication excerpt",
            transport([ckks["receipt"]["context"]], ckks=True),
            f"exact-clear match={ckks.get('matches_exact_clear', MISSING)}; released maximum "
            f"horizon-numerator error={ckks.get('max_numerator_error', MISSING)}; "
            "soft-clip absolute-error bound="
            f"{computation.get('soft_clip_absolute_error_bound', MISSING)}; "
            f"released counts={shape(ckks['statistics'].get('counts'))}; not exact hard clipping",
            (PUBLICATION,),
        )
    )

    smoke = load_json(root, SMOKE)["evidence_summary"]
    accounting = smoke["shield_fhe"]["accounting"]
    rows.append(
        FitRow(
            "Broader shield smoke / Concrete (negative)",
            "5x2x4 margins (publication description); "
            "input/degree/coefficients not recorded in summary",
            "not recorded per shield call in retained summary",
            "per-shield evaluation keys/request/response not recorded in retained summary",
            f"{number(accounting['valid_calls'])} valid calls; "
            f"{number(accounting['margin_matches'])}/"
            f"{number(accounting['decoded_margins'])} margins; "
            f"{number(accounting['action_matches'])}/{number(accounting['valid_calls'])} actions; "
            f"exact-clear conformance={smoke['shield_fhe']['exact_clear_conformance']}",
            (SMOKE, PUBLICATION),
        )
    )
    systems = smoke["systems"]
    rows.append(
        FitRow(
            "Mixed systems smoke (negative; not a new canary)",
            "OPE cell 64x8; D and policy coefficients not recorded in summary",
            "server latency not recorded; retained OPE cell p95="
            + f"{systems['ope_cell_p95_seconds']['64x8']:.6f} s"
            + " (metric is not labeled server-only)",
            "per-circuit bytes not recorded; mixed maximum evaluation keys="
            + number(
                next(
                    gate["observed"]
                    for gate in systems["gates"]
                    if gate["name"] == "maximum_evaluation_key_bytes"
                )
            )
            + " B",
            f"{number(systems['successful_measured_requests'])}/"
            f"{number(systems['measured_request_denominator'])} measured successes; "
            "cell p95 is a retained summary, not a comparable all-success latency distribution",
            (SMOKE,),
        )
    )

    reference = load_json(root, REFERENCE)
    reference_rows = reference["real_fhe_trials"]
    trajectory = reference["closed_loop_real_fhe"]["trajectory"]
    rows.append(
        FitRow(
            "Linear CartPole reference / Concrete",
            policy_shape(load_json(root, REFERENCE_POLICY)),
            "not a colocated measurement: local-entrypoint client / Modal server; "
            "no comparable colocated quantiles",
            transport(reference_rows + trajectory),
            f"trial equality: {matches(reference_rows, 'matches_integer_clear')}; "
            f"closed-loop equality: {matches(trajectory, 'matches_integer_clear')}; "
            "same-input distinct ciphertexts="
            f"{number(reference['same_input_canary']['distinct_ciphertexts'])}; "
            "not exhaustive REAL-domain verification",
            (REFERENCE, REFERENCE_POLICY),
        )
    )
    return rows


def render(rows: list[FitRow]) -> str:
    lines = [
        "# What fits under FHE today",
        "",
        "These are the small circuits measured in this repository, "
        "not a general FHE capacity limit.",
        "Exact integer circuits, synthetic policies, "
        "and approximate CKKS arithmetic are kept separate.",
        "The generator only reads committed JSON/JSONL; it does not compile, generate keys,",
        "execute cryptography, use the network, or request paid compute.",
        "",
        "Regenerate from the repository root (Python standard library only):",
        "",
        "```sh",
        "nice -n 19 uv run python tools/fhe_fit_table.py --output docs/fhe-fit.md",
        "```",
        "",
        "A system Python 3.11+ can replace `uv run python`; no project dependencies are imported.",
        "Latencies are seconds, rounded to six decimal places. Byte counts are serialized bytes,",
        "not resident memory. Requests, responses, evaluation keys, "
        "and public contexts are distinct.",
        "",
        "| Circuit / semantics | Shape and size | Colocated server evaluation | "
        "Serialized material | Correctness accounting | Sources |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        sources = "; ".join(f"[{path}](../{path})" for path in row.sources)
        cells = (row.name, row.shape, row.latency, row.material, row.correctness, sources)
        lines.append(
            "| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Reading the measurements",
            "",
            "- The nonlinear study mixes complete-domain calls "
            "with fresh encryptions in one context.",
            "  Its committed p50/p95 use linear interpolation. "
            "Repeated timing instead uses nearest-rank",
            "  quantiles over successful non-warmup requests, "
            "clustered across independent containers;",
            "  its summary also retains hierarchical container/request bootstrap intervals.",
            "  These are different experiments, not pooled samples or a scaling comparison.",
            "- The nonlinear and repeated timing configurations requested 16 CPUs and 32,768 MiB",
            "  per Modal worker. A CPU model and portable local-laptop latency are not recorded.",
            "  Colocation does not demonstrate local-client/remote-server input secrecy "
            "or network latency.",
            "- The shield's domain-wide check is simulation, not domain-wide REAL FHE execution.",
            "  One exact encrypted tensor does not overturn the broader smoke's margin mismatches.",
            "  Shield action selection and OPE division are client-side, "
            "not encrypted argmax/division.",
            "- The retained publication contains canary excerpts "
            "and hashes of source summaries, not",
            "  the original complete canary bundles. "
            "CKKS server latency and request/response bytes",
            "  are absent there. Earlier Markdown canary timings/CKKS bytes do not all match this",
            "  snapshot; this table uses the JSON values and leaves absent fields absent.",
            "  CKKS public server context includes public, "
            "relinearization, and Galois key material;",
            "  it is not interchangeable with Concrete evaluation keys or a secret client context.",
            "- CKKS reports approximate polynomial soft clipping, not exact hard-clipped OPE.",
            "  Its released numerator error is against a distinct exact clear reference; "
            "the soft-clip",
            "  bound is an analytic approximation bound, not a measured CKKS arithmetic error.",
            "  Synthetic conformance and arithmetic equality do not establish useful control or",
            "  statistical OPE accuracy, production throughput, "
            "or empirical cryptographic failure rates.",
            "",
            "The source list is deliberately fixed to these retained experiments. "
            "Missing source files",
            "are errors; missing measurement fields are shown explicitly "
            "rather than filled from prose.",
            "The reference's 64-byte evaluation-key serialization "
            "belongs to its linear circuit only.",
            "The systems smoke's mixed maximum is not assigned to any individual canary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write Markdown instead of printing it")
    args = parser.parse_args()
    markdown = render(extract_rows())
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
