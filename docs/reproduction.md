# Reproduction Guide

## Prerequisites

- Linux x86-64 for the recorded Concrete server artifact path;
- Python 3.11 or 3.12 (the project selects 3.12);
- `uv` 0.12 or compatible;
- authenticated Modal profile only for the cloud path;
- enough cache/storage for Concrete, Torch, and CUDA wheels.

Concrete and Zama dependencies have separate license and patent terms. The repository is MIT; review dependency terms before commercial use.

## Level 1: deterministic clear semantics

```bash
uv sync --extra dev
uv run unseen-loop demo \
  --backend clear \
  --env-id CartPole-v1 \
  --seed-root demo-2026-08 \
  --output artifacts/demo
uv run unseen-loop verify artifacts/demo
uv run unseen-loop inspect artifacts/demo
```

This trains a clear CEM teacher, searches eight students, runs student-occupancy refinement, computes held-out and exhaustive certificates, and writes a checksum ledger. It makes no privacy claim.

Expected invariant—not an expected fixed timing—is:

```json
{
  "backend": "clear",
  "label": "QUANTIZED CLEAR",
  "privacy_evidence": false
}
```

## Level 2: local serialized real FHE

```bash
uv sync --extra dev --extra fhe
uv run unseen-loop demo \
  --backend fhe \
  --env-id CartPole-v1 \
  --seed-root demo-2026-08 \
  --output artifacts/fhe-local
uv run unseen-loop verify artifacts/fhe-local
```

Required gates:

- `summary.label == "REAL FHE"`;
- `summary.privacy_evidence == true`;
- `summary.simulated_matches_integer == true`;
- `summary.real_fhe_all_match == true`;
- every `fhe/measurements.jsonl` row has `output_matches_clear=true`;
- repeated encryptions have different request ciphertext hashes;
- `circuits/receipt.json` contains zero server secret-key markers.

Local execution instantiates serialized client and server objects in one process to validate the API boundary. It is real encryption, but it is not physical machine separation.

## Level 3: Modal GPU and remote FHE evaluator

Confirm the selected authenticated workspace:

```bash
uv sync --extra cloud --extra fhe
uv run modal --version
uv run modal profile current
```

Run the bounded quick pipeline:

```bash
uv run modal run \
  -w artifacts/modal-evidence.json \
  modal_app.py::research \
  --run-id modal-smoke-reproduction
```

The entrypoint executes:

1. `train_teacher_gpu` on one NVIDIA L4;
2. `search_on_cpu` in a bounded CPU worker and named Volume;
3. `compile_finalist` in the Concrete x86 image;
4. **local** `fhe.Client` key generation and encryption;
5. remote `evaluate_ciphertext` with no client object or secret key;
6. local decryption and integer-clear comparison;
7. `persist_cloud_evidence` and `--write-result` export.

Do not place the FHE secret key in a Modal Secret. The project does not require any Modal Secret for the recorded pipeline.

Publish the exported record to the dashboard:

```bash
uv run unseen-loop report \
  artifacts/modal-evidence.json \
  --output site/data/evidence.json
python -m http.server 8000
# open http://127.0.0.1:8000/site/
```

## Full preregistered search

```bash
uv run unseen-loop research \
  --backend clear \
  --env-id CartPole-v1 \
  --seed-root release-2026-08 \
  --output artifacts/release-cartpole

uv run modal run \
  -w artifacts/modal-release.json \
  modal_app.py::research \
  --run-id release-cartpole-modal \
  --full
```

`--full` expands GPU training, evaluation seeds, degree/precision/ridge search, and encrypted trials. It is intentionally expensive. Before launching, inspect Modal workspace/environment budgets and the function limits in `modal_app.py`.

## Artifact schema

```text
<artifact>/
  provenance.json             software, hardware, command, git state
  seeds.json                  content-derived disjoint namespaces
  config.json                 immutable preset and candidate count
  teacher/checkpoint.json     clear teacher weights and training metadata
  teacher/training.jsonl      CEM iteration records when trained locally
  search/candidates.jsonl     every candidate, failure-free and Pareto label
  policies/<sha256>.json      immutable PolicySpec for every candidate
  certificates/heldout.json   occupancy coverage and mismatch counts
  certificates/box.json       exhaustive integer-domain receipt when feasible
  circuits/receipt.json       compiler/range/error/security receipt
  circuits/server.zip         architecture-specific FHE evaluator
  circuits/client-specs.bin   client crypto specifications, no secret key
  fhe/measurements.jsonl      one row per real ciphertext attempt
  summary.json                headline values and evidence label
  claims.json                 supported and unsupported claims
  checksums.sha256            completion marker and integrity ledger
```

Client keys are forbidden by path policy and `.gitignore`. Evaluation keys are not persisted by the experiment harness.

## Determinism contract

Expected bitwise deterministic:

- seed expansion;
- teacher checkpoint on matched software/hardware execution where operations are deterministic;
- calibration data and quantized integer inputs;
- policy specs and digests;
- integer-clear outputs;
- certificate rows and box input digest;
- summary aggregation.

Expected non-deterministic:

- encryption randomness and ciphertext hashes;
- keys;
- wall-clock timing;
- Modal call IDs, cold starts, and host assignment.

A second run should match semantic outputs and content-addressed deterministic artifacts; it must not match ciphertext bytes.

## Troubleshooting

### `pkg_resources` missing

Concrete-Python 2.10.0 imports `pkg_resources`. The `fhe` extra pins `setuptools==75.3.0`. Re-run:

```bash
uv sync --extra fhe
```

### Concrete unavailable

The system fails rather than substituting clear execution. Install the `fhe` extra on Python <3.13.

### `server.zip` fails to load

Recompile in the same architecture/image family as the evaluator. Concrete server artifacts are not portable from x86 to ARM or automatically between CPU and CUDA targets.

### Modal image/storage pressure

Torch and Concrete pull large native wheels. Keep Modal images separate. Locally, use `uv`'s default cache; on storage-constrained ephemeral workspaces a symlink link mode can avoid copying cached wheels:

```bash
uv sync --link-mode=symlink --extra dev --extra cloud --extra fhe
```

### Modal write-result error

`--write-result` accepts only strings or bytes. Use the `research` local entrypoint, which returns canonical JSON text, rather than invoking a dict-returning internal remote function directly.
