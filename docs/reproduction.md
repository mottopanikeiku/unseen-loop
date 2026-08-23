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

This trains a clear CEM teacher, searches eight students, runs student-occupancy refinement, selects on eight dedicated selection episodes, and recomputes headline metrics on eight disjoint evaluation episodes under exact integer-student occupancy. It persists paired teacher/student episode rows, computes held-out and exhaustive certificates, and writes a checksum ledger. It makes no privacy claim.

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

Local execution instantiates serialized client and server objects in one process to validate the API boundary. It is real encryption, but it is not physical machine separation. The archive filename-marker scan is a fail-closed hygiene gate, not a proof that arbitrary archive bytes contain no secret.

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
4. **local** `fhe.Client` key generation, range checking, and encryption;
5. local construction and HMAC signing of a fixed-shape request envelope;
6. remote `evaluate_ciphertext`, which verifies the envelope/context/freshness, atomically claims the request digest in the ten-minute Volume replay ledger, then runs without a client object or FHE secret key;
7. local verification of the authenticated response, decryption, and integer-clear comparison;
8. 25 sequential encrypted control steps with the environment remaining local;
9. `persist_cloud_evidence` and `--write-result` export.

Do not place either the FHE secret key or the per-run HMAC authentication key in a Modal Secret. The project does not require any Modal Secret for the recorded pipeline. Only `REAL FHE` rows are privacy evidence.

Publish the exported record to the dashboard:

```bash
uv run unseen-loop report \
  artifacts/modal-evidence.json \
  --output site/data/evidence.json
python -m http.server 8000
# open http://127.0.0.1:8000/site/
```

## Release suite versus single-checkpoint scale-up

The committed measured smoke matrix is exactly reproduced with:

```bash
uv run unseen-loop suite \
  --config experiments/multitask-smoke.toml \
  --backend clear \
  --output artifacts/multitask-smoke-reproduction
```

It runs three environments × five checkpoints with eight selection and eight disjoint evaluation episodes per checkpoint. It retains 120 paired episodes / 240 long-form rows and is explicitly clear-only conformance evidence.


The preregistered release path is the typed suite command:

```bash
uv run unseen-loop suite \
  --config experiments/release.toml \
  --backend clear \
  --output artifacts/release
```

It consumes `unseen-loop/release-suite-v1` and materializes every declared workload: three environments × five independent checkpoints, 120 candidate rows, 100 selection episodes, and 100 disjoint evaluation episodes per checkpoint. Across 15 child runs this retains 1,800 candidate rows, 1,500 selection episodes, 1,500 paired evaluation episodes, and 3,000 long-form teacher/student episode rows. The root contains the copied `release.toml`, `suite-summary.json`, `suite-runs.jsonl`, paired `suite-episodes.jsonl`, and `checksums.sha256`; children live under `runs/<environment>--checkpoint-NN/`, and the root ledger transitively covers every child file. The command rejects a nonempty output directory rather than mixing stale and current runs. Because the shown command uses `--backend clear`, it makes no privacy claim and does not by itself execute the manifest's real-FHE challenge, stress, ablation, or repeated-container timing requirements.

By contrast, this command scales the single CartPole Modal path only:

```bash
uv run modal run \
  -w artifacts/modal-full-single-checkpoint.json \
  modal_app.py::research \
  --run-id cartpole-modal-full-single \
  --full
```

`--full` expands that one path's GPU population/iterations, candidate search, selection/evaluation episodes, and encrypted canaries. It does **not** read `experiments/release.toml`, run MountainCar or Acrobot, train five checkpoints per environment, or materialize release-suite aggregates. It is intentionally expensive; inspect Modal budgets and `modal_app.py` function limits before launch.

## Artifact schemas

The local experiment bundle is:

```text
<artifact>/
  provenance.json             software, hardware, command, git state
  seeds.json                  content-derived disjoint namespaces
  config.json                 immutable preset, including selection/evaluation counts
  teacher/checkpoint.json     clear teacher weights and training metadata
  teacher/training.jsonl      CEM iteration records when trained locally
  search/candidates.jsonl     every candidate, diagnostics, range validity and Pareto label
  policies/<sha256>.json      immutable PolicySpec for every candidate
  evaluation/episodes.jsonl   paired FLOAT TEACHER / QUANTIZED CLEAR rows per evaluation seed
  certificates/heldout.json   post-selection integer-student-occupancy certificate receipt
  certificates/box.json       exhaustive integer-domain receipt when feasible
  circuits/receipt.json       compiler/range/error/security receipt
  circuits/server.zip         architecture-specific FHE evaluator
  circuits/client-specs.bin   client crypto specifications, no secret key
  fhe/measurements.jsonl      one row per real ciphertext attempt
  summary.json                post-selection headline values and evidence label
  claims.json                 supported and unsupported claims
  checksums.sha256            completion marker and integrity ledger
```

The Modal privacy-evidence bundle at `/artifacts/runs/<run-id>/modal/` is deliberately smaller:

```text
evidence.json                 unseen-loop/modal-evidence-v2
receipt.json                  public circuit receipt
server.zip                    nonsecret evaluator artifact
client-specs.bin              nonsecret client specifications
policy.json                   selected public PolicySpec
checksums.sha256              SHA-256 ledger for the five files above
```

The v2 evidence includes `closed_loop_real_fhe`, `same_input_canary`, `artifact_secret_marker_audit`, and `nonsecret_bundle`. `authenticated_envelope_protocol` records the HMAC algorithm, request/response schemas, and binding classes; every REAL FHE row records request/response envelope plus policy/circuit/client/evaluation-key digests. Public timings, sizes, hashes, actions, and rewards remain auditable, while plaintext private observations and decrypted score vectors are excluded. Client keys, HMAC authentication keys, encryption randomness, and evaluation-key payloads are forbidden from persisted/cloud artifacts. The producer verifies the persisted evidence byte length/SHA and exact bundle file list before returning.

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
