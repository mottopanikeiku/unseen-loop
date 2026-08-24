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

## Executed Modal publication studies

The publication tables are bound to [`../artifacts/studies/unseen-loop-release-analysis-003/publication.json`](../artifacts/studies/unseen-loop-release-analysis-003/publication.json), SHA-256 `4a38c55363a7c442c9322a7d12b49e8761cb3813746dca66ba9d1fb12ba94aa3`, and its enclosing ledger, SHA-256 `ccafb13012ff678555c7de6370f79147412661d693b3327c44fbffa967f20fcf`. The following are the exact canonical invocations. They require an authenticated workspace whose `unseen-loop-artifacts` Volume does not already contain these IDs; every runner rejects a nonempty destination rather than overwrite evidence.

```bash
uv sync --extra cloud --extra fhe
uv run modal profile current

uv run modal run -w artifacts/expanded-modal-summary.json \
  modal_studies.py::suite \
  --config experiments/expanded-multitask.toml \
  --study-id expanded-multitask-modal-002

uv run modal run -w artifacts/ablation-modal-summary.json \
  modal_studies.py::ablations \
  --config-directory experiments \
  --study-id expanded-cartpole-ablation-modal-004

uv run modal run -w artifacts/nonlinear-modal-summary.json \
  modal_fhe_studies.py::nonlinear_challenge \
  --study-id modal-nonlinear-qmax2-002

uv run modal run -w artifacts/timing-modal-summary.json \
  modal_fhe_studies.py::timing_study \
  --study-id modal-fhe-timing-003

uv run modal run -w artifacts/analysis-modal-summary.json \
  modal_analysis.py::main
```

The expanded run executes 3 environments × 5 checkpoints × 8 candidates × 50 selection episodes, then 100 disjoint paired evaluation episodes per checkpoint: 15/15 runs, 120 candidates, 6,000 selection rows, 1,500 paired rows, and 3,000 long-form teacher/student rows. The ablation command runs all four matched CartPole cells. The nonlinear command executes 25 complete-domain + 15 canary calls. The timing command executes four independent contexts, each with three excluded warmups and 16 measured requests.

`modal_analysis.py` deliberately binds the seven canonical IDs and refuses to overwrite `unseen-loop-release-analysis-003`. In an already populated workspace, use fresh study IDs for a semantic rerun and keep it separate; do not delete or replace the canonical evidence. The canonical analysis is a bounded publication analysis, not completion of the full preregistration.

### Download the canonical Volume evidence

```bash
mkdir -p artifacts/downloaded-studies
uv run modal volume get unseen-loop-artifacts \
  studies/expanded-multitask-modal-002 \
  artifacts/downloaded-studies/expanded-multitask-modal-002
uv run modal volume get unseen-loop-artifacts \
  studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-refined \
  artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-refined
uv run modal volume get unseen-loop-artifacts \
  studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-unrefined \
  artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-unrefined
uv run modal volume get unseen-loop-artifacts \
  studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-refined \
  artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-refined
uv run modal volume get unseen-loop-artifacts \
  studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-unrefined \
  artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-unrefined
uv run modal volume get unseen-loop-artifacts \
  studies/modal-nonlinear-qmax2-002 \
  artifacts/downloaded-studies/modal-nonlinear-qmax2-002
uv run modal volume get unseen-loop-artifacts \
  studies/modal-fhe-timing-003 \
  artifacts/downloaded-studies/modal-fhe-timing-003
uv run modal volume get unseen-loop-artifacts \
  studies/unseen-loop-release-analysis-003 \
  artifacts/downloaded-studies/unseen-loop-release-analysis-003
```

### Verify every ledger and publication binding

```bash
(cd artifacts/downloaded-studies/expanded-multitask-modal-002/suite && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-refined/suite && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-unrefined/suite && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-refined/suite && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-unrefined/suite && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/modal-nonlinear-qmax2-002 && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/modal-fhe-timing-003 && sha256sum --check checksums.sha256)
(cd artifacts/downloaded-studies/unseen-loop-release-analysis-003 && sha256sum --check checksums.sha256)

printf '%s  %s\n' \
  4a38c55363a7c442c9322a7d12b49e8761cb3813746dca66ba9d1fb12ba94aa3 \
  artifacts/downloaded-studies/unseen-loop-release-analysis-003/publication.json \
  | sha256sum --check -
printf '%s  %s\n' \
  ccafb13012ff678555c7de6370f79147412661d693b3327c44fbffa967f20fcf \
  artifacts/downloaded-studies/unseen-loop-release-analysis-003/checksums.sha256 \
  | sha256sum --check -
```

The suite root ledgers transitively enumerate every child file. `evidence-index.json` additionally binds the exact planned/observed denominators, config/source/ledger digests, failed-gate counts, and trust labels used in the paper.

## Expanded study versus the full preregistration

The expanded study uses eight candidates per environment/checkpoint and 50 selection episodes per candidate (120 candidates / 6,000 selection rows total). The full [`../experiments/release.toml`](../experiments/release.toml) search specifies 120 candidates per environment/checkpoint and 100 selection episodes per candidate (1,800 candidates / 180,000 selection rows total). Both use 100 disjoint paired evaluation episodes per checkpoint. The expanded result does not complete the preregistered 64-row physically remote client/server challenge or the release-wide stress/range/tie gates. Clear expanded and ablation studies make no privacy claim; the nonlinear and timing workers colocate client and server, so they make no local-client/remote-server secrecy claim.

The typed full clear matrix command remains:

```bash
uv run unseen-loop suite \
  --config experiments/release.toml \
  --backend clear \
  --output artifacts/release
```

It materializes clear RL rows only. It does not execute FHE, provide privacy evidence, or discharge the full preregistration by itself. `modal_app.py::research --full` still expands just one CartPole checkpoint and is not the release suite.

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
