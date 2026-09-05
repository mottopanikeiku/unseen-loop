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

The publication tables are bound to [`../artifacts/studies/unseen-loop-release-analysis-004/publication.json`](../artifacts/studies/unseen-loop-release-analysis-004/publication.json), SHA-256 `7a8c4ee7fd8f5d27778b94c98913b292b120172b136dcffe936b2591f5811536`, and its enclosing ledger, SHA-256 `3dd9ac68c0e2db09449b228180707b3f459f094606cb1bd7953e9a2f3a70e823`. The following are the exact canonical invocations. They require an authenticated workspace whose `unseen-loop-artifacts` Volume does not already contain these IDs; every runner rejects a nonempty destination rather than overwrite evidence.

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

`modal_analysis.py` deliberately binds the seven canonical IDs and refuses to overwrite `unseen-loop-release-analysis-004`. In an already populated workspace, use fresh study IDs for a semantic rerun and keep it separate; do not delete or replace the canonical evidence. The canonical analysis is a bounded publication analysis, not completion of the full preregistration.

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
  studies/unseen-loop-release-analysis-004 \
  artifacts/downloaded-studies/unseen-loop-release-analysis-004
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
(cd artifacts/downloaded-studies/unseen-loop-release-analysis-004 && sha256sum --check checksums.sha256)

printf '%s  %s\n' \
  7a8c4ee7fd8f5d27778b94c98913b292b120172b136dcffe936b2591f5811536 \
  artifacts/downloaded-studies/unseen-loop-release-analysis-004/publication.json \
  | sha256sum --check -
printf '%s  %s\n' \
  3dd9ac68c0e2db09449b228180707b3f459f094606cb1bd7953e9a2f3a70e823 \
  artifacts/downloaded-studies/unseen-loop-release-analysis-004/checksums.sha256 \
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

## Integrated flagship Modal DAG

All empirical shield, OPE, integration, timing, analysis, and publication work runs on Modal. Local entrypoints only validate/submit plans and copy closed evidence.

Inspect the immutable job plan:

```bash
uv run modal run modal_flagship.py::inspect_plan \
  --config experiments/flagship-smoke.toml
```

Run the three cryptographic canaries independently. Distinct study IDs are immutable Volume destinations:

```bash
uv run modal run -w artifacts/shield-canary.json \
  modal_flagship_canary.py::shield_canary --study-id my-shield-canary
uv run modal run -w artifacts/exact-ope-canary.json \
  modal_flagship_canary.py::exact_ope_canary --study-id my-exact-ope-canary
uv run modal run -w artifacts/ckks-ope-canary.json \
  modal_flagship_canary.py::ckks_ope_canary --study-id my-ckks-ope-canary
```

The shield canary compiles once, compares all `5^6 = 15,625` declared inputs against Concrete simulation, then performs one serialized REAL-FHE roundtrip. The exact OPE canary uses the bounded `(N=1,H=2,D=1)` proof shape because the former `(4,4,6)` graph exceeded a 32 GiB Modal worker; this proof is not a scale claim. The CKKS canary uses `(N=64,H=8,D=1)`, one logical 64-slot vector per ciphertext, and a tc128-compatible degree-16384 modulus chain.

`modal_flagship.py::launch` requires a lowercase run ID, a SHA-256 of the submitted source snapshot, three pinned image/build-spec digests, and an exact executor-module map. This prevents an orchestrator from silently selecting an implementation:

```bash
SOURCE_DIGEST="$(git archive --format=tar HEAD | sha256sum | cut -d' ' -f1)"
IMAGE_DIGESTS='{"core":"f2b43cb57dcdda520bec33425f1a93b3aa4609a7d3faa4109a8518ef9eeec71e","fhe":"7f2fbfb9df06adf0709137d5467a48dbfd3396680754fca9048f1eddd83ced62","integration":"bd3eac7cc613ab60e5f949cd1bd5cd75da04b749513a672225100818fd3821ff"}'
EXECUTORS='{"clear_shield_matrix":"unseen_loop.flagship.executor_clear_shield","shield_fhe_challenge":"unseen_loop.flagship.executor_shield_fhe","ope_validation":"unseen_loop.flagship.executor_ope","integration":"unseen_loop.flagship.executor_integration","timing":"unseen_loop.flagship.executor_timing","analysis":"unseen_loop.flagship.executor_analysis"}'
uv run modal run modal_flagship.py::launch \
  --config experiments/flagship-smoke.toml \
  --run-id my-flagship-smoke \
  --source-digest "$SOURCE_DIGEST" \
  --image-digests-json "$IMAGE_DIGESTS" \
  --executor-modules-json "$EXECUTORS"
```

The append-only registry stops the historical DAG after any stage that does not reach its preregistered terminal state. `finalize_evidence` rejects incomplete jobs, unexpected extra files, checksum mismatches, and conflicting repeated closure; an identical existing index is fully recomputed and byte-verified. The historical finalizer wrapper itself remains single-attempt.

Build a browser publication only from completed canaries:

```bash
uv run modal run -w artifacts/flagship-publication.json \
  modal_flagship_publication.py \
  --publication-id my-flagship-publication \
  --shield-canary-id my-shield-canary \
  --exact-ope-canary-id my-exact-ope-canary \
  --ckks-ope-canary-id my-ckks-ope-canary \
  --smoke-run-id my-closed-flagship-smoke \
  --positive-study-id my-qualified-positive-recovery
uv run modal volume get unseen-loop-flagship-evidence \
  publications/my-flagship-publication/flagship-evidence.json \
  site/data/flagship-evidence.json
```

Pin the downloaded publication byte SHA-256 in `data-flagship-sha256` on `index.html`, `control-room.html`, and `ope.html`. The browser removes the complete measurement surface if that digest, schema, certificate arithmetic, shield selection replay, OPE denominators/division, security receipt, or trust label fails.

### Executed bounded smoke

The closed smoke run is `flagship-smoke-20260902-023`. Its source snapshot digest is `0c6cd923ee26b2b6bec7c6e87e49c011770cb47118d91e75d64e89133aee2df3`, manifest digest is `4f570be4e716a93bc878d4fbbdcc339fa7d6afbb995912bd1861f0c029aa90b5`, registry digest is `7b2ff845bb90e8d1291788a948e7857357172c87d4451e1522743d9922f1c6cd`, and root evidence-index digest is `fca26b7ec0d672219b9abcb55ef576946589c725a5267b71eba21e0b0c8741f5`.

The compact committed closure is under [`../artifacts/flagship-smoke-20260902-023/`](../artifacts/flagship-smoke-20260902-023/). Verify it without recomputation:

```bash
(cd artifacts/flagship-smoke-20260902-023 && sha256sum --check checksums.sha256)
```

The full 3,278 upstream job artifacts and append-only registry remain on the `unseen-loop-flagship-evidence` Modal Volume. Copy them when a complete row-level audit is required:

```bash
uv run modal volume get unseen-loop-flagship-evidence \
  flagship-smoke-20260902-023/ \
  artifacts/downloaded-flagship-smoke
```

The smoke closed every planned terminal status but did not pass the scientific release gates. This is intentional negative evidence, not a release qualification. The complete `experiments/flagship.toml` plan has 328,622 jobs and remains unexecuted.

### Qualified positive recovery

The post-baseline recovery used [`../experiments/positive-pilot.toml`](../experiments/positive-pilot.toml) for the OPE and CKKS hypotheses and [`../experiments/shield-consensus-pilot.toml`](../experiments/shield-consensus-pilot.toml) for the independent replicated shield confirmation. Both files were committed before their named Modal calls.

The closed result is `positive-recovery-20260903-001`, summary SHA-256 `574655d23e68d48d0c71bc7226a29cf6920ee63e00e79b905784061a6b93dddd`. Verify the committed compact bundle:

```bash
(cd artifacts/positive-recovery-20260903-001 && sha256sum --check checksums.sha256)
```

The bundle records Modal function-call identities, both configuration hashes, individual track results, all frozen gates, qualified claims, and excluded claims. Re-run the full three-track pilot:

```bash
uv run modal deploy modal_positive_pilot.py
uv run modal run -w artifacts/positive-recovery-result.json \
  modal_positive_pilot.py::run \
  --config experiments/positive-pilot.toml \
  --shield-config experiments/shield-consensus-pilot.toml \
  --study-id my-positive-recovery
```

The entrypoint runs OPE/CKKS under the base configuration and shield consensus under its independent confirmation configuration, then `finalize_recovery` binds both configuration digests and all three Modal function-call identities. Do not substitute the exploratory shield row from the first pilot.

### Failed independent confirmation001

The subsequent fixed holdout `independent-confirmation-20260904-001` failed.
Its six original files are archived byte-for-byte under
[`../artifacts/independent-confirmation-20260904-001/`](../artifacts/independent-confirmation-20260904-001/).
The original summary SHA-256 is `9233207bce45cdc8cd95fa026f2026571715185d055a0bcd83655d33c87d6b17`;
the original ledger SHA-256 is `4227d8e154979d49031547a32c026d63f0feeebb7e79cef3e04607114f1f6885`.

```bash
(cd artifacts/independent-confirmation-20260904-001 && sha256sum --check checksums.sha256 && sha256sum --check diagnostic-checksums.sha256)
```

Clear OPE passed its component gates on 100 N4096/H8 batches: normalized bias
0.0003522328, RMSE 0.0115734561, coverage 92/100, and minimum ESS fraction
0.9350948. CKKS completed only 6/10 contexts (indices 0, 1, 4, 5, 6, 8);
its completed-context maximum numerator/denominator errors were 1.31039988
and 28.76692046. Shield completed 150/150 calls with action agreement 50/50,
certification agreement 49/50, and exact margins 5555/6000. The original
`all_tracks_passed`, `independently_confirmed_positive_result`, and
`scale_up_allowed` booleans remain false.

The separate `diagnostic-receipt.json` binds the originals and a sanitized
read-only Modal log excerpt. Four logged worker exceptions are
`ValueError("decrypted CKKS count is outside the accepted precision tolerance")`
at `ope/ckks.py:611`, called from `modal_independent_confirmation.py:298`.
Neither failed raw count values nor replica-to-container mapping is retained.
These logs establish count-validation failures, not their lower-level cause
and not OOM or preemption. Inspection of the replay source at lines 508–509
and 555–556 explains the empty failure-code lists: each `starmap` generator
is traversed once for successes, then again after exhaustion for failures.
Generated original results are not corrected retroactively.

The 100 clear batch rows and authenticated historical source/image identity
were not retained; aggregate replay is not row-level replay. State 30's
certification mismatch is conservative relative to the discrete reference
by inference from the inspected comparison logic, not a general false-safe
guarantee. `modal_independent_confirmation.py` and its TOML remain unchanged
historical replay assets; do not invoke them for another holdout.

### Source-bound private policy comparison: not yet run

The new source implementation is `UNCLIPPED_RATIO_LIFT_WPDIS_V1`, not the
historical soft-clipped estimator. The honest client validates logged-action
ratios for both frozen policies, forms policy-independent action/feature/behavior
ratio lifts, and encrypts one reusable request. The server uses a known inclusive
prefix network and returns 2H encrypted mean statistics per policy. Counts come
from the public fixed shape; discounting and division remain client-side. Request
digests bind the full ciphertext bytes and chunk cover, not just the shape.
This is an honest-client/honest-but-curious protocol, not malicious-server
integrity, differential privacy, private training, or a new prefix-scan algorithm.

The H64 profile is degree 16384, scale `2**40`, and `(60, *([40]*8), 58)`:
eight multiplication levels and 438 total modulus bits. Static range receipts
bound coefficient accumulation, prefix nodes, normalized products and slot
reduction before and after rescaling. Analysis reconstructs the canonical receipt
from the frozen specification and recorded actual modulus integers and verifies
its digest; an opaque checksum alone cannot pass numerical admission. These are
supported-domain obligations, not a global CKKS accuracy certificate or evidence
of actual SEAL acceptance. The preserved V1 count path is diagnosed separately.

The fixed queue compares ordered policies A and B on identical data. Clear
references include IS, PDIS, WPDIS, hard clips 2 and 10, fitted-model DM,
full sequential DR, globally normalized cross-fitted WDR, and state-action MIS.
The client bootstrap shares whole-trajectory draws between policies and reports
raw and discounted-unit-bound-normalized estimates, endpoints and standard
errors. Confirmation decisions use the client interval translated around the
ciphertext contrast; this is not an encrypted bootstrap.

| Frozen phase | Planned jobs | Empirical scope | Current status |
|---|---:|---|---|
| Diagnostic D | 16 | verification, 12 count contexts, two infrastructure probes, analysis | Not run |
| Pilot P | 156 | 128 clear batches, 27 contexts / 53 policy evaluations, analysis | Not run |
| Confirmation C | 241 | 240 contexts / 480 policy evaluations, analysis | Not run |

`private_ope_capture_sources` captures the exact committed source into the
ignored `.private-ope-image/` snapshot and `.private-ope-source.json` bundle.
Image inputs come from the committed snapshot, not live worktree directories;
runtime verification rejects incomplete manifests and extra packaged files.
The build explicitly selects Modal builder `2025.06` through
`MODAL_IMAGE_BUILDER_VERSION`. Every manifest binds four complete code manifests,
the lockfile, image specification and an immutable deployed version.
Config-only commits do not become the image's code commit. The local entrypoints
are `private_ope_inspect`, `private_ope_launch`, and `private_ope_close`; launch
and close resolve the frozen deployed Function rather than call an ephemeral
`modal run` worker. Configuration files are frozen only after those identities
and their predecessor evidence actually exist; do not fill them with invented
digests or deployment versions.

The singleton coordinator reserves each seed namespace before initialization,
commits an initialization marker before dispatch, and records dispatch intent
before spawn. Workers claim a slot before backend imports and random draws,
write only disjoint transport receipts, and never update the canonical registry.
Recovery cancels known calls and retains every fixed slot, including unknown
dispatch windows and unattempted jobs, without replacement scientific attempts.
Rescheduling the originating coordinator input enters recovery, while a separate
duplicate launch returns the reservation without computation. A rescheduled
worker reuses its immutable entry rather than acquiring another scientific
attempt. The 60-second artifact-visibility grace never extends a computation
deadline or recovery completion cutoff. Budget and wave capacity are checked
before each worker spawn, and each new wave must fit entirely before cycle reset.
The checksum ledger excludes itself and the index; `evidence-index.json` is the
sole closure marker. A validated failed-science artifact is registry success,
not scientific promotion.

Both timing arms use the same canonical length-prefixed binary framing and
retain their exact request/response wire payloads. This avoids charging only the
raw baseline for base64 expansion or redundant framing work.

Before any image build or dispatch, an Owner or Manager must verify a hard
workspace **usage** cap covering physical rescheduling, startup, builds and
storage. A spend limit after credits is not sufficient. Remaining authorized
allowances are at most $10 for D, $30 for P, and $100 for C; the complete
projected stage plus overhead must fit. The guard is checksum-bound at
`private-ope-budget-guards/<receipt-sha256>.json` on the evidence Volume, with
the authenticated observation artifact retained separately by its digest.
No unattended billing-cycle reset, parameter search, backend switch or failed
holdout replacement is authorized.

Local verification excludes real cryptographic execution:

```bash
uv run pytest tests/test_ope_estimators.py tests/test_ope_ckks.py tests/test_flagship_registry.py tests/test_ratio_lift_wpdis.py -m 'not fhe and not slow and not modal'
uv run pytest -m 'not fhe and not slow and not modal'
```

The source also includes focused manifest, study, executor and crash-accounting
regressions. Actual prime/level acceptance, request reuse precision, full
8192/8193 chunk execution, and the operational closure path are reserved for the
Modal D verification slot. Passing local tests does not promote D, P or C.
The current public v1 publication and manuscript results remain historical;
they have not been replaced by a provisional private-OPE result.

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
