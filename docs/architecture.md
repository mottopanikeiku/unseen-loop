# Architecture

## Components

```text
GPU TRAINER                    CPU RESEARCH                     FHE DEPLOYMENT
┌─────────────────┐           ┌─────────────────────┐          ┌──────────────────────┐
│ Modal NVIDIA L4 │ checkpoint│ disjoint seed pools │ policy   │ compile_finalist      │
│ 2,048 policies  ├──────────►│ distill / refine    ├─────────►│ Concrete-Python 2.10  │
│ vectorized CEM  │           │ select, then evaluate│         │ server.zip + specs   │
└─────────────────┘           │ integer occupancy   │          └──────────┬───────────┘
                              └──────────┬──────────┘                     │
                                         │                                 │
                                         ▼                                 ▼
                              ┌─────────────────────┐          ┌──────────────────────┐
                              │ artifact ledger     │          │ local client          │
                              │ hashes / raw rows   │          │ keygen / encrypt      │
                              │ policy specs        │          │ decrypt / argmax      │
                              └─────────────────────┘          └──────────┬───────────┘
                                                                          │ ciphertext
                                                                          ▼
                                                               ┌──────────────────────┐
                                                               │ Modal evaluator      │
                                                               │ Server.load + run    │
                                                               │ no Client, no sk     │
                                                               └──────────────────────┘
```

## Source map

| Path | Responsibility |
|---|---|
| `specs.py` | immutable quantizer/policy/candidate schemas and stable policy digest |
| `policy.py` | polynomial feature map, weighted ridge fit, coefficient freezing, exact integer kernel |
| `certificate.py` | analytical per-state bounds, counterexample weights, exhaustive integer-box receipt |
| `teacher.py` | clear NumPy MLP teacher, Gym rollout contract, CPU CEM, trajectory collection |
| `gpu_teacher.py` | Torch-vectorized CartPole dynamics and GPU CEM population evaluation |
| `search.py` | certificate-guided student-occupancy refinement and selection-only Pareto filtering |
| `fhe_backend.py` | Concrete compiler, client/server serialization, simulation, real roundtrip, circuit receipt |
| `protocol.py` | versioned request/response envelopes, HMAC authentication, replay and fixed-shape guard |
| `artifacts.py` | atomic writes, secret-key path denial, checksums and verification |
| `experiment.py` | disjoint seed namespaces, post-selection evaluation, paired episode rows, evidence ladder and claims |
| `suite.py` | typed `release.toml` loader and multi-environment/checkpoint release-suite materialization |
| `cli.py` | smoke, research, release-suite, verification, inspection, and report commands |
| `modal_app.py` | bounded single-checkpoint GPU/CPU/FHE path with local-key entrypoint |
| `modal_studies.py` | bounded clear expanded-suite and four-cell factorial runners |
| `modal_fhe_studies.py` | colocated degree-2 complete-domain challenge and four-context timing runners |
| `modal_analysis.py` | single-container checksum verification, matched bootstrap analysis, evidence index, and publication tables |

## Semantic invariant

A `PolicySpec` is the only policy definition consumed by the execution backends. It freezes:

- environment and policy name;
- polynomial degree;
- action count;
- per-feature quantizer center, step, and signed code limit;
- clear fitted coefficients;
- integer coefficients;
- coefficient dequantization scale;
- schema version and content digest.

The integer-clear path and compiled FHE path both compute

```text
integer_coefficients @ polynomial_features(quantized_observation)
```

Both paths use the same stable lowest-index argmax on the client. Backend adapters may not retrain, recalibrate, change output shape, apply softmax, or move argmax.

## Search/evaluation isolation

`SeedPlan` derives mutually disjoint distillation, refinement, selection, and evaluation namespaces. Candidate policies are fitted/refined without selection or evaluation rows, compared under exact integer-student occupancy on selection seeds, and rejected from Pareto/champion eligibility if their reached observations saturate the declared quantizer range. Only after the champion is frozen does `run_experiment` recompute return, constraint cost, teacher agreement, and certificate coverage on evaluation seeds under that champion's occupancy. It writes paired `FLOAT TEACHER` and `QUANTIZED CLEAR` rows to `evaluation/episodes.jsonl`.

`unseen-loop suite` type-checks `experiments/release.toml` and instantiates all three environments × five checkpoints beneath a checksummed suite root. A clear-backend suite materializes the paired RL matrix but is not FHE privacy evidence and does not execute the separate stress, ablation, or repeated-container timing requirements. `unseen-loop research` and the Modal `research --full` entrypoint remain single-environment/checkpoint paths.

## Executed publication-study topology

The executed studies are separate trust/execution paths joined only by a checksumming analysis worker:

```text
expanded clear 3×5 ───────┐
four matched clear cells ─┼─► modal_analysis.py ─► publication.json + evidence-index.json
degree-2 colocated FHE ────┤                         (no endpoint; max_containers=1)
four-context colocated FHE ┘
```

`modal_analysis.py` first verifies every source ledger and exact planned/observed denominator. It then copies source-scoped FHE summaries, aggregates the 15-run clear environments, and computes matched 2×2 CartPole effects. The resulting [`../artifacts/studies/unseen-loop-release-analysis-004/publication.json`](../artifacts/studies/unseen-loop-release-analysis-004/publication.json) has SHA-256 `7a8c4ee7fd8f5d27778b94c98913b292b120172b136dcffe936b2591f5811536`; the enclosing ledger has SHA-256 `3dd9ac68c0e2db09449b228180707b3f459f094606cb1bd7953e9a2f3a70e823`.

| Path | Exact execution boundary | Evidence boundary |
|---|---|---|
| `expanded-multitask-modal-002` | one bounded Modal CPU suite worker; 15/15 clear runs | descriptive paired clear results only; no privacy |
| `expanded-cartpole-ablation-modal-004--…` | four matched clear suites; 5 checkpoints / 500 pairs each | CartPole factorial effects; causal scope limited to tested refinement bundle |
| `modal-nonlinear-qmax2-002` | one worker creates and consumes one client/server context; 40/40 `REAL FHE` calls | degree-2 circuit conformance/cost; no local/remote separation or efficacy |
| `modal-fhe-timing-003` | four workers, each with its own colocated context; 12 warmups + 64/64 measured successes | clustered latency/size distribution; no shared-context, service, throughput, or local/remote claim |

The positive +83.619 [26.144, 145.954] occupancy-refinement-bundle return effect and the negative −108.461 [−288.649, 68.250] weighting point estimate arise from matched clear CartPole evidence. The Acrobot expanded loss, −231.896 [−388.536, −75.831], is retained. None of those return claims crosses into the FHE studies; none of the FHE studies supplies evidence of task efficacy or generalization.

## Modal topology

Three images avoid a kitchen-sink trust and dependency boundary:

| Image | Major dependencies | Function |
|---|---|---|
| `gpu_image` | Torch 2.7.1, Gymnasium, NumPy | vectorized clear RL teacher training on fixed L4 |
| `core_image` | Gymnasium, NumPy | clear candidate search, artifact persistence |
| `fhe_image` | Concrete-Python 2.10.0, pinned Setuptools, NumPy | compilation and opaque ciphertext evaluation |

All pools set explicit CPU, memory, timeout, retry, warm-buffer, and maximum-container bounds. GPU training and ciphertext evaluation use `max_containers=1`; search and compiler pools are bounded at two. Serializing the evaluator makes its Volume-backed replay-ledger claim atomic across calls and container restarts. No function has a secret containing the FHE decryption key.

A named Volume stores canonical clear-search and Modal evidence under:

```text
/artifacts/runs/<run_id>/
  clear-search/
    provenance.json
    seeds.json
    config.json
    teacher/
    policies/
    search/candidates.jsonl
    certificates/
    summary.json
    claims.json
    checksums.sha256
  modal/
    evidence.json
    receipt.json
    server.zip
    client-specs.bin
    policy.json
    checksums.sha256
```

`checksums.sha256` covers the other five nonsecret files. `evidence.json` uses `unseen-loop/modal-evidence-v2`; its `authenticated_envelope_protocol` descriptor names HMAC-SHA256, request/response schema versions, and bound digest classes, while every REAL FHE row carries request/response envelope and policy/circuit/client/evaluation-key digests. The record also includes `closed_loop_real_fhe`, `same_input_canary`, `artifact_secret_marker_audit`, and `nonsecret_bundle`, but excludes plaintext private observations and decrypted score vectors. Client secret keys, encryption randomness, authentication keys, and evaluation-key payloads are never persisted. The producer commits the completed bundle, then verifies the persisted evidence byte length/SHA and exact file inventory before returning; Modal's `--write-result` separately exports the same canonical JSON text.

The replay ledger is separate at `/artifacts/protocol/replay-ledger.json`. It contains authenticated request digests and claim times only, never ciphertext payloads, plaintext observations, score vectors, or keys; entries older than ten minutes are pruned on the next claim.

## Client/server split

### Development/compile

1. Freeze the champion `PolicySpec`.
2. Build input-range corners and axis extrema.
3. Compile with category-128 defaults, unsafe features disabled, insecure key cache disabled, and `global_p_error=10^-6`.
4. Save `server.zip` and serialized client specifications.
5. Hash policy, MLIR representation, server artifact, specifications, calibration input set, and output range.

### Local client

1. Deserialize client specifications and generate secret and evaluation keys.
2. Quantize and reject an observation outside the compiled range before encryption.
3. Encrypt to `fhe.Value`; serialize the randomized ciphertext.
4. Create and HMAC-sign a fixed-shape request envelope binding request/nonce/time, policy, circuit, client-context and evaluation-key digests, payload shape, length, and digest.
5. Send the server artifact, signed request JSON, serialized evaluation keys, per-run authentication key, and circuit receipt to the evaluator function.
6. Verify the authenticated response envelope and its request/context binding.
7. Deserialize and decrypt integer scores. The research entrypoint requires exact equality with its local integer-clear vector before stable argmax and the environment transition; a deployment client must independently enforce the declared shape/range before actuation.

### Modal evaluator

1. Verify the HMAC-authenticated request and deserialize its envelope.
2. Recompute server-artifact and evaluation-key digests and apply the freshness/context/shape/length checks in `FixedShapeGuard`.
3. Atomically claim the authenticated request digest in `/artifacts/protocol/replay-ledger.json`, commit it, and reject a duplicate. Claims are retained for ten minutes, beyond the five-minute request-freshness window.
4. Only after that claim, write the architecture-specific `server.zip` to ephemeral disk, audit its filenames, and load `fhe.Server`.
5. Deserialize `fhe.Value` and `fhe.EvaluationKeys`, then call `server.run`.
6. Serialize the encrypted result in a response envelope bound to the request, HMAC-sign it, and return public timing, size, hash, mode, and protocol metadata.

There is no server-side fallback. Missing Concrete support raises `FHEUnavailableError`; it never calls integer clear under an FHE label.

## Failure policy

- out-of-domain observation: reject before encryption;
- wrong tensor shape: reject;
- stale/replayed/downgraded envelope: reject;
- compile failure: preserve failure record; do not drop candidate silently;
- simulation mismatch: fail run;
- real-FHE mismatch: fail run;
- checksum mismatch: fail verification;
- any detected secret-key marker in server artifact: fail privacy gate;
- network/server timeout: record failure; a retry never replaces the failed attempt in release metrics.

## Portability

Concrete `server.zip` is architecture-specific. Compile and evaluate on the same x86 image family. CUDA FHE compilation would produce a distinct artifact and parameter selection; it is not implied by GPU teacher training. The current FHE evaluator is CPU-only by design.
