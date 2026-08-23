# Architecture

## Components

```text
GPU TRAINER                    CPU RESEARCH                     FHE DEPLOYMENT
┌─────────────────┐           ┌─────────────────────┐          ┌──────────────────────┐
│ Modal NVIDIA L4 │ checkpoint│ teacher rollouts    │ policy   │ compile_finalist      │
│ 2,048 policies  ├──────────►│ weighted ridge      ├─────────►│ Concrete-Python 2.10  │
│ vectorized CEM  │           │ certificate / CEGIS │          │ server.zip + specs   │
└─────────────────┘           │ Pareto selection    │          └──────────┬───────────┘
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
| `search.py` | certificate-guided student-occupancy refinement and Pareto filtering |
| `fhe_backend.py` | Concrete compiler, client/server serialization, simulation, real roundtrip, circuit receipt |
| `protocol.py` | versioned request/response envelopes, HMAC authentication, replay and fixed-shape guard |
| `artifacts.py` | atomic writes, secret-key path denial, checksums and verification |
| `experiment.py` | seed namespaces, end-to-end experiment, evidence ladder and claims record |
| `cli.py` | quick/full experiment, verification, inspection and report publication commands |
| `modal_app.py` | bounded GPU/CPU/FHE functions and local-key research entrypoint |

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

with the same stable lowest-index argmax on the client. Backend adapters may not retrain, recalibrate, change output shape, apply softmax, or move argmax.

## Modal topology

Three images avoid a kitchen-sink trust and dependency boundary:

| Image | Major dependencies | Function |
|---|---|---|
| `gpu_image` | Torch 2.7.1, Gymnasium, NumPy | vectorized clear RL teacher training on fixed L4 |
| `core_image` | Gymnasium, NumPy | clear candidate search, artifact persistence |
| `fhe_image` | Concrete-Python 2.10.0, pinned Setuptools, NumPy | compilation and opaque ciphertext evaluation |

All pools set explicit CPU, memory, timeout, retry, warm-buffer, and maximum-container bounds. The GPU trainer has `max_containers=1`; compiler and evaluator pools are bounded at two and four. No function has a secret containing the FHE decryption key.

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
    server.zip
```

Each producer commits before returning. The local artifact returned by `--write-result` is also committed under `artifacts/reference/` for long-term reproducibility because Modal call results are temporary.

## Client/server split

### Development/compile

1. Freeze the champion `PolicySpec`.
2. Build input-range corners and axis extrema.
3. Compile with category-128 defaults, unsafe features disabled, insecure key cache disabled, and `global_p_error=10^-6`.
4. Save `server.zip` and serialized client specifications.
5. Hash policy, MLIR representation, server artifact, specifications, calibration input set, and output range.

### Local client

1. Deserialize client specifications.
2. Generate secret and evaluation keys.
3. Quantize and range-check one observation.
4. Encrypt to `fhe.Value`; serialize ciphertext.
5. Send `server.zip`, ciphertext, and serialized evaluation keys to the evaluator function.
6. Deserialize encrypted response, decrypt integer scores, validate shape/range, stable-argmax, and advance the environment.

### Modal evaluator

1. Write the received architecture-specific `server.zip` to ephemeral disk.
2. Load `fhe.Server`.
3. Deserialize `fhe.Value` and `fhe.EvaluationKeys`.
4. Call `server.run`.
5. Serialize encrypted response.
6. Return ciphertext plus public timing/size/hash metadata.

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
