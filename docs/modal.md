# Modal Operations and Cost Controls

## Why three functions and a local entrypoint

- **GPU teacher:** clear RL training is massively parallel and benefits from an L4.
- **CPU search:** Gym rollouts and small ridge solves are CPU workloads; using a GPU would waste credit.
- **CPU FHE compiler/evaluator:** the recorded Concrete target is x86 CPU. GPU teacher hardware does not imply GPU FHE.
- **Local entrypoint:** owns the secret/decryption key. A Modal Secret would place it in the remote trust domain and invalidate the architecture claim.

## Current resource caps

| Function | Accelerator/CPU | Memory | Max containers | Timeout | Retries |
|---|---:|---:|---:|---:|---:|
| `train_teacher_gpu` | 1 × L4, 4 cores | 16 GiB | 1 | 1 h | 0 |
| `search_on_cpu` | 8 cores | 16 GiB | 2 | 6 h | 0 |
| `compile_finalist` | 16 cores | 32 GiB | 2 | 6 h | 0 |
| `evaluate_ciphertext` | 8 cores | 16 GiB | 1 | 1 h | 0 |
| `persist_cloud_evidence` | 0.25 core | 512 MiB | 1 | 5 min | 0 |

All use `min_containers=0` and `buffer_containers=0`, so idle benchmark capacity scales to zero. Exact CPU/memory requests equal limits to reduce hidden resource variance. `evaluate_ciphertext` is deliberately serialized at one container so its Volume-backed replay-ledger claim remains atomic.

The publication-study runners use separate bounded functions:

| Function | CPU | Memory | Max containers | Timeout | Retries |
|---|---:|---:|---:|---:|---:|
| expanded / one ablation suite worker | 8 | 16 GiB | 4 across mapped cells | 6 h | 0 |
| nonlinear challenge worker | 16 | 32 GiB | 1 | 6 h | 0 |
| timing context worker | 16 | 32 GiB | exactly 4 | 6 h | 0 |
| timing aggregator | 4 | 8 GiB | 1 | 12 h | 0 |
| release-analysis worker | 4 | 8 GiB | 1 | 3 h | 0 |

All publication-study destinations are append-once by study ID; a nonempty destination is an error.

## Smoke workflow

```bash
uv run modal profile current
uv run modal run -w artifacts/modal-evidence.json \
  modal_app.py::research --run-id modal-smoke-$(date -u +%Y%m%d)
```

The command is synchronous and bounded. It runs one CartPole checkpoint, two randomized canaries, and one 25-step encrypted control prefix. It is a smoke/conformance path, not the three-environment release suite. The result is persisted as a nonsecret checksummed bundle on the named Volume; `--write-result` exports the canonical v2 JSON locally.

Adding `--full` expands only this single-checkpoint Modal path. It does not consume [`../experiments/release.toml`](../experiments/release.toml) or materialize the three-environment/five-checkpoint release matrix. The release orchestrator is `uv run unseen-loop suite --config experiments/release.toml --backend clear --output artifacts/release`; see the [reproduction guide](reproduction.md#expanded-study-versus-the-full-preregistration).

## Publication-study workflow

The exact canonical execution sequence is:

```bash
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
uv run modal run -w artifacts/analysis-modal-summary.json modal_analysis.py::main
```

These IDs are canonical and cannot be reused in the same Volume. The [reproduction guide](reproduction.md#executed-modal-publication-studies) gives exact download and `sha256sum --check` commands. The final `publication.json` digest is `4a38c55363a7c442c9322a7d12b49e8761cb3813746dca66ba9d1fb12ba94aa3`; its enclosing ledger digest is `ccafb13012ff678555c7de6370f79147412661d693b3327c44fbffa967f20fcf`.

The expanded and factorial runners are clear and have no privacy claim. The nonlinear runner generates, uses, and destroys one client secret inside a single colocated worker. Each timing worker independently generates and consumes its own colocated client/server context; the aggregator receives no client material. This topology exercises `REAL FHE`, but it is not the local-client/remote-server topology of `modal_app.py::research`.

## Inspect artifacts

```bash
uv run modal volume ls unseen-loop-artifacts runs
uv run modal volume get unseen-loop-artifacts \
  runs/<run-id>/modal/evidence.json \
  artifacts/<run-id>-evidence.json
uv run modal volume get unseen-loop-artifacts \
  runs/<run-id>/modal/checksums.sha256 \
  artifacts/<run-id>-checksums.sha256
```

`FunctionCall` results are temporary; the Volume bundle is canonical. It contains `evidence.json`, `receipt.json`, `server.zip`, `client-specs.bin`, `policy.json`, and `checksums.sha256`; the ledger covers the other five files. `unseen-loop/modal-evidence-v2` includes `closed_loop_real_fhe`, `same_input_canary`, `artifact_secret_marker_audit`, `nonsecret_bundle`, a top-level `authenticated_envelope_protocol` descriptor, and per-call request/response envelope and context digests. It contains no plaintext private observation or decrypted score vector.

## Spend controls

Before a scaled single-checkpoint Modal run:

1. configure a Workspace usage budget;
2. configure an Environment compute budget if available;
3. inspect the `--full` population, iteration, selection/evaluation seed, and trial counts;
4. confirm all `max_containers`, timeouts, and `retries=0` values;
5. run the quick path first;
6. stop a crash-looping deployed app immediately.

Billing report:

```bash
uv run modal billing report --for today --show-resources --json
```

Emergency stop for a deployed app:

```bash
uv run modal app stop -y unseen-loop
```

The repository does not deploy a persistent public web FHE endpoint. The static dashboard replays a recorded result, which avoids an unauthenticated credit sink and avoids sending browser plaintext to Modal.

## Image discipline

- Python 3.12 for every image;
- pinned NumPy and Gymnasium;
- pinned Torch/CUDA wheel in GPU image;
- pinned Concrete-Python and Setuptools in FHE image;
- project source added after dependency layers;
- no FHE key files baked into an image;
- compiler and evaluator share the same FHE image family because `server.zip` is architecture-specific.

The images intentionally duplicate small core dependencies. Combining CUDA and Concrete into one image would increase cold build/pull size, mix trust boundaries, and make dependency conflicts harder to audit.

## Measured publication-study results and rules

| Study | Exact denominator | Server evaluation | End-to-end | Valid scope |
|---|---:|---:|---:|---|
| `modal-nonlinear-qmax2-002` | 25 exhaustive domain + 15 canary = 40/40 matching calls | p50 362.091 ms; p95 374.514 ms | p50 366.597 ms; p95 378.941 ms | degree-2 `qmax=2` circuit conformance in one colocated context |
| `modal-fhe-timing-003` | 4 contexts; 12 excluded warmups; 64/64 measured successes | p50 544.536 ms; p95 830.709 ms | p50 550.076 ms; p95 837.010 ms | clustered distribution across four independent colocated contexts |

The timing study satisfies the preregistered warm denominator: four containers, three recorded/excluded warmups and 16 measured requests each. It uses 2,000 hierarchical container/request bootstrap repetitions and does not report p99 from 64 measurements. Every failure would remain in the denominator; the observed record has none.

Neither table row is throughput, steady-state production-service latency, “real-time” evidence, or a shared cryptographic-context result. Neither proves `global_p_error`. The workers colocate client and server, so they do not demonstrate local-client/remote-server secrecy. The separate expanded and factorial return studies are clear and supply no privacy evidence.

## Key-separation and envelope audit

The remote evaluator never constructs `fhe.Client`. Its implemented signature is:

```python
def evaluate_ciphertext(
    server_artifact: bytes,
    signed_request_json: str,
    evaluation_keys: bytes,
    authentication_key: bytes,
    receipt: dict[str, Any],
) -> dict[str, Any]:
```

The evaluator verifies the HMAC-authenticated `RequestEnvelope`, recomputes the server/evaluation-key digests, applies `FixedShapeGuard`, atomically claims the authenticated request digest in the shared Volume replay ledger, and commits that claim before deserializing any FHE object. Claims persist for ten minutes, exceeding the five-minute freshness window, so replay is rejected across RPCs and evaluator container restarts. It then runs `fhe.Server` and returns an authenticated `ResponseEnvelope`. The HMAC key is operational transport-authentication material, not a decryption key; an evaluator that holds it can still authenticate an incorrect result. No function argument carries the FHE secret key. Neither client keys nor authentication keys are written to a Volume, local evidence record, environment variable, or Modal Secret. The evidence records `secret_key_sent_to_modal=false`, and the server archive is scanned for secret-key filename markers; that filename scan is not a proof of arbitrary-byte secret absence.


## Deployment alternative

If a durable service is later required, expose only authenticated job submission/polling using `requires_proxy_auth=True`. Do not wait for compilation in an HTTP request, and do not expose the evaluator directly without strict payload, context, quota, and replay controls. Such a service is a separate deployment/security review, not implied by the research functions.
