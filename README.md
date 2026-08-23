# Unseen Loop

**The cloud acts on state it cannot read.**

Unseen Loop is a reproducible research system for **encrypted closed-loop policy serving**. It distills reinforcement-learning teachers into low-degree integer policies, searches the return–certificate–circuit-cost frontier, and evaluates the selected policy on client-encrypted observations with Fully Homomorphic Encryption (FHE).

> **Scope:** inference only. Training data, teacher execution, and policy compilation are cleartext development activities. The deployed evaluator receives ciphertext observations and public evaluation material; the secret key remains with the client. Only results explicitly labeled `REAL FHE` are privacy evidence. Simulation is never reported as encrypted execution.

## Research thesis

Ordinary FHE inference benchmarks stop at per-example accuracy. A control policy is sequential: one changed action changes every later state. Unseen Loop therefore optimizes and measures four coupled properties:

1. **post-selection closed-loop return and constraint cost** on held-out seeds under exact integer-student occupancy;
2. **action agreement and margin**, not only score error;
3. **certified action invariance** under the deployed fixed-point circuit;
4. **measured FHE systems cost**: compile, key generation, encryption, evaluation, decryption, payload, evaluation-key, and artifact sizes.

The central mechanism is **certificate-guided distillation**. For each reached state, an analytical coefficient-quantization error bound $\epsilon$ is compared with the clear student's top-two score margin $m$. If $m > 2\epsilon$, the integer circuit and clear student must select the same action whenever the FHE program evaluates correctly. High-occupancy uncertified states are fed back into weighted distillation. The claim is intentionally narrow: this certificate does not prove teacher agreement, task safety, malicious-server correctness, or endpoint security.

## Evidence ladder

Every candidate is evaluated through the same semantics:

| Label | Execution | What it establishes |
|---|---|---|
| `FLOAT TEACHER` | clear high-capacity policy | utility ceiling |
| `FLOAT STUDENT` | clear polynomial scores | distillation loss |
| `QUANTIZED CLEAR` | exact integer circuit in clear | quantization and overflow behavior |
| `FHE SIMULATED` | Concrete compiler simulation | compiled numerical semantics; **not privacy or latency** |
| `REAL FHE` | keygen → encrypt → homomorphic evaluate → decrypt | encrypted correctness and measured systems cost |

## Recorded Modal result

The committed `modal-smoke-001` record was produced by the real end-to-end path on 23 August 2026:

| Measurement | Recorded value |
|---|---:|
| GPU teacher search | 2,048 policies × 18 iterations × 12 states on NVIDIA L4 |
| GPU training wall | 9.668 s |
| Teacher / quantized student return | 500 / 500 over the same eight post-selection evaluation episodes |
| Post-selection integer-student-occupancy certificate coverage | 95.875% |
| Exhaustive integer-box coverage | 98.7534% of 923,521 codes |
| FHE configuration | Concrete-Python 2.10.0, category 128, `global_p_error=10^-6` |
| REAL FHE closed-loop prefix | 25 / 25 encrypted steps match integer clear; 24 / 25 reached codes certified |
| Median across those 25 sequential steps | 11.058 ms server evaluation / 1,795.318 ms client-observed online time |
| Serialized request / response | 33,592 B / 16,920 B |

These are smoke measurements, not a latency distribution or multi-task paper result. The latency medians describe one 25-step trajectory with a separate Modal RPC per step; they are not independent-container samples, p50 estimates, or throughput. The selected affine circuit used the Concrete FHE runtime but did not require bootstrapping; no unbounded-depth claim is made.

The `unseen-loop/modal-evidence-v2` record contains the complete 25-step encrypted prefix, a top-level `authenticated_envelope_protocol` descriptor, and per-call request/response envelope and context digests. It does not persist plaintext private observations or decrypted score vectors. Its nonsecret Modal bundle contains `evidence.json`, `receipt.json`, `server.zip`, `client-specs.bin`, `policy.json`, and `checksums.sha256`; the ledger checksums the other five files. Inspect the [raw recorded evidence](artifacts/reference/modal-smoke-001.json) and the [paper](docs/paper.md).

## One-command paths

```bash
# Deterministic local research smoke; no privacy claim.
uv sync --extra dev
uv run unseen-loop demo --backend clear --output artifacts/demo

# Typed preregistered clear matrix: 3 environments × 5 checkpoints; no privacy claim.
uv run unseen-loop suite \
  --config experiments/release.toml \
  --backend clear \
  --output artifacts/release

# Serialized real FHE locally for one quick experiment.
uv sync --extra dev --extra fhe
uv run unseen-loop demo --backend fhe --output artifacts/fhe-local

# One-checkpoint Modal smoke: L4 teacher → CPU compile → local keygen → remote FHE.
uv sync --extra cloud --extra fhe
uv run modal run -w artifacts/modal-evidence.json \
  modal_app.py::research --run-id my-modal-run

# Publish and inspect the evidence-first report.
uv run unseen-loop report artifacts/modal-evidence.json
python -m http.server 8000
```

`unseen-loop suite` is the typed release-matrix orchestrator: it validates `experiments/release.toml` and materializes every declared environment/checkpoint run. With `--backend clear`, it does not provide privacy evidence or discharge the manifest's real-FHE, stress, timing, and ablation requirements. `unseen-loop research` and `modal_app.py::research --full` scale one environment/checkpoint path only; neither is the release suite. The FHE paths fail if Concrete is missing and never alias clear execution to an FHE label.

## Security envelope

The intended evaluator is **honest-but-curious** and runs a pinned, data-independent circuit. The client HMAC-authenticates fixed-shape request and response envelopes that bind freshness, payload length, and policy, circuit, client-context, and evaluation-key digests. Under the FHE scheme's assumptions, fresh client encryption hides observation and encrypted-score values. Public leakage includes policy/version, tensor shape, parameter set, request/response sizes, timing, traffic volume, status, and linkable evaluation-key identity. The eventual action may be observable through the environment.

Envelope authentication detects corruption, substitution, response swaps, and context confusion. After authentication, context, and freshness checks, the serialized evaluator atomically claims the request digest in a Volume-backed replay ledger retained for ten minutes—longer than the five-minute freshness window—before deserializing FHE inputs. Authentication still does not prove that a malicious evaluator ran the committed circuit. Unseen Loop does not claim malicious-server integrity, circuit privacy, model-extraction resistance, endpoint protection, availability, or traffic-flow confidentiality. See the threat model before using the protocol outside research.

## Repository map

```text
src/unseen_loop/       policies, certificates, protocol, experiments, reports
modal_app.py           isolated cloud training/search/FHE orchestration
experiments/           versioned experiment specifications
tests/                 semantic, certificate, protocol, and FHE boundary tests
docs/                  paper, architecture, threat model, reproduction guide
site/                   generated evidence-first research report
artifacts/              machine-readable run manifests and raw measurements
```

## Research documentation

- [Paper and precise novelty boundary](docs/paper.md)
- [Threat model and negative tests](docs/threat-model.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Preregistered release benchmark](docs/benchmark-protocol.md)
- [Local and Modal reproduction](docs/reproduction.md)
- [Step-by-step encrypted action tutorial](docs/tutorial.md)
- [Modal resource and cost controls](docs/modal.md)

The interactive report is served from [`site/index.html`](site/index.html); it loads every displayed result from the committed evidence JSON rather than hard-coded hidden state.

## License

MIT for this repository. Concrete Python is an optional external runtime with its own license and patent terms; review them before commercial use.
