# Unseen Loop

**The cloud acts on state it cannot read.**

Unseen Loop is a reproducible research system for **encrypted closed-loop policy serving**. It distills reinforcement-learning teachers into low-degree integer policies, searches the return–certificate–circuit-cost frontier, and evaluates the selected policy on client-encrypted observations with Fully Homomorphic Encryption (FHE).

> **Scope:** inference only. Training data, teacher execution, and policy compilation are cleartext development activities. The deployed evaluator receives ciphertext observations and public evaluation material; the secret key remains with the client. Only results explicitly labeled `REAL FHE` are privacy evidence. Simulation is never reported as encrypted execution.

## Research thesis

Ordinary FHE inference benchmarks stop at per-example accuracy. A control policy is sequential: one changed action changes every later state. Unseen Loop therefore optimizes and measures four coupled properties:

1. **closed-loop return and constraint cost** under student-induced occupancy;
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

The committed `modal-smoke-001` evidence record was produced by the real end-to-end path on 23 August 2026:

| Measurement | Recorded value |
|---|---:|
| GPU teacher search | 2,048 policies × 18 iterations × 12 states on NVIDIA L4 |
| GPU training wall | 9.497 s |
| Teacher / quantized student return | 500 / 469 over eight held-out quick-artifact episodes |
| Held-out certificate coverage | 94.1% |
| Exhaustive integer-box coverage | 98.9867% of 50,625 codes |
| FHE configuration | Concrete-Python 2.10.0, category 128, `global_p_error=10^-6` |
| Cold / warm Modal server evaluation | 43.736 ms / 7.799 ms |
| Serialized request / response | 32,088 B / 16,168 B |
| Real-FHE equality | 2 / 2 decrypted outputs equal exact integer clear |

These are smoke measurements, not a latency distribution or multi-task paper result. The selected affine circuit used the Concrete FHE runtime but did not require bootstrapping; no unbounded-depth claim is made. Inspect the [raw recorded evidence](artifacts/reference/modal-smoke-001.json) and the [paper](docs/paper.md).

## One-command paths

```bash
# Deterministic local research smoke; no privacy claim.
uv sync --extra dev
uv run unseen-loop demo --backend clear --output artifacts/demo

# Serialized real FHE locally.
uv sync --extra dev --extra fhe
uv run unseen-loop demo --backend fhe --output artifacts/fhe-local

# NVIDIA L4 teacher → CPU search/compile → local keygen → remote Modal FHE evaluator.
uv sync --extra cloud --extra fhe
uv run modal run -w artifacts/modal-evidence.json \
  modal_app.py::research --run-id my-modal-run

# Publish and inspect the evidence-first report.
uv run unseen-loop report artifacts/modal-evidence.json
python -m http.server 8000
```

The FHE path fails if Concrete is missing; it never aliases clear execution to an FHE label.

## Security envelope

The intended evaluator is **honest-but-curious** and runs a pinned, data-independent circuit. Under the FHE scheme's assumptions, fresh client encryption hides observation and encrypted-score values. Public leakage includes policy/version, tensor shape, parameter set, request/response sizes, timing, traffic volume, status, and linkable evaluation-key identity. The eventual action may be observable through the environment.

FHE is malleable and does **not** prove correct evaluation. Unseen Loop does not claim malicious-server integrity, circuit privacy, model-extraction resistance, endpoint protection, availability, or traffic-flow confidentiality. See the threat model before using the protocol outside research.

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
