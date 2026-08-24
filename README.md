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

## Checksummed Modal studies

The publication source is [`artifacts/studies/unseen-loop-release-analysis-004/publication.json`](artifacts/studies/unseen-loop-release-analysis-004/publication.json), SHA-256 `7a8c4ee7fd8f5d27778b94c98913b292b120172b136dcffe936b2591f5811536`. Its enclosing `checksums.sha256` ledger has SHA-256 `3dd9ac68c0e2db09449b228180707b3f459f094606cb1bd7953e9a2f3a70e823`. Tables below are transcribed from that named, checksummed analysis—not recomputed in Markdown; decimal estimates are rounded to three places for display.

Before any table is accepted, the Modal analyzer independently reloads every teacher, reruns all teacher selection seeds, reconstructs all 280 candidate objectives from 14,000 candidate-by-seed rows, recomputes every Pareto flag, and applies the exact champion tolerance and ordering rule. A summary-named but misselected champion fails publication.

The expanded clear study completed all 15 environment/checkpoint runs: 120 candidate rows, 6,000 selection-episode rows, 1,500 paired post-selection evaluations, and 3,000 long-form teacher/student evaluation rows.

| Environment | Checkpoints / pairs | Teacher mean | Integer-student mean | Paired return Δ, 95% bootstrap CI | Action certificate |
|---|---:|---:|---:|---:|---:|
| CartPole-v1 | 5 / 500 | 461.488 | 432.008 | −29.480 [−71.358, 0.072] | 214,268 / 216,004 (99.196%) |
| MountainCar-v0 | 5 / 500 | −194.986 | −194.190 | +0.796 [−1.620, 3.544] | 96,925 / 97,095 (99.825%) |
| Acrobot-v1 | 5 / 500 | −94.260 | −326.156 | **−231.896 [−388.536, −75.831]** | 163,088 / 163,295 (99.873%) |

The intervals are deterministic 10,000-repetition checkpoint-then-paired-episode percentile bootstraps. CartPole and MountainCar intervals include zero. Acrobot is a large, precisely retained negative result: the selected integer student lost return relative to its teacher.

The matched clear CartPole 2×2 study isolates certificate weighting and the complete occupancy-refinement bundle over five checkpoints and 500 paired evaluations per cell. The refinement-bundle main effect is **+83.619 [26.144, 145.954]** return. This supports a causal statement only for that tested CartPole bundle in these four matched cells. The weighting main-effect estimate is **−108.461 [−288.649, 68.250]**: negative at the point estimate, but its interval includes zero. The interaction is +8.654 [−180.737, 156.256].

Two source-scoped `REAL FHE` studies complete the systems evidence:

| Study | Exact accounting | Server evaluation | End-to-end | Scope |
|---|---:|---:|---:|---|
| degree-2, two-feature/two-action, `qmax=2` | 25 complete-domain + 15 randomized-canary = 40/40 matching calls | p50 362.091 ms; p95 374.514 ms | p50 366.597 ms; p95 378.941 ms | one colocated Modal client/server worker |
| repeated timing | 4 independent contexts; 12 excluded warmups; 64/64 measured successes | p50 544.536 ms; p95 830.709 ms | p50 550.076 ms; p95 837.010 ms | four colocated Modal client/server workers |

The nonlinear study exercised three encrypted-encrypted quadratic feature products per inference over the complete declared 25-point domain. The timing study reports a hierarchical container/request bootstrap over non-warmup successes, not throughput, a production service, or a shared cryptographic context. Because client and server were colocated inside each Modal worker, neither study demonstrates local-client/remote-server secrecy. Conversely, the expanded and factorial studies are `QUANTIZED CLEAR` and provide no privacy evidence.

This is a completed **expanded bounded evidence study**, not completion of the full preregistration. The expanded matrix searched 8 candidates per environment/checkpoint (120 total) with 50 selection episodes per candidate; the full release specifies 120 candidates per environment/checkpoint (1,800 total) with 100 selection episodes each (180,000 selection rows). Still uncompleted are the preregistered 64-row physically remote client/server challenge and the remaining release-wide stress and gate matrix. See the [paper](docs/paper.md), [benchmark protocol](docs/benchmark-protocol.md), and [exact Modal reproduction guide](docs/reproduction.md).

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
