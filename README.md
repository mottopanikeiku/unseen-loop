# Unseen Loop

**Act safely. Estimate cautiously. Keep the inputs private.**

Unseen Loop is a reproducible research system for **private sequential decision evaluation**. Its integrated flagship combines a two-step encrypted counterfactual safety shield with horizon-aware private off-policy evaluation (OPE). The original certificate-guided policy-distillation and encrypted closed-loop serving system remains the training and systems foundation.

> **Scope:** confidential inference and evaluation only. Training data, teacher execution, calibration, policy compilation, target-policy fitting, and uncertainty resampling are clear development or client activities. The evaluator receives ciphertext inputs plus public evaluation material; the secret key, shield selection, OPE division, and any explicit aggregate disclosure remain with the client. Only checksum-closed `REAL FHE` rows are privacy evidence. Simulation is never reported as ciphertext execution.

## Integrated flagship

### CipherShield-RL

For one encrypted state in frozen order `(x, y, vx, vy, battery, tilt)`, the server evaluates every public action `(BRAKE, EAST, WEST, NORTH, SOUTH)` through two public polynomial dynamics steps. The logical output is a complete `5 × 2 × 4` tensor of signed obstacle, speed, tilt, and battery margins. Complete-domain analysis removes spatial constraints that never attain the minimum. If several remain, sign-preserving saturation bounds Concrete lookup width without changing strict-positive safety decisions. The server does not select an action. After decryption, the client:

1. retains the requested action when its eight obligations are strictly positive;
2. otherwise selects the certified candidate with the greatest minimum buffered margin;
3. breaks exact ties in frozen action-enum order; and
4. records an explicitly uncertified `BRAKE` fallback if no candidate is certified.

The Concrete backend is an encrypted tensor/vectorized circuit, not SIMD packing. Its complete declared input domain is `qmax=2`, or `5^6 = 15,625` signed states.

### Private horizon-aware OPE

The client encrypts a fixed-shape trajectory batch containing logged states, requested actions, rewards, and behavior propensities. The server evaluates a public degree-1 or degree-2 target propensity model and returns `3H` additive sufficient statistics: horizon numerators, denominators, and counts. Division and the final estimate are client-only.

Two deliberately distinct backends prevent an approximation claim from masquerading as exactness:

- **Concrete exact bounded canary:** integer hard clipping, three encrypted horizon vectors, and exact clear/simulation/REAL agreement on the declared proof shape.
- **TenSEAL CKKS:** slot-packed approximate polynomial soft clipping under identifier `POLYNOMIAL_APPROX_OPE_V1`; tc128 modulus-budget enforcement, client/public context separation, and explicit approximation receipts.

The shield is part of the logged MDP. Integration evidence therefore logs `requested_action` with its behavior propensity separately from `executed_action`; it never substitutes post-shield executed-action propensity after the many-to-one shield map.

### Checksum-closed flagship canaries

The browser publication is [`site/data/flagship-evidence.json`](site/data/flagship-evidence.json), SHA-256 `c19d256ee6fe6fc301e715155c3538cf3fdac28946b1e99eda3a0f4473aa5407`. Modal built it from three immutable canary summaries; the browser independently verifies the copied byte digest and every displayed certificate/division invariant before rendering measurements.

| Canary | Exact declared accounting | Server evaluation | End-to-end | Transport |
|---|---:|---:|---:|---:|
| CipherShield Concrete, `qmax=2` | 15,625/15,625 simulations match + 1/1 REAL FHE tensor match | 73.916 s | 77.491 s | 758,473,160 B evaluation keys; 492,056 B request; 3,278,720 B response |
| exact Concrete OPE, `(N=1,H=2,D=1)` | simulation = REAL = integer reference | 1.866 s | 6.329 s | 589,084,056 B evaluation keys; 124,032 B request; 115,480 B response |
| CKKS OPE, `(N=64,H=8,D=1)` | 24 approximate output ciphertexts; distinct clear approximation | 7.465 s | client phase receipts reported separately | 813,936,378 B public server context; 86,405,050 B request; 25,943,816 B response |

These are colocated Modal cryptographic canaries, not production latency, throughput, remote-network secrecy, or empirical policy-value results. The CKKS maximum released numerator error was 8.409 under a declared conservative soft-clip absolute-error bound of 32. The exact OPE shape is intentionally tiny; the former `(4,4,6)` graph exceeded a 32 GiB worker.

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

# Flagship plan inspection; emits exact stage IDs and denominators, no empirical work.
uv run modal run modal_flagship.py::inspect_plan \
  --config experiments/flagship-smoke.toml

# Modal-only REAL-FHE canaries: complete-domain CipherShield, exact OPE, and CKKS OPE.
uv run modal run -w artifacts/flagship-canaries.json \
  modal_flagship_canary.py::run --prefix my-flagship-canary

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
