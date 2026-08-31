# Unseen Loop: Certificate-Guided Distillation for Encrypted Closed-Loop Policy Serving

**Research artifact · 23 August 2026**

## Abstract

Fully Homomorphic Encryption (FHE) can evaluate a policy without revealing its input to the evaluator, but conventional private-inference benchmarks treat predictions as independent samples. Reinforcement-learning policies violate that assumption: a single changed action changes the future state distribution and can compound into a different return. We introduce **Unseen Loop**, an open research artifact for deterministic discrete-action policy serving over client-key FHE. The system trains a clear RL teacher, distills low-degree integer score policies, searches policy degree and precision against student-induced closed-loop return, analytical action-invariance coverage, and circuit cost, then executes a serialized client/server protocol locally or on Modal.

The central mechanism is a coefficient-rounding certificate. At a quantized state, if the clear student's top-two score margin exceeds twice an analytical per-score error bound, the error-free integer circuit must choose the same greedy action. Uncertified and mismatched student-occupancy states receive greater weight in the next distillation round. The certificate is composed—never conflated—with Concrete's probabilistic whole-circuit correctness configuration.

The checksummed expanded study completed 15/15 clear runs—five checkpoints each for CartPole, MountainCar, and Acrobot—with 120 candidates, 6,000 selection rows, and 1,500 paired/3,000 long-form post-selection evaluation rows. The occupancy-refinement bundle improved paired CartPole return by +83.619, 95% CI [26.144, 145.954], across a matched 2×2 factorial; certificate weighting's main-effect estimate was −108.461 [−288.649, 68.250]. Expanded paired student-minus-teacher return was −29.480 [−71.358, 0.072] for CartPole, +0.796 [−1.620, 3.544] for MountainCar, and −231.896 [−388.536, −75.831] for Acrobot. A degree-2 two-feature/two-action circuit completed 25 exhaustive-domain and 15 canary `REAL FHE` calls exactly. A separate four-context timing study retained 12 warmups and 64/64 successful measured calls: server p50/p95 were 544.536/830.709 ms and end-to-end p50/p95 were 550.076/837.010 ms. The clear studies provide no privacy evidence; the colocated FHE studies do not demonstrate local-client/remote-server secrecy. These are bounded executed studies, not efficacy across tasks or completion of the full preregistration.

## 1. Problem

Let an environment expose private observation $s_t$ to a client. A remote evaluator holds a deterministic score policy $f: \mathcal{S}\rightarrow\mathbb{Z}^{|\mathcal{A}|}$. The client wants

$$
a_t = \operatorname*{argmax}_{a\in\mathcal{A}} f_a(s_t)
$$

without disclosing $s_t$ or the returned score values to the evaluator. The client generates a secret key, encrypts a fixed-shape quantized observation, sends the ciphertext and public evaluation material, receives encrypted scores, decrypts them, applies stable argmax, and advances the environment.

This is not independent classification. If mode $A$ and mode $B$ select different actions at step $t$, then $s_{t+1}^{A}\neq s_{t+1}^{B}$ may hold even in a deterministic environment. Pointwise agreement on a frozen teacher dataset cannot characterize the resulting return or safety cost. Unseen Loop therefore makes **integer-student-induced occupancy** a first-class search distribution and the mandatory post-selection evaluation distribution.

### Scope

- deterministic, discrete-action inference;
- clear training, calibration, distillation, and compilation;
- client-private observations and encrypted score outputs against an honest-but-curious evaluator;
- client-side argmax, so the client learns every returned score;
- bounded integer programs compiled by Concrete-Python 2.10.0.

Not in scope: private RL training, stochastic action sampling, continuous-action certification, malicious-server integrity, circuit privacy, endpoint compromise, traffic-flow confidentiality, or model-extraction resistance.

## 2. Contributions

1. **Closed-loop FHE co-design.** Search degree, observation precision, coefficient precision, and ridge regularization against return, constraint cost, teacher agreement, certificate coverage, estimated integer width, encrypted multiplication count, and measured finalist FHE cost.
2. **Counterexample-guided action certificates.** Derive a sound coefficient-rounding error bound at every reached state, reweight fragile states, and exactly enumerate low-dimensional declared integer domains.
3. **Five-stage semantic ladder.** Keep float teacher, float student, exact integer clear, compiled simulation, and real FHE distinct. Only the final stage is confidentiality and encrypted-latency evidence.
4. **Physical client/server protocol.** Serialize the architecture-specific server artifact, client specifications, ciphertexts, and evaluation material. The Modal evaluator never constructs a client and never receives the secret key.
5. **Evidence-first artifact.** Emit content hashes, range and bit-width receipts, configuration, raw timings, ciphertext sizes and hashes, explicit non-claims, a static interactive report, and deterministic reproduction commands. Persisted/cloud privacy evidence excludes plaintext private observations and decrypted score vectors.

We do **not** claim the first use of FHE in RL. Encrypted tabular learning, encrypted policy synthesis, and encrypted deep-RL training predate this artifact [1–3]. The contribution is the combination of closed-loop co-design, action-stability evidence, and real serialized deployment.

## 3. Policy representation

### 3.1 Observation quantization

For feature $i$, calibration freezes center $c_i$, positive step $\Delta_i$, and signed code limit $Q$:

$$
q_i(s)=\operatorname{clip}_{[-Q,Q]}\left(\operatorname{round}\frac{s_i-c_i}{\Delta_i}\right).
$$

Runtime inference rejects an observation outside the compiled domain by default. Clipping exists only as an explicit search/diagnostic behavior; a release client must fail before encryption rather than allow wraparound or silent saturation.

### 3.2 Polynomial score policy

For degree $d\in\{1,2\}$, the integer feature map is

$$
\phi_1(q)=[1,q_1,\ldots,q_n],
$$

$$
\phi_2(q)=[1,q_1,\ldots,q_n,q_1^2,q_1q_2,\ldots,q_n^2].
$$

Weighted ridge regression fits clear coefficients $W\in\mathbb{R}^{A\times D}$ to teacher score vectors. A signed coefficient budget $b_w$ gives $C=2^{b_w-1}-1$. A global scale $\alpha=C/\max|W|$ freezes integer coefficients $\widehat{W}=\operatorname{round}(\alpha W)$. The deployed integer program returns

$$
F(q)=\widehat{W}\phi_d(q).
$$

The same `PolicySpec` drives integer-clear, simulation, and real FHE. No backend retrains weights, changes the quantizer, moves argmax, or changes output semantics.

## 4. Action-invariance certificate

For fixed integer input $q$, define the dequantized integer score $\widehat{g}_a(q)=\widehat{W}_a\phi(q)/\alpha$ and clear student score $g_a(q)=W_a\phi(q)$. Coefficient rounding gives the analytical bound

$$
|g_a(q)-\widehat{g}_a(q)|
\leq
\epsilon_a(q)
=
\sum_j \left|W_{aj}-\frac{\widehat{W}_{aj}}{\alpha}\right|\,|\phi_j(q)|.
$$

Let $a^*=\arg\max_a g_a(q)$ under stable lowest-index tie-breaking and let

$$
m(q)=g_{a^*}(q)-\max_{a\neq a^*}g_a(q),\qquad
\epsilon(q)=\max_a\epsilon_a(q).
$$

### Proposition 1: coefficient-rounding action invariance

If $m(q)>2\epsilon(q)$, then $\arg\max_a g_a(q)=\arg\max_a \widehat{g}_a(q)$.

**Proof.** The winning score can decrease by at most $\epsilon$ and every competing score can increase by at most $\epsilon$. Therefore its post-rounding lead is greater than $m-2\epsilon>0$. ∎

This proposition says nothing about whether the student matches the teacher. It also conditions on correct evaluation of the integer circuit.

### 4.1 Domain certificate

For a low-dimensional quantizer box, Unseen Loop enumerates every integer code, hashes the ordered code stream, evaluates the analytical obligation, and checks for certified mismatches. The CartPole quick artifact enumerates $31^4=923{,}521$ codes over $[-15,15]^4$. Coverage below 100% is reported directly; an exhaustive run is not mislabeled a complete global action guarantee.

### 4.2 FHE correctness composition

Let the compiler's declared whole-circuit error probability be at most $p_g$. For $H$ encrypted decisions, the union bound gives

$$
\Pr[\text{any circuit-error event in }H\text{ calls}]\leq Hp_g.
$$

No independence assumption is required. A small empirical canary set cannot estimate $p_g$. The bound only composes a separately configured compiler premise with deterministic coefficient certification.

## 5. Counterexample-guided distillation

For every candidate tuple $(d,b_x,b_w,\lambda)$:

1. collect teacher trajectories on distillation-only seeds;
2. calibrate the quantizer and fit weighted ridge scores;
3. compute per-state action certificates;
4. run the integer student on refinement-only seeds to collect **student-induced** trajectories;
5. query clear teacher scores on those states, increase weights on uncertified and integer/float-mismatched states, and refit without changing the frozen quantizer;
6. evaluate candidate utility, constraint cost, agreement, and certificate coverage on selection-only seeds under exact integer-student occupancy;
7. exclude any candidate that saturates the declared quantizer range, then Pareto-filter and choose the champion using only selection metrics;
8. after selection is frozen, run the champion and teacher on the same disjoint evaluation seeds and report only these post-selection paired results.

Quick and release presets use 8/8 and 100/100 selection/evaluation episodes respectively. The artifact preserves two long-form episode rows per evaluation seed—`FLOAT TEACHER` and `QUANTIZED CLEAR`—so paired return deltas and denominators remain auditable. This loop does not claim optimality; it is a deterministic, inspectable grid-search baseline intended to make the RL-specific objective falsifiable.

## 6. Systems design

```mermaid
sequenceDiagram
    participant C as Local client + environment
    participant M as Modal evaluator
    C->>C: Generate secret + evaluation keys
    C->>C: Quantize, range-check, encrypt, authenticate envelope
    C->>M: server.zip + receipt, signed request, evaluation keys
    M->>M: Verify envelope/context; homomorphically evaluate
    M-->>C: Authenticated encrypted-score envelope
    C->>C: Verify, decrypt, stable argmax, environment.step
```

Compilation is remote because `server.zip` is architecture-specific. Key generation, encryption, decryption, and environment transition execute in the local Modal entrypoint process. The evaluator accepts the server artifact, an authenticated request-envelope JSON document, serialized evaluation keys, a per-run HMAC authentication key, and the circuit receipt. It verifies the HMAC, freshness, shape, payload length, and policy/circuit/client-context/evaluation-key digests, then atomically claims the request digest in a Volume replay ledger before FHE deserialization. The serialized evaluator retains claims for ten minutes, beyond the five-minute request-freshness window, including across container restarts. It authenticates the response envelope after `Server.run`. This transcript integrity is not a proof of correct evaluation, and the authentication key is distinct from the FHE secret key.

The recorded `modal-smoke-001` champion is affine. Concrete evaluated its bounded encrypted integer dot product with a fully homomorphic scheme/runtime, but the circuit did not need programmable bootstrapping. Separately, `modal-nonlinear-qmax2-002` executes a degree-2, two-feature/two-action polynomial with three encrypted-encrypted quadratic feature products per inference. That synthetic complete-domain challenge establishes exact circuit conformance for its declared `qmax=2` domain; it is not evidence that the expanded RL champions are quadratic or that quadratic policies improve return.

## 7. Experimental protocol

### 7.1 Evidence ladder

| Stage | Input | Execution | Valid evidence |
|---|---|---|---|
| Float teacher | float observation | clear MLP | teacher utility |
| Float student | quantized code interpreted in clear | clear polynomial | distillation |
| Quantized clear | integer code | exact exported integer kernel | coefficient rounding and overflow |
| FHE simulated | integer code | compiler graph in clear | compiled semantics only |
| Real FHE | serialized ciphertext | encrypt → server run → decrypt | ciphertext correctness and measured cost |

### 7.2 Publication evidence and integrity

All values in Sections 7.2–7.5 are copied from [`../artifacts/studies/unseen-loop-release-analysis-004/publication.json`](../artifacts/studies/unseen-loop-release-analysis-004/publication.json), SHA-256 `7a8c4ee7fd8f5d27778b94c98913b292b120172b136dcffe936b2591f5811536`. That digest is recorded in the analysis `checksums.sha256`; the ledger itself has SHA-256 `3dd9ac68c0e2db09449b228180707b3f459f094606cb1bd7953e9a2f3a70e823`. The source registry is `evidence-index.json` (SHA-256 `6e7176284db10dbb51ab0e4e00066fcf51c39876cb3027e0581757829b2a92fa`). Decimal estimates are rounded to three places for display; denominators and digests are exact.

The analysis does not trust stored champion names. It digest-checks and reloads each teacher checkpoint, reruns teacher selection baselines, reconstructs return mean/standard deviation, cost, agreement, certificate coverage, saturation, and circuit objectives for all 280 candidates from 14,000 selection rows, recomputes the Pareto frontier, and executes the exact championship tolerance and tie-breaking key. Any objective, Pareto, or champion mismatch aborts publication.

| Source study | Backend / trust label | Exact observed denominator | Config SHA-256 | Source-summary SHA-256 | Ledger SHA-256 |
|---|---|---|---|---|---|
| `expanded-multitask-modal-002` | `QUANTIZED CLEAR`; `clear Modal CPU research worker; no privacy evidence` | 15 runs; 120 candidates; 6,000 selection rows; 1,500 paired / 3,000 long-form evaluation rows | `55e44942918359c0e8cd2a11335bba2bf2b71a64225afc1c11e78c0cbcb98367` | `700fe962d2d6a42da23a36d67e7224e835f04571b71339207140ea98946c0b4d` | `ee1824879d4bb023f92e44b78df861ec578c1c87d564694ffa6a2f574f8d7988` |
| `expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-refined` | `QUANTIZED CLEAR`; `clear Modal CPU research worker; no privacy evidence` | 5 runs; 40 candidates; 2,000 selection rows; 500 paired / 1,000 long-form evaluation rows | `82e0a6449dd608cd88b91ce9c2074d7af1122450e54ec7a1297014b7f4591d52` | `9d0abe904eb67ccce394d6066ce13f2d721ca02e4d541016254d1b97c388ac92` | `7f006ea7ecf358b64601ac9ed6e7663b0339098db14c082f9f1012aecdc8812e` |
| `expanded-cartpole-ablation-modal-004--ablation-cartpole-unweighted-unrefined` | `QUANTIZED CLEAR`; `clear Modal CPU research worker; no privacy evidence` | 5 runs; 40 candidates; 2,000 selection rows; 500 paired / 1,000 long-form evaluation rows | `1de8253a55c74e068ff6fca808a70206233b5b58cd54f8ee1b94d2938751bdb4` | `19e6b03e798496933467fe77553b9ea93bcee740668a44b15a377f7e10e18cc8` | `9415d48a3de811912017f64cd8bcafd52a3adefb3dbb20ceeb414afabf229f16` |
| `expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-refined` | `QUANTIZED CLEAR`; `clear Modal CPU research worker; no privacy evidence` | 5 runs; 40 candidates; 2,000 selection rows; 500 paired / 1,000 long-form evaluation rows | `4ccee71276f6c605c1566336e7ee6e438ff4d68265db4ef7ccc37f78d0190696` | `72b5db1dfb6601abaf87f5e02bfe79d3f274acc8de95e1f8ec17be4915d6f968` | `81431a70ac63d6a8b6182d411ef455f068dfe8b78a1f96e9a18f141a72781e44` |
| `expanded-cartpole-ablation-modal-004--ablation-cartpole-weighted-unrefined` | `QUANTIZED CLEAR`; `clear Modal CPU research worker; no privacy evidence` | 5 runs; 40 candidates; 2,000 selection rows; 500 paired / 1,000 long-form evaluation rows | `34c314f61274af2417c41866e00d755bc1df3a3e7c18d45d190b536a39665790` | `0a650515b7e93798ef99405a6e087f782b81e52a845d8a44c87b5d2d9df63490` | `13ec7f52044c896c77bab9ecba15ca38b3fca7e049e11dc52cb325d426318dd7` |
| `modal-nonlinear-qmax2-002` | `REAL FHE`; `colocated Modal client/server research worker` | 40/40 attempts; zero failures | `0c777fa1d56be79f65bf25d5d6afc8ebba7ce916a8405f2286550c42a5a46f6b` | `20bfad9186e2cc25801d550a4c14378ae83e426c4b7274e53af98d12cd683a9c` | `42333d8fb6ff53f9cb1ea6556887bfcaa72975f9415f8243e830cc05d417f1e8` |
| `modal-fhe-timing-003` | `REAL FHE`; `four independent colocated Modal client/server research contexts` | 76/76 attempts: 12 excluded warmups + 64/64 measured successes; 4 containers | `b03154fd348df52259030f9104c73f51e9401ef8494fb950d5371c8e1b020232` | `79e81d934f05eef82c5f7249330487e79cac5ae6a7eb98bb386efcf40bfed00b` | `92828e92d83256f7b7b64b26576ee0caf68334ef1c4fd05faf03b29e1474df6f` |

Every planned denominator in this index equals its observed denominator; checksum and incomplete-denominator failures are zero. Three expanded-suite gates and one to three gates in each factorial cell still fail. Completion means evidence completeness, not gate success.

### 7.3 Expanded three-environment result

[`expanded-environments.jsonl`](../artifacts/studies/unseen-loop-release-analysis-004/expanded-environments.jsonl) reports post-selection student-minus-teacher return under exact integer-student occupancy. Intervals use 10,000 deterministic checkpoint-then-paired-episode percentile-bootstrap repetitions.

| Environment | Checkpoints / paired episodes | Teacher return, mean | Integer-student return, mean | Paired Δ, 95% CI | Teacher agreement | Action certificate |
|---|---:|---:|---:|---:|---:|---:|
| CartPole-v1 | 5 / 500 | 461.488 | 432.008 | −29.480 [−71.358, 0.072] | 174,943 / 216,004 (80.991%) | 214,268 / 216,004 (99.196%); 0 certified mismatches |
| MountainCar-v0 | 5 / 500 | −194.986 | −194.190 | +0.796 [−1.620, 3.544] | 87,773 / 97,095 (90.399%) | 96,925 / 97,095 (99.825%); 0 certified mismatches |
| Acrobot-v1 | 5 / 500 | −94.260 | −326.156 | **−231.896 [−388.536, −75.831]** | 112,901 / 163,295 (69.139%) | 163,088 / 163,295 (99.873%); 0 certified mismatches |

CartPole's and MountainCar's intervals include zero: these data do not establish an improvement or regression in either environment. Acrobot is an unambiguous negative result within this study: the selected integer policies lost 231.896 mean return relative to their matched teachers, with the entire interval below zero. High float-student/integer action-certificate coverage did not imply teacher agreement or task efficacy. These 15 clear runs test the bounded distillation/evaluation pipeline; they provide neither privacy evidence nor a generalization claim beyond the measured checkpoints.

### 7.4 Matched CartPole 2×2 factorial

[`ablation-cells.jsonl`](../artifacts/studies/unseen-loop-release-analysis-004/ablation-cells.jsonl) contains five matched checkpoints and 500 paired post-selection evaluations in each cell. “Refinement” denotes the complete occupancy-refinement bundle, not a single isolated submechanism.

| Certificate weighting | Occupancy-refinement bundle | Paired return Δ, 95% CI | Selection-occupancy certificate | Post-selection held-out certificate |
|---|---|---:|---:|---:|
| off | off | −82.968 [−226.666, 0.916] | 101,858 / 103,671 | 201,770 / 205,855 |
| off | on | −3.676 [−14.638, 1.572] | 121,192 / 122,402 | 242,990 / 245,501 |
| on | off | −195.756 [−361.146, −30.122] | 75,387 / 76,454 | 147,368 / 149,461 |
| on | on | −107.810 [−248.575, 9.772] | 95,419 / 96,653 | 190,971 / 193,434 |

[`ablation-effects.jsonl`](../artifacts/studies/unseen-loop-release-analysis-004/ablation-effects.jsonl) averages matched contrasts across the other factor. Return and selection-certificate intervals are 10,000-repetition matched-checkpoint-then-episode percentile bootstraps. Held-out receipts preserve exact aggregate numerators/denominators but not per-episode certificate rows, so the analysis correctly does not manufacture held-out bootstrap intervals.

| Factorial contrast | Paired-return effect, 95% CI | Selection-certificate-rate effect, 95% CI | Exact interpretation |
|---|---:|---:|---|
| Weighting main effect | **−108.461 [−288.649, 68.250]** | +0.000325 [−0.016047, 0.019685] | negative return point estimate; interval includes zero |
| Occupancy-refinement-bundle main effect | **+83.619 [26.144, 145.954]** | +0.004396 [−0.011036, 0.022976] | positive return effect for this tested matched CartPole bundle |
| Interaction | +8.654 [−180.737, 156.256] | −0.006414 [−0.038128, 0.026261] | interval includes zero |

The positive causal claim is deliberately narrow: enabling the represented occupancy-refinement bundle caused higher paired return across these matched clear CartPole cells. It is not a claim about other environments, different teachers/search grids, any one component inside the bundle, or privacy. Certificate weighting has a negative point estimate but is statistically inconclusive here; it must not be advertised as beneficial.

### 7.5 Nonlinear circuit and timing

[`scoped-fhe-summaries.json`](../artifacts/studies/unseen-loop-release-analysis-004/scoped-fhe-summaries.json) copies the two checksummed source summaries exactly.

| Measurement | Degree-2 complete-domain challenge | Four-context timing study |
|---|---:|---:|
| Circuit / context | 2 features, 2 actions, degree 2, `qmax=2`; 3 quadratic products/inference | same fixed bounded timing circuit in 4 independent contexts |
| Exact accounting | 25 complete-domain + 15 canary = 40/40 matching `REAL FHE` calls; 0 failures | 4 containers × (3 excluded warmups + 16 measured); 64/64 measured successes; 0 failures |
| Simulation | 25/25 complete-domain rows match | not the reported timing population |
| Encrypt p50 / p95 | 3.147 / 3.214 ms | 3.636 / 4.142 ms |
| Server evaluate p50 / p95 | 362.091 / 374.514 ms | **544.536 / 830.709 ms** |
| Decrypt p50 / p95 | 1.266 / 1.308 ms | 1.603 / 2.277 ms |
| End-to-end p50 / p95 | 366.597 / 378.941 ms | **550.076 / 837.010 ms** |
| Evaluation key | digest `0f34877acac4e9f66c6338120c4144bb556b88ec1c558ef5b359028e1b17c475` | 472,645,952 B p50/p95 |
| Request / response | source bundle retains raw serialized calls | 131,336 B / 131,336 B p50/p95 |

The nonlinear result proves exact agreement over the complete declared 25-point integer domain and 15 randomized canaries for this source-scoped circuit. It does not establish policy efficacy. The timing quantiles condition on all 64 measured requests having succeeded; confidence intervals use a 2,000-repetition hierarchical container/request bootstrap, and p99 is not reported. They describe four independent colocated research contexts—not a shared context, production endpoint, throughput result, or “real-time” service. In both studies the secret key stayed within its dedicated worker, but client and server were colocated there: these records are `REAL FHE` circuit/cost evidence, not local-client/remote-server secrecy evidence.

## 8. Integrated flagship extension

The flagship adds action-time safety control and retrospective policy evaluation without changing the confidentiality boundary: the evaluator applies a frozen public computation to encrypted client inputs and returns ciphertext outputs; the client decrypts and performs the final non-polynomial decision.

### 8.1 Counterfactual safety shield

The client state is $s_t=(x,y,v_x,v_y,b,\theta)$. For every public candidate $a\in\{\text{BRAKE},\text{EAST},\text{WEST},\text{NORTH},\text{SOUTH}\}$, the server applies the same public polynomial dynamics for horizons $h\in\{1,2\}$. It returns the logical encrypted tensor

$$
M(s_t)\in\mathbb{Z}^{5\times 2\times 4},
$$

where the final axis contains obstacle/boundary, squared-speed, squared-tilt, and battery margins. Strict positivity is the safety convention. Complete discrete-domain analysis removes a spatial candidate only if it never attains the minimum. When several candidates remain, values saturate at a public signed-integer limit before each binary minimum lookup; monotonic sign-preserving saturation leaves strict-positive certificates unchanged. When one proved-active constraint remains, its exact margin needs no lookup or saturation.

For candidate $a$, conservative client buffers $\delta_{h,f}\ge 0$ yield

$$
\underline m(a)=\min_{h,f}\left(M_{a,h,f}-\delta_{h,f}\right).
$$

The client retains the requested action if all buffered obligations are positive. Otherwise it selects the certified candidate maximizing $\underline m(a)$, with exact ties broken by the frozen action order. If no candidate is certified, the client executes and records an explicitly uncertified BRAKE fallback. There is no server-side argmin.

The Concrete program evaluates encrypted tensors but does not claim SIMD packing. Its declared quantized state domain is $[-2,2]^6$, containing $5^6=15{,}625$ points, and its logical output remains the full $5\times2\times4$ margin tensor.

### 8.2 Private horizon-aware OPE

For trajectory $i$, requested logged action $A_{i,h}$, behavior propensity $\mu_{i,h}$, public target propensity $\pi_{i,h}$, reward $R_{i,h}$, and clip $\tau$, define cumulative and clipped importance weights

$$
\rho_{i,h}=\prod_{k=1}^{h}\frac{\pi_{i,k}(A_{i,k}\mid S_{i,k})}{\mu_{i,k}},
\qquad
\bar\rho_{i,h}=\min(\rho_{i,h},\tau).
$$

The encrypted server output is three additive horizon vectors,

$$
N_h=\sum_i \gamma^{h-1}\bar\rho_{i,h}R_{i,h},\qquad
D_h=\sum_i \bar\rho_{i,h},\qquad
C_h=\sum_i 1.
$$

The client decrypts and computes $\widehat V=\sum_h N_h/D_h$ only when every required denominator is positive. Counts make batch accounting explicit. Trajectory-percentile bootstrap intervals remain clear client-side evidence because aggregate $3H$ statistics do not support trajectory resampling.

The exact Concrete backend uses integer hard clipping and three encrypted $H$-vectors. Its executable proof shape is deliberately bounded to $(N=1,H=2,D=1)$ after the larger $(4,4,6)$ graph exceeded a 32 GiB Modal worker; it is a semantics/transport proof, not a scale result. The TenSEAL backend instead names `POLYNOMIAL_APPROX_OPE_V1`, uses slot-packed CKKS and polynomial soft clipping, and reports a separate approximation receipt. High-level TenSEAL provides no exact encrypted comparison/min primitive, so the CKKS path never claims exact hard-clip semantics.

### 8.3 Shield/OPE integration

The shield changes the transition kernel and is therefore part of the evaluated MDP. The behavior policy logs the sampled **requested action** and its propensity before shielding, while `executed_action` records the environment action after the many-to-one shield map. OPE evaluates the same frozen requested-action target separately in the shield-off, one-step, and two-step MDPs. Substituting executed-action propensity after shielding is prohibited because multiple requested actions can map to one executed action.

The integration experiment compares OPE estimates against paired direct online target-policy truth within each shield-defined MDP. Return effects and unsafe-event effects are noncompensating outcomes; encrypted execution evidence does not substitute for statistical validity, and statistical evidence does not substitute for privacy evidence.

### 8.4 Executed cryptographic canaries

The digest-pinned browser source is [`../site/data/flagship-evidence.json`](../site/data/flagship-evidence.json), SHA-256 `c19d256ee6fe6fc301e715155c3538cf3fdac28946b1e99eda3a0f4473aa5407`. Modal built it from immutable summaries `flagship-shield-canary-014`, `flagship-exact-ope-canary-004`, and `flagship-ckks-ope-canary-006`.

| Canary | Semantic check | Server evaluation | End-to-end / public context |
|---|---|---:|---:|
| CipherShield Concrete | 15,625/15,625 complete-domain simulations and 1/1 REAL FHE tensor match | 73.916 s | 77.491 s |
| exact Concrete OPE `(1,2,1)` | simulation = REAL = integer reference | 1.866 s | 6.329 s |
| CKKS OPE `(64,8,1)` | separately named approximation; 24 output ciphertexts | 7.465 s | 813,936,378 B public server context |

CipherShield used 758,473,160 B of evaluation keys, a 492,056 B request, and a 3,278,720 B response. Exact OPE used 589,084,056 B of evaluation keys, a 124,032 B request, and a 115,480 B response. CKKS used an 86,405,050 B request and 25,943,816 B response; the maximum released numerator error was 8.409 under a conservative declared soft-clip absolute-error bound of 32. These are cryptographic semantics/cost canaries, not empirical task efficacy or production systems results.

## 9. Threat model

The evaluator is honest-but-curious, runs a pinned data-independent circuit, and may observe policy/version, tensor shapes, parameter set, ciphertext and evaluation-key sizes, request timing/volume/status, and linkable evaluation-key identity. Assuming secure scheme parameters, fresh client randomness, uncompromised endpoints, and no decryption oracle, FHE computationally hides observation and encrypted-score values.

FHE does not provide result integrity. A malicious evaluator can return any ciphertext or valid but wrong score. Artifact hashes, HMAC envelopes, TLS, and signatures can authenticate a transcript; none proves exact FHE evaluation. The system also does not provide model confidentiality against adaptive query extraction, endpoint protection, availability, side-channel resistance, or secrecy of an action visible through environment effects.

## 10. Related work and novelty boundary

Suh and Tanaka study encrypted RL and comparison-free relative-entropy-regularized synthesis [1]. Encrypted RERL adds bootstrapping and error analysis for encrypted policy synthesis [2]. Nguyen et al. report homomorphic-encryption-compatible SAC training [3]. AutoFHE co-optimizes polynomial degree and bootstrapping placement for neural inference [4]. VIPER distills neural policies into compact verifiable trees [5]. Concrete already exposes quantization, simulation, compilation, and circuit error controls [6].

Accordingly, Unseen Loop does not claim novelty in “FHE + RL,” policy distillation, bit-width search, or the margin inequality alone. Its research unit is the integration of student-induced closed-loop evaluation, deployed-circuit coefficient bounds, counterexample feedback, exact discrete-domain enumeration, and real client/server ciphertext traces in one reproducible artifact.

The flagship likewise does not claim the first predictive safety shield, encrypted control system, private OPE estimator, or packed homomorphic workload. Predictive multi-action shielding and encrypted control are prior problem classes; private policy evaluation inherits the established importance-sampling and secure-computation literature. The narrower artifact contribution is the shared clear/encrypted semantics for a full five-action/two-horizon/four-family margin tensor, stable client-only selection, an additive `3H` private-OPE contract with client-only division, a separately named CKKS approximation, and a paired experiment that treats each shield as part of the MDP.

## 11. Limitations and remaining preregistration

1. The expanded efficacy evidence comprises five checkpoints per environment, eight policy candidates per checkpoint, and 50 selection episodes per candidate: 120 candidates and 6,000 selection rows in total. It does not execute the preregistered 120 candidates per checkpoint × 100 selection episodes per candidate: 1,800 candidates and 180,000 selection rows across 15 runs.
2. The expanded and factorial studies execute in clear. Their return, agreement, and certificate measurements provide no privacy or confidentiality evidence.
3. Acrobot regresses sharply, while the CartPole and MountainCar paired-return intervals include zero. The data do not support a cross-environment efficacy or generalization claim.
4. The positive factorial result is for the complete occupancy-refinement bundle in matched CartPole cells. It cannot identify one component as the cause or extend the effect to other tasks.
5. Certificate coverage is not 100%; behavior at uncertified states remains empirical. The domain certificate enumerates integer codes, not a continuous reachable-set proof.
6. The nonlinear and timing studies colocate client and server within each Modal worker. They exercise real ciphertext computation and cost, but not physical local-client/remote-server secrecy.
7. The four-context timing study completes the specified 12-warmup/64-measurement distribution. It does not measure a shared-context service, cold production traffic, throughput, peak RSS, or network separation.
8. Client-side argmax reveals returned scores to the client. Remote correctness remains trusted; authentication is not verifiable computation.
9. CipherShield certifies only the frozen one- or two-step public model. Model mismatch, sensor error beyond the declared buffer, later-horizon hazards, and malicious computation remain outside its guarantee.
10. Exact Concrete OPE is an `(N=1,H=2,D=1)` bounded semantics/transport canary. It is not an empirical-scale or latency result; the former `(4,4,6)` canary exceeded a 32 GiB worker.
11. CKKS OPE uses approximate soft clipping rather than exact hard clipping. Its H8 canary requires a degree-16384 context and large public evaluation material; it is not a production bandwidth or service result.
12. The flagship smoke and full manifests are separate. A completed canary publication does not by itself discharge the full clear shield matrix, OPE coverage/power cells, shield/OPE truth comparison, REAL-FHE challenge denominators, concurrency timing, or noncompensating release gates.

The executed expanded study is therefore distinct from completion of [`../experiments/release.toml`](../experiments/release.toml). Still outstanding are the full 120-candidates-per-checkpoint × 100-selection-episodes-per-candidate search (1,800 candidates / 180,000 selection rows), the preregistered 64-row physically remote client/server challenge, and the remaining stress/range/tie and release-wide gate matrix. A clear release-suite run provides no privacy evidence, and `modal_app.py::research --full` scales only one environment/checkpoint path. Exact executed-study, download, ledger-verification, and analysis commands are in [`reproduction.md`](reproduction.md).

## References

1. J. Suh and T. Tanaka. “Efficient Implementation of Reinforcement Learning over Homomorphic Encryption.” 2025. <https://arxiv.org/abs/2504.09335>
2. J. Suh and T. Tanaka. “Encrypted Relative-Entropy-Regularized Reinforcement Learning.” 2025. <https://arxiv.org/abs/2506.12358>
3. H. Nguyen et al. “Empowering artificial intelligence with homomorphic encryption for secure deep reinforcement learning.” *Nature Machine Intelligence*, 2025. <https://doi.org/10.1038/s42256-025-01135-2>
4. Y. Ao and S. Boddeti. “AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE.” *USENIX Security*, 2024. <https://www.usenix.org/conference/usenixsecurity24/presentation/ao>
5. O. Bastani, Y. Pu, and A. Solar-Lezama. “Verifiable Reinforcement Learning via Policy Extraction.” 2018. <https://arxiv.org/abs/1805.08328>
6. Zama. “Concrete ML: Compilation and production deployment.” <https://docs.zama.org/concrete-ml/explanations/compilation>
7. HomomorphicEncryption.org. “Homomorphic Encryption Security Standard.” <https://homomorphicencryption.org/standard/>
