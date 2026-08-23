# Preregistered Benchmark Protocol

This document separates the committed quick artifact from the evidence required for a paper-level release. Thresholds must be changed before a run, never after inspecting its outcome.

## Workloads

| Environment | Purpose | Actions | Horizon | Required property |
|---|---|---:|---:|---|
| CartPole-v1 | low-dimensional exhaustive certificate and conformance | 2 | 500 | full signed-code enumeration |
| MountainCar-v0 | sparse reward and boundary-sensitive decisions | 3 | 200 | success-rate and first-divergence analysis |
| Acrobot-v1 | long-horizon multi-action control | 3 | 500 | margin deciles and student-occupancy shift |

All use deterministic greedy discrete policies for the certificate contract. Pendulum-v1 is an optional continuous-action comparison and must report action-error bounds rather than action invariance.

## Checkpoints and seeds

- five independently trained checkpoints per environment;
- checkpoints selected by a precommitted rule, never best-of-seed after evaluation;
- content-derived, disjoint namespaces for training, calibration, distillation, refinement, evaluation, stress, real-FHE challenges, timing order, and bootstrap resampling;
- 100 paired evaluation episodes per checkpoint for float teacher, float student, integer clear, and compiled simulation;
- real FHE runs on preregistered observations and closed-loop prefixes, with every attempt retained.

Seed derivation hashes:

```text
unseen-loop/v1 | root | environment | purpose | index
```

Cryptographic randomness is never deterministic or seeded for reproducibility.

## Frozen datasets

Per checkpoint:

1. **Calibration/compile:** at least 4,096 teacher and student-occupancy observations plus axis extrema and in-domain range probes.
2. **Common fidelity:** 10,000 held-out observations, half from teacher occupancy and half from integer-student occupancy; preserve episode and step IDs.
3. **Stress:** quantizer half-steps, feature extrema, low-margin states, ties, perturbed reached states, and one-step-outside-domain cases marked `expected_reject`.
4. **Real challenge:** preselect uniform, low-margin, input-range, and stress codes before FHE execution.

Calibration may not use fidelity, stress, or final evaluation rows.

## Execution modes

| Mode | Required semantics |
|---|---|
| `FLOAT TEACHER` | frozen clear teacher in evaluation mode |
| `FLOAT STUDENT` | fitted polynomial coefficients on fixed integer feature codes |
| `QUANTIZED CLEAR` | exact exported integer coefficients and feature program |
| `FHE SIMULATED` | same compiled graph in clear simulation; never privacy/latency evidence |
| `REAL FHE` | serialized client encryption, server evaluation, client decryption |

The quantizer, policy digest, output vector, stable argmax, and candidate checkpoint remain identical across the last three modes.

## Search grid

Primary axes:

- degree: 1 and 2;
- signed observation width: 3, 4, 5, and 6 bits;
- signed coefficient width: 3, 4, 6, 8, and 10 bits;
- ridge: $10^{-4}$, $10^{-3}$, and $10^{-2}$;
- certificate refinement: on/off and 0–3 rounds;
- calibration padding: fixed before launch;
- whole-circuit error target: $10^{-6}$ headline, $10^{-3}$ diagnostic ablation.

Search uses integer clear and compiled simulation. Every Pareto finalist used in a headline receives real ciphertext measurements. Compiler failures remain explicit rows.

## Baselines

### Policy/utility

- float teacher;
- unweighted score ridge on teacher-only occupancy;
- behavior cloning on teacher actions;
- certificate-guided weighting disabled;
- student-occupancy refinement disabled;
- degree 1 versus degree 2;
- uniform bit-width sweeps;
- proxy circuit cost versus measured finalist cost.

### Execution

- float student;
- exact integer clear;
- compiled simulation;
- real FHE serialized client/server;
- client-side argmax only in the primary track.

An in-circuit argmax or another FHE scheme is a separate circuit/leakage track, not a drop-in timing comparison.

## Metrics

### Pointwise

- exact integer input hash equality;
- float-student versus integer-clear score MAE/max error;
- integer-clear versus simulation exact equality;
- simulation versus real decrypted exact equality;
- stable-argmax agreement;
- top-two margin and certificate result;
- disagreement by margin decile;
- saturation/range-rejection rate;
- first divergent step under paired rollouts.

### Closed loop

- undiscounted return;
- episode length and termination/truncation;
- environment success;
- predefined constraint cost;
- teacher action agreement;
- certified occupancy coverage;
- time to first uncertified state and action divergence.

Report mean, standard deviation, median, and IQR per checkpoint. Primary policy comparison uses paired episode seeds and a hierarchical bootstrap over checkpoint then episode. Across environments report interquartile mean and stratified bootstrap interval; never normalize to the best observed mode.

### Systems

- compile wall and peak RSS;
- key generation and evaluation-key serialization;
- client encryption/decryption;
- server cold and warm evaluation;
- client-observed online end-to-end latency;
- request, response, evaluation-key, client-specification, and server-artifact bytes;
- compiler complexity and maximum integer bit width;
- encrypted multiplication and available PBS/TLU statistics;
- exact CPU/GPU SKU, thread controls, image/dependency/commit hashes, and Modal region/call ID.

Use at least four independent warm evaluator containers with 16 shuffled measured requests each for p50/p95. Three warmups per container are recorded but excluded. Do not report p99 from 64 requests.

## Statistical reporting

- paired mean return difference with hierarchical 95% bootstrap interval;
- Wilson interval for agreement/certificate proportions clustered or bootstrapped by episode;
- exact mismatch numerator/denominator for real FHE;
- no empirical claim that a small trial validates `global_p_error`;
- every failed/timeout attempt remains in the denominator;
- no relative error as the primary metric near zero scores.

## Release gates

| Gate | Threshold |
|---|---|
| Completeness | all preregistered IDs/attempts present; no silent replacement |
| Compiled semantics | 100% integer-clear/simulation integer equality |
| Real correctness | every decrypted real output equals simulation/integer clear |
| Security configuration | category 128, unsafe features off, insecure key cache off, `global_p_error≤10^-6` |
| Secret separation | no secret key in evaluator API, image, Volume, server artifact, logs, or evidence |
| Ciphertext canary | repeated same-input encryptions have distinct hashes and equal decrypted output |
| Discrete action fidelity | ≥99% overall teacher action agreement; every checkpoint ≥97% |
| Certificate | ≥99% held-out occupancy, zero certified mismatches |
| Domain safety | all outside-domain cases rejected before encryption; no wraparound |
| Policy noninferiority | lower confidence bound on paired return delta above precommitted per-environment tolerance |
| Local viability | canonical compile/keygen within resource cap and warm end-to-end ≤60 s |
| Modal viability | every request returns; recorded SKU/image; warm end-to-end ≤60 s |
| Reproducibility | deterministic artifact hashes and summary regenerate from clean commit |

The quick `modal-smoke-001` record does **not** pass the ≥99% held-out certificate target (94.1%) and has only one teacher checkpoint. It is a strong systems/conformance artifact, not a completed release study.

## Invalid comparisons

Reject:

- simulation time labeled FHE latency;
- simulation labeled secure;
- float model-core timing compared with FHE setup+network timing;
- GPU float versus CPU FHE called isolated cryptographic overhead;
- changed checkpoint, quantizer, graph, output, batch, hardware, or threads across modes;
- return differences interpreted as numerical error after trajectories diverge;
- only mean ± standard deviation across a few correlated steps;
- retry success replacing a failed attempt;
- zero observed failures called proof of error-free FHE;
- server-clear weights called an encrypted model;
- input-private inference called private RL training.
