# Preregistered Benchmark Protocol

This document separates the committed quick artifact from the evidence required for a paper-level release. Thresholds must be changed before a run, never after inspecting its outcome.

## Workloads

| Environment | Purpose | Actions | Horizon | Required property |
|---|---|---:|---:|---|
| CartPole-v1 | low-dimensional exhaustive certificate and conformance | 2 | 500 | full signed-code enumeration |
| MountainCar-v0 | sparse reward and boundary-sensitive decisions | 3 | 200 | success-rate and first-divergence analysis |
| Acrobot-v1 | long-horizon multi-action control | 3 | 500 | margin deciles and student-occupancy shift |

All use deterministic greedy discrete policies for the certificate contract. Pendulum-v1 is an optional continuous-action comparison and must report action-error bounds rather than action invariance.

The executable release manifest is [`../experiments/release.toml`](../experiments/release.toml), schema `unseen-loop/release-suite-v1`. Materialize all declared environment/checkpoint runs with:

```bash
uv run unseen-loop suite \
  --config experiments/release.toml \
  --backend clear \
  --output artifacts/release
```

This typed suite command is the release-matrix orchestrator. With `--backend clear` it materializes every declared environment/checkpoint and paired episode row, but provides no privacy evidence and does not itself execute the manifest's FHE challenge, stress, ablation, or repeated-container timing requirements. `unseen-loop research` and `modal_app.py::research --full` each scale only one environment/checkpoint path and must not be described as the preregistered suite.

## Checkpoints and seeds

- five independently trained checkpoints per environment;
- checkpoints selected by a precommitted rule, never best-of-seed after evaluation;
- content-derived, mutually disjoint namespaces for training, calibration, distillation, refinement, **candidate selection**, post-selection evaluation, stress, real-FHE challenges, timing order, and bootstrap resampling;
- 100 selection episodes per candidate; Pareto filtering and champion choice use only integer-student trajectories from this namespace;
- after selection is frozen, 100 paired evaluation episodes per checkpoint for float teacher and exact integer student, with compiled simulation evaluated against the same frozen policy;
- real FHE runs on preregistered observations and closed-loop prefixes, with every attempt retained.

Seed derivation hashes:

```text
unseen-loop/v1 | root | environment | purpose | index
```

Cryptographic randomness is never deterministic or seeded for reproducibility.

## Frozen datasets

Per checkpoint:

1. **Distillation/refinement:** clear teacher targets and integer-student counterexample trajectories from their own namespaces.
2. **Selection:** 100 integer-student episodes per candidate. Candidate utility, constraint cost, teacher agreement, and certificate coverage are computed from exact student-induced occupancy; a range-saturating candidate is ineligible for the Pareto frontier and championship.
3. **Post-selection evaluation:** 100 fresh paired episode seeds used only after the champion is frozen. Persist one `FLOAT TEACHER` and one `QUANTIZED CLEAR` long-form episode row per seed. Headline return, constraint cost, agreement, and certificate coverage come only from these rows and the selected integer student's occupancy.
4. **Calibration/compile:** at least 4,096 teacher and student-occupancy observations plus axis extrema and in-domain range probes.
5. **Common fidelity:** 10,000 held-out observations, half from teacher occupancy and half from integer-student occupancy; preserve episode and step IDs.
6. **Stress:** quantizer half-steps, feature extrema, low-margin states, ties, perturbed reached states, and one-step-outside-domain cases marked `expected_reject`.
7. **Real challenge:** preselect uniform, low-margin, input-range, and stress codes before FHE execution.

Calibration, selection, and evaluation namespaces may not overlap. Calibration may not use fidelity, stress, or final evaluation rows.

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

Search uses integer clear and compiled simulation. Range-invalid candidates remain explicit records but are excluded from Pareto and champion eligibility. Every eligible Pareto finalist used in a headline receives real ciphertext measurements; compiler failures also remain explicit rows.

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

- post-selection undiscounted return from paired evaluation seeds;
- episode length and termination/truncation;
- environment success;
- predefined constraint cost;
- teacher action agreement measured at the selected integer student's reached states;
- certificate coverage measured on that same integer-student occupancy;
- time to first uncertified state and action divergence.

Report mean, standard deviation, median, and IQR per checkpoint from the persisted paired episode rows. Primary policy comparison uses paired evaluation seeds and a hierarchical bootstrap over checkpoint then episode. Across environments report interquartile mean and stratified bootstrap interval; never normalize to the best observed mode. Selection-seed metrics must never be relabeled as post-selection evaluation.

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
| Certificate | ≥99% post-selection evaluation coverage under integer-student occupancy, zero certified mismatches |
| Domain safety | all outside-domain cases rejected before encryption; no wraparound |
| Policy noninferiority | lower confidence bound on paired return delta above precommitted per-environment tolerance |
| Local viability | canonical compile/keygen within resource cap and warm end-to-end ≤60 s |
| Modal viability | every request returns; recorded SKU/image; warm end-to-end ≤60 s |
| Reproducibility | deterministic artifact hashes and summary regenerate from clean commit |

The quick `modal-smoke-001` record does **not** pass the ≥99% post-selection integer-student-occupancy certificate target (95.875%) and has only one teacher checkpoint. Its 25/25 encrypted prefix is conformance evidence; 24/25 reached codes certify. This is not a completed release study or a latency distribution.

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
