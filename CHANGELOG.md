# Changelog

## 0.1.0 — 2026-08-23

### Research mechanism

- Added weighted degree-1/2 policy distillation with signed observation and coefficient quantization.
- Added analytical action-invariance certificates and exhaustive integer-box enumeration.
- Added counterexample-guided student-occupancy refinement, disjoint candidate-selection and post-selection evaluation namespaces, paired per-episode evidence rows, and range-valid Pareto filtering.
- Added NumPy CPU CEM teachers and Torch-vectorized GPU CartPole CEM.

### FHE and protocol

- Added Concrete-Python 2.10.0 compilation, clear simulation, real encryption/evaluation/decryption, and architecture-specific client/server serialization.
- Added compiler receipts with security/error configuration, range, bit width, complexity, sizes, and content hashes.
- Added fixed-shape HMAC-authenticated request/response envelopes, freshness/replay checks, policy/circuit/client/evaluation-key context binding, and fail-closed range policy.
- Added local-client/Modal-server FHE key separation; the HMAC authentication key is distinct from the decryption key, and no clear fallback is allowed under an FHE label.

### Cloud and evidence

- Added bounded Modal L4 training, CPU search, FHE compilation, remote ciphertext evaluation, and Volume persistence.
- Recorded `modal-smoke-001`: a 9.668 s L4 teacher search, 500/500 post-selection quantized-student/teacher return over paired evaluation seeds, 95.875% integer-student-occupancy certificate coverage, 98.7534% exhaustive box coverage, and 27 exact REAL FHE calls—25/25 sequential encrypted control steps plus two fresh-randomness canaries.
- Added `unseen-loop/modal-evidence-v2` with `closed_loop_real_fhe`, top-level and per-call authenticated-envelope metadata, same-input and secret-marker audits, and a nonsecret bundle of `evidence.json`, `receipt.json`, `server.zip`, `client-specs.bin`, `policy.json`, and their checksum ledger. Persisted/cloud evidence excludes plaintext private observations and decrypted score vectors.
- Recorded the earlier checksummed clear-only 3-environment × 5-checkpoint smoke matrix with 120 candidates and 240 retained episode rows; retained its unsolved MountainCar teachers and regressed Acrobot students without converting clear results into privacy claims.
- Recorded `expanded-multitask-modal-002`: all 15/15 clear environment/checkpoint runs, 120 candidates, 6,000 selection rows, 1,500 paired evaluations, and 3,000 long-form teacher/student rows. Paired student-minus-teacher return was CartPole −29.480 [−71.358, 0.072], MountainCar +0.796 [−1.620, 3.544], and Acrobot −231.896 [−388.536, −75.831]. The Acrobot regression and the two intervals spanning zero remain explicit; this is no privacy or cross-task efficacy claim.
- Recorded the matched clear CartPole 2×2 factorial: occupancy-refinement-bundle main effect +83.619 [26.144, 145.954], certificate-weighting main effect −108.461 [−288.649, 68.250], and interaction +8.654 [−180.737, 156.256]. The positive causal claim is restricted to the tested refinement bundle in matched CartPole cells; weighting's negative point estimate is inconclusive.
- Recorded `modal-nonlinear-qmax2-002`: a degree-2, two-feature/two-action circuit with three quadratic products per inference, 25/25 complete-domain rows plus 15/15 randomized canaries, for 40 exact `REAL FHE` calls and zero failures.
- Recorded `modal-fhe-timing-003`: four independent contexts, 12 recorded/excluded warmups, and 64/64 measured successes; server p50/p95 544.536/830.709 ms and end-to-end p50/p95 550.076/837.010 ms.
- Added `unseen-loop-release-analysis-004`, whose checksummed `publication.json`, evidence index, expanded tables, factorial contrasts/effects, and source-scoped FHE summaries bind every displayed denominator and digest. Its clear sources provide no privacy evidence, and its colocated FHE sources provide no local-client/remote-server secrecy evidence.
- Distinguished the completed expanded bounded studies (8 candidates per environment/checkpoint × 50 selection episodes) from the still-uncompleted full release search (120 candidates per environment/checkpoint × 100 selection episodes, 1,800 candidates / 180,000 rows total), 64-row physically remote challenge, and release-wide stress/gate matrix.

### Research product

- Added an interactive evidence report, recorded protocol trace, latency anatomy, trust matrix, circuit receipt, and claim-to-command reproduction surface.
- Added paper, threat model, architecture, preregistered benchmark, reproduction guide, implementation tutorial, and Modal operations guide.
- Added behavioral, protocol, artifact, RL-search, and actual ciphertext execution tests.
