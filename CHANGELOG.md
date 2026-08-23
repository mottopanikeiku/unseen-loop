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
- Recorded a checksummed clear-only 3-environment × 5-checkpoint conformance matrix with 120 candidates, 240 retained episode rows, and a deterministic hierarchical paired-return interval; retained negative MountainCar and Acrobot results without converting them into privacy claims.

### Research product

- Added an interactive evidence report, recorded protocol trace, latency anatomy, trust matrix, circuit receipt, and claim-to-command reproduction surface.
- Added paper, threat model, architecture, preregistered benchmark, reproduction guide, implementation tutorial, and Modal operations guide.
- Added behavioral, protocol, artifact, RL-search, and actual ciphertext execution tests.
