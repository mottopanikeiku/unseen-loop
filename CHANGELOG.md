# Changelog

## 0.1.0 — 2026-08-23

### Research mechanism

- Added weighted degree-1/2 policy distillation with signed observation and coefficient quantization.
- Added analytical action-invariance certificates and exhaustive integer-box enumeration.
- Added counterexample-guided student-occupancy refinement and Pareto filtering across utility, stability, and circuit cost.
- Added NumPy CPU CEM teachers and Torch-vectorized GPU CartPole CEM.

### FHE and protocol

- Added Concrete-Python 2.10.0 compilation, clear simulation, real encryption/evaluation/decryption, and architecture-specific client/server serialization.
- Added compiler receipts with security/error configuration, range, bit width, complexity, sizes, and content hashes.
- Added fixed-shape authenticated envelopes, freshness/replay checks, context binding, and fail-closed range policy.
- Added local-client/Modal-server key separation; no clear fallback under an FHE label.

### Cloud and evidence

- Added bounded Modal L4 training, CPU search, FHE compilation, remote ciphertext evaluation, and Volume persistence.
- Recorded `modal-smoke-001`: 469/500 quantized-student/teacher return, 94.1% held-out certificate coverage, 98.9867% exhaustive box coverage, and 2/2 exact real-FHE trials.
- Added checksum-ledger experiment artifacts and a committed raw Modal evidence record.

### Research product

- Added an interactive evidence report, recorded protocol trace, latency anatomy, trust matrix, circuit receipt, and claim-to-command reproduction surface.
- Added paper, threat model, architecture, preregistered benchmark, reproduction guide, implementation tutorial, and Modal operations guide.
- Added behavioral, protocol, artifact, RL-search, and actual ciphertext execution tests.
