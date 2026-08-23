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

## Planned one-command paths

```bash
# Fast deterministic local research smoke; no privacy claim.
uv run unseen-loop demo --backend clear --output artifacts/demo

# Compile and execute a real encrypted step locally.
uv sync --extra fhe
nix_or_linux_command='uv run unseen-loop demo --backend fhe --output artifacts/fhe-smoke'

# Train/search on Modal, compile finalists on CPU, and persist signed receipts.
modal run modal_app.py::research --env-id CartPole-v1 --seeds 0,1,2,3,4
```

The commands are activated as their implementation lands; CI never aliases a missing FHE runtime to a clear backend.

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

## License

MIT for this repository. Concrete Python is an optional external runtime with its own license and patent terms; review them before commercial use.
