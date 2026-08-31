# Threat Model

## Security claim

Under a reviewed FHE parameter set, fresh client randomness, uncompromised endpoints, and an **honest-but-curious evaluator** running the committed fixed-shape circuit, Unseen Loop provides computational confidentiality of:

- the client's quantized observation value while encrypted; and
- the encrypted integer score value before client decryption.

The claim is against the evaluator's protocol view. It is not information-theoretic and does not extend to plaintext endpoint memory, environment effects, or public transcript metadata.

## Roles and trust boundaries

| Role | Holds | Must never hold |
|---|---|---|
| Development machine | clear training data, teacher, student weights, calibration data, compiler | release client secret keys |
| Local inference client | environment, observation, quantizer, FHE secret key, evaluation keys, decrypted scores, action | unpinned server artifact |
| Modal evaluator | plaintext policy circuit, `server.zip`, circuit receipt, HMAC authentication key, public evaluation material, ciphertext input/output, timing metadata | FHE secret/decryption key, plaintext observation, plaintext score |
| Artifact store | policy/circuit receipts, server artifact, client specifications, authenticated-protocol metadata, raw non-secret measurements | client keys, authentication keys, encryption randomness, plaintext private observations, decrypted score vectors |

The local client and Modal evaluator are separate program objects and processes. `modal_app.py::evaluate_ciphertext` imports `fhe.Server`, `fhe.Value`, and `fhe.EvaluationKeys`; it never constructs `fhe.Client`. The per-run HMAC key is operational transcript-authentication material shared with the evaluator, not the FHE decryption key, and is not persisted.

## Assets

1. **Observation and trajectory history** — value confidentiality before and during remote evaluation.
2. **Secret key and encryption randomness** — exclusive client possession; never serialized to function arguments, Modal Secrets, Volumes, logs, or artifacts.
3. **Evaluation and bootstrapping material** — not a decryption key, but integrity-sensitive and linkable; bounded lifetime and context binding are operational requirements.
4. **Policy weights and architecture** — clear on the evaluator. The client receives only black-box score outputs, but FHE does not make the function circuit-private or extraction-resistant.
5. **Encrypted response and decrypted action** — response confidential from evaluator before decryption; eventual action may be observable through environment effects.
6. **Circuit/version commitment** — integrity-sensitive. Hash pinning detects substitution relative to a trusted manifest but does not prove the server executed it.

## Adversaries

### A1: honest-but-curious evaluator — in scope

The evaluator follows the protocol and committed circuit but records its complete view. It tries to infer observation and output values from ciphertext, evaluation material, shapes, sizes, timings, traffic volume, status, policy, and repeated linkable sessions.

### A2: malicious network — partially in scope

A network adversary may replay, swap, truncate, corrupt, or downgrade envelopes. The implemented `authenticated-envelope-v1` protocol HMAC-authenticates canonical request and response documents and validates freshness, policy digest, circuit digest, client-context digest, evaluation-key digest, fixed shape, canonical payload length, a 1 MiB ciphertext safety cap, response-to-request binding, and schema. Random nonces are included and the local client rejects a nonce it generates twice.

After those checks, the serialized Modal evaluator atomically claims the authenticated request digest in `/artifacts/protocol/replay-ledger.json` and commits the shared Volume before deserializing any FHE input. Claims survive evaluator container restarts and are retained for ten minutes, beyond the five-minute freshness window, so every request still eligible for evaluation remains replay-protected. The HMAC assumes the per-run authentication key remains secret from the network adversary; it demonstrates message authentication but is not a replacement for deployment transport security. Traffic analysis remains out of scope.

### A3: malicious evaluator — out of scope

A malicious evaluator can:

- skip the circuit;
- evaluate another circuit;
- return a stale, random, chosen, or valid-but-wrong ciphertext;
- selectively fail based on public metadata;
- mount availability and timing attacks.

FHE is malleable and provides no computation integrity. TLS, HMAC, signatures, nonces, and artifact hashes are **not** proofs of evaluation. A malicious-evaluator claim requires an implemented proof system or verifiable FHE mechanism benchmarked for the exact circuit.

### A4: adaptive client — out of scope for model privacy

A client can choose quantized inputs and observe complete decrypted score vectors. Query restrictions, domain guards, fixed shapes, and quotas reduce abuse but do not prevent model extraction. Noncanonical but valid ciphertexts may expand the attack surface if the server does not bind input ranges and context.

### A5: compromised endpoint or cloud host — out of scope

FHE does not protect plaintext already resident in client memory, secret keys stolen from the client, policy weights read from a compromised evaluator host, unsafe native library behavior, speculative-execution leakage, or crash dumps containing secrets.

## Flagship protocol assets and disclosure

CipherShield and private OPE reuse the honest-but-curious boundary but protect different plaintexts:

| Protocol | Client-private during evaluation | Evaluator view | Optional client release |
|---|---|---|---|
| CipherShield-RL | six-feature physical state, secret key, decrypted `5 × 2 × 4` margins, final stable selection | public dynamics/limits/candidates, tensor shape, evaluation material, ciphertext sizes/digests, timing | redacted decision receipt or explicitly labeled derived replay geometry |
| exact OPE | logged states, requested actions, rewards, behavior propensities, secret key, decrypted `3H` integers | public batch shape, target propensity polynomial, clipping/scale constants, ciphertext metadata | per-horizon integer/decoded aggregates and client-derived estimate |
| CKKS OPE | same logged row fields and secret key | public `POLYNOMIAL_APPROX_OPE_V1` circuit, CKKS parameters, slot count, ciphertext metadata | approximate per-horizon aggregates plus clear comparison/error evidence |

An explicit release changes confidentiality: `CLIENT_RELEASED_DERIVED_GEOMETRY` makes the plotted positions and candidate tubes public after evaluation; `CLIENT_RELEASED_STATISTICS` makes the named decrypted sufficient statistics public. Neither disclosure implies that the evaluator decrypted them.

The shield server does not select an action. The OPE server does not divide sufficient statistics. FHE does not hide the public candidate set, dynamics, target model, batch shape, circuit topology, parameter set, or transport sizes. CKKS additionally exposes approximate-arithmetic parameters and carries approximation error; it is never substituted for the exact hard-clip semantics without its distinct identifier.

## Guarantee matrix

| Property | Status | Conditions / leakage |
|---|---|---|
| Observation value confidentiality from evaluator | Provided | secure FHE parameters, fresh randomness, correct implementation, no endpoint compromise |
| Encrypted score confidentiality from evaluator | Provided | only client decrypts; evaluator may see response size/timing |
| Secret-key separation | Enforced by architecture and tests | local client only; evaluator API has no secret-key parameter |
| Circuit/policy integrity in artifact | Detectable | content digests relative to trusted receipt |
| Correct remote computation | Trusted, not proved | honest evaluator assumption |
| Policy confidentiality from cloud host | Not provided | evaluator stores plaintext circuit/weights |
| Policy confidentiality from client | Black-box only | score-vector queries permit extraction |
| Action secrecy | Conditional | evaluator does not decrypt scores; environment behavior may reveal action |
| Shape and size privacy | Not provided | input shape, request/response and evaluation-key bytes are recorded |
| Timing/access-pattern privacy | Not provided | episode cadence, length, status, cold/warm latency leak |
| Availability | Not provided | quotas/timeouts limit cost, not denial of service |
| Differential privacy | Not provided | no DP mechanism or privacy accounting |
| Private training | Not provided | training and distillation are clear |
| Side-channel resistance | Not evaluated | library and host implementation dependent |

## Implemented protocol

The client creates a `RequestEnvelope` containing:

- schema version;
- random request ID and nonce;
- creation time;
- policy and circuit digests;
- client-context and evaluation-key digests;
- fixed observation shape;
- ciphertext byte length and canonical base64 payload.

`TranscriptAuthenticator` signs the canonical payload with HMAC-SHA256. The Modal evaluator verifies the signed wrapper, then `FixedShapeGuard` checks freshness, context, shape, and fixed length. It claims the request digest in the durable replay ledger before releasing ciphertext bytes to `fhe.Server`.

The evaluator's `ResponseEnvelope` binds the request digest, ID and nonce, policy/circuit digests, fixed output shape, status, completion time, ciphertext length, and canonical base64 payload. It HMAC-signs that response. The client verifies the signature and request/context binding before decryption. The research entrypoint then requires exact equality with its local integer-clear score vector before stable argmax; a deployment client without that oracle must independently enforce the declared decrypted shape/range and fail closed.

This is operational transcript integrity, not evaluation integrity. An evaluator holding the HMAC key can authenticate an incorrect result.

## Required negative tests

| ID | Attack | Required result |
|---|---|---|
| TM-01 | Replay identical request nonce | reject before evaluator invocation |
| TM-02 | Swap response between request IDs | reject before decryption |
| TM-03 | Change policy/circuit digest | reject downgrade/substitution |
| TM-04 | Wrong context/evaluation-key digest | reject context confusion |
| TM-05 | Truncated or oversized base64 payload | reject canonical-length mismatch or 1 MiB cap violation |
| TM-06 | Stale/future request clock | reject freshness violation |
| TM-07 | Valid authentication over wrong encrypted result | authenticate transcript but document that correctness is not established |
| TM-08 | Wrong-shape decrypted result | fail closed; do not actuate |
| TM-09 | Out-of-domain plaintext before encryption | reject; never silently wrap |
| TM-10 | Same observation encrypted twice | ciphertext hashes differ; decrypted integer result agrees |
| TM-11 | Scan `server.zip` names for secret-key markers | zero markers |
| TM-12 | Inspect evaluator function signature | server artifact, signed request JSON, evaluation keys, authentication key, and public receipt only; no FHE client or decryption key |
| TM-13 | FHE boundary/tie canaries | real output equals exact integer clear or release fails |
| TM-14 | Adaptive extraction budget | report fidelity versus query budget before model-privacy claims |

Focused suites exercise replay rejection plus protocol swap/downgrade/freshness/shape checks, range rejection, randomized-ciphertext canaries, server-artifact secret-marker scans, and real-FHE equality. TM-07 is a documented expected limitation. Endpoint memory, extraction, and production transport audits require deployment-specific tests.

## Logging and artifacts

Safe to record:

- mode label;
- compiler/library version;
- public parameter and error configuration;
- policy/circuit/server hashes;
- tensor shape;
- ciphertext, evaluation-key, and artifact sizes;
- ciphertext hashes;
- phase timings;
- Boolean equality against a local oracle in a dedicated test.

Never record:

- client secret key or keyset;
- CSPRNG seed/randomness;
- plaintext private observations in persisted/cloud evidence or server logs;
- decrypted score vectors in persisted/cloud evidence or evaluator logs;
- raw client memory or crash dumps;
- HMAC authentication keys or API secrets.

`unseen-loop/modal-evidence-v2` records actions/rewards and nonsecret protocol metadata for the 25-step client-driven prefix, but no plaintext observation or decrypted score vector. Its `authenticated_envelope_protocol` descriptor and per-call protocol objects retain only schema/algorithm labels and envelope/context digests. The checksummed bundle is limited to `evidence.json`, `receipt.json`, `server.zip`, `client-specs.bin`, `policy.json`, and `checksums.sha256`; evaluation-key bytes and key material are not persisted. The separate replay ledger persists only authenticated request digests and claim times and is pruned after ten minutes. Evaluation material is cryptographically public in the selected protocol but operationally identifying, so evidence records only its size and digest. The server-archive audit scans filenames for secret-key markers; a clean scan is a useful gate, not a proof that arbitrary bytes contain no secret.

## Honest deployment statement

> Unseen Loop protects encrypted observation and score values from an uncompromised honest-but-curious evaluator under the selected FHE scheme and parameter assumptions. It exposes shapes, sizes, timing, policy/version, and repeated-session metadata. It trusts the evaluator for correct computation, does not protect clear training, and does not prevent model extraction or endpoint compromise.
