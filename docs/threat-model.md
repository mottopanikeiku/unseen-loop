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
| Modal evaluator | plaintext policy circuit, `server.zip`, public evaluation material, ciphertext input/output, timing metadata | secret/decryption key, plaintext observation, plaintext score |
| Artifact store | policy/circuit receipts, server artifact, client specifications, raw non-secret measurements | client keys, encryption randomness, decrypted private observations |

The local client and Modal evaluator are separate program objects and processes. `modal_app.py::evaluate_ciphertext` imports `fhe.Server`, `fhe.Value`, and `fhe.EvaluationKeys`; it never constructs `fhe.Client`.

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

A network adversary may replay, swap, truncate, corrupt, or downgrade authenticated envelopes. A deployment must use authenticated transport and validate the request nonce, context digest, policy digest, circuit digest, fixed shape, fixed byte lengths, output schema, and freshness. The repository's transcript HMAC demonstrates these checks.

Traffic analysis remains out of scope. Authentication material is operational and distinct from the FHE secret key.

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

## Protocol requirements

A release request envelope binds:

- schema version;
- random request ID and nonce;
- creation time;
- policy and circuit digests;
- client-context and evaluation-key digests;
- fixed observation shape;
- ciphertext byte length and payload digest.

The response binds request digest/ID/nonce, policy/circuit digest, fixed output shape, status, completion time, ciphertext length, and payload digest. The client verifies the response before decryption, validates the decrypted vector's exact shape and integer range, applies stable argmax, and fails closed.

`FixedShapeGuard` intentionally describes transcript authentication as operational integrity, not evaluation integrity.

## Required negative tests

| ID | Attack | Required result |
|---|---|---|
| TM-01 | Replay identical request nonce | reject before evaluator invocation |
| TM-02 | Swap response between request IDs | reject before decryption |
| TM-03 | Change policy/circuit digest | reject downgrade/substitution |
| TM-04 | Wrong context/evaluation-key digest | reject context confusion |
| TM-05 | Truncated/oversized base64 payload | reject fixed-length violation |
| TM-06 | Stale/future request clock | reject freshness violation |
| TM-07 | Valid authentication over wrong encrypted result | authenticate transcript but document that correctness is not established |
| TM-08 | Wrong-shape decrypted result | fail closed; do not actuate |
| TM-09 | Out-of-domain plaintext before encryption | reject; never silently wrap |
| TM-10 | Same observation encrypted twice | ciphertext hashes differ; decrypted integer result agrees |
| TM-11 | Scan `server.zip` names for secret-key markers | zero markers |
| TM-12 | Inspect evaluator function signature | only server bytes, ciphertext bytes, evaluation-key bytes |
| TM-13 | FHE boundary/tie canaries | real output equals exact integer clear or release fails |
| TM-14 | Adaptive extraction budget | report fidelity versus query budget before model-privacy claims |

The repository executes TM-01/02/03/05/09/10/11/13 in its automated suite. TM-07 is a documented expected limitation. Endpoint memory, extraction, and production transport audits require deployment-specific tests.

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
- clear private observation in server logs;
- decrypted score/action in evaluator logs;
- raw client memory or crash dumps;
- authentication or API secrets.

Evaluation material is cryptographically public in the selected protocol but operationally identifying. Release evidence records only its size and digest unless exact publication is required.

## Honest deployment statement

> Unseen Loop protects encrypted observation and score values from an uncompromised honest-but-curious evaluator under the selected FHE scheme and parameter assumptions. It exposes shapes, sizes, timing, policy/version, and repeated-session metadata. It trusts the evaluator for correct computation, does not protect clear training, and does not prevent model extraction or endpoint compromise.
