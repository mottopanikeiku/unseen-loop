# Tutorial: From RL Scores to an Encrypted Action

This tutorial uses one frozen policy object across every semantic stage. It deliberately keeps argmax on the client and labels simulation as clear execution.

## 1. Collect a teacher occupancy dataset

```python
from unseen_loop.teacher import collect_trajectories, train_cem_teacher

teacher, history = train_cem_teacher(
    "CartPole-v1",
    seed=7,
    hidden_size=16,
    iterations=24,
    population=64,
)
trace = collect_trajectories("CartPole-v1", teacher, seeds=(101, 102, 103, 104))

print(trace.observations.shape)  # steps × 4
print(trace.scores.shape)        # steps × 2
```

The teacher is clear. Training is not privacy-preserving.

## 2. Fit an integer policy

```python
from unseen_loop.policy import fit_polynomial_policy

policy, diagnostics = fit_polynomial_policy(
    trace.observations,
    trace.scores,
    env_id="CartPole-v1",
    name="tutorial-d2-x4-w8",
    degree=2,
    input_bits=4,
    coefficient_bits=8,
    ridge=1e-3,
)

print(policy.spec.digest)
print(policy.estimated_output_bits)
print(policy.encrypted_multiplications)
```

`PolicySpec` freezes centers, steps, valid integer range, clear coefficients, integer coefficients, scale, degree, and output actions. Save `policy.spec.to_json()`; never serialize a mutable Python model as the release contract.

## 3. Separate deterministic errors

```python
q = policy.quantize(trace.observations)
float_student = policy.float_scores_from_quantized(q)
integer_scores = policy.integer_scores_from_quantized(q)
integer_dequantized = policy.dequantized_integer_scores(q)
```

`float_student → integer_dequantized` isolates coefficient rounding. No encryption has happened.

## 4. Certify action invariance

```python
from unseen_loop.certificate import certify_actions

certificate = certify_actions(policy, q, global_p_error=1e-6)
certificate.assert_sound()

print(certificate.coverage)
print(certificate.mismatches)
print(certificate.certified_mismatches)  # must be zero
```

For each state, `certificate.error_bounds` is analytical coefficient error, not maximum observed FHE error. The Boolean obligation is `margin > 2 * error_bound`.

For a low-dimensional integer region:

```python
from unseen_loop.certificate import certify_quantized_box

box = certify_quantized_box(policy, max_points=1_000_000)
print(box.points, box.coverage, box.complete)
```

`box.complete` is true only when every integer code certifies. An exhaustive enumeration with coverage below 1.0 is still valuable but is not a complete action guarantee.

## 5. Compile the exact integer program

Install the optional runtime:

```bash
uv sync --extra fhe
```

Compile on signed-range corners and axis extrema:

```python
import itertools
import numpy as np
from unseen_loop.fhe_backend import compile_policy

qmax = policy.spec.quantizer.qmax
n = policy.spec.quantizer.n_features
corners = np.asarray(tuple(itertools.product((-qmax, qmax), repeat=n)), dtype=np.int64)
compiled = compile_policy(policy, corners, "artifacts/tutorial-circuit", global_p_error=1e-6)

print(compiled.receipt.to_json())
```

The receipt includes compiler version, category-128 configuration, error target, range, max bit width, complexity, hashes, sizes, and a server archive filename scan.

## 6. Run clear compiler simulation

```python
sample = q[0]
expected = policy.integer_scores_from_quantized(sample)
simulated = compiled.simulate(sample)
np.testing.assert_array_equal(simulated, expected)
```

This proves compiler semantic agreement for the sample. It is **not** encrypted execution, privacy evidence, or a latency proxy.

## 7. Run a serialized real-FHE roundtrip

```python
measurement = compiled.real_roundtrip(sample)
assert measurement.backend == "REAL FHE"
assert measurement.output_matches_clear
assert not measurement.server_secret_key_present
print(measurement.to_json())
```

Internally:

1. `ClientSpecs.deserialize` reconstructs only client requirements;
2. `Client.keys.generate` creates secret and evaluation keys;
3. `Client.encrypt` creates a randomized ciphertext;
4. `Server.load` reads `server.zip`;
5. the server receives deserialized ciphertext and evaluation material only;
6. `Server.run` returns ciphertext;
7. the client decrypts integer scores.

Call the roundtrip twice and require distinct request hashes. This is a canary for fresh randomized encryption, not a proof of semantic security.

## 8. Advance a closed loop

The production client performs:

```python
observation, _ = env.reset(seed=episode_seed)
while True:
    q = policy.quantize(observation, reject=True)
    encrypted_scores = remote_server(policy, q)  # serialized protocol
    scores = local_client.decrypt(encrypted_scores)
    action = int(np.argmax(scores))               # stable first-index tie rule
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Never decrypt on the evaluator. Never send browser plaintext to a hosted demo. Never catch a missing-FHE error and run clear while retaining an FHE label.

## 9. Authenticate and bind the remote transcript

The Modal path uses `RequestEnvelope`, `ResponseEnvelope`, `SignedEnvelope`, `TranscriptAuthenticator`, and `FixedShapeGuard` rather than sending a bare ciphertext:

```python
import hashlib
import secrets

from unseen_loop.protocol import (
    FixedShapeGuard,
    RequestEnvelope,
    ResponseEnvelope,
    SignedEnvelope,
    TranscriptAuthenticator,
)

authentication_key = secrets.token_bytes(32)  # distinct from the FHE secret key
authenticator = TranscriptAuthenticator(authentication_key)
request = RequestEnvelope.create(
    serialized_ciphertext,
    policy_digest=policy.spec.digest,
    circuit_digest=compiled.receipt.server_artifact_sha256,
    client_context_digest=compiled.receipt.client_specs_sha256,
    evaluation_key_digest=hashlib.sha256(evaluation_keys).hexdigest(),
    observation_shape=(policy.spec.quantizer.n_features,),
)
signed_request_json = authenticator.sign(request).to_json()

# The serialized evaluator verifies the HMAC/guard, durably claims request.digest
# in its ten-minute replay ledger, and only then calls Server.run.
# It returns signed_response_json containing encrypted scores.
signed_response = SignedEnvelope.from_json(server_record["signed_response_json"])
response = authenticator.verify(signed_response, ResponseEnvelope)
assert isinstance(response, ResponseEnvelope)
guard = FixedShapeGuard(
    policy_digest=policy.spec.digest,
    circuit_digest=compiled.receipt.server_artifact_sha256,
    client_context_digest=compiled.receipt.client_specs_sha256,
    evaluation_key_digest=hashlib.sha256(evaluation_keys).hexdigest(),
    observation_shape=(policy.spec.quantizer.n_features,),
    output_shape=(policy.spec.actions,),
    request_bytes=len(serialized_ciphertext),
    response_bytes=response.ciphertext_bytes,
)
encrypted_scores = guard.validate_response(request, response)
```

HMAC authentication covers the canonical ciphertext-bearing envelopes and detects substitution/context confusion before decryption. The durable request-digest claim rejects replay across RPCs and evaluator container restarts for longer than the request-freshness window. Neither mechanism proves that an evaluator holding the authentication key ran the committed circuit. Never persist the authentication key, plaintext private observation, or decrypted score vector in cloud evidence.
