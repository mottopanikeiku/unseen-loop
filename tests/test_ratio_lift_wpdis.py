from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from unseen_loop.crypto.ckks import (
    CKKSClient,
    CKKSEncryptedVector,
    CKKSServer,
    SerializedCKKSVector,
    generate_contexts,
)
from unseen_loop.ope.lifted import (
    RatioLiftRequest,
    RatioLiftResponse,
    RatioLiftWPDISClient,
    RatioLiftWPDISServer,
    RatioLiftWPDISSpec,
    _inclusive_prefix,
    lifted_ckks_parameters,
)
from unseen_loop.ope.types import PolynomialPolicySpec, TrajectoryBatch, TrajectorySpec


def _fixture(n=3, horizon=3, degree=1):
    shape = TrajectorySpec(
        n, horizon, 1, 2, state_min=(0.0,), state_max=(1.0,), reward_min=-1.0, reward_max=0.0
    )
    left_extra, right_extra = ((-0.05,), (0.05,)) if degree == 2 else ((), ())
    policies = tuple(
        PolynomialPolicySpec(2, 1, degree, ((1.0 - p, -0.1, *left_extra), (p, 0.1, *right_extra)))
        for p in (0.4, 0.5)
    )
    spec = RatioLiftWPDISSpec(shape, policies, 0.9, 0.5, 1.2)
    states = np.full((n, horizon, 1), 0.5)
    actions = np.zeros((n, horizon), dtype=int)
    rewards = -np.arange(1, n + 1)[:, None] / n * np.ones((1, horizon))
    return spec, TrajectoryBatch(shape, states, actions, rewards, np.full((n, horizon), 0.5))


class _Vector:
    """Exact test arithmetic; never used by production or empirical executors."""

    def __init__(self, values, level=0):
        self.values = np.asarray(values, dtype=np.float64)
        self.level = level

    def size(self):
        return self.values.size

    def serialize(self):
        return self.values.tobytes()

    def __mul__(self, other):
        if isinstance(other, _Vector):
            # Mimic TenSEAL's hazardous mutation of the right operand.
            level = max(self.level, other.level)
            other.level = level
            return _Vector(self.values * other.values, level + 1)
        return _Vector(self.values * other, self.level + 1)

    def __add__(self, other):
        level = max(self.level, other.level)
        other.level = level
        return _Vector(self.values + other.values, level)

    def sum(self):
        return _Vector([self.values.sum()], self.level)


class _Client:
    def __init__(self, parameters):
        self.parameters = parameters
        parameters_view = SimpleNamespace(
            poly_modulus_degree=lambda: parameters.poly_modulus_degree
        )
        context_data = SimpleNamespace(
            parms=lambda: parameters_view,
            chain_index=lambda: len(parameters.coeff_mod_bit_sizes) - 2,
            qualifiers=lambda: SimpleNamespace(parameters_set=True),
        )
        self._context = SimpleNamespace(
            global_scale=parameters.global_scale,
            seal_context=lambda: SimpleNamespace(
                first_context_data=lambda: context_data, key_context_data=lambda: context_data
            ),
        )
        self._tenseal = SimpleNamespace()
        self.encryptions = self.decryptions = 0

    def encrypt(self, values):
        self.encryptions += 1
        values = np.asarray(values, dtype=np.float64)
        return SerializedCKKSVector(values.tobytes(), values.size), None

    def decrypt(self, value):
        self.decryptions += 1
        return np.frombuffer(value.ciphertext, dtype=np.float64).copy(), None


def _boundaries(spec):
    parameters = lifted_ckks_parameters(spec)
    client = _Client(parameters)
    backend = SimpleNamespace(
        parameters=parameters,
        _context=client._context,
        _owner=object(),
        _tenseal=SimpleNamespace(
            ckks_vector_from=lambda _, payload: _Vector(np.frombuffer(payload, dtype=np.float64))
        ),
    )
    return client, RatioLiftWPDISClient(client, spec), RatioLiftWPDISServer(backend, spec)


def _policy_hash(policy):
    return hashlib.sha256(policy.to_json().encode()).hexdigest()


def test_prefix_zero_non_power_of_two_and_saved_operand_ownership():
    original = [_Vector([value]) for value in (2.0, 3.0, 0.0, 7.0, 11.0)]
    prefixes = _inclusive_prefix(original)
    assert [v.values.item() for v in prefixes] == [2.0, 6.0, 0.0, 0.0, 0.0]
    assert [v.level for v in original] == [0] * 5
    saved = [(v.values.copy(), v.level) for v in prefixes]
    for prefix in prefixes:
        _Vector([0.25]) * prefix
    for prefix, (value, level) in zip(prefixes, saved, strict=True):
        np.testing.assert_array_equal(prefix.values, value)
        assert prefix.level == level
    with pytest.raises(ValueError):
        _inclusive_prefix(())


def test_both_policy_ratio_validation_precedes_every_encryption():
    spec, batch = _fixture()
    violating = PolynomialPolicySpec(2, 1, 1, ((0.9, 0.0), (0.1, 0.0)))
    spec = dataclasses.replace(spec, target_policies=(spec.target_policies[0], violating))
    raw, client, _ = _boundaries(spec)
    with pytest.raises(ValueError, match=r"^domain\.ratio_bound"):
        client.encrypt_batch(batch)
    assert raw.encryptions == 0


def test_cancelling_coefficients_cannot_hide_intermediate_overflow():
    shape = TrajectorySpec(
        1, 1, 2, 2, state_min=(1.0, 1.0), state_max=(1.0, 1.0), reward_min=-1.0, reward_max=0.0
    )
    # Mathematically constant .5 probabilities on this degenerate box can
    # pass the historical probability proof, but enormous cancelling terms
    # must never pass the encrypted intermediate-range proof.
    policy = PolynomialPolicySpec(2, 2, 1, ((0.5, 1e100, -1e100), (0.5, -1e100, 1e100)))
    policy.probability_bounds(shape)
    with pytest.raises(ValueError, match=r"^domain\.range_bound"):
        RatioLiftWPDISSpec(shape, (policy,), 0.99, 0.5, 2.0)


@pytest.mark.parametrize("degree", [1, 2])
def test_two_policy_roundtrip_applies_gamma_once_and_never_mutates_request(degree):
    spec, batch = _fixture(degree=degree)
    _, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    wire = request.to_bytes()
    loaded = RatioLiftRequest.from_bytes(wire)
    for policy in spec.target_policies:
        response, receipt = server.evaluate(loaded, _policy_hash(policy))
        response = RatioLiftResponse.from_bytes(response.to_bytes())
        statistics, _ = client.decrypt_statistics(request, response)
        weights = np.cumprod(
            policy.logged_action_probabilities(batch) / batch.behavior_array, axis=1
        )
        means = np.mean(weights * batch.reward_array, axis=0)
        denominators = np.mean(weights, axis=0)
        np.testing.assert_allclose(
            [
                np.frombuffer(v.ciphertext, dtype=np.float64).item()
                for v in response.mean_weighted_rewards
            ],
            means,
        )
        assert statistics.estimate == pytest.approx(
            sum(spec.gamma**t * means[t] / denominators[t] for t in range(batch.spec.horizon))
        )
        assert statistics.counts == (batch.spec.trajectories,) * batch.spec.horizon
        assert receipt.schema_version == "unseen-loop/ratio-lift-wpdis-ckks-operation-v1"
        assert request.to_bytes() == loaded.to_bytes() == wire


def test_wrong_same_shape_request_rejected_before_decryption():
    spec, batch = _fixture()
    raw, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    other_batch = dataclasses.replace(batch, rewards=np.full((3, 3), -0.125))
    other, _ = client.encrypt_batch(other_batch)
    response, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    with pytest.raises(ValueError, match=r"^ckks\.request_binding"):
        client.decrypt_statistics(other, response)
    assert raw.decryptions == 0
    with pytest.raises(ValueError, match=r"^ckks\.request_binding"):
        server.evaluate(request, "0" * 64)
    assert raw.decryptions == 0


def test_replacing_serialized_records_preserves_new_ciphertext_and_request_binding():
    spec, batch = _fixture()
    raw, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    original_wire = request.to_bytes()
    response, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    original_response_wire = response.to_bytes()
    chunk = request.chunks[0]
    reward = SerializedCKKSVector(np.full(3, -0.125).tobytes(), 3)
    changed_chunk = dataclasses.replace(
        chunk, normalized_rewards=(reward, *chunk.normalized_rewards[1:])
    )
    changed = dataclasses.replace(request, chunks=(changed_chunk,))
    changed_wire = changed.to_bytes()
    loaded = RatioLiftRequest.from_bytes(changed_wire)
    assert loaded.chunks[0].normalized_rewards[0] == reward
    assert loaded.digest == hashlib.sha256(changed_wire).hexdigest()
    assert loaded.digest != request.digest
    assert request.to_bytes() == original_wire
    with pytest.raises(ValueError, match=r"^ckks\.request_binding"):
        client.decrypt_statistics(loaded, response)
    assert raw.decryptions == 0

    denominator = SerializedCKKSVector(np.array([0.75]).tobytes(), 1)
    changed_response = dataclasses.replace(
        response, mean_weights=(denominator, *response.mean_weights[1:])
    )
    loaded_response = RatioLiftResponse.from_bytes(changed_response.to_bytes())
    assert loaded_response.mean_weights[0] == denominator
    assert response.to_bytes() == original_response_wire


@pytest.mark.parametrize("record_kind", ["request", "response"])
def test_noncanonical_wire_metadata_is_rejected(record_kind):
    spec, batch = _fixture()
    _, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    record: RatioLiftRequest | RatioLiftResponse = request
    if record_kind == "response":
        record, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    wire = record.to_bytes()
    length = int.from_bytes(wire[:8], "big")
    header = json.dumps(json.loads(wire[8 : 8 + length]), indent=2).encode()
    malformed = len(header).to_bytes(8, "big") + header + wire[8 + length :]
    with pytest.raises(ValueError):
        type(record).from_bytes(malformed)


@pytest.mark.parametrize("n", [8192, 8193])
def test_global_batch_normalization_at_chunk_boundary(n):
    spec, batch = _fixture(n=n, horizon=1)
    _, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    response, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    statistics, _ = client.decrypt_statistics(request, response)
    assert statistics.denominators == pytest.approx((1.1,))
    assert statistics.numerators == pytest.approx((1.1 * float(batch.reward_array.mean()),))
    assert statistics.counts == (n,)
    assert statistics.estimate == pytest.approx(float(batch.reward_array.mean()))


def test_malformed_transport_shapes_and_nonpositive_means_fail_closed():
    spec, batch = _fixture()
    raw, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    with pytest.raises(ValueError):
        RatioLiftRequest.from_bytes(request.to_bytes() + b"extra")
    with pytest.raises(ValueError):
        dataclasses.replace(request, state_dim=2)
    response, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    with pytest.raises(ValueError):
        dataclasses.replace(response, mean_weights=(SerializedCKKSVector(b"bad", 2),) * 3)
    assert raw.decryptions == 0
    for value, code in (
        (0.0, "nonpositive_denominator"),
        (-0.1, "nonpositive_denominator"),
        (float("nan"), "nonfinite"),
    ):
        bad = dataclasses.replace(
            response, mean_weights=(SerializedCKKSVector(np.array([value]).tobytes(), 1),) * 3
        )
        with pytest.raises(ValueError, match=code):
            client.decrypt_statistics(request, bad)


@pytest.mark.parametrize(
    "corruption", ["extra", "missing", "bool_dimension", "identifier", "chunk_gap"]
)
def test_wire_metadata_rejects_unknown_missing_and_invalid_identities(corruption):
    spec, batch = _fixture()
    _, client, _ = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    wire = request.to_bytes()
    length = int.from_bytes(wire[:8], "big")
    metadata = json.loads(wire[8 : 8 + length])
    if corruption == "extra":
        metadata["clear_target"] = 0.5
    elif corruption == "missing":
        del metadata["state_dim"]
    elif corruption == "bool_dimension":
        metadata["state_dim"] = True
    elif corruption == "identifier":
        metadata["identifier"] = "POLYNOMIAL_APPROX_OPE_V1"
    else:
        metadata["chunk_plan"]["chunks"][0]["start"] = 1
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    malformed = len(header).to_bytes(8, "big") + header + wire[8 + length :]
    with pytest.raises(ValueError):
        RatioLiftRequest.from_bytes(malformed)


def test_zero_logged_ratios_are_preserved_without_clipping_or_fake_zero():
    spec, batch = _fixture(n=2, horizon=3)
    target = PolynomialPolicySpec(2, 1, 1, ((0.0, 0.0), (1.0, 0.0)))
    spec = dataclasses.replace(
        spec, target_policies=(target, spec.target_policies[1]), maximum_importance_ratio=2.0
    )
    batch = dataclasses.replace(batch, actions=((0, 0, 0), (1, 1, 1)))
    _, client, server = _boundaries(spec)
    request, _ = client.encrypt_batch(batch)
    response, _ = server.evaluate(request, _policy_hash(target))
    statistics, _ = client.decrypt_statistics(request, response)
    assert statistics.denominators == pytest.approx((1.0, 2.0, 4.0))
    assert statistics.estimate == pytest.approx(-sum(spec.gamma**t for t in range(3)))


@pytest.mark.fhe
@pytest.mark.slow
@pytest.mark.parametrize("n", [8192, 8193])
def test_real_ckks_full_chunk_boundary_global_means_and_two_policy_reuse(n):
    pytest.importorskip("tenseal")
    spec, batch = _fixture(n=n, horizon=3)
    parameters = lifted_ckks_parameters(spec)
    artifacts = generate_contexts(parameters)
    assert artifacts.receipt.security_enforced
    assert artifacts.receipt.effective_security_level == "tc128"
    assert not artifacts.receipt.server_context_is_private
    client = RatioLiftWPDISClient(
        CKKSClient.from_serialized(artifacts.client_context, parameters=parameters), spec
    )
    server = RatioLiftWPDISServer(
        CKKSServer.from_serialized(artifacts.server_context, parameters=parameters), spec
    )
    request, _ = client.encrypt_batch(batch)
    wire = request.to_bytes()
    request = RatioLiftRequest.from_bytes(wire)
    for policy in spec.target_policies:
        response, _ = server.evaluate(request, _policy_hash(policy))
        response = RatioLiftResponse.from_bytes(response.to_bytes())
        statistics, _ = client.decrypt_statistics(request, response)
        target = policy.logged_action_probabilities(batch)
        weights = np.cumprod(target / batch.behavior_array, axis=1)
        expected_mean = np.mean(weights * batch.reward_array, axis=0)
        assert statistics.numerators == pytest.approx(
            tuple(spec.gamma**t * v for t, v in enumerate(expected_mean)), abs=0.001
        )
        assert statistics.denominators == pytest.approx(tuple(np.mean(weights, axis=0)), abs=0.001)
        assert statistics.counts == (n,) * 3
        assert np.isfinite(statistics.numerators).all()
        assert request.to_bytes() == wire


@pytest.mark.fhe
@pytest.mark.slow
@pytest.mark.parametrize("horizon", [8, 32, 64])
def test_real_ckks_chain_acceptance_and_consumed_levels(horizon):
    pytest.importorskip("tenseal")
    spec, batch = _fixture(n=2, horizon=horizon)
    parameters = lifted_ckks_parameters(spec)
    artifacts = generate_contexts(parameters)
    raw_client = CKKSClient.from_serialized(artifacts.client_context, parameters=parameters)
    raw_server = CKKSServer.from_serialized(artifacts.server_context, parameters=parameters)
    client, server = RatioLiftWPDISClient(raw_client, spec), RatioLiftWPDISServer(raw_server, spec)
    request, _ = client.encrypt_batch(batch)
    response, _ = server.evaluate(request, _policy_hash(spec.target_policies[0]))
    statistics, _ = client.decrypt_statistics(request, response)
    assert statistics.counts == (2,) * horizon
    assert np.isfinite(statistics.numerators).all()
    wrapped = raw_server._context.seal_context()
    seal = getattr(wrapped, "data", wrapped)
    last = raw_server._tenseal.ckks_vector_from(
        raw_server._context, response.mean_weights[-1].ciphertext
    )
    ciphertext = last.ciphertext()[0]
    # H=2^k consumes precisely k+2 primes and leaves the first data prime.
    assert ciphertext.coeff_modulus_size() == 1
    assert seal.first_context_data().chain_index() == (horizon - 1).bit_length() + 2
    if artifacts.receipt.actual_coeff_modulus_primes is not None:
        assert (
            tuple(q.bit_length() for q in artifacts.receipt.actual_coeff_modulus_primes)
            == parameters.coeff_mod_bit_sizes
        )


@pytest.mark.fhe
@pytest.mark.slow
def test_real_ckks_saved_prefix_zero_non_power_of_two_reuse_tolerance():
    pytest.importorskip("tenseal")
    spec, _ = _fixture(n=1, horizon=8)
    parameters = lifted_ckks_parameters(spec)
    artifacts = generate_contexts(parameters)
    client = CKKSClient.from_serialized(artifacts.client_context, parameters=parameters)
    server = CKKSServer.from_serialized(artifacts.server_context, parameters=parameters)

    def load(value):
        payload, _ = client.encrypt([value])
        raw = server._tenseal.ckks_vector_from(server._context, payload.ciphertext)
        return CKKSEncryptedVector(raw, 1, server._owner)

    def serialize(value):
        return SerializedCKKSVector(value._vector.serialize(), 1)

    inputs = tuple(load(value) for value in (1.1, 0.9, 0.0, 1.2, 0.8))
    original = tuple(serialize(value).to_bytes() for value in inputs)
    prefixes = _inclusive_prefix(inputs)
    before = tuple(float(client.decrypt(serialize(value))[0][0]) for value in prefixes)
    assert before == pytest.approx((1.1, 0.99, 0.0, 0.0, 0.0), abs=0.001)
    for prefix in prefixes:
        # A later numerator and denominator both reuse this saved node.
        (load(-0.25) * prefix).reduce_sum()
        (prefix * 0.25).reduce_sum()
    after = tuple(float(client.decrypt(serialize(value))[0][0]) for value in prefixes)
    assert after == pytest.approx(before, abs=1e-9, rel=0.0)
    assert tuple(serialize(value).to_bytes() for value in inputs) == original
