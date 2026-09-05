"""Evidence-boundary and fixed-denominator regressions; no empirical study runs."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from unseen_loop.flagship import executor_private_ope as executor
from unseen_loop.flagship.manifest import PlannedJob
from unseen_loop.ope.study import PairedWPDISBootstrap

DIGEST = "a" * 64
OTHER = "b" * 64


def _job(arm="lifted_prefix", kind="paired_context"):
    coordinates = dict(
        kind=kind,
        cohort="timing",
        case_index=0,
        trajectories=1,
        horizon=1,
        behavior="primary",
        arm=arm,
        wave_index=1,
        case_id="case-" + "c" * 24,
        data_seed="123",
        bootstrap_seed="456",
    )
    return PlannedJob("job-" + arm, "private_ope_pilot", 789, tuple(sorted(coordinates.items())))


def _bootstrap(se=0.0):
    return PairedWPDISBootstrap(
        normalization=1.0,
        raw_left=-0.5,
        raw_right=-0.25,
        raw_contrast=0.25,
        raw_left_se=se,
        raw_right_se=se,
        raw_contrast_se=se,
        raw_lower=0.20,
        raw_upper=0.30,
        normalized_left=-0.5,
        normalized_right=-0.25,
        normalized_contrast=0.25,
        normalized_left_se=se,
        normalized_right_se=se,
        normalized_contrast_se=se,
        normalized_lower=0.20,
        normalized_upper=0.30,
        normalized_width=0.10,
    )


def _reference(se=0.0):
    policies = tuple(
        executor.PolicyReference(
            label,
            digest,
            (value,),
            (1.0,),
            (1.0,),
            (1,),
            value,
            value,
            0.5,
            1.0,
            1.0,
            0,
        )
        for label, digest, value in (("A", DIGEST, -0.5), ("B", OTHER, -0.25))
    )
    baselines = tuple(
        executor.BaselinePair(name, -0.5, -0.25, 0.25, None) for name in executor.BASELINE_IDS
    )
    return executor.BatchReference(
        "batch_reference",
        DIGEST,
        DIGEST,
        DIGEST,
        1.0,
        -0.5,
        -0.25,
        0.25,
        policies,
        _bootstrap(se),
        baselines,
    )


def _metrics(error=0.0, se=0.0):
    reference = _reference(se)
    values = tuple(
        executor.CipherValue(
            p.policy_sha256,
            (p.mean_weighted_rewards[0] + (error if i else 0.0),),
            (1.0,),
            (1,),
            p.raw_value + (error if i else 0.0),
            p.normalized_value + (error if i else 0.0),
        )
        for i, p in enumerate(reference.policy_rows)
    )
    errors = executor.recompute_cipher_errors(reference, *values, 1.0)
    contrast = values[1].normalized_value - values[0].normalized_value
    decision = executor._decision(0.2 + error, 0.3 + error)
    # Explicitly translated endpoints, not a freshly computed bootstrap.
    return executor.CipherBatchMetrics(
        "cipher_batch",
        reference,
        *values,
        errors,
        contrast,
        0.2 + error,
        0.3 + error,
        decision,
        executor.TimingReceipt(100, 150, 10, 1024, ()),
    )


def _manifest():
    return SimpleNamespace(
        phase="pilot",
        seed_root="unit-evidence-boundary",
        domain=SimpleNamespace(gamma=1.0),
        policies=SimpleNamespace(primary_propensity_floor=0.25, primary_ratio_bound=1.2),
        statistics=SimpleNamespace(timing_bootstrap_repetitions=10000),
        execution=SimpleNamespace(crypto_timeout_s=1800),
        gates=SimpleNamespace(
            maximum_normalized_cipher_error=0.001,
            maximum_cipher_error_se_fraction=0.1,
            maximum_mean_statistic_abs_error=0.001,
            maximum_denominator_relative_error=0.01,
            maximum_peak_rss_gib=24.0,
            maximum_context_gib_exclusive=1.5,
        ),
    )


def _numeric_attempt(error=0.0, se=0.0, arm="lifted_prefix"):
    receipts = SimpleNamespace(
        context=SimpleNamespace(
            security_enforced=True,
            effective_security_level="tc128",
            server_context_is_private=False,
            parameters=executor._expected_parameters(_job(arm)),
            actual_coeff_modulus_primes=None,
            data_chain_length=len(executor._expected_parameters(_job(arm)).coeff_mod_bit_sizes) - 1,
        ),
        counts_source="public_fixed_shape",
        computation_sha256=DIGEST,
        request_sha256=DIGEST,
        response_sha256=(DIGEST, OTHER),
        runtime=SimpleNamespace(source_match=True),
        client_context_bytes=1024,
        public_context_bytes=1024,
    )
    receipts.computation_sha256 = executor._sha(
        executor.canonical_bytes(
            executor._reconstructed_computation(_manifest(), _job(arm), receipts.context)
        )
    )
    return SimpleNamespace(
        completed=True, metrics=_metrics(error, se), receipts=receipts, job=_job(arm)
    )


def test_nested_schema_rejects_unknown_bool_nonfinite_and_shape_errors():
    original = _reference().to_dict()
    assert executor.BatchReference.from_dict(original).policy_rows[0].counts == (1,)
    malformed = _reference().to_dict()
    malformed["bootstrap"]["extra"] = 1
    with pytest.raises(ValueError):
        executor.BatchReference.from_dict(malformed)
    malformed = _reference().to_dict()
    malformed["policy_rows"][0]["counts"] = [True]
    with pytest.raises(ValueError):
        executor.BatchReference.from_dict(malformed)
    malformed = _reference().to_dict()
    malformed["policy_rows"][1]["mean_weights"] = [float("nan")]
    with pytest.raises(ValueError):
        executor.BatchReference.from_dict(malformed)
    malformed = _reference().to_dict()
    malformed["policy_rows"][1]["mean_weights"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        executor.BatchReference.from_dict(malformed)


def test_job_seeds_are_decimal_and_reserved_seed_mismatch_is_rejected(monkeypatch, tmp_path):
    import unseen_loop.flagship.manifest as manifests

    job = _job()
    encoded = executor.job_to_dict(job)
    assert encoded["seed"] == "789"
    for bad in (789, True, "0789", "+789", str(2**128)):
        with pytest.raises(ValueError):
            executor.job_from_dict(dict(encoded, seed=bad))
    config = b"unit source bytes"
    digest = executor._sha(config)
    run_id = "private-ope-pilot-" + digest[:24]
    payload = executor.PrivateOPEJobPayload(
        "unseen-loop/private-ope-job-v1",
        run_id,
        digest,
        replace(job, seed=790),
        DIGEST,
        "2026-09-04T12:00:00Z",
    ).to_dict()
    monkeypatch.setattr(
        manifests, "parse_private_ope_manifest_bytes", lambda _: SimpleNamespace(phase="pilot")
    )
    monkeypatch.setattr(manifests, "iter_private_ope_jobs", lambda _: (job,))
    with pytest.raises(ValueError, match="reserved expansion"):
        executor.validate_job_payload(config, payload, str(tmp_path / "private-ope" / run_id))


def test_transport_cannot_escape_invocation_or_omit_digest():
    run_id = "private-ope-pilot-" + "c" * 24
    prefix = f"private-ope-transport/{run_id}/job-1/in-1"
    result = executor.PrivateOPETransportResult(
        "unseen-loop/private-ope-transport-v1",
        run_id,
        "job-1",
        "fc-1",
        "in-1",
        prefix + "/entry.json",
        prefix + "/result.json",
        DIGEST,
        "result",
    )
    assert executor.PrivateOPETransportResult.from_dict(result.to_dict()).result_sha256 == DIGEST
    for bad in (
        replace(result, result_path=prefix + "/../in-2/result.json"),
        replace(result, result_path=prefix.replace("in-1", "in-2") + "/result.json"),
        replace(result, result_sha256=None),
    ):
        with pytest.raises(ValueError):
            executor.PrivateOPETransportResult.from_dict(bad.to_dict())


def test_missing_terminal_row_is_not_a_smaller_analysis_cohort(monkeypatch):
    import unseen_loop.flagship.manifest as manifests

    monkeypatch.setattr(manifests, "iter_private_ope_jobs", lambda _: (_job(),))
    with pytest.raises(ValueError, match="every fixed terminal row"):
        executor.analyze_private_ope(SimpleNamespace(), [])


def test_strict_pilot_and_inclusive_confirmation_timing_boundaries():
    assert not executor._threshold_gate("timing_lower", 1.0, ">", 1.0).passed
    assert executor._threshold_gate("timing_lower", 1.10, ">=", 1.10).passed
    assert executor._threshold_gate("timing_median", 1.25, ">=", 1.25).passed
    assert not executor._threshold_gate("interval_width", 0.01, "<", 0.01).passed
    assert not executor._threshold_gate("timing_median", None, ">=", 1.25).passed


def test_cipher_interval_translation_is_replayed_and_cannot_be_forged():
    metrics = _metrics(error=0.0001, se=0.1)
    parsed = executor.CipherBatchMetrics.from_dict(metrics.to_dict())
    assert parsed.cipher_interval_lower == pytest.approx(0.2001)
    assert parsed.cipher_interval_upper == pytest.approx(0.3001)
    with pytest.raises(ValueError):
        executor.CipherBatchMetrics.from_dict(replace(metrics, cipher_interval_lower=0.2).to_dict())
    with pytest.raises(ValueError):
        executor.CipherBatchMetrics.from_dict(replace(metrics, left=None).to_dict())
    partial = replace(
        metrics,
        right=None,
        errors=None,
        cipher_contrast_normalized=None,
        cipher_interval_lower=None,
        cipher_interval_upper=None,
        cipher_decision="unavailable",
    )
    assert executor.CipherBatchMetrics.from_dict(partial.to_dict()).left is not None


def test_zero_sampling_se_allows_only_zero_cipher_error():
    assert executor._numeric_pass(_numeric_attempt(), _manifest())
    assert not executor._numeric_pass(_numeric_attempt(error=1e-10), _manifest())
    assert executor._numeric_pass(_numeric_attempt(error=0.0001, se=0.01), _manifest())
    assert not executor._numeric_pass(_numeric_attempt(error=0.0001, se=0.0001), _manifest())


@pytest.mark.parametrize("arm", ["lifted_prefix", "raw_prefix"])
def test_numeric_pass_requires_replayed_context_computation_digest(arm):
    attempt = _numeric_attempt(arm=arm)
    assert executor._numeric_pass(attempt, _manifest())
    attempt.receipts.computation_sha256 = OTHER
    assert not executor._numeric_pass(attempt, _manifest())


@pytest.mark.parametrize("arm", ["lifted_prefix", "raw_prefix"])
def test_computation_replay_binds_recorded_moduli_even_below_float_resolution(arm):
    attempt = _numeric_attempt(arm=arm)
    context = attempt.receipts.context
    # Public synthetic modulus integers exercise replay only, not SEAL or key generation.
    moduli = tuple((1 << bits) - 1 for bits in context.parameters.coeff_mod_bit_sizes)
    context.actual_coeff_modulus_primes = moduli
    attempt.receipts.computation_sha256 = executor._sha(
        executor.canonical_bytes(
            executor._reconstructed_computation(_manifest(), attempt.job, context)
        )
    )
    assert executor._numeric_pass(attempt, _manifest())
    # Float-rounded intermediate limits may be identical; the integer moduli must still bind.
    context.actual_coeff_modulus_primes = (moduli[0] - 2, *moduli[1:])
    assert not executor._numeric_pass(attempt, _manifest())


def _raw_wire_fixture():
    from unseen_loop.crypto.ckks import SerializedCKKSVector
    from unseen_loop.ope.study import queue_policies
    from unseen_loop.ope.types import TrajectorySpec

    spec = executor.RawPrefixSpec(
        TrajectorySpec(1, 1, 1, 2, (0.0,), (1.0,), -1.0, 0.0),
        queue_policies(),
        1.0,
        0.25,
        1.2,
    )
    metadata = executor._raw_metadata(spec)
    metadata["chunks"] = [[0, 1]]
    request = executor._RawRequest(
        metadata, tuple(SerializedCKKSVector(bytes([i + 1]), 1) for i in range(5))
    )
    response = executor._RawResponse(
        executor._sha(executor.canonical_bytes(spec.target_policies[0].to_dict())),
        request.digest,
        (SerializedCKKSVector(b"numerator", 1),),
        (SerializedCKKSVector(b"denominator", 1),),
    )
    return spec, metadata, request, response


def test_raw_response_uses_canonical_shared_frame_and_rejects_wrong_request():
    from unseen_loop.ope.lifted import _frame, _unframe

    spec, _, request, response = _raw_wire_fixture()
    wire = response.to_bytes()
    metadata, vectors = _unframe(wire)
    assert metadata == {
        "identifier": executor.RAW_IDENTIFIER,
        "policy_sha256": response.policy_sha256,
        "request_sha256": request.digest,
        "horizon": 1,
    }
    assert vectors == response.mean_weighted_rewards + response.mean_weights
    decoded = executor._RawResponse.from_bytes(_frame(metadata, vectors))
    assert decoded == response
    different_request = replace(request, vectors=tuple(reversed(request.vectors)))
    with pytest.raises(ValueError, match="request_binding"):
        executor._raw_decrypt(object(), spec, different_request, decoded)
    header_size = int.from_bytes(wire[:8], "big")
    header = wire[8 : 8 + header_size] + b" "
    noncanonical = len(header).to_bytes(8, "big") + header + wire[8 + header_size :]
    with pytest.raises(ValueError):
        executor._RawResponse.from_bytes(noncanonical)
    with pytest.raises(ValueError):
        executor._RawResponse.from_bytes(wire[:-1])


def test_raw_wire_reuse_preserves_owned_metadata_and_validates_replacements(monkeypatch):
    from unseen_loop.ope import lifted

    spec, metadata, request, response = _raw_wire_fixture()
    request_wire, response_wire = request.to_bytes(), response.to_bytes()
    metadata["chunks"][0][1] = 2
    metadata["horizon"] = 2
    assert request.digest == executor._sha(request_wire)

    def unexpected_reencoding(*args, **kwargs):
        raise AssertionError("an already serialized boundary was encoded again")

    with monkeypatch.context() as patch:
        patch.setattr(lifted, "_frame", unexpected_reencoding)
        received_request = executor._RawRequest.from_bytes(request_wire, spec)
        received_response = executor._RawResponse.from_bytes(response_wire)
        assert received_request.to_bytes() is request_wire
        assert received_response.to_bytes() is response_wire
        assert request.to_bytes() is request_wire
        assert response.to_bytes() is response_wire
    changed = replace(
        response,
        policy_sha256=executor._sha(executor.canonical_bytes(spec.target_policies[1].to_dict())),
    )
    assert changed.to_bytes() != response_wire
    with pytest.raises(ValueError, match="request_binding"):
        replace(response, mean_weights=())


def test_invalid_pair_is_retained_not_infinite_speedup():
    candidate = _numeric_attempt(arm="lifted_prefix")
    baseline = _numeric_attempt(arm="raw_prefix")
    baseline.completed = False
    summary = executor._aggregate_timing((candidate, baseline), _manifest())
    assert summary.planned_pairs == 1
    assert summary.valid_pairs == 0
    assert summary.pairs[0].speedup is None
    assert summary.median_speedup is None
    assert summary.speedup_lower is None


def test_reference_mismatch_invalidates_even_completed_timing_pair():
    candidate = _numeric_attempt(arm="lifted_prefix")
    baseline = _numeric_attempt(arm="raw_prefix")
    baseline.metrics = replace(
        baseline.metrics, reference=replace(baseline.metrics.reference, batch_sha256=OTHER)
    )
    summary = executor._aggregate_timing((candidate, baseline), _manifest())
    assert not summary.pairs[0].reference_digest_matches
    assert summary.valid_pairs == 0


def test_conditional_ratio_bound_keeps_denominator_condition_explicit():
    reference = _reference()
    left = executor.CipherValue(DIGEST, (-0.5,), (2.1,), (1,), -0.5 / 2.1, -0.5 / 2.1)
    right = executor.CipherValue(OTHER, (-0.25,), (1.0,), (1,), -0.25, -0.25)
    errors = executor.recompute_cipher_errors(reference, left, right, 1.0)
    assert errors.normalized_ratio_perturbation_bounds[0] is None
    assert errors.normalized_ratio_perturbation_bounds[1] == 0.0


def test_failed_diagnostic_keeps_false_and_unknown_attempt_denominators():
    from unseen_loop.flagship.manifest import iter_private_ope_jobs

    manifest = SimpleNamespace(
        phase="diagnostic",
        digest=DIGEST,
        seed_root="unseen-loop-private-ope-ratio-lift-v1-diagnostic",
        gates=SimpleNamespace(diagnostic_sum_abs_error=0.25),
    )
    planned = tuple(
        j for j in iter_private_ope_jobs(manifest) if dict(j.coordinates)["kind"] != "analysis"
    )
    rows = []
    unknown_used = False
    for job in planned:
        count = dict(job.coordinates)["kind"] == "count_precision"
        unknown = count and not unknown_used
        unknown_used |= unknown
        receipts = executor.AttemptReceipts(
            None,
            None,
            None,
            None,
            None,
            (),
            (),
            None,
            None,
            None,
            None,
            (),
            "diagnostic_sum" if count else "not-applicable",
        )
        rows.append(
            executor.PrivateOPEAttempt(
                "unseen-loop/private-ope-attempt-v1",
                "private-ope-diagnostic-" + DIGEST[:24],
                DIGEST,
                OTHER,
                job,
                None,
                None,
                None if unknown else False,
                False,
                "runtime.dispatch_unknown" if unknown else "runtime.not_dispatched",
                None,
                None,
                receipts,
                False,
                False,
            )
        )
    analysis = executor.analyze_private_ope(manifest, rows)
    counts = analysis.counts_by_kind["count_precision"]
    assert (counts.planned, counts.attempted_false, counts.attempted_unknown, counts.completed) == (
        12,
        11,
        1,
        0,
    )
    assert len(analysis.attempt_row_sha256) == 15
    assert analysis.diagnostic.new_profile_passes == 0
    assert analysis.status == "failed"
    assert not analysis.promotion_allowed
