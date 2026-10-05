# Unseen Loop

Unseen Loop is a feasibility study of running small control policies over encrypted inputs.

**Question:** can a reinforcement-learning teacher be distilled into a polynomial cheap enough for fully homomorphic encryption (FHE), without losing closed-loop performance?

The code fits degree-one or degree-two score policies to clear teacher outputs, quantizes their coefficients, and evaluates the integer student on states it actually visits. Start with [the polynomial policy](src/unseen_loop/policy.py), [the coefficient-rounding certificate](src/unseen_loop/certificate.py), and [the Concrete backend](src/unseen_loop/fhe_backend.py). Concrete-Python supplies TFHE execution; TenSEAL supplies a separate approximate CKKS policy-evaluation experiment. These are external cryptographic libraries, not encryption implementations written here.

**Result:** tiny encrypted circuits ran, but the larger reinforcement-learning comparison ran in the clear. Distillation failed badly on Acrobot; high certificate coverage did not prevent that loss.

## What actually ran

The [committed analysis](artifacts/studies/unseen-loop-release-analysis-004/publication.json) contains five checkpoints and 500 paired teacher/student episodes per environment. These are **clear integer-student rollouts, not encrypted rollouts**:

| Environment | Teacher mean return | Integer-student mean return | Student minus teacher, 95% CI |
|---|---:|---:|---:|
| CartPole | 461.488 | 432.008 | −29.480 [−70.988, 0.072] |
| MountainCar | −194.986 | −194.190 | +0.796 [−1.620, 3.502] |
| Acrobot | −94.260 | −326.156 | −231.896 [−387.814, −76.253] |

CartPole and MountainCar intervals include zero; MountainCar's teacher was itself weak. In the matched clear CartPole ablation, occupancy refinement helped (+83.619 [26.971, 147.058]), but certificate weighting did **not** show a benefit (−108.461 [−288.061, 68.172]). Those intervals come from the same analysis, not the older Markdown transcriptions.

The strongest nonlinear encrypted policy result is **synthetic**, not an expanded-study champion: two inputs, two outputs, degree two, twelve coefficients, and `qmax=2`. Its [summary](artifacts/studies/modal-nonlinear-qmax2-002/summary.json) records 40/40 ciphertext calls matching the clear circuit, with server p50 362.091 ms. The [raw measurements](artifacts/studies/modal-nonlinear-qmax2-002/raw.jsonl) record 472,645,952-byte evaluation keys. Client and server were colocated on a Modal worker.

An [earlier CartPole reference](artifacts/reference/modal-smoke-001.json) records 25/25 matching encrypted control steps using a smaller affine policy and a local-client/Modal-server path. That short trace is not encrypted reproduction of the expanded study.

Larger circuits were expensive: the [shield canary](site/data/flagship-evidence.json) took 73.916 seconds of server evaluation and 758,473,160 bytes of evaluation keys for one encrypted call. Exact Concrete and approximate TenSEAL off-policy-evaluation roundtrips also ran. The generated [what fits under FHE today table](docs/fhe-fit.md) puts their circuit shapes, latency, key/context size, and correctness checks side by side; it is not a scaling law or a remote-service benchmark.

The [independent confirmation](artifacts/independent-confirmation-20260904-001/) failed. The newer ratio-lift policy-comparison implementation has not been executed as an encrypted study.

## Reproduce without cloud compute

From the repository root, on a CPU with Python compatible with `pyproject.toml` and `uv` installed:

```bash
uv sync --extra dev
nice -n 19 uv run pytest -q -x tests/test_policy_certificate.py tests/test_teacher_search.py tests/test_experiment_cli.py tests/test_fhe_fit_table.py
nice -n 19 uv run python tools/fhe_fit_table.py --output docs/fhe-fit.md
```

These commands cost $0 in paid compute. They test clear behavior and regenerate the table from committed files; they do not rerun FHE or teacher training. The encrypted studies used provisioned Modal CPU workers, not this laptop. Full local/cloud reproduction and older protocols are retained in the [documentation guide](docs/NEXT.md#retained-documentation).

## Limits

- The certificate proves **float-student versus integer-student argmax agreement**, conditional on correct circuit execution. It proves neither teacher agreement nor task safety.
- Training, calibration, distillation, and final client decisions are clear; this is not private training.
- Colocated ciphertext runs do not demonstrate a physically separate remote client. [The next experiment](docs/NEXT.md) specifies that missing demonstration.
- The shield's broader exact-margin checks and subsequent confirmation failed; the one-call canary is not a general safety result.
- The threat model is an honest-but-curious evaluator, not a malicious-server correctness guarantee or a production service.

## Prior work and license

This does not claim novelty in FHE plus RL or policy distillation. It builds on [Suh and Tanaka's encrypted RL](https://arxiv.org/abs/2504.09335), [VIPER policy extraction](https://arxiv.org/abs/1805.08328), and the [Concrete](https://github.com/zama-ai/concrete) and [TenSEAL](https://github.com/OpenMined/TenSEAL) runtimes. See the paper's [references and novelty boundary](docs/paper.md#10-related-work-and-novelty-boundary). Repository code is MIT; external runtimes have their own terms.
