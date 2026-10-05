# What fits under FHE today

These are the small circuits measured in this repository, not a general FHE capacity limit.
Exact integer circuits, synthetic policies, and approximate CKKS arithmetic are kept separate.
The generator only reads committed JSON/JSONL; it does not compile, generate keys,
execute cryptography, use the network, or request paid compute.

Regenerate from the repository root (Python standard library only):

```sh
nice -n 19 uv run python tools/fhe_fit_table.py --output docs/fhe-fit.md
```

A system Python 3.11+ can replace `uv run python`; no project dependencies are imported.
Latencies are seconds, rounded to six decimal places. Byte counts are serialized bytes,
not resident memory. Requests, responses, evaluation keys, and public contexts are distinct.

| Circuit / semantics | Shape and size | Colocated server evaluation | Serialized material | Correctness accounting | Sources |
|---|---|---|---|---|---|
| Synthetic polynomial / Concrete | features=2; actions=2; degree=2; coefficients=12; qmax=2 | p50 0.362091 s; p95 0.374514 s; n=40, one colocated worker | evaluation keys: 472,645,952 B; request: 131,336 B; response: 131,336 B | REAL: 40/40 match (0 unrecorded); domain: 25 calls; fresh-ciphertext canary: 15 calls; simulation: 25 rows, all-match=True | [artifacts/studies/modal-nonlinear-qmax2-002/summary.json](../artifacts/studies/modal-nonlinear-qmax2-002/summary.json); [artifacts/studies/modal-nonlinear-qmax2-002/raw.jsonl](../artifacts/studies/modal-nonlinear-qmax2-002/raw.jsonl); [artifacts/studies/modal-nonlinear-qmax2-002/policy.json](../artifacts/studies/modal-nonlinear-qmax2-002/policy.json) |
| Synthetic polynomial / repeated Concrete timing | features=2; actions=2; degree=2; coefficients=12; qmax=2 | p50 0.544536 s; p95 0.830709 s; n=64, 4 colocated contexts | evaluation keys: 472,645,952 B; request: 131,336 B; response: 131,336 B | measured successes: 64/64; excluded warmups: 12; per-call output equality not recorded in sanitized raw rows | [artifacts/studies/modal-fhe-timing-003/summary.json](../artifacts/studies/modal-fhe-timing-003/summary.json); [artifacts/studies/modal-fhe-timing-003/raw.jsonl](../artifacts/studies/modal-fhe-timing-003/raw.jsonl); [artifacts/studies/modal-fhe-timing-003/context.json](../artifacts/studies/modal-fhe-timing-003/context.json) |
| Exact shield canary / Concrete | input=(6); margins=(5,2,4) (actions,horizon,families); qmax=2; policy degree/coefficients: not applicable | 73.916227 s; one colocated call, not a percentile | evaluation keys: 758,473,160 B; request: 492,056 B; response: 3,278,720 B | one REAL tensor equality flag=True; complete-domain simulation flag=True over 15,625 points; no per-point raw rows retained | [site/data/flagship-evidence.json](../site/data/flagship-evidence.json) |
| Exact hard-clipped OPE canary / Concrete | (N,H,D)=(1,2,1); input=(10); output=(3,2) (statistics,horizon); policy degree/actions/coefficients not recorded in excerpt | 1.826500 s; one colocated call, not a percentile | evaluation keys: 589,084,056 B; request: 124,032 B; response: 115,480 B | one REAL/reference equality flag=True; simulation/REAL equality flag=True; released counts=(1,1) | [site/data/flagship-evidence.json](../site/data/flagship-evidence.json) |
| Approximate polynomial OPE / TenSEAL CKKS | (N,H,D)=(64,8,1); actions=2; policy degree=1; policy coefficient count=not recorded; soft-clip coefficients=3; output ciphertexts=24 | not recorded in retained publication excerpt | public server context: 813,969,437 B; request: not recorded for some/all calls; response: not recorded for some/all calls | exact-clear match=False; released maximum horizon-numerator error=8.399916065832734; soft-clip absolute-error bound=32.0; released counts=(64,64,64,64,64,64,64,64); not exact hard clipping | [site/data/flagship-evidence.json](../site/data/flagship-evidence.json) |
| Broader shield smoke / Concrete (negative) | 5x2x4 margins (publication description); input/degree/coefficients not recorded in summary | not recorded per shield call in retained summary | per-shield evaluation keys/request/response not recorded in retained summary | 5 valid calls; 185/200 margins; 5/5 actions; exact-clear conformance=False | [artifacts/flagship-smoke-20260902-023/analysis.json](../artifacts/flagship-smoke-20260902-023/analysis.json); [site/data/flagship-evidence.json](../site/data/flagship-evidence.json) |
| Mixed systems smoke (negative; not a new canary) | OPE cell 64x8; D and policy coefficients not recorded in summary | server latency not recorded; retained OPE cell p95=27.036424 s (metric is not labeled server-only) | per-circuit bytes not recorded; mixed maximum evaluation keys=821,133,768 B | 2/5 measured successes; cell p95 is a retained summary, not a comparable all-success latency distribution | [artifacts/flagship-smoke-20260902-023/analysis.json](../artifacts/flagship-smoke-20260902-023/analysis.json) |
| Linear CartPole reference / Concrete | features=4; actions=2; degree=1; coefficients=10; qmax=15 | not a colocated measurement: local-entrypoint client / Modal server; no comparable colocated quantiles | evaluation keys: 64 B; request: 33,592 B; response: 16,920 B | trial equality: 2/2 match (0 unrecorded); closed-loop equality: 25/25 match (0 unrecorded); same-input distinct ciphertexts=2; not exhaustive REAL-domain verification | [artifacts/reference/modal-smoke-001.json](../artifacts/reference/modal-smoke-001.json); [artifacts/reference/modal-smoke-001/modal/policy.json](../artifacts/reference/modal-smoke-001/modal/policy.json) |

## Reading the measurements

- The nonlinear study mixes complete-domain calls with fresh encryptions in one context.
  Its committed p50/p95 use linear interpolation. Repeated timing instead uses nearest-rank
  quantiles over successful non-warmup requests, clustered across independent containers;
  its summary also retains hierarchical container/request bootstrap intervals.
  These are different experiments, not pooled samples or a scaling comparison.
- The nonlinear and repeated timing configurations requested 16 CPUs and 32,768 MiB
  per Modal worker. A CPU model and portable local-laptop latency are not recorded.
  Colocation does not demonstrate local-client/remote-server input secrecy or network latency.
- The shield's domain-wide check is simulation, not domain-wide REAL FHE execution.
  One exact encrypted tensor does not overturn the broader smoke's margin mismatches.
  Shield action selection and OPE division are client-side, not encrypted argmax/division.
- The retained publication contains canary excerpts and hashes of source summaries, not
  the original complete canary bundles. CKKS server latency and request/response bytes
  are absent there. Earlier Markdown canary timings/CKKS bytes do not all match this
  snapshot; this table uses the JSON values and leaves absent fields absent.
  CKKS public server context includes public, relinearization, and Galois key material;
  it is not interchangeable with Concrete evaluation keys or a secret client context.
- CKKS reports approximate polynomial soft clipping, not exact hard-clipped OPE.
  Its released numerator error is against a distinct exact clear reference; the soft-clip
  bound is an analytic approximation bound, not a measured CKKS arithmetic error.
  Synthetic conformance and arithmetic equality do not establish useful control or
  statistical OPE accuracy, production throughput, or empirical cryptographic failure rates.

The source list is deliberately fixed to these retained experiments. Missing source files
are errors; missing measurement fields are shown explicitly rather than filled from prose.
The reference's 64-byte evaluation-key serialization belongs to its linear circuit only.
The systems smoke's mixed maximum is not assigned to any individual canary.
