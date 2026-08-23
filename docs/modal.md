# Modal Operations and Cost Controls

## Why three functions and a local entrypoint

- **GPU teacher:** clear RL training is massively parallel and benefits from an L4.
- **CPU search:** Gym rollouts and small ridge solves are CPU workloads; using a GPU would waste credit.
- **CPU FHE compiler/evaluator:** the recorded Concrete target is x86 CPU. GPU teacher hardware does not imply GPU FHE.
- **Local entrypoint:** owns the secret/decryption key. A Modal Secret would place it in the remote trust domain and invalidate the architecture claim.

## Current resource caps

| Function | Accelerator/CPU | Memory | Max containers | Timeout | Retries |
|---|---:|---:|---:|---:|---:|
| `train_teacher_gpu` | 1 × L4, 4 cores | 16 GiB | 1 | 1 h | 0 |
| `search_on_cpu` | 8 cores | 16 GiB | 2 | 6 h | 0 |
| `compile_finalist` | 16 cores | 32 GiB | 2 | 6 h | 0 |
| `evaluate_ciphertext` | 8 cores | 16 GiB | 4 | 1 h | 0 |
| `persist_cloud_evidence` | 0.25 core | 512 MiB | 1 | 5 min | 0 |

All use `min_containers=0` and `buffer_containers=0`, so idle benchmark capacity scales to zero. Exact CPU/memory requests equal limits to reduce hidden resource variance.

## Smoke workflow

```bash
uv run modal profile current
uv run modal run -w artifacts/modal-evidence.json \
  modal_app.py::research --run-id modal-smoke-$(date -u +%Y%m%d)
```

The command is synchronous and bounded. The result is persisted both to the named Volume and the local `--write-result` path.

## Inspect artifacts

```bash
uv run modal volume ls unseen-loop-artifacts runs
uv run modal volume get unseen-loop-artifacts \
  runs/<run-id>/modal/evidence.json \
  artifacts/<run-id>-evidence.json
```

`FunctionCall` results are temporary; the Volume and exported local record are canonical. A producer calls `artifacts.commit()` before returning.

## Spend controls

Before a full run:

1. configure a Workspace usage budget;
2. configure an Environment compute budget if available;
3. inspect `--full` population, iteration, seed, and trial counts;
4. confirm all `max_containers`, timeouts, and `retries=0` values;
5. run the quick path first;
6. stop a crash-looping deployed app immediately.

Billing report:

```bash
uv run modal billing report --for today --show-resources --json
```

Emergency stop for a deployed app:

```bash
uv run modal app stop -y unseen-loop
```

The repository does not deploy a persistent public web FHE endpoint. The static dashboard replays a recorded result, which avoids an unauthenticated credit sink and avoids sending browser plaintext to Modal.

## Image discipline

- Python 3.12 for every image;
- pinned NumPy and Gymnasium;
- pinned Torch/CUDA wheel in GPU image;
- pinned Concrete-Python and Setuptools in FHE image;
- project source added after dependency layers;
- no FHE key files baked into an image;
- compiler and evaluator share the same FHE image family because `server.zip` is architecture-specific.

The images intentionally duplicate small core dependencies. Combining CUDA and Concrete into one image would increase cold build/pull size, mix trust boundaries, and make dependency conflicts harder to audit.

## Measurement rules

- Record first evaluator call as cold and later calls as warm.
- Do not use dynamic batching for single-request latency.
- Do not divide batch latency and call it single-query latency.
- Do not compare GPU teacher time with FHE evaluation time as cryptographic overhead.
- Record every failed/timeout request; retries are zero in benchmark functions.
- Store CPU/GPU name, library/CUDA versions, compiler config, call ID, region where available, byte counts, and hashes.
- Use multiple containers and shuffled requests before reporting p50/p95.

## Key-separation audit

The only line that creates `fhe.Client` is inside the local entrypoint or local backend. The remote evaluator signature is:

```python
def evaluate_ciphertext(
    server_artifact: bytes,
    encrypted_input: bytes,
    evaluation_keys: bytes,
) -> dict[str, Any]:
```

No function argument, Volume path, environment variable, or Modal Secret carries the secret key. The evidence record explicitly sets `secret_key_sent_to_modal=false`, and the server archive is scanned for secret-key filename markers.

## Deployment alternative

If a durable service is later required, expose only authenticated job submission/polling using `requires_proxy_auth=True`. Do not wait for compilation in an HTTP request, and do not expose the evaluator directly without strict payload, context, quota, and replay controls. Such a service is a separate deployment/security review, not implied by the research functions.
