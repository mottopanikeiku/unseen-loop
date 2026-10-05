# Changelog

## Unreleased

- Reframed the README as an encrypted-control feasibility study, separating clear RL results from synthetic encrypted circuit tests and retaining the Acrobot loss and failed independent confirmation.
- Added `tools/fhe_fit_table.py` and its tests to derive a circuit-shape, latency, and key/context-size comparison from committed artifacts. Missing CKKS timing remains missing.
- Corrected statistical intervals and canary transcriptions in current documentation to match the committed analysis/publication files; no run data was changed.
- Preserved the former README in [docs/research-history.md](docs/research-history.md) and the previous development log in [docs/change-history.md](docs/change-history.md).
- Documented the borrowed-hardware prerequisites for a physically separate remote-client demonstration in [docs/NEXT.md](docs/NEXT.md), including unused convenience methods and the historical replay's ineffective failure-collection paths.
- Removed a redundant catch-and-reraise around integer-student quantization. Quantization errors still propagate unchanged.

## Earlier changes

See the [retained development log](docs/change-history.md). Historical source additions and proposed experiments are not evidence of completed encrypted runs.
