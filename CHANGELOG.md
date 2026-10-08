# Changelog

## Unreleased

- Fixed the rounding certificate's comparison so float64 rounding cannot certify a state whose exact margin only equals twice its bound. On aarch64 an integer-score tie was certified while the integer argmax differed, which made `test_range_invalid_candidate_is_excluded_from_pareto_front` fail. Re-checking all 414 committed policies over their quantizer boxes (66.8M points, exhaustive or 200k sampled) found no state whose certification changes, so no committed coverage figure is affected.
- Added source-linked SVGs for clear paired-return intervals and the separate synthetic nonlinear / earlier affine encrypted results, with a deterministic standard-library generator and focused regression tests. These redraw committed measurements; no experiment was rerun.
- Led the README with what was shown and not shown, keeping failed distillation, inconclusive weighting, and the encrypted-study limits visible.
- Reframed the README as an encrypted-control feasibility study, separating clear RL results from synthetic encrypted circuit tests and retaining the Acrobot loss and failed independent confirmation.
- Added `tools/fhe_fit_table.py` and its tests to derive a circuit-shape, latency, and key/context-size comparison from committed artifacts. Missing CKKS timing remains missing.
- Corrected statistical intervals and canary transcriptions in current documentation to match the committed analysis/publication files; no run data was changed.
- Preserved the former README in [docs/research-history.md](docs/research-history.md) and the previous development log in [docs/change-history.md](docs/change-history.md).
- Documented the borrowed-hardware prerequisites for a physically separate remote-client demonstration in [docs/NEXT.md](docs/NEXT.md), including unused convenience methods and the historical replay's ineffective failure-collection paths.
- Removed a redundant catch-and-reraise around integer-student quantization. Quantization errors still propagate unchanged.
- Fixed CI lint and formatting errors without changing the generated table. CI now installs the optional Modal dependency for orchestration tests and CPU Torch for teacher type checks/tests.
- Fixed the scheduled FHE workflow's missing Modal dependency, which prevented test collection before any encrypted smoke ran. The schedule and manual trigger remain because this was a dependency failure, not a missing-secret or paid-cloud requirement.

## Earlier changes

See the [retained development log](docs/change-history.md). Historical source additions and proposed experiments are not evidence of completed encrypted runs.
