.PHONY: sync check test smoke release-suite fhe-smoke modal-smoke report serve

sync:
	uv sync --extra dev

check:
	uv run ruff check .
	uv run ruff format --check src tests modal_app.py
	uv run mypy

# Includes the real Concrete serialized client/server canary when the FHE extra is installed.
test:
	uv run pytest --cov=unseen_loop --cov-report=term-missing -q

smoke:
	uv run unseen-loop demo --backend clear --output artifacts/demo
	uv run unseen-loop verify artifacts/demo

# Typed 3x5 clear release matrix; no privacy claim or Modal/FHE timing gates.
release-suite:
	uv run unseen-loop suite --config experiments/release.toml \
		--backend clear --output artifacts/release

fhe-smoke:
	uv sync --extra dev --extra fhe
	uv run unseen-loop demo --backend fhe --output artifacts/fhe-local
	uv run unseen-loop verify artifacts/fhe-local

# One CartPole checkpoint and encrypted prefix; add --full only to scale this single path.
modal-smoke:
	uv sync --extra cloud --extra fhe
	uv run modal run -w artifacts/modal-evidence.json \
		modal_app.py::research --run-id modal-smoke-reproduction

report:
	uv run unseen-loop report artifacts/reference/modal-smoke-001.json

serve: report
	python -m http.server 8000
