"""Command-line entrypoints for reproducible Unseen Loop experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from unseen_loop.artifacts import dataclass_dict
from unseen_loop.experiment import ResearchPreset, load_summary, run_experiment, verify_artifact


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unseen-loop",
        description="Certificate-guided RL policy distillation for real encrypted inference.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    demo = subcommands.add_parser("demo", help="run the bounded quick research pipeline")
    demo.add_argument("--env-id", default="CartPole-v1")
    demo.add_argument("--backend", choices=("clear", "simulate", "fhe"), default="clear")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed-root", default="demo-2026-08")

    research = subcommands.add_parser("research", help="run the preregistered full search")
    research.add_argument("--env-id", default="CartPole-v1")
    research.add_argument("--backend", choices=("clear", "simulate", "fhe"), default="clear")
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--seed-root", default="release-2026-08")

    verify = subcommands.add_parser("verify", help="verify an artifact checksum ledger")
    verify.add_argument("artifact", type=Path)

    inspect = subcommands.add_parser("inspect", help="print a run summary")
    inspect.add_argument("artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command in {"demo", "research"}:
        commit, dirty = _git_state()
        preset = ResearchPreset.quick() if arguments.command == "demo" else ResearchPreset.release()
        summary = run_experiment(
            env_id=arguments.env_id,
            output=arguments.output,
            backend=arguments.backend,
            preset=preset,
            seed_root=arguments.seed_root,
            git_commit=commit,
            git_dirty=dirty,
        )
        print(json.dumps(dataclass_dict(summary), sort_keys=True, indent=2))
        return 0
    if arguments.command == "verify":
        valid, failures = verify_artifact(arguments.artifact)
        print(
            json.dumps(
                {"artifact": str(arguments.artifact), "valid": valid, "failures": failures},
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if valid else 1
    if arguments.command == "inspect":
        print(
            json.dumps(dataclass_dict(load_summary(arguments.artifact)), sort_keys=True, indent=2)
        )
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
