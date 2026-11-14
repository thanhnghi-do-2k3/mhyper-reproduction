#!/usr/bin/env python3
"""Run the three Paper 1 full reproductions sequentially with W&B logging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mhyper_lab.service import create_run, run_worker  # noqa: E402, I001


CONFIGS = (
    ("DB15K", REPOSITORY_ROOT / "configs/reproduction/db15k.toml"),
    ("MKG-W", REPOSITORY_ROOT / "configs/reproduction/mkg_w.toml"),
    ("MKG-Y", REPOSITORY_ROOT / "configs/reproduction/mkg_y.toml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("MHYPER_WORKSPACE", REPOSITORY_ROOT / "workspace")),
        help="Experiment workspace containing lab.sqlite3 and artifacts.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next dataset if one run fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    if wandb_mode != "online":
        raise SystemExit(
            "W&B is required for this script. Run `wandb login`, then set WANDB_MODE=online."
        )

    workspace = args.workspace.resolve()
    project = os.environ.get("WANDB_PROJECT", "graph-paper-1-mhyper")
    group = os.environ.get("WANDB_RUN_GROUP", "paper1-full-reproduction")
    print(f"workspace={workspace}")
    print(f"wandb_project={project} wandb_group={group}")

    failures: list[str] = []
    for dataset_name, config_path in CONFIGS:
        print(f"\n=== {dataset_name} ===")
        print(f"config={config_path}")
        run_id = create_run(config_path, root=workspace)
        print(f"run_id={run_id}")
        try:
            outcome = run_worker(run_id, root=workspace)
            if outcome.get("status") == "BLOCKED":
                raise RuntimeError(str(outcome))
            print(
                "completed "
                f"completed_epochs={outcome['completed_epochs']} "
                f"best_epoch={outcome['best_epoch']} "
                f"validation_mrr={outcome['best_validation_mrr']:.6f} "
                f"test_mrr={outcome['test_mrr']:.6f}"
            )
        except Exception as error:
            failures.append(f"{dataset_name} ({run_id}): {error}")
            print(f"FAILED: {error}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed runs:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll three Paper 1 reproductions completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
