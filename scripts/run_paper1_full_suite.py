#!/usr/bin/env python3
"""Run the supported Paper 1 experiment suite in one command.

This suite runs main multi-seed experiments, ablations, and robustness
experiments. Baselines that are not present in third_party are not fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mhyper_lab.service import create_run, make_store, run_worker  # noqa: E402, I001


CONFIGS = {
    "DB15K": REPOSITORY_ROOT / "configs/reproduction/db15k_paper.toml",
    "MKG-W": REPOSITORY_ROOT / "configs/reproduction/mkg_w.toml",
    "MKG-Y": REPOSITORY_ROOT / "configs/reproduction/mkg_y.toml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("MHYPER_WORKSPACE", "workspace")))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    parser.add_argument("--group", default=os.environ.get("WANDB_RUN_GROUP", "paper1-full-suite"))
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not args.skip_main:
        for seed in args.seeds:
            for dataset in CONFIGS:
                result.append({"kind": "main", "dataset": dataset, "seed": seed})
    if not args.skip_ablations:
        for variant in (
            "no_joint", "no_structure", "no_vision", "no_text", "no_ferf",
            "no_noise", "no_gate", "no_reconstruction", "no_translation",
            "no_rotation", "fusion", "ensemble",
        ):
            result.append({"kind": "ablation", "dataset": "DB15K", "seed": 42, "variant": variant})
    if not args.skip_robustness:
        for missing in ("vision", "text", "all"):
            result.append({"kind": "missing", "dataset": "DB15K", "seed": 42, "missing": missing})
        for rate in (0.1, 0.2, 0.4):
            result.append({"kind": "noise", "dataset": "DB15K", "seed": 42, "noise_rate": rate})
        for rate in (0.1, 0.2, 0.4):
            result.append({"kind": "link_sparse", "dataset": "DB15K", "seed": 42, "link_rate": rate})
    return result


def clear_experiment_env() -> None:
    for key in (
        "MHYPER_MODEL_VARIANT",
        "MHYPER_MISSING_MODALITY",
        "MHYPER_NOISE_RATE",
        "MHYPER_LINK_SPARSE_RATE",
        "WANDB_RUN_NAME",
    ):
        os.environ.pop(key, None)


def run_job(job: dict[str, Any], workspace: Path, group: str) -> dict[str, Any]:
    clear_experiment_env()
    if "variant" in job:
        os.environ["MHYPER_MODEL_VARIANT"] = job["variant"]
    if "missing" in job:
        os.environ["MHYPER_MISSING_MODALITY"] = job["missing"]
    if "noise_rate" in job:
        os.environ["MHYPER_NOISE_RATE"] = str(job["noise_rate"])
    if "link_rate" in job:
        os.environ["MHYPER_LINK_SPARSE_RATE"] = str(job["link_rate"])
    suffix = "-".join(
        f"{key}{value}" for key, value in job.items()
        if key not in {"kind", "dataset", "seed"}
    )
    name = f"paper1-{job['kind']}-{job['dataset']}-seed{job['seed']}"
    if suffix:
        name += f"-{suffix}"
    os.environ["WANDB_RUN_GROUP"] = group
    os.environ["WANDB_RUN_NAME"] = name
    run_id = create_run(CONFIGS[job["dataset"]], seed=job["seed"], root=workspace)
    print(f"\n=== {name} ===")
    print(f"run_id={run_id}")
    outcome = run_worker(run_id, root=workspace)
    if outcome.get("status") == "BLOCKED":
        raise RuntimeError(str(outcome))
    store = make_store(workspace)
    try:
        run = store.get_run(run_id)
        test = next(
            item for item in run["evaluations"]
            if item["checkpoint_kind"] == "best_validation"
            and item["split"] == "test" and item["direction"] == "both"
        )
        return {
            **job,
            "run_id": run_id,
            "name": name,
            "status": run["status"],
            "best_epoch": run["best_epoch"],
            "best_validation_mrr": run["best_validation_mrr"],
            "test_mrr": test["mrr"],
            "test_hits_at_1": test["hits_at_1"],
            "test_hits_at_3": test["hits_at_3"],
            "test_hits_at_10": test["hits_at_10"],
        }
    finally:
        store.close()


def main() -> int:
    args = parse_args()
    if os.environ.get("WANDB_MODE") != "online":
        raise SystemExit("Set WANDB_MODE=online and run wandb login first.")
    workspace = args.workspace.resolve()
    suite_jobs = jobs(args)
    print(f"workspace={workspace}")
    print(f"group={args.group}")
    print(f"jobs={len(suite_jobs)}")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for job in suite_jobs:
        try:
            rows.append(run_job(job, workspace, args.group))
        except Exception as error:
            failure = f"{job}: {error}"
            failures.append(failure)
            print(f"FAILED: {failure}", file=sys.stderr)
            if not args.continue_on_error:
                break
    clear_experiment_env()
    report_dir = workspace / "reports" / args.group
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "suite_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (report_dir / "suite_failures.txt").write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8"
    )
    print(f"report_dir={report_dir}")
    if failures:
        print("\n".join(f"- {item}" for item in failures))
        return 1
    print("Full supported Paper 1 suite completed.")
    print("External baselines remain unavailable because their implementations are not present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
