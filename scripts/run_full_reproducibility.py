#!/usr/bin/env python3
"""Run Paper 1 main experiments across datasets and seeds with W&B reporting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mhyper_lab.config import load_config  # noqa: E402, I001
from mhyper_lab.service import create_run, make_store, run_worker  # noqa: E402, I001


CONFIGS = {
    "DB15K": REPOSITORY_ROOT / "configs/reproduction/db15k_paper.toml",
    "MKG-W": REPOSITORY_ROOT / "configs/reproduction/mkg_w.toml",
    "MKG-Y": REPOSITORY_ROOT / "configs/reproduction/mkg_y.toml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("MHYPER_WORKSPACE", REPOSITORY_ROOT / "workspace")),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(CONFIGS),
        default=list(CONFIGS),
    )
    parser.add_argument(
        "--group",
        default=os.environ.get("WANDB_RUN_GROUP", "paper1-main-3seeds"),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def check_configs(datasets: list[str]) -> None:
    for dataset in datasets:
        config, _ = load_config(CONFIGS[dataset])
        evaluation = config["evaluation"]
        source = evaluation.get("validation_source", "provided")
        if source != "provided":
            raise SystemExit(
                f"{dataset} is not using the paper split: validation_source={source!r}"
            )
        if float(evaluation.get("validation_holdout_fraction", 0.0)) != 0.0:
            raise SystemExit(f"{dataset} has a non-zero validation holdout fraction")


def extract_metrics(
    workspace: Path, run_id: str, dataset: str, seed: int
) -> dict[str, Any]:
    store = make_store(workspace)
    try:
        run = store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        test = next(
            metric
            for metric in run["evaluations"]
            if metric["checkpoint_kind"] == "best_validation"
            and metric["split"] == "test"
            and metric["direction"] == "both"
        )
        return {
            "dataset": dataset,
            "seed": seed,
            "run_id": run_id,
            "status": run["status"],
            "validation_integrity": run["validation_integrity"],
            "best_epoch": run["best_epoch"],
            "best_validation_mrr": run["best_validation_mrr"],
            "test_mrr": test["mrr"],
            "test_hits_at_1": test["hits_at_1"],
            "test_hits_at_3": test["hits_at_3"],
            "test_hits_at_10": test["hits_at_10"],
            "completed_epochs": run["training_completed"],
        }
    finally:
        store.close()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_report(
    workspace: Path, group: str, rows: list[dict[str, Any]]
) -> Path:
    report_dir = workspace / "reports" / group
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if rows:
        with (report_dir / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    lines = [
        f"# Paper 1 main reproduction: {group}",
        "",
        "| Dataset | n | MRR mean | MRR std | H@1 mean | H@3 mean | H@10 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(grouped):
        values = grouped[dataset]
        mrr = [float(row["test_mrr"]) for row in values]
        h1 = [float(row["test_hits_at_1"]) for row in values]
        h3 = [float(row["test_hits_at_3"]) for row in values]
        h10 = [float(row["test_hits_at_10"]) for row in values]
        lines.append(
            f"| {dataset} | {len(values)} | {mean(mrr):.6f} | "
            f"{sample_std(mrr):.6f} | {mean(h1):.6f} | "
            f"{mean(h3):.6f} | {mean(h10):.6f} |"
        )
    (report_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_dir


def main() -> int:
    args = parse_args()
    if os.environ.get("WANDB_MODE") != "online":
        raise SystemExit("Set WANDB_MODE=online and run wandb login first.")
    check_configs(args.datasets)

    workspace = args.workspace.resolve()
    os.environ["WANDB_RUN_GROUP"] = args.group
    print(f"workspace={workspace}")
    print(f"group={args.group} seeds={args.seeds} datasets={args.datasets}")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    stop = False
    for seed in args.seeds:
        for dataset in args.datasets:
            run_id = create_run(CONFIGS[dataset], seed=seed, root=workspace)
            os.environ["WANDB_RUN_NAME"] = (
                f"paper1-main-{dataset}-seed{seed}-ep200"
            )
            print(f"\n=== {dataset} seed={seed} ===")
            print(f"run_id={run_id}")
            try:
                outcome = run_worker(run_id, root=workspace)
                if outcome.get("status") == "BLOCKED":
                    raise RuntimeError(str(outcome))
                row = extract_metrics(workspace, run_id, dataset, seed)
                rows.append(row)
                print(json.dumps(row, indent=2))
            except Exception as error:
                failure = f"{dataset}/seed{seed}/{run_id}: {error}"
                failures.append(failure)
                print(f"FAILED: {failure}", file=sys.stderr)
                if not args.continue_on_error:
                    stop = True
                    break
        if stop:
            break

    os.environ.pop("WANDB_RUN_NAME", None)
    report_dir = write_report(workspace, args.group, rows)
    print(f"\nreport_dir={report_dir}")
    if failures:
        print("Failures:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("All requested main reproduction runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
