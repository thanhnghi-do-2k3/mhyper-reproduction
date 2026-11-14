from __future__ import annotations

import os
from typing import Any

import wandb


def start_run(name: str, config: dict[str, Any], tags: list[str]) -> Any:
    mode = os.getenv("WANDB_MODE", "disabled")
    if mode in {"disabled", "off"}:
        return None
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "graph-paper-1-mhyper"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=os.getenv("WANDB_RUN_NAME") or name,
        group=os.getenv("WANDB_RUN_GROUP") or None,
        job_type=os.getenv("WANDB_JOB_TYPE", "reproduction"),
        mode=mode,
        config=config,
        tags=tags,
        reinit="return_previous",
    )


def log(run: Any, values: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(values, step=step)


def finish(run: Any, summary: dict[str, Any] | None = None) -> None:
    if run is not None:
        if summary:
            run.summary.update(summary)
        run.finish()
