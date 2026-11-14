#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-paper-1-mhyper}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-paper1-full-reproduction-$(date +%Y%m%d-%H%M%S)}"

if [[ "$WANDB_MODE" != "online" ]]; then
  echo "WANDB_MODE must be online for this full Paper 1 run." >&2
  exit 1
fi

exec python3 scripts/run_all_reproduction.py "$@"
