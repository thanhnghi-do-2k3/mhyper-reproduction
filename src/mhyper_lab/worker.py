from __future__ import annotations

import argparse
from pathlib import Path

from .service import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(prog="mhyper-worker")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()
    run_worker(args.run_id, args.workspace)


if __name__ == "__main__":
    main()
