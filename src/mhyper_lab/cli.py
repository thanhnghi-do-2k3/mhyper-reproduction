from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import (
    create_run,
    doctor,
    inspect_dataset,
    launch_worker,
    make_store,
    register_dataset,
    run_worker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mhyper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    smoke = subparsers.add_parser("smoke")
    smoke_sub = smoke.add_subparsers(dest="smoke_command", required=True)
    smoke_sub.add_parser("test")
    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    register = data_sub.add_parser("register")
    register.add_argument("--name", required=True)
    register.add_argument("--path", required=True, type=Path)
    data_sub.add_parser("list")
    inspect = data_sub.add_parser("inspect")
    inspect.add_argument("--name", required=True)
    run = subparsers.add_parser("run")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    create = run_sub.add_parser("create")
    create.add_argument("--config", required=True, type=Path)
    create.add_argument("--variant")
    create.add_argument("--seed", type=int)
    run_start = run_sub.add_parser("start")
    run_start.add_argument("--id", required=True)
    worker = subparsers.add_parser("worker")
    worker_sub = worker.add_subparsers(dest="worker_command", required=True)
    worker_start = worker_sub.add_parser("start")
    worker_start.add_argument("--id", required=True)
    run_list = run_sub.add_parser("list")
    run_list.add_argument("--dataset")
    show = run_sub.add_parser("show")
    show.add_argument("--id", required=True)
    compare = run_sub.add_parser("compare")
    compare.add_argument("--ids", nargs="+", required=True)
    export = run_sub.add_parser("export")
    export.add_argument("--id", required=True)
    export.add_argument("--format", choices=["markdown"], default="markdown")
    web = subparsers.add_parser("web")
    web_sub = web.add_subparsers(dest="web_command", required=True)
    serve = web_sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
    elif args.command == "data" and args.data_command == "register":
        print(json.dumps(register_dataset(args.name, args.path), indent=2))
    elif args.command == "data" and args.data_command == "list":
        store = make_store()
        try:
            print(json.dumps(store.list_datasets(), indent=2))
        finally:
            store.close()
    elif args.command == "data" and args.data_command == "inspect":
        print(json.dumps(inspect_dataset(args.name), indent=2))
    elif args.command == "smoke":
        run_id = create_run(Path("configs/experiments/smoke.toml"), root=None)
        print(run_id)
        print(json.dumps(run_worker(run_id), indent=2))
    elif args.command == "run" and args.run_command == "create":
        print(create_run(args.config, args.variant, args.seed))
    elif args.command == "run" and args.run_command == "start":
        print(
            json.dumps(
                {"run_id": args.id, "pid": launch_worker(args.id), "status": "STARTED"}, indent=2
            )
        )
    elif args.command == "worker" and args.worker_command == "start":
        print(
            json.dumps(
                {"run_id": args.id, "pid": launch_worker(args.id), "status": "STARTED"}, indent=2
            )
        )
    elif args.command == "run" and args.run_command == "list":
        store = make_store()
        try:
            print(json.dumps(store.list_runs(dataset=args.dataset), indent=2))
        finally:
            store.close()
    elif args.command == "run" and args.run_command == "show":
        store = make_store()
        try:
            print(json.dumps(store.get_run(args.id), indent=2))
        finally:
            store.close()
    elif args.command == "run" and args.run_command == "compare":
        store = make_store()
        try:
            print(json.dumps(store.compare_runs(args.ids), indent=2))
        finally:
            store.close()
    elif args.command == "run" and args.run_command == "export":
        store = make_store()
        try:
            print(store.export_markdown(args.id))
        finally:
            store.close()
    elif args.command == "web":
        import uvicorn

        uvicorn.run("mhyper_lab.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
