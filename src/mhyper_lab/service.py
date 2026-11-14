from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

from .config import resolve_run_spec, workspace_path
from .domain.models import RunStatus
from .storage import LabStore, new_run_id, utc_now
from .training.runner import run_synthetic_training
from .training.upstream_runner import run_upstream_training


def make_store(root: Path | None = None) -> LabStore:
    base = root or workspace_path()
    return LabStore(base / "lab.sqlite3", base / "artifacts")


def create_run(
    config_path: Path, variant: str | None = None, seed: int | None = None, root: Path | None = None
) -> str:
    store = make_store(root)
    try:
        spec = resolve_run_spec(config_path, new_run_id(), variant=variant, seed=seed)
        registered = store.get_dataset(spec.dataset_name)
        if registered is not None:
            spec = replace(
                spec,
                dataset_fingerprint=str(registered["fingerprint"]),
                provenance_status=(
                    "OBSERVED"
                    if spec.variant in {"evaluation_corrected", "research_variant"}
                    else spec.provenance_status
                ),
            )
        store.create_run(spec)
        store.add_event(spec.run_id, "INFO", "lifecycle", f"run queued with variant={spec.variant}")
        (store.artifact_root / spec.run_id).mkdir(parents=True, exist_ok=True)
        (store.artifact_root / spec.run_id / "resolved_config.toml").write_text(
            spec.config_text, encoding="utf-8"
        )
        store.register_artifact(
            spec.run_id, store.artifact_root / spec.run_id / "resolved_config.toml", "configuration"
        )
        return spec.run_id
    finally:
        store.close()


def run_worker(run_id: str, root: Path | None = None) -> dict[str, object]:
    store = make_store(root)
    try:
        run = store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] != RunStatus.QUEUED:
            raise ValueError(f"run {run_id} is {run['status']}, expected QUEUED")
        store.transition(run_id, RunStatus.RUNNING, started_at=utc_now())
        store.add_event(run_id, "INFO", "worker", f"worker started pid={os.getpid()}")
        if run["variant"] == "released_code":
            store.transition(
                run_id,
                RunStatus.BLOCKED,
                error_text="released_code is blocked: upstream run.py has a syntax error and selects checkpoints using test data.",
            )
            store.add_event(
                run_id,
                "WARNING",
                "provenance",
                "released-code run blocked; no scientific correction was silently applied",
            )
            return {"status": "BLOCKED", "run_id": run_id}
        if run["dataset_name"] == "SYNTHETIC_SMOKE":
            config = _config_from_run(run)
            spec = resolve_run_spec(config, run_id, variant=run["variant"], seed=run["seed"])
            outcome = run_synthetic_training(spec, store, store.artifact_root / run_id)
        else:
            dataset = store.get_dataset(run["dataset_name"])
            if dataset is None:
                store.transition(run_id, RunStatus.BLOCKED, error_text="dataset is not registered")
                store.add_event(
                    run_id,
                    "WARNING",
                    "data",
                    "real-data run blocked because dataset is not registered",
                )
                return {"status": "BLOCKED", "run_id": run_id}
            config = _config_from_run(run)
            spec = resolve_run_spec(config, run_id, variant=run["variant"], seed=run["seed"])
            if spec.device.startswith("cuda") and not _cuda_available():
                store.transition(
                    run_id, RunStatus.BLOCKED, error_text="requested CUDA device is unavailable"
                )
                store.add_event(
                    run_id,
                    "WARNING",
                    "environment",
                    "real-data run blocked because CUDA is unavailable",
                )
                return {"status": "BLOCKED", "run_id": run_id}
            if run["variant"] != "evaluation_corrected":
                store.transition(
                    run_id,
                    RunStatus.BLOCKED,
                    error_text="only evaluation_corrected is wired to the pinned upstream adapter",
                )
                store.add_event(
                    run_id,
                    "WARNING",
                    "provenance",
                    "research variant requires an explicit upstream experiment module",
                )
                return {"status": "BLOCKED", "run_id": run_id}
            outcome = run_upstream_training(
                spec,
                store,
                Path(dataset["root_path"]),
                Path(dataset["root_path"]),
                store.artifact_root / run_id,
            )
        store.register_artifact(run_id, store.artifact_root / run_id / "best.pt", "checkpoint")
        log_path = store.artifact_root / run_id / "stdout.log"
        if log_path.exists():
            store.register_artifact(run_id, log_path, "log")
        store.transition(
            run_id,
            RunStatus.COMPLETED,
            finished_at=utc_now(),
            best_epoch=outcome["best_epoch"],
            best_validation_mrr=outcome["best_validation_mrr"],
            training_completed=1,
            evaluation_completed=1,
        )
        store.add_event(run_id, "INFO", "lifecycle", "training and held-out evaluation completed")
        return outcome
    except Exception as error:
        store.add_event(run_id, "ERROR", "exception", str(error))
        store.connection.execute(
            "UPDATE runs SET status=?, finished_at=?, error_text=? WHERE run_id=?",
            (RunStatus.FAILED, utc_now(), traceback.format_exc(), run_id),
        )
        store.connection.commit()
        raise
    finally:
        store.close()


def launch_worker(run_id: str, root: Path | None = None) -> int:
    workspace = (root or workspace_path()).resolve()
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_root if not existing_path else f"{source_root}{os.pathsep}{existing_path}"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mhyper_lab.worker",
            "--run-id",
            run_id,
            "--workspace",
            str(workspace),
        ],
        cwd=Path.cwd(),
        env=environment,
        start_new_session=True,
    )
    store = make_store(root)
    try:
        store.connection.execute(
            "INSERT INTO run_attempts (run_id,started_at,pid,heartbeat_at,status) VALUES (?,?,?,?,?)",
            (run_id, utc_now(), process.pid, utc_now(), "STARTED"),
        )
        store.connection.commit()
    finally:
        store.close()
    return process.pid


def _config_from_run(run: dict[str, object]) -> Path:
    path = Path(tempfile.gettempdir()) / f"mhyper-{run['run_id']}.toml"
    path.write_text(str(run["config_text"]), encoding="utf-8")
    return path


def register_dataset(name: str, path: Path, root: Path | None = None) -> dict[str, object]:
    if not path.exists() or not path.is_dir():
        raise ValueError("dataset path must be an existing directory")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    store = make_store(root)
    try:
        store.insert_dataset(
            name,
            "local-v1",
            str(path.resolve()),
            digest.hexdigest(),
            "REGISTERED",
            f"files={len(files)}",
        )
        return {
            "name": name,
            "file_count": len(files),
            "fingerprint": digest.hexdigest(),
            "path": str(path.resolve()),
        }
    finally:
        store.close()


def inspect_dataset(name: str, root: Path | None = None) -> dict[str, object]:
    store = make_store(root)
    try:
        record = store.get_dataset(name)
        if record is None:
            raise KeyError(name)
        dataset_root = Path(record["root_path"])
        required = ["entity2id.txt", "relation2id.txt", "train", "valid", "test"]
        missing = [filename for filename in required if not (dataset_root / filename).is_file()]
        feature_files = sorted(dataset_root.glob("*_features.pth"))
        feature_shapes: dict[str, list[int]] = {}
        for feature_file in feature_files:
            import torch

            # Author-provided .pth files use the legacy pickle tensor format.
            tensor = torch.load(feature_file, map_location="cpu", weights_only=False)
            feature_shapes[feature_file.name] = list(tensor.shape)
        entity_ids = {}
        relation_ids = {}
        for mapping_name, destination in (("entity2id.txt", entity_ids), ("relation2id.txt", relation_ids)):
            mapping_path = dataset_root / mapping_name
            if mapping_path.is_file():
                for line in mapping_path.read_text(encoding="utf-8").splitlines():
                    fields = line.rsplit(maxsplit=1)
                    if len(fields) == 2:
                        destination[fields[0]] = int(fields[1])
        counts = {}
        split_lines: dict[str, set[str]] = {}
        for split in ("train", "valid", "test"):
            raw_lines = (
                [
                    line.strip()
                    for line in (dataset_root / split).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if (dataset_root / split).is_file()
                else []
            )
            lines = (
                {
                    " ".join(
                        (
                            str(entity_ids[fields[0]]),
                            str(relation_ids[fields[1]]),
                            str(entity_ids[fields[2]]),
                        )
                    )
                    for line in raw_lines
                    for fields in [line.split(" .")[0].split()]
                    if len(fields) == 3
                    and fields[0] in entity_ids
                    and fields[1] in relation_ids
                    and fields[2] in entity_ids
                }
            )
            split_lines[split] = lines
            counts[split] = len(raw_lines)
        entity_count = (
            sum(
                1
                for line in (dataset_root / "entity2id.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
            if (dataset_root / "entity2id.txt").is_file()
            else 0
        )
        relation_count = (
            sum(
                1
                for line in (dataset_root / "relation2id.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
            if (dataset_root / "relation2id.txt").is_file()
            else 0
        )
        return {
            "name": name,
            "root_path": str(dataset_root),
            "fingerprint": record["fingerprint"],
            "missing": missing,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "split_counts": counts,
            "split_overlaps": {
                "train_valid": len(split_lines["train"] & split_lines["valid"]),
                "train_test": len(split_lines["train"] & split_lines["test"]),
                "valid_test": len(split_lines["valid"] & split_lines["test"]),
            },
            "split_duplicates": {
                split: counts[split] - len(split_lines[split]) for split in split_lines
            },
            "feature_shapes": feature_shapes,
            "ready_for_upstream_pilot": not missing and len(feature_files) >= 2,
        }
    finally:
        store.close()


def doctor(root: Path | None = None) -> dict[str, object]:
    store = make_store(root)
    try:
        store.add_reference_results()
        disk = shutil.disk_usage(store.database_path.parent)
        return {
            "python": platform.python_version(),
            "torch": _module_version("torch"),
            "cuda_available": _cuda_available(),
            "database": str(store.database_path),
            "schema_version": store.connection.execute(
                "SELECT MAX(version) FROM schema_versions"
            ).fetchone()[0],
            "datasets": store.list_datasets(),
            "upstream_source": "PINNED",
            "upstream_sha": "cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76",
            "free_disk_bytes": disk.free,
        }
    finally:
        store.close()


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except ImportError:
        return "NOT_INSTALLED"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
