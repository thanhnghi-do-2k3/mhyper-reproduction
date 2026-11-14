from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .domain.models import RunSpec, RunStatus

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, version TEXT NOT NULL,
  root_path TEXT, fingerprint TEXT NOT NULL, status TEXT NOT NULL, metadata_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, name TEXT NOT NULL, dataset_name TEXT NOT NULL, dataset_version TEXT NOT NULL,
  dataset_fingerprint TEXT NOT NULL DEFAULT 'UNKNOWN',
  variant TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, provenance_status TEXT NOT NULL,
  validation_integrity TEXT NOT NULL, reproduction_assessment TEXT NOT NULL, config_fingerprint TEXT NOT NULL,
  implementation_fingerprint TEXT NOT NULL, config_text TEXT NOT NULL, created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT, best_epoch INTEGER, best_validation_mrr REAL,
  training_completed INTEGER NOT NULL DEFAULT 0, evaluation_completed INTEGER NOT NULL DEFAULT 0,
  error_text TEXT
);
CREATE TABLE IF NOT EXISTS run_parameters (
  run_id TEXT NOT NULL REFERENCES runs(run_id), parameter_name TEXT NOT NULL,
  value_text TEXT, value_real REAL, value_integer INTEGER, PRIMARY KEY (run_id, parameter_name)
);
CREATE TABLE IF NOT EXISTS run_attempts (
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
  started_at TEXT NOT NULL, finished_at TEXT, pid INTEGER, heartbeat_at TEXT, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS epoch_metrics (
  metric_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
  attempt_id INTEGER, epoch INTEGER NOT NULL, split TEXT NOT NULL, fit_loss REAL,
  regularizer_loss REAL, reconstruction_loss REAL, consistency_loss REAL, auxiliary_loss REAL, total_loss REAL,
  duration_seconds REAL, peak_vram_bytes INTEGER, learning_rate REAL, UNIQUE(run_id, epoch, split)
);
CREATE TABLE IF NOT EXISTS evaluation_metrics (
  metric_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
  checkpoint_kind TEXT NOT NULL, split TEXT NOT NULL, direction TEXT NOT NULL, mrr REAL,
  hits_at_1 REAL, hits_at_3 REAL, hits_at_10 REAL, sample_count INTEGER NOT NULL,
  filtering_policy TEXT NOT NULL, evaluation_version TEXT NOT NULL,
  UNIQUE(run_id, checkpoint_kind, split, direction)
);
CREATE TABLE IF NOT EXISTS run_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
  created_at TEXT NOT NULL, level TEXT NOT NULL, category TEXT NOT NULL, message TEXT NOT NULL,
  numeric_value REAL
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
  relative_path TEXT NOT NULL, kind TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  UNIQUE(run_id, relative_path)
);
CREATE TABLE IF NOT EXISTS reference_results (
  reference_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_name TEXT NOT NULL, metric TEXT NOT NULL,
  value REAL NOT NULL, label TEXT NOT NULL, citation TEXT NOT NULL, protocol_caveat TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
CREATE INDEX IF NOT EXISTS runs_dataset_idx ON runs(dataset_name);
CREATE INDEX IF NOT EXISTS epoch_run_idx ON epoch_metrics(run_id, epoch);
CREATE INDEX IF NOT EXISTS events_run_idx ON run_events(run_id, event_id);
"""
SCHEMA_VERSION = 2


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex


class LabStore:
    def __init__(self, database_path: Path, artifact_root: Path) -> None:
        self.database_path = database_path
        self.artifact_root = artifact_root
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(epoch_metrics)")}
        if "auxiliary_loss" not in columns:
            self.connection.execute("ALTER TABLE epoch_metrics ADD COLUMN auxiliary_loss REAL")
        run_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "dataset_fingerprint" not in run_columns:
            self.connection.execute(
                "ALTER TABLE runs ADD COLUMN dataset_fingerprint TEXT NOT NULL DEFAULT 'UNKNOWN'"
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_versions VALUES (?, ?)", (SCHEMA_VERSION, utc_now())
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def create_run(self, spec: RunSpec) -> None:
        now = utc_now()
        self.connection.execute(
            """INSERT INTO runs (run_id,name,dataset_name,dataset_version,variant,seed,status,
            provenance_status,validation_integrity,reproduction_assessment,dataset_fingerprint,config_fingerprint,
            implementation_fingerprint,config_text,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                spec.run_id,
                spec.name,
                spec.dataset_name,
                spec.dataset_version,
                spec.variant,
                spec.seed,
                RunStatus.QUEUED,
                spec.provenance_status,
                (
                    "CLEAN"
                    if spec.provenance_status == "SMOKE_ONLY"
                    or spec.variant in {"evaluation_corrected", "research_variant"}
                    else "TAINTED"
                    if spec.variant == "released_code"
                    else "UNKNOWN"
                ),
                "NOT_ASSESSED",
                spec.dataset_fingerprint,
                spec.config_fingerprint,
                spec.implementation_fingerprint,
                spec.config_text,
                now,
            ),
        )
        for name, value in spec.as_parameters().items():
            text_value = str(value) if isinstance(value, str) else None
            real_value = float(value) if isinstance(value, float) else None
            integer_value = int(value) if isinstance(value, int) else None
            self.connection.execute(
                "INSERT INTO run_parameters VALUES (?,?,?,?,?)",
                (spec.run_id, name, text_value, real_value, integer_value),
            )
        self.connection.commit()

    def transition(self, run_id: str, status: RunStatus, **fields: Any) -> None:
        current = self.connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if current is None:
            raise KeyError(run_id)
        allowed = {
            RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.BLOCKED, RunStatus.CANCELLED},
            RunStatus.RUNNING: {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.INTERRUPTED,
                RunStatus.BLOCKED,
            },
            RunStatus.BLOCKED: {RunStatus.QUEUED},
        }
        current_status = RunStatus(current["status"])
        if status != current_status and status not in allowed.get(current_status, set()):
            raise ValueError(f"invalid run transition {current_status} -> {status}")
        assignments = ["status=?"]
        values: list[Any] = [status]
        for key, value in fields.items():
            if key not in {
                "started_at",
                "finished_at",
                "best_epoch",
                "best_validation_mrr",
                "training_completed",
                "evaluation_completed",
                "error_text",
                "validation_integrity",
                "reproduction_assessment",
            }:
                raise ValueError(f"unsupported run field: {key}")
            assignments.append(f"{key}=?")
            values.append(value)
        values.append(run_id)
        self.connection.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id=?", values)
        self.connection.commit()

    def add_event(
        self,
        run_id: str,
        level: str,
        category: str,
        message: str,
        numeric_value: float | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO run_events (run_id,created_at,level,category,message,numeric_value) VALUES (?,?,?,?,?,?)",
            (run_id, utc_now(), level, category, message, numeric_value),
        )
        self.connection.commit()

    def add_epoch_metric(self, run_id: str, epoch: int, split: str, values: dict[str, Any]) -> None:
        columns = {
            key: values.get(key)
            for key in (
                "fit_loss",
                "regularizer_loss",
                "reconstruction_loss",
                "consistency_loss",
                "auxiliary_loss",
                "total_loss",
                "duration_seconds",
                "peak_vram_bytes",
                "learning_rate",
            )
        }
        self.connection.execute(
            """INSERT OR REPLACE INTO epoch_metrics
            (run_id,epoch,split,fit_loss,regularizer_loss,reconstruction_loss,consistency_loss,auxiliary_loss,total_loss,duration_seconds,peak_vram_bytes,learning_rate)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, epoch, split, *columns.values()),
        )
        self.connection.commit()

    def add_evaluation(
        self, run_id: str, checkpoint_kind: str, split: str, direction: str, values: dict[str, Any]
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO evaluation_metrics
            (run_id,checkpoint_kind,split,direction,mrr,hits_at_1,hits_at_3,hits_at_10,sample_count,filtering_policy,evaluation_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                checkpoint_kind,
                split,
                direction,
                values.get("mrr"),
                values.get("hits_at_1"),
                values.get("hits_at_3"),
                values.get("hits_at_10"),
                values["sample_count"],
                "filtered",
                "lab-v1",
            ),
        )
        self.connection.commit()

    def register_artifact(self, run_id: str, path: Path, kind: str) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(self.artifact_root).as_posix()
        self.connection.execute(
            "INSERT OR REPLACE INTO artifacts (run_id,relative_path,kind,size_bytes,sha256) VALUES (?,?,?,?,?)",
            (run_id, relative, kind, path.stat().st_size, digest),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["epochs"] = [
            dict(item)
            for item in self.connection.execute(
                "SELECT * FROM epoch_metrics WHERE run_id=? ORDER BY epoch, split", (run_id,)
            )
        ]
        result["evaluations"] = [
            dict(item)
            for item in self.connection.execute(
                "SELECT * FROM evaluation_metrics WHERE run_id=? ORDER BY split, direction",
                (run_id,),
            )
        ]
        result["events"] = [
            dict(item)
            for item in self.connection.execute(
                "SELECT * FROM run_events WHERE run_id=? ORDER BY event_id", (run_id,)
            )
        ]
        result["artifacts"] = [
            dict(item)
            for item in self.connection.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,))
        ]
        return result

    def list_runs(
        self, query: str = "", status: str | None = None, dataset: str | None = None
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        parameters: list[Any] = []
        if query:
            clauses.append("(name LIKE ? OR run_id LIKE ? OR variant LIKE ? OR error_text LIKE ?)")
            parameters.extend([f"%{query}%"] * 4)
        if status:
            clauses.append("status=?")
            parameters.append(status)
        if dataset:
            clauses.append("dataset_name=?")
            parameters.append(dataset)
        rows = self.connection.execute(
            f"SELECT * FROM runs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC", parameters
        )
        return [dict(row) for row in rows]

    def overview(self) -> dict[str, Any]:
        counts = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) count FROM runs GROUP BY status"
            )
        }
        return {
            "total_runs": sum(counts.values()),
            "counts": counts,
            "recent": self.list_runs()[:5],
        }

    def add_reference_results(self) -> None:
        refs = [("DB15K", 0.4125), ("MKG-W", 0.3702), ("MKG-Y", 0.3946)]
        for dataset, value in refs:
            self.connection.execute(
                "INSERT OR IGNORE INTO reference_results (dataset_name,metric,value,label,citation,protocol_caveat) VALUES (?,?,?,?,?,?)",
                (
                    dataset,
                    "filtered_mrr_both",
                    value,
                    "PAPER",
                    "ACL 2026 paper / Table reference",
                    "Reference claim only; not an observed run.",
                ),
            )
        self.connection.commit()

    def insert_dataset(
        self,
        name: str,
        version: str,
        root_path: str,
        fingerprint: str,
        status: str,
        metadata_text: str,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO datasets (name,version,root_path,fingerprint,status,metadata_text) VALUES (?,?,?,?,?,?)",
            (name, version, root_path, fingerprint, status, metadata_text),
        )
        self.connection.commit()

    def list_datasets(self) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.connection.execute("SELECT * FROM datasets ORDER BY name")
        ]

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()
        return dict(row) if row is not None else None

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        if not 2 <= len(run_ids) <= 8:
            raise ValueError("comparison requires 2 to 8 run IDs")
        records = [self.get_run(run_id) for run_id in run_ids]
        if any(record is None for record in records):
            raise KeyError("one or more run IDs were not found")
        complete_records = [record for record in records if record is not None]
        warnings = []
        for field, reason in (
            ("dataset_name", "dataset names differ"),
            ("dataset_version", "dataset versions/fingerprints differ"),
            ("variant", "implementation variants differ"),
            ("validation_integrity", "validation integrity differs"),
        ):
            if len({record[field] for record in complete_records}) > 1:
                warnings.append(
                    {"severity": "blocks_direct_comparison", "field": field, "reason": reason}
                )
        if len({record["seed"] for record in complete_records}) > 1:
            warnings.append(
                {"severity": "informational", "field": "seed", "reason": "report N and dispersion"}
            )
        return {"run_ids": run_ids, "runs": complete_records, "warnings": warnings}

    def export_markdown(self, run_id: str) -> str:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        lines = [
            f"# Run dossier: `{run_id}`",
            "",
            f"- Status: **{run['status']}**",
            f"- Dataset: `{run['dataset_name']}` / `{run['dataset_version']}`",
            f"- Variant: `{run['variant']}`",
            f"- Seed: `{run['seed']}`",
            f"- Provenance: **{run['provenance_status']}**",
            f"- Best validation epoch: `{run['best_epoch'] or 'Not Recorded'}`",
            "",
            "## Evaluation",
            "",
            "| Split | Checkpoint | Direction | MRR | Hits@1 | Hits@3 | Hits@10 | N |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for metric in run["evaluations"]:
            lines.append(
                f"| {metric['split']} | {metric['checkpoint_kind']} | {metric['direction']} | "
                f"{metric['mrr']:.6f} | {metric['hits_at_1']:.6f} | {metric['hits_at_3']:.6f} | "
                f"{metric['hits_at_10']:.6f} | {metric['sample_count']} |"
            )
        lines.extend(["", "## Artifacts", ""])
        lines.extend(
            f"- `{item['relative_path']}` SHA256 `{item['sha256']}`" for item in run["artifacts"]
        )
        return "\n".join(lines) + "\n"
