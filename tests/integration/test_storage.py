from pathlib import Path

from mhyper_lab.config import resolve_run_spec
from mhyper_lab.storage import LabStore


def test_sqlite_schema_and_run_lifecycle(tmp_path: Path):
    store = LabStore(tmp_path / "lab.sqlite3", tmp_path / "artifacts")
    spec = resolve_run_spec(Path("configs/experiments/smoke.toml"), "run_storage")
    store.create_run(spec)
    assert store.get_run(spec.run_id)["status"] == "QUEUED"
    assert store.get_run(spec.run_id)["dataset_fingerprint"] == "UNKNOWN"
    store.add_epoch_metric(spec.run_id, 1, "train", {"auxiliary_loss": 0.25})
    assert store.get_run(spec.run_id)["epochs"][0]["auxiliary_loss"] == 0.25
    store.close()
    reopened = LabStore(tmp_path / "lab.sqlite3", tmp_path / "artifacts")
    assert reopened.get_run(spec.run_id)["config_fingerprint"] == spec.config_fingerprint
    assert reopened.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    reopened.close()
