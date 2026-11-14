from pathlib import Path

from mhyper_lab.service import create_run, make_store, run_worker


def test_cpu_smoke_is_real_sqlite_backed_run(tmp_path: Path):
    run_id = create_run(Path("configs/experiments/smoke.toml"), root=tmp_path)
    outcome = run_worker(run_id, root=tmp_path)
    assert outcome["best_epoch"] in {1, 2, 3}
    store = make_store(tmp_path)
    run = store.get_run(run_id)
    assert run["status"] == "COMPLETED"
    assert run["provenance_status"] == "SMOKE_ONLY"
    assert run["training_completed"] == 1
    assert run["evaluation_completed"] == 1
    assert len(run["epochs"]) == 6
    assert {item["direction"] for item in run["evaluations"]} == {"lhs", "rhs", "both"}
    assert {item["kind"] for item in run["artifacts"]} == {"configuration", "checkpoint", "log"}
    store.close()
