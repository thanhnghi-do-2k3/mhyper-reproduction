import time
from pathlib import Path

from fastapi.testclient import TestClient

from mhyper_lab.api.app import build_app


def test_api_overview_and_run_detail(tmp_path: Path):
    client = TestClient(build_app(tmp_path))
    created = client.post("/api/runs", json={"config_path": "configs/experiments/smoke.toml"})
    assert created.status_code == 201
    run_id = created.json()["run_id"]
    started = client.post(f"/api/runs/{run_id}/start")
    assert started.status_code == 202
    for _ in range(30):
        if client.get(f"/api/runs/{run_id}").json()["status"] == "COMPLETED":
            break
        time.sleep(0.05)
    overview = client.get("/api/overview")
    assert overview.json()["total_runs"] == 1
    detail = client.get(f"/api/runs/{run_id}")
    assert detail.json()["status"] == "COMPLETED"
    comparison = client.get(f"/api/compare?ids={run_id},{run_id}")
    assert comparison.status_code == 200
    assert comparison.json()["warnings"] == []
    assert client.get("/").status_code == 200
