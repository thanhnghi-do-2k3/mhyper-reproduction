from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..service import create_run, launch_worker, make_store


class CreateRunRequest(BaseModel):
    config_path: str = Field(..., description="Path to a TOML profile inside the workspace")
    variant: str | None = None
    seed: int | None = None


def build_app(workspace: Path | None = None) -> FastAPI:
    app = FastAPI(title="M-Hyper Research Lab", version="0.1.0")

    @app.get("/api/overview")
    def overview() -> dict[str, object]:
        store = make_store(workspace)
        try:
            return store.overview()
        finally:
            store.close()

    @app.get("/api/runs")
    def runs(
        query: str = "", status: str | None = None, dataset: str | None = None
    ) -> list[dict[str, object]]:
        store = make_store(workspace)
        try:
            return store.list_runs(query, status, dataset)
        finally:
            store.close()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, object]:
        store = make_store(workspace)
        try:
            item = store.get_run(run_id)
            if item is None:
                raise HTTPException(404, "run not found")
            return item
        finally:
            store.close()

    @app.post("/api/runs", status_code=201)
    def create(request: CreateRunRequest) -> dict[str, str]:
        path = Path(request.config_path).resolve()
        if not path.is_file() or path.suffix != ".toml":
            raise HTTPException(400, "config_path must point to a TOML file")
        try:
            return {"run_id": create_run(path, request.variant, request.seed, workspace)}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/runs/{run_id}/start", status_code=202)
    def start(run_id: str) -> dict[str, object]:
        try:
            return {"run_id": run_id, "pid": launch_worker(run_id, workspace), "status": "STARTED"}
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, object]]:
        store = make_store(workspace)
        try:
            return store.list_datasets()
        finally:
            store.close()

    @app.get("/api/compare")
    def compare(ids: str = Query(..., description="Comma-separated run IDs")) -> dict[str, object]:
        store = make_store(workspace)
        try:
            try:
                return store.compare_runs([item for item in ids.split(",") if item])
            except (KeyError, ValueError) as error:
                raise HTTPException(400, str(error)) from error
        finally:
            store.close()

    static_path = Path(__file__).resolve().parents[3] / "web" / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return static_path.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = build_app()
