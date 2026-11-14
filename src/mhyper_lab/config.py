from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Any

from .domain.models import ALLOWED_VARIANTS, RunSpec
from .provenance import implementation_fingerprint


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{k}:{_canonical_bytes(value[k]).decode()}" for k in sorted(value))
            + "}"
        ).encode()
    if isinstance(value, list):
        return ("[" + ",".join(_canonical_bytes(item).decode() for item in value) + "]").encode()
    return repr(value).encode()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required = {"run", "dataset", "model", "training", "evaluation", "execution"}
    optional = {"research"}
    unknown = set(config) - required - optional
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    digest = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    return config, digest


def resolve_run_spec(
    path: Path, run_id: str, *, variant: str | None = None, seed: int | None = None
) -> RunSpec:
    config, config_fingerprint = load_config(path)
    run = config["run"]
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    execution = config["execution"]
    research = config.get("research", {})
    chosen_variant = variant or run.get("variant")
    if chosen_variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(ALLOWED_VARIANTS)}")
    chosen_seed = int(seed if seed is not None else run.get("seed", 0))
    if chosen_seed < 0 or int(model["rank"]) <= 0 or int(training["max_epochs"]) <= 0:
        raise ValueError("seed, rank and max_epochs must be positive (seed may be zero)")
    if evaluation["ranking"] != "filtered":
        raise ValueError("only filtered ranking is supported")
    validation_source = str(evaluation.get("validation_source", "provided"))
    if validation_source not in {"provided", "train_holdout"}:
        raise ValueError("evaluation.validation_source must be provided or train_holdout")
    holdout_fraction = float(evaluation.get("validation_holdout_fraction", 0.0))
    if validation_source == "train_holdout" and not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            "evaluation.validation_holdout_fraction must be between 0 and 1 for train_holdout"
        )
    if validation_source == "provided" and holdout_fraction != 0.0:
        raise ValueError("validation_holdout_fraction is only valid with train_holdout")
    resolved_text = path.read_text(encoding="utf-8")
    query_limit = int(evaluation.get("query_limit", -1))
    if query_limit == 0 or query_limit < -1:
        raise ValueError("evaluation.query_limit must be -1 or positive")
    training_example_limit = int(training.get("example_limit", -1))
    if training_example_limit == 0 or training_example_limit < -1:
        raise ValueError("training.example_limit must be -1 or positive")
    return RunSpec(
        run_id=run_id,
        name=str(run["name"]),
        dataset_name=str(dataset["name"]),
        dataset_version=str(dataset["registered_version"]),
        dataset_fingerprint="UNKNOWN",
        variant=chosen_variant,
        seed=chosen_seed,
        rank=int(model["rank"]),
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        validation_interval_epochs=int(training["validation_interval_epochs"]),
        regularization_strength=float(training["regularization_strength"]),
        reconstruction_coefficient=float(research.get("reconstruction_coefficient", 0.0)),
        device=str(execution["device"]),
        evaluation_ranking=str(evaluation["ranking"]),
        checkpoint_selection=str(evaluation["checkpoint_selection"]),
        validation_source=validation_source,
        validation_holdout_fraction=holdout_fraction,
        validation_query_limit=query_limit,
        test_query_limit=query_limit,
        training_example_limit=training_example_limit,
        config_text=resolved_text,
        config_fingerprint=config_fingerprint,
        implementation_fingerprint=implementation_fingerprint(chosen_variant),
        provenance_status=(
            "SMOKE_ONLY"
            if chosen_variant == "smoke_only" or str(dataset["name"]) == "SYNTHETIC_SMOKE"
            else "BLOCKED"
        ),
    )


def workspace_path() -> Path:
    return Path(os.environ.get("MHYPER_WORKSPACE", "workspace"))
