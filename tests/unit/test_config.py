from pathlib import Path

import pytest

from mhyper_lab.config import resolve_run_spec
from mhyper_lab.wandb_logging import start_run


def test_smoke_config_resolves_immutable_provenance():
    spec = resolve_run_spec(Path("configs/experiments/smoke.toml"), "run_test")
    assert spec.variant == "smoke_only"
    assert spec.provenance_status == "SMOKE_ONLY"
    assert spec.implementation_fingerprint


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        resolve_run_spec(
            Path("configs/experiments/smoke.toml"), "run_test", variant="not_a_variant"
        )


def test_research_variant_has_explicit_change():
    spec = resolve_run_spec(Path("configs/experiments/research_variant.toml"), "run_research")
    assert spec.variant == "research_variant"
    assert spec.reconstruction_coefficient == 0.01


def test_train_holdout_validation_requires_explicit_fraction(tmp_path: Path):
    config = tmp_path / "holdout.toml"
    config.write_text(
        Path("configs/reproduction/db15k.toml")
        .read_text(encoding="utf-8")
        .replace('registered_version = "REPLACE_WITH_REGISTERED_DATASET_VERSION"', 'registered_version = "local-v1"'),
        encoding="utf-8",
    )
    spec = resolve_run_spec(config, "run_holdout")
    assert spec.validation_source == "train_holdout"
    assert spec.validation_holdout_fraction == 0.1


def test_reproduction_configs_use_full_paper_epoch_budget():
    for name in ("db15k", "mkg_w", "mkg_y"):
        spec = resolve_run_spec(Path(f"configs/reproduction/{name}.toml"), f"run_{name}")
        assert spec.max_epochs == 200
        assert spec.validation_interval_epochs == 5
        assert spec.validation_query_limit == -1
        assert spec.test_query_limit == -1
        assert spec.training_example_limit == -1


def test_wandb_is_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert start_run("test", {}, []) is None
