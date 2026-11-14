from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..domain.models import RunSpec
from ..evaluation.ranking import RankingSummary
from ..storage import LabStore, utc_now
from ..wandb_logging import finish as finish_wandb
from ..wandb_logging import log as log_wandb
from ..wandb_logging import start_run
from .runner import seed_everything

UPSTREAM_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "m_hyper" / "source"


def _load_upstream_modules() -> tuple[Any, Any]:
    source_src = str(UPSTREAM_SOURCE_ROOT / "src")
    if source_src not in sys.path:
        sys.path.insert(0, source_src)
    import models  # noqa: PLC0415
    import regularizers  # noqa: PLC0415

    return models, regularizers


def _read_id_map(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, identifier = line.rsplit(maxsplit=1)
        result[name] = int(identifier)
    return result


def _read_triples(
    root: Path, entities: dict[str, int], relations: dict[str, int], name: str
) -> np.ndarray:
    rows: list[list[int]] = []
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        fields = line.split(" .")[0].split()
        if len(fields) != 3:
            continue
        rows.append([entities[fields[0]], relations[fields[1]], entities[fields[2]]])
    return np.asarray(rows, dtype=np.int64)


def _load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    entities = _read_id_map(root / "entity2id.txt")
    relations = _read_id_map(root / "relation2id.txt")
    return (
        _read_triples(root, entities, relations, "train"),
        _read_triples(root, entities, relations, "valid"),
        _read_triples(root, entities, relations, "test"),
        len(entities),
        len(relations),
    )


def _split_overlap(left: np.ndarray, right: np.ndarray) -> int:
    left_keys = {tuple(row) for row in left.tolist()}
    return sum(tuple(row) in left_keys for row in right.tolist())


def _prepare_splits(
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    spec: RunSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.validation_source == "train_holdout":
        rng = np.random.default_rng(spec.seed)
        order = rng.permutation(len(train))
        holdout_size = max(1, int(round(len(train) * spec.validation_holdout_fraction)))
        valid_indices = order[:holdout_size]
        train_indices = order[holdout_size:]
        prepared_train = train[train_indices]
        prepared_valid = train[valid_indices]
        return prepared_train, prepared_valid, test

    overlaps = {
        "train_valid": _split_overlap(train, valid),
        "train_test": _split_overlap(train, test),
        "valid_test": _split_overlap(valid, test),
    }
    if any(overlaps.values()):
        raise ValueError(
            "provided dataset splits overlap; use evaluation.validation_source=train_holdout "
            f"for an explicit corrected split: {overlaps}"
        )
    return train, valid, test


def _filters(
    all_splits: tuple[np.ndarray, np.ndarray, np.ndarray], relation_count: int
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    lhs: dict[tuple[int, int], set[int]] = {}
    rhs: dict[tuple[int, int], set[int]] = {}
    for split in all_splits:
        for head, relation, tail in split.tolist():
            rhs.setdefault((int(head), int(relation)), set()).add(int(tail))
            lhs.setdefault((int(tail), int(relation + relation_count)), set()).add(int(head))
    return lhs, rhs


def _evaluate(
    model: Any,
    triples: np.ndarray,
    filters: tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]],
    relation_count: int,
    entity_count: int,
    limit: int,
) -> dict[str, float | int]:
    selected = triples if limit < 0 else triples[:limit]
    device = next(model.parameters()).device
    summaries: dict[str, RankingSummary] = {}
    for direction in ("rhs", "lhs"):
        query = torch.from_numpy(selected.copy()).long().to(device)
        if direction == "lhs":
            query = query[:, [2, 1, 0]]
            query[:, 1] += relation_count
            filter_map = filters[0]
        else:
            filter_map = filters[1]
        ranks: list[int] = []
        for batch_start in range(0, len(query), 256):
            batch = query[batch_start : batch_start + 256]
            scores, _ = model.inference(batch)
            targets = scores[
                torch.arange(len(batch), device=scores.device), batch[:, 2]
            ].clone()
            for row_index, row in enumerate(batch.tolist()):
                filtered = filter_map.get((row[0], row[1]), set())
                scores[row_index, list(filtered - {row[2]})] = -torch.inf
            ranks.extend((scores >= targets.unsqueeze(1)).sum(dim=1).tolist())
        rank_tensor = torch.tensor(ranks, dtype=torch.float32)
        summaries[direction] = RankingSummary(
            mrr=float((1.0 / rank_tensor).mean()),
            hits_at_1=float((rank_tensor <= 1).float().mean()),
            hits_at_3=float((rank_tensor <= 3).float().mean()),
            hits_at_10=float((rank_tensor <= 10).float().mean()),
            sample_count=len(ranks),
        )
    both = {
        "mrr": (summaries["lhs"].mrr + summaries["rhs"].mrr) / 2,
        "hits_at_1": (summaries["lhs"].hits_at_1 + summaries["rhs"].hits_at_1) / 2,
        "hits_at_3": (summaries["lhs"].hits_at_3 + summaries["rhs"].hits_at_3) / 2,
        "hits_at_10": (summaries["lhs"].hits_at_10 + summaries["rhs"].hits_at_10) / 2,
        "sample_count": summaries["lhs"].sample_count + summaries["rhs"].sample_count,
    }
    return {
        **both,
        "directions": {direction: summary.__dict__ for direction, summary in summaries.items()},
    }


def run_upstream_training(
    spec: RunSpec, store: LabStore, dataset_root: Path, feature_root: Path, artifact_directory: Path
) -> dict[str, float | int]:
    if spec.variant == "released_code":
        raise RuntimeError(
            "released_code is intentionally blocked: upstream run.py has a syntax error and uses test data for valid selection"
        )
    if not dataset_root.is_dir() or not feature_root.is_dir():
        raise FileNotFoundError("registered dataset and feature roots are required")
    models, regularizers = _load_upstream_modules()
    raw_train, raw_valid, raw_test, entity_count, relation_count = _load_dataset(dataset_root)
    train, valid, test = _prepare_splits(raw_train, raw_valid, raw_test, spec)
    sparse_rate = float(os.getenv("MHYPER_LINK_SPARSE_RATE", "0"))
    if not 0.0 <= sparse_rate < 1.0:
        raise ValueError("MHYPER_LINK_SPARSE_RATE must be in [0, 1)")
    if sparse_rate:
        rng = np.random.default_rng(spec.seed + 7919)
        keep = rng.random(len(train)) >= sparse_rate
        train = train[keep]
    image_features = torch.load(
        feature_root / "img_features.pth", map_location="cpu", weights_only=False
    ).float()
    text_features = torch.load(
        feature_root / "text_features.pth", map_location="cpu", weights_only=False
    ).float()
    if image_features.shape[0] != entity_count or text_features.shape[0] != entity_count:
        raise ValueError("feature row count does not match entity map")
    seed_everything(spec.seed)
    model = models.M_Hyper_B(
        (entity_count, relation_count * 2, entity_count),
        spec.rank,
        1e-3,
        image_features,
        text_features,
    )
    model.to(spec.device)
    regularizer = regularizers.wN3(spec.regularization_strength)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=spec.learning_rate)
    criterion = nn.CrossEntropyLoss()
    training_examples = np.concatenate((train, train[:, [2, 1, 0]]), axis=0)
    training_examples[len(train) :, 1] += relation_count
    if spec.training_example_limit > 0:
        training_examples = training_examples[: spec.training_example_limit]
    all_splits = (train, valid, test)
    filter_maps = _filters(all_splits, relation_count)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_directory / "best.pt"
    log_path = artifact_directory / "stdout.log"
    log_handle = log_path.open("a", encoding="utf-8")
    best_mrr = -1.0
    best_epoch = 0
    completed_epochs = 0
    wandb_run = start_run(
        spec.run_id,
        {
            **spec.as_parameters(),
            "dataset": spec.dataset_name,
            "model_variant": os.getenv("MHYPER_MODEL_VARIANT", "full"),
            "missing_modality": os.getenv("MHYPER_MISSING_MODALITY", ""),
            "noise_rate": float(os.getenv("MHYPER_NOISE_RATE", "0.2")),
            "link_sparse_rate": sparse_rate,
        },
        ["paper_1", "mhyper", "upstream", spec.variant],
    )
    for epoch in range(1, spec.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(training_examples))
        totals = {"fit_loss": 0.0, "regularizer_loss": 0.0, "auxiliary_loss": 0.0}
        for start in range(0, len(order), spec.batch_size):
            batch = (
                torch.from_numpy(training_examples[order[start : start + spec.batch_size]])
                .long()
                .to(spec.device)
            )
            predictions, factors, auxiliary = model(batch)
            fit_loss = criterion(predictions, batch[:, 2])
            regularizer_loss = regularizer(factors)
            objective = fit_loss + regularizer_loss + auxiliary
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            batch_size = len(batch)
            totals["fit_loss"] += float(fit_loss.detach()) * batch_size
            totals["regularizer_loss"] += float(regularizer_loss.detach()) * batch_size
            totals["auxiliary_loss"] += float(auxiliary.detach()) * batch_size
        count = len(training_examples)
        duration = time.perf_counter() - started
        train_metrics = {
            "fit_loss": totals["fit_loss"] / count,
            "regularizer_loss": totals["regularizer_loss"] / count,
            "auxiliary_loss": totals["auxiliary_loss"] / count,
            "total_loss": sum(totals.values()) / count,
            "duration_seconds": duration,
            "learning_rate": spec.learning_rate,
        }
        store.add_epoch_metric(
            spec.run_id,
            epoch,
            "train",
            train_metrics,
        )
        log_wandb(wandb_run, {f"train/{key}": value for key, value in train_metrics.items()}, epoch)
        log_handle.write(
            f"{utc_now()} epoch={epoch} split=train total_loss={sum(totals.values()) / count:.6f}\n"
        )
        log_handle.flush()
        completed_epochs = epoch
        if epoch % spec.validation_interval_epochs == 0 or epoch == spec.max_epochs:
            valid_metrics = _evaluate(
                model, valid, filter_maps, relation_count, entity_count, spec.validation_query_limit
            )
            for direction in ("lhs", "rhs"):
                store.add_evaluation(
                    spec.run_id,
                    "validation",
                    "valid",
                    direction,
                    valid_metrics["directions"][direction],
                )
            store.add_evaluation(spec.run_id, "validation", "valid", "both", valid_metrics)
            log_wandb(wandb_run, {f"validation/{key}": value for key, value in valid_metrics.items() if key != "directions"}, epoch)
            if valid_metrics["mrr"] > best_mrr:
                best_mrr = float(valid_metrics["mrr"])
                best_epoch = epoch
                with tempfile.NamedTemporaryFile(
                    dir=artifact_directory, prefix="best.", suffix=".tmp", delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_validation_mrr": best_mrr,
                        "seed": spec.seed,
                    },
                    temporary_path,
                )
                os.replace(temporary_path, checkpoint_path)
    if completed_epochs != spec.max_epochs:
        raise RuntimeError(
            f"training ended at epoch {completed_epochs}, expected {spec.max_epochs}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=spec.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _evaluate(
        model, test, filter_maps, relation_count, entity_count, spec.test_query_limit
    )
    for direction in ("lhs", "rhs"):
        store.add_evaluation(
            spec.run_id,
            "best_validation",
            "test",
            direction,
            test_metrics["directions"][direction],
        )
    store.add_evaluation(spec.run_id, "best_validation", "test", "both", test_metrics)
    log_handle.write(
        f"{utc_now()} completed_epochs={completed_epochs} configured_epochs={spec.max_epochs} "
        f"selected_epoch={best_epoch} validation_mrr={best_mrr:.6f} "
        f"test_mrr={test_metrics['mrr']:.6f}\n"
    )
    log_handle.flush()
    log_handle.close()
    log_wandb(
        wandb_run,
        {
            "test/mrr": float(test_metrics["mrr"]),
            "test/hits_at_1": float(test_metrics["hits_at_1"]),
            "test/hits_at_3": float(test_metrics["hits_at_3"]),
            "test/hits_at_10": float(test_metrics["hits_at_10"]),
        },
        spec.max_epochs,
    )
    finish_wandb(
        wandb_run,
        {
            "completed_epochs": completed_epochs,
            "configured_epochs": spec.max_epochs,
            "best_epoch": best_epoch,
            "best_validation_mrr": best_mrr,
            "test_mrr": float(test_metrics["mrr"]),
        },
    )
    return {
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_validation_mrr": best_mrr,
        "test_mrr": float(test_metrics["mrr"]),
    }
