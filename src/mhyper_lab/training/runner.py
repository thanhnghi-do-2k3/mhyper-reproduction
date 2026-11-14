from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..domain.models import RunSpec
from ..evaluation.ranking import calculate_filtered_ranks, summarize_ranks
from ..storage import LabStore, utc_now
from ..wandb_logging import finish as finish_wandb
from ..wandb_logging import log as log_wandb
from ..wandb_logging import start_run
from .model import MHyperModel
from .synthetic import SyntheticDataset, make_synthetic_dataset

MetricCallback = Callable[[int, str, dict[str, float | int | None]], None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_sets(triples: torch.Tensor, side: str) -> dict[tuple[int, int], set[int]]:
    result: dict[tuple[int, int], set[int]] = {}
    for head, relation, tail in triples.tolist():
        key = (relation, tail) if side == "lhs" else (head, relation)
        result.setdefault(key, set()).add(head if side == "lhs" else tail)
    return result


def evaluate(
    model: MHyperModel,
    triples: torch.Tensor,
    all_triples: torch.Tensor,
    entity_count: int,
    side: str,
) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        if side == "rhs":
            scores = model.score(triples, torch.arange(entity_count))
            targets = triples[:, 2]
        else:
            swapped = triples[:, [2, 1, 0]]
            scores = model.score(swapped, torch.arange(entity_count))
            targets = triples[:, 0]
        known = _filter_sets(all_triples, side)
        if side == "lhs":
            query_keys = [(int(row[1]), int(row[2])) for row in triples]
        else:
            query_keys = [(int(row[0]), int(row[1])) for row in triples]
        sets = [known.get(key, set()) for key in query_keys]
        ranks = calculate_filtered_ranks(scores, targets, sets)
        summary = summarize_ranks(ranks)
    return summary.__dict__


def run_synthetic_training(
    spec: RunSpec, store: LabStore, artifact_directory: Path, callback: MetricCallback | None = None
) -> dict[str, float | int]:
    if spec.device != "cpu":
        raise RuntimeError("synthetic smoke runner requires device=cpu")
    seed_everything(spec.seed)
    dataset: SyntheticDataset = make_synthetic_dataset()
    model = MHyperModel(dataset.entity_count, dataset.relation_count, spec.rank)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=spec.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_directory / "stdout.log"
    log = stdout_path.open("a", encoding="utf-8")
    best_mrr = -1.0
    best_epoch = 0
    completed_epochs = 0
    best_path = artifact_directory / "best.pt"
    wandb_run = start_run(spec.run_id, {**spec.as_parameters(), "dataset": spec.dataset_name}, ["paper_1", "mhyper", "synthetic", spec.variant])
    try:
        for epoch in range(1, spec.max_epochs + 1):
            started = time.perf_counter()
            model.train()
            permutation = torch.randperm(len(dataset.train))
            total_fit = 0.0
            total_reg = 0.0
            total_reconstruction = 0.0
            for batch_start in range(0, len(permutation), spec.batch_size):
                indices = permutation[batch_start : batch_start + spec.batch_size]
                batch = dataset.train[indices]
                scores = model.score(batch, torch.arange(dataset.entity_count))
                loss = loss_function(scores, batch[:, 2])
                regularizer = model.regularization_loss() * spec.regularization_strength
                reconstruction = (
                    model.entity_embeddings.weight.square().mean() * spec.reconstruction_coefficient
                )
                objective = loss + regularizer + reconstruction
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                total_fit += float(loss.detach()) * len(batch)
                total_reg += float(regularizer.detach()) * len(batch)
                total_reconstruction += float(reconstruction.detach()) * len(batch)
            count = len(dataset.train)
            duration = time.perf_counter() - started
            reconstruction_value = spec.reconstruction_coefficient * float(
                model.entity_embeddings.weight.square().mean().detach()
            )
            values = {
                "fit_loss": total_fit / count,
                "regularizer_loss": total_reg / count,
                "reconstruction_loss": reconstruction_value
                if spec.reconstruction_coefficient
                else None,
                "total_loss": (total_fit + total_reg + total_reconstruction) / count,
                "duration_seconds": duration,
                "learning_rate": spec.learning_rate,
                "consistency_loss": None,
            }
            store.add_epoch_metric(spec.run_id, epoch, "train", values)
            log_wandb(wandb_run, {f"train/{key}": value for key, value in values.items()}, epoch)
            log.write(
                f"{utc_now()} epoch={epoch} split=train total_loss={values['total_loss']:.6f}\n"
            )
            log.flush()
            completed_epochs = epoch
            if epoch % spec.validation_interval_epochs == 0 or epoch == spec.max_epochs:
                rhs = evaluate(
                    model, dataset.valid, dataset.all_triples, dataset.entity_count, "rhs"
                )
                lhs = evaluate(
                    model, dataset.valid, dataset.all_triples, dataset.entity_count, "lhs"
                )
                both = {
                    "mrr": (rhs["mrr"] + lhs["mrr"]) / 2,
                    "hits_at_1": (rhs["hits_at_1"] + lhs["hits_at_1"]) / 2,
                    "hits_at_3": (rhs["hits_at_3"] + lhs["hits_at_3"]) / 2,
                    "hits_at_10": (rhs["hits_at_10"] + lhs["hits_at_10"]) / 2,
                    "sample_count": rhs["sample_count"] + lhs["sample_count"],
                }
                for direction, result in (("lhs", lhs), ("rhs", rhs), ("both", both)):
                    store.add_evaluation(spec.run_id, "validation", "valid", direction, result)
                log_wandb(wandb_run, {f"validation/{key}": value for key, value in both.items()}, epoch)
                store.add_epoch_metric(
                    spec.run_id, epoch, "valid", {"total_loss": None, "duration_seconds": duration}
                )
                if float(both["mrr"]) > best_mrr:
                    best_mrr = float(both["mrr"])
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
                    os.replace(temporary_path, best_path)
                if callback:
                    callback(epoch, "valid", both)
        if completed_epochs != spec.max_epochs:
            raise RuntimeError(
                f"training ended at epoch {completed_epochs}, expected {spec.max_epochs}"
            )
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        for direction in ("lhs", "rhs"):
            store.add_evaluation(
                spec.run_id,
                "best_validation",
                "test",
                direction,
                evaluate(model, dataset.test, dataset.all_triples, dataset.entity_count, direction),
            )
        lhs = evaluate(model, dataset.test, dataset.all_triples, dataset.entity_count, "lhs")
        rhs = evaluate(model, dataset.test, dataset.all_triples, dataset.entity_count, "rhs")
        both = (lhs["mrr"] + rhs["mrr"]) / 2
        store.add_evaluation(
            spec.run_id,
            "best_validation",
            "test",
            "both",
            {
                "mrr": both,
                "hits_at_1": (lhs["hits_at_1"] + rhs["hits_at_1"]) / 2,
                "hits_at_3": (lhs["hits_at_3"] + rhs["hits_at_3"]) / 2,
                "hits_at_10": (lhs["hits_at_10"] + rhs["hits_at_10"]) / 2,
                "sample_count": lhs["sample_count"] + rhs["sample_count"],
            },
        )
        log.write(
            f"{utc_now()} completed_epochs={completed_epochs} configured_epochs={spec.max_epochs} "
            f"selected_epoch={best_epoch} validation_mrr={best_mrr:.6f} test_mrr={both:.6f}\n"
        )
        log.flush()
        log_wandb(
            wandb_run,
            {
                "test/mrr": both,
                "test/hits_at_1": (lhs["hits_at_1"] + rhs["hits_at_1"]) / 2,
                "test/hits_at_3": (lhs["hits_at_3"] + rhs["hits_at_3"]) / 2,
                "test/hits_at_10": (lhs["hits_at_10"] + rhs["hits_at_10"]) / 2,
            },
            spec.max_epochs,
        )
        return {
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "best_validation_mrr": best_mrr,
            "test_mrr": both,
        }
    finally:
        log.close()
        finish_wandb(
            wandb_run,
            {
                "completed_epochs": completed_epochs,
                "configured_epochs": spec.max_epochs,
                "best_epoch": best_epoch,
                "best_validation_mrr": best_mrr,
            },
        )
