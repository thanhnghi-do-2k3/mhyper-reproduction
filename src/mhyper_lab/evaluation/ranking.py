from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RankingSummary:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    sample_count: int


def calculate_filtered_ranks(
    scores: torch.Tensor, targets: torch.Tensor, known_true: list[set[int]]
) -> torch.Tensor:
    ranks: list[int] = []
    for row, target, true_entities in zip(scores, targets.tolist(), known_true):
        filtered = row.clone()
        for entity_id in true_entities:
            if entity_id != target:
                filtered[entity_id] = -torch.inf
        target_score = filtered[target]
        ranks.append(int((filtered > target_score).sum().item()) + 1)
    return torch.tensor(ranks, dtype=torch.long)


def summarize_ranks(ranks: torch.Tensor) -> RankingSummary:
    if ranks.numel() == 0:
        raise ValueError("cannot summarize empty ranks")
    reciprocal = 1.0 / ranks.float()
    return RankingSummary(
        mrr=float(reciprocal.mean().item()),
        hits_at_1=float((ranks <= 1).float().mean().item()),
        hits_at_3=float((ranks <= 3).float().mean().item()),
        hits_at_10=float((ranks <= 10).float().mean().item()),
        sample_count=int(ranks.numel()),
    )
