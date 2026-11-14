from __future__ import annotations

import torch
from torch import nn


def hamilton_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(*left.shape[:-1], 4, -1)
    right = right.reshape(*right.shape[:-1], 4, -1)
    a, b, c, d = left.unbind(-2)
    e, f, g, h = right.unbind(-2)
    return torch.cat(
        (
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ),
        dim=-1,
    )


class MHyperModel(nn.Module):
    def __init__(self, entity_count: int, relation_count: int, rank: int) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(entity_count, rank * 4)
        self.relation_embeddings = nn.Embedding(relation_count, rank * 4)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(
        self, triples: torch.Tensor, candidate_entities: torch.Tensor | None = None
    ) -> torch.Tensor:
        heads, relations, tails = triples.T
        head = self.entity_embeddings(heads)
        relation = self.relation_embeddings(relations)
        if candidate_entities is None:
            tail = self.entity_embeddings(tails)
            return (hamilton_product(head, relation) * tail).sum(dim=-1)
        transformed = hamilton_product(head, relation)
        candidates = self.entity_embeddings(candidate_entities)
        return transformed @ candidates.T

    def regularization_loss(self) -> torch.Tensor:
        return (
            self.entity_embeddings.weight.pow(3).abs().mean()
            + self.relation_embeddings.weight.pow(3).abs().mean()
        )
