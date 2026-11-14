from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticDataset:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    entity_count: int
    relation_count: int

    @property
    def all_triples(self) -> torch.Tensor:
        return torch.cat((self.train, self.valid, self.test), dim=0)


def make_synthetic_dataset() -> SyntheticDataset:
    triples = []
    for relation in range(3):
        for head in range(8):
            triples.append((head, relation, (head + relation + 1) % 12))
    tensor = torch.tensor(triples, dtype=torch.long)
    return SyntheticDataset(tensor[:12], tensor[12:18], tensor[18:], 12, 3)
