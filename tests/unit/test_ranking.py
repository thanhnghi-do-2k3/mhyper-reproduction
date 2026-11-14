import pytest
import torch

from mhyper_lab.evaluation.ranking import calculate_filtered_ranks, summarize_ranks


def test_filtered_rank_excludes_other_true_entities():
    scores = torch.tensor([[0.9, 0.8, 0.7]])
    ranks = calculate_filtered_ranks(scores, torch.tensor([0]), [{0, 1}])
    assert ranks.tolist() == [1]


def test_rank_summary_has_observed_sample_count():
    summary = summarize_ranks(torch.tensor([1, 2, 10]))
    assert summary.sample_count == 3
    assert summary.hits_at_1 == pytest.approx(1 / 3)
