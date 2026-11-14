import torch

from mhyper_lab.training.model import MHyperModel, hamilton_product


def test_hamilton_product_identity():
    left = torch.tensor([[1.0, 0, 0, 0]])
    right = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    assert torch.equal(hamilton_product(left, right), right)


def test_model_score_shape_and_gradient():
    model = MHyperModel(5, 2, 3)
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
    scores = model.score(triples, torch.arange(5))
    assert scores.shape == (2, 5)
    scores.sum().backward()
    assert model.entity_embeddings.weight.grad is not None
