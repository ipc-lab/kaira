import pytest
import torch

from kaira.losses.multimodal import (
    AlignmentLoss,
    CMCLoss,
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
)


@pytest.fixture
def embeddings1():
    return torch.randn(10, 128)


@pytest.fixture
def embeddings2():
    return torch.randn(10, 128)


@pytest.fixture
def labels():
    return torch.randint(0, 2, (10,))


def test_contrastive_loss_forward(embeddings1, embeddings2, labels):
    contrastive_loss = ContrastiveLoss()
    loss = contrastive_loss(embeddings1, embeddings2, labels)
    assert isinstance(loss, torch.Tensor)


def test_triplet_loss_forward(embeddings1, embeddings2, labels):
    triplet_loss = TripletLoss()
    loss = triplet_loss(embeddings1, embeddings2, labels=labels)
    assert isinstance(loss, torch.Tensor)


def test_infonce_loss_forward(embeddings1, embeddings2):
    infonce_loss = InfoNCELoss()
    loss = infonce_loss(embeddings1, embeddings2)
    assert isinstance(loss, torch.Tensor)


def test_cmc_loss_forward(embeddings1, embeddings2):
    class DummyProjection(torch.nn.Module):
        def forward(self, x):
            return x

    proj1 = DummyProjection()
    proj2 = DummyProjection()
    cmc_loss = CMCLoss()
    loss = cmc_loss(embeddings1, embeddings2, proj1, proj2)
    assert isinstance(loss, torch.Tensor)


def test_alignment_loss_forward(embeddings1, embeddings2):
    alignment_loss = AlignmentLoss()
    loss = alignment_loss(embeddings1, embeddings2)
    assert isinstance(loss, torch.Tensor)
