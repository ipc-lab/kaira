import pytest
import torch
from kaira.losses.text import CrossEntropyLoss, LabelSmoothingLoss, CosineSimilarityLoss, Word2VecLoss

@pytest.fixture
def sample_logits():
    return torch.randn(3, 5)  # 3 samples, 5 classes

@pytest.fixture
def sample_targets():
    return torch.tensor([1, 0, 4])  # Target classes for the samples

def test_cross_entropy_loss_forward(sample_logits, sample_targets):
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(sample_logits, sample_targets)
    assert isinstance(loss, torch.Tensor)

def test_label_smoothing_loss_forward(sample_logits, sample_targets):
    label_smoothing_loss = LabelSmoothingLoss(smoothing=0.1, classes=5)
    loss = label_smoothing_loss(sample_logits, sample_targets)
    assert isinstance(loss, torch.Tensor)

def test_cosine_similarity_loss_forward():
    x = torch.randn(10, 128)  # 10 samples with 128-dimensional embeddings
    target = torch.randn(10, 128)  # Target embeddings
    cosine_similarity_loss = CosineSimilarityLoss(margin=0.5)
    loss = cosine_similarity_loss(x, target)
    assert isinstance(loss, torch.Tensor)

def test_word2vec_loss_forward():
    input_idx = torch.tensor([1, 2, 3])
    output_idx = torch.tensor([4, 5, 6])
    word2vec_loss = Word2VecLoss(embedding_dim=128, vocab_size=100)
    loss = word2vec_loss(input_idx, output_idx)
    assert isinstance(loss, torch.Tensor)
