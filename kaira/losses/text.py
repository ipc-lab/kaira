"""Text Losses module for Kaira.

This module contains various loss functions for training text-based systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss
from .registry import LossRegistry


@LossRegistry.register_loss()
class CrossEntropyLoss(BaseLoss):
    """Cross Entropy Loss Module.

    This module calculates the cross entropy loss for classification tasks.
    """

    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0):
        """Initialize the CrossEntropyLoss module.

        Args:
            weight (torch.Tensor, optional): Class weights. Default is None.
            ignore_index (int): Index to ignore. Default is -100.
            label_smoothing (float): Label smoothing value. Default is 0.0.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CrossEntropyLoss module.

        Args:
            x (torch.Tensor): The input logits tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The cross entropy loss.
        """
        return self.ce(x, target)


@LossRegistry.register_loss()
class LabelSmoothingLoss(BaseLoss):
    """Label Smoothing Loss Module.

    This module implements label smoothing to prevent overconfidence.
    """

    def __init__(self, smoothing=0.1, classes=0, dim=-1):
        """Initialize the LabelSmoothingLoss module.

        Args:
            smoothing (float): Smoothing factor. Default is 0.1.
            classes (int): Number of classes. Default is 0.
            dim (int): Dimension to reduce. Default is -1.
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LabelSmoothingLoss module.

        Args:
            x (torch.Tensor): The input logits tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The label smoothing loss.
        """
        assert x.size(1) == self.classes

        log_probs = F.log_softmax(x, dim=self.dim)

        # Hard targets
        nll_loss = -log_probs.gather(dim=self.dim, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # Smoothed targets
        smooth_loss = -log_probs.sum(dim=self.dim)

        # Combine losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss / self.classes

        return loss.mean()


@LossRegistry.register_loss()
class CosineSimilarityLoss(BaseLoss):
    """Cosine Similarity Loss Module.

    This module calculates loss based on cosine similarity between embeddings.
    """

    def __init__(self, margin=0.0):
        """Initialize the CosineSimilarityLoss module.

        Args:
            margin (float): Margin for similarity. Default is 0.0.
        """
        super().__init__()
        self.margin = margin

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CosineSimilarityLoss module.

        Args:
            x (torch.Tensor): The input embeddings tensor.
            target (torch.Tensor): The target embeddings tensor.

        Returns:
            torch.Tensor: The cosine similarity loss.
        """
        # Normalize embeddings
        x_norm = F.normalize(x, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Calculate cosine similarity
        cosine_sim = torch.sum(x_norm * target_norm, dim=1)

        # Calculate loss
        loss = torch.mean(torch.clamp(self.margin - cosine_sim, min=0.0))

        return loss


@LossRegistry.register_loss()
class Word2VecLoss(BaseLoss):
    """Word2Vec Loss Module.

    This module implements the negative sampling loss used in Word2Vec.
    """

    def __init__(self, embedding_dim, vocab_size, n_negatives=5):
        """Initialize the Word2VecLoss module.

        Args:
            embedding_dim (int): Dimensionality of embeddings.
            vocab_size (int): Size of vocabulary.
            n_negatives (int): Number of negative samples. Default is 5.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_negatives = n_negatives

        # Initialize embeddings
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights
        self.in_embed.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.out_embed.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, input_idx: torch.Tensor, output_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Word2VecLoss module.

        Args:
            input_idx (torch.Tensor): Input word indices.
            output_idx (torch.Tensor): Output context word indices.

        Returns:
            torch.Tensor: The Word2Vec loss.
        """
        batch_size = input_idx.size(0)

        # Get embeddings
        input_emb = self.in_embed(input_idx)  # [batch_size, embed_dim]
        output_emb = self.out_embed(output_idx)  # [batch_size, embed_dim]

        # Positive samples
        pos_score = torch.sum(input_emb * output_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative samples
        neg_samples = torch.randint(0, self.vocab_size, (batch_size, self.n_negatives), device=input_idx.device)
        neg_emb = self.out_embed(neg_samples)  # [batch_size, n_negatives, embed_dim]

        # Calculate negative scores
        neg_score = torch.bmm(neg_emb, input_emb.unsqueeze(2)).squeeze(2)  # [batch_size, n_negatives]
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        # Total loss
        loss = -(pos_loss + neg_loss).mean()

        return loss


__all__ = ["CrossEntropyLoss", "LabelSmoothingLoss", "CosineSimilarityLoss", "Word2VecLoss"]
