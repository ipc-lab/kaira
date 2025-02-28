"""Multimodal Losses module for Kaira.

This module contains various loss functions for training multimodal systems.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss


class ContrastiveLoss(BaseLoss):
    """Contrastive Loss Module.

    This module calculates contrastive loss between embeddings from different modalities.
    """

    def __init__(self, margin=0.2, temperature=0.07):
        """Initialize the ContrastiveLoss module.

        Args:
            margin (float): Margin for contrastive loss. Default is 0.2.
            temperature (float): Temperature scaling factor. Default is 0.07.
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the ContrastiveLoss module.

        Args:
            embeddings1 (torch.Tensor): Embeddings from the first modality.
            embeddings2 (torch.Tensor): Embeddings from the second modality.
            labels (torch.Tensor, optional): Matching labels. Default is None (assumes paired data).

        Returns:
            torch.Tensor: The contrastive loss between the modalities.
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Calculate cosine similarity
        similarity = torch.mm(embeddings1, embeddings2.t()) / self.temperature

        # For paired data (default)
        if labels is None:
            # Positive pairs are on the diagonal
            labels = torch.arange(similarity.size(0), device=similarity.device)

        # Compute loss
        loss = F.cross_entropy(similarity, labels)

        return loss


class TripletLoss(BaseLoss):
    """Triplet Loss Module for multimodal data.

    This module implements triplet loss with hard negative mining.
    """

    def __init__(self, margin=0.3, distance="cosine"):
        """Initialize the TripletLoss module.

        Args:
            margin (float): Margin for triplet loss. Default is 0.3.
            distance (str): Distance metric ('cosine' or 'euclidean'). Default is 'cosine'.
        """
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the TripletLoss module.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor, optional): Explicit negative embeddings.
            labels (torch.Tensor, optional): Labels for online mining. Default is None.

        Returns:
            torch.Tensor: The triplet loss.
        """
        if self.distance == "cosine":
            # Normalize for cosine distance
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)

            # Calculate cosine similarity
            pos_sim = torch.sum(anchor * positive, dim=1)
            pos_dist = 1.0 - pos_sim

            if negative is not None:
                negative = F.normalize(negative, p=2, dim=1)
                neg_sim = torch.sum(anchor * negative, dim=1)
                neg_dist = 1.0 - neg_sim
            elif labels is not None:
                # Online mining using labels
                all_dists = []
                for i in range(anchor.size(0)):
                    neg_mask = labels != labels[i]
                    if not torch.any(neg_mask):
                        continue

                    curr_anchor = anchor[i].unsqueeze(0)
                    neg_candidates = anchor[neg_mask]

                    neg_sims = torch.mm(curr_anchor, neg_candidates.t()).squeeze()
                    hardest_neg_sim = torch.max(neg_sims)
                    all_dists.append(1.0 - hardest_neg_sim)

                if all_dists:
                    neg_dist = torch.stack(all_dists)
                else:
                    return pos_dist.mean()  # No negatives found
            else:
                raise ValueError("Either negative samples or labels must be provided")

        else:  # euclidean
            pos_dist = torch.pairwise_distance(anchor, positive)

            if negative is not None:
                neg_dist = torch.pairwise_distance(anchor, negative)
            elif labels is not None:
                # Online mining using labels
                all_dists = []
                for i in range(anchor.size(0)):
                    neg_mask = labels != labels[i]
                    if not torch.any(neg_mask):
                        continue

                    curr_anchor = anchor[i].unsqueeze(0).expand(torch.sum(neg_mask), -1)
                    neg_candidates = anchor[neg_mask]

                    dists = torch.pairwise_distance(curr_anchor, neg_candidates)
                    hardest_neg_dist = torch.min(dists)
                    all_dists.append(hardest_neg_dist)

                if all_dists:
                    neg_dist = torch.stack(all_dists)
                else:
                    return pos_dist.mean()  # No negatives found
            else:
                raise ValueError("Either negative samples or labels must be provided")

        # Calculate triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return loss.mean()


class InfoNCELoss(BaseLoss):
    """InfoNCE Loss Module for multimodal contrastive learning.

    This module implements the Noise Contrastive Estimation loss.
    """

    def __init__(self, temperature=0.07):
        """Initialize the InfoNCELoss module.

        Args:
            temperature (float): Temperature scaling factor. Default is 0.07.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the InfoNCELoss module.

        Args:
            query (torch.Tensor): Query embeddings from one modality.
            key (torch.Tensor): Key embeddings from another modality (positives).
            queue (torch.Tensor, optional): Queue of negative samples. Default is None.

        Returns:
            torch.Tensor: The InfoNCE loss.
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)

        # Positive logits: NxN matrix
        l_pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)

        # Negative logits
        if queue is not None:
            queue = F.normalize(queue, p=2, dim=1)
            l_neg = torch.einsum("nc,kc->nk", [query, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)

            # Labels: positives are the 0-th
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        else:
            # Use other samples in the batch as negatives
            l_neg = torch.einsum("nc,kc->nk", [query, key])
            # Remove diagonal (self-similarity)
            mask = torch.eye(l_neg.shape[0], device=query.device)
            l_neg.masked_fill_(mask.bool(), float("-inf"))

            logits = torch.cat([l_pos, l_neg], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)

        # Scale with temperature
        logits /= self.temperature

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss


class CMCLoss(BaseLoss):
    """Cross-Modal Consistency Loss Module.

    This module implements a loss to ensure consistency across modalities.
    """

    def __init__(self, lambda_cmc=1.0):
        """Initialize the CMCLoss module.

        Args:
            lambda_cmc (float): Weight for the CMC loss. Default is 1.0.
        """
        super().__init__()
        self.lambda_cmc = lambda_cmc

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, proj1: BaseLoss, proj2: BaseLoss) -> torch.Tensor:
        """Forward pass through the CMCLoss module.

        Args:
            x1 (torch.Tensor): Features from the first modality.
            x2 (torch.Tensor): Features from the second modality.
            proj1 (BaseLoss): Projection head for the first modality.
            proj2 (BaseLoss): Projection head for the second modality.

        Returns:
            torch.Tensor: The cross-modal consistency loss.
        """
        z1 = proj1(x1)
        z2 = proj2(x2)

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Cross-modal similarity
        sim_1to2 = torch.mm(z1, z2.t())
        sim_2to1 = torch.mm(z2, z1.t())

        # Target: identity matrix (matching indices should have high similarity)
        targets = torch.arange(z1.size(0), device=z1.device)

        # Calculate loss
        loss = (F.cross_entropy(sim_1to2, targets) + F.cross_entropy(sim_2to1, targets)) / 2

        return self.lambda_cmc * loss


class AlignmentLoss(BaseLoss):
    """Alignment Loss for multimodal embeddings.

    This module aligns embeddings from different modalities.
    """

    def __init__(self, alignment_type="l2"):
        """Initialize the AlignmentLoss module.

        Args:
            alignment_type (str): Type of alignment ('l1', 'l2', or 'cosine'). Default is 'l2'.
        """
        super().__init__()
        self.alignment_type = alignment_type

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AlignmentLoss module.

        Args:
            x1 (torch.Tensor): Embeddings from the first modality.
            x2 (torch.Tensor): Embeddings from the second modality.

        Returns:
            torch.Tensor: The alignment loss.
        """
        if self.alignment_type == "l1":
            return F.l1_loss(x1, x2)
        elif self.alignment_type == "l2":
            return F.mse_loss(x1, x2)
        elif self.alignment_type == "cosine":
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            return 1 - torch.mean(torch.sum(x1 * x2, dim=1))
        else:
            raise ValueError(f"Unsupported alignment type: {self.alignment_type}")
