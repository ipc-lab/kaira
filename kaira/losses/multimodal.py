"""Multimodal Losses module for Kaira.

This module contains various loss functions for training multimodal systems.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .registry import LossRegistry


@LossRegistry.register_loss()
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
            labels = torch.arange(similarity.size(0), device=similarity.device)
        else:
            labels = labels.long()  # Ensure labels are of type Long

        # Compute loss
        loss = F.cross_entropy(similarity, labels)

        return loss


@LossRegistry.register_loss()
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
        if distance not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported distance metric: {distance}")

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


@LossRegistry.register_loss()
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

    def forward(self, query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the InfoNCELoss module.

        Args:
            query (torch.Tensor): Query embeddings from one modality.
            key (torch.Tensor): Key embeddings from another modality (positives).
            queue (torch.Tensor, optional): Queue of negative samples. Default is None.
            mask (torch.Tensor, optional): Binary mask defining positive pairs. Default is None.
                Shape should be [query.size(0), key.size(0)] where 1 indicates a positive pair.

        Returns:
            torch.Tensor: The InfoNCE loss.
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)

        # Handle different masking scenarios
        if queue is not None:
            # Compute positive logits
            l_pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)

            # Compute negative logits with queue
            queue = F.normalize(queue, p=2, dim=1)
            l_neg = torch.einsum("nc,kc->nk", [query, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)

            # Labels: positives are the 0-th
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        else:
            # Compute all pairwise similarities
            similarities = torch.einsum("nc,kc->nk", [query, key])

            if mask is not None:
                # Apply custom masking to define positives and negatives
                # Make sure the mask is properly shaped
                assert mask.shape == similarities.shape, "Mask shape must match similarity matrix shape"

                # For each query, get the positive key with the highest similarity
                positive_mask = mask.bool()
                negative_mask = ~positive_mask

                # Replace non-positive similarities with -inf
                masked_similarities = similarities.clone()
                masked_similarities.masked_fill_(negative_mask, float("-inf"))

                # Get positive logits (max similarity for each query among its positive keys)
                l_pos = masked_similarities.max(dim=1, keepdim=True)[0]

                # Prepare negative logits
                # Replace diagonal with -inf to avoid self-contrast if not already masked
                diag_mask = torch.eye(similarities.shape[0], device=similarities.device).bool()
                negative_mask = negative_mask & ~diag_mask  # Remove diagonal from negatives

                # Extract only negative similarities
                l_neg = similarities.masked_select(negative_mask).reshape(similarities.shape[0], -1)

                if l_neg.shape[1] == 0:  # No negatives found
                    # Just minimize distance between positive pairs
                    return -l_pos.mean()

                # Concatenate positive and negative logits
                logits = torch.cat([l_pos, l_neg], dim=1)

                # Labels: positives are at index 0
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
            else:
                # Default behavior: use diagonal elements as positives
                # Get positive logits (diagonal elements)
                l_pos = torch.diag(similarities).unsqueeze(-1)

                # Remove diagonal from similarities to get negative logits
                mask = torch.eye(similarities.shape[0], device=similarities.device)
                similarities.masked_fill_(mask.bool(), float("-inf"))
                l_neg = similarities

                # Concatenate positive and negative logits
                logits = torch.cat([l_pos, l_neg], dim=1)

                # Labels: positives are at index 0
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)

        # Scale with temperature
        logits /= self.temperature

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss


@LossRegistry.register_loss()
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


@LossRegistry.register_loss()
class AlignmentLoss(BaseLoss):
    """Alignment Loss for multimodal embeddings.

    This module aligns embeddings from different modalities.
    """

    def __init__(self, alignment_type="l2", projection_dim=None):
        """Initialize the AlignmentLoss module.

        Args:
            alignment_type (str): Type of alignment ('l1', 'l2', or 'cosine'). Default is 'l2'.
            projection_dim (int, optional): Dimension to project embeddings to before computing loss.
                If None, no projection is performed. Default is None.
        """
        super().__init__()
        self.alignment_type = alignment_type
        self.projection_dim = projection_dim

        if alignment_type not in ["l1", "l2", "cosine"]:
            raise ValueError(f"Unsupported alignment type: {alignment_type}")

        # Create projection layer if needed
        self.projector = None
        if self.projection_dim is not None:
            self.projector = torch.nn.Linear(in_features=1, out_features=projection_dim, bias=False)
            # We'll initialize the actual weights in the forward pass when we know the input dimension

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AlignmentLoss module.

        Args:
            x1 (torch.Tensor): Embeddings from the first modality.
            x2 (torch.Tensor): Embeddings from the second modality.

        Returns:
            torch.Tensor: The alignment loss.
        """
        # Apply projection if needed
        if self.projection_dim is not None:
            # Initialize the projector if it's the first call
            if self.projector.in_features != x1.shape[1]:
                # Replace the projector with a properly sized one
                device = x1.device
                self.projector = torch.nn.Linear(in_features=x1.shape[1], out_features=self.projection_dim, bias=False).to(device)
                # Initialize with orthogonal weights for better preservation of distances
                torch.nn.init.orthogonal_(self.projector.weight)

            # Apply projection
            x1 = self.projector(x1)
            x2 = self.projector(x2)

        # Compute alignment loss based on the chosen type
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


__all__ = ["ContrastiveLoss", "TripletLoss", "InfoNCELoss", "CMCLoss", "AlignmentLoss"]
