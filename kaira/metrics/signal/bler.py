"""Block Error Rate (BLER) metric for communication systems.

BLER is a key performance indicator for block-based transmission schemes like those used
in modern wireless systems :cite:`lin2004error` :cite:`moon2005error`.
"""

from typing import Optional

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("bler")
class BlockErrorRate(BaseMetric):
    """Block Error Rate (BLER) Module.

    This metric calculates the ratio of blocks with errors to the total number of blocks. A block
    is considered erroneous if any bit/symbol within the block is incorrect.

    BLER is commonly used in communication systems to evaluate the performance of channel coding
    schemes, especially in scenarios with burst errors :cite:`lin2004error`. In modern cellular
    systems like 5G, BLER is a critical performance metric :cite:`3gpp2018nr`.
    """

    def __init__(
        self,
        block_size: Optional[int] = None,
        threshold: float = 0.0,
        reduction: str = "mean",
        name: Optional[str] = None,
    ) -> None:
        """Initialize the BlockErrorRate module.

        Args:
            block_size (Optional[int]): Size of each block in the input.
                If None, each row of the input is treated as a separate block.
            threshold (float): Threshold for considering values as different.
                Useful for floating-point comparisons.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
            name (Optional[str]): Name for the metric.
        """
        super().__init__(name=name or "BLER")
        self.block_size = block_size
        self.threshold = threshold
        self.reduction = reduction

        self.register_buffer("total_blocks", torch.tensor(0))
        self.register_buffer("error_blocks", torch.tensor(0))

    def _reshape_into_blocks(self, x: Tensor) -> Tensor:
        """Reshape input tensor into blocks based on block_size.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Reshaped tensor with blocks as the second dimension
        """
        if self.block_size is None:
            # Each row is a separate block
            return x

        # Handle empty tensor case
        if x.numel() == 0:
            return x.reshape(0, 0, 0)

        batch_size = x.size(0)
        # Make sure tensor can be evenly divided into blocks
        if x.numel() % (batch_size * self.block_size) != 0:
            raise ValueError(f"Input size {x.numel()} is not divisible by batch_size ({batch_size}) " f"multiplied by block_size ({self.block_size})")

        # Reshape to [batch_size, num_blocks, block_size, ...]
        remainder_dims = x.shape[1:]
        elements_per_batch = torch.tensor(remainder_dims).prod().item()
        num_blocks = elements_per_batch // self.block_size

        # Handle case where input has more than 2 dimensions
        if len(remainder_dims) > 1:
            # Flatten the input to [batch_size, -1] first
            x_flat = x.reshape(batch_size, -1)
            # Then reshape to [batch_size, num_blocks, block_size]
            return x_flat.reshape(batch_size, num_blocks, self.block_size)
        else:
            # Simple case: input is [batch_size, sequence_length]
            return x.reshape(batch_size, num_blocks, self.block_size)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Calculate Block Error Rate between predictions and targets.

        Args:
            preds (Tensor): Predicted values
            targets (Tensor): Target values

        Returns:
            Tensor: Block Error Rate value
        """
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {preds.shape} vs {targets.shape}")

        # Handle empty tensors
        if preds.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0)

        # Reshape inputs into blocks if needed
        if self.block_size is not None:
            preds_blocks = self._reshape_into_blocks(preds)
            targets_blocks = self._reshape_into_blocks(targets)

            # Check if any element in each block has an error that exceeds the threshold
            errors = torch.abs(preds_blocks - targets_blocks) > self.threshold

            # Reduce along block_size dimension to check if any element has error in each block
            block_errors = errors.any(dim=-1)
        else:
            # Each row is already a block
            errors = torch.abs(preds - targets) > self.threshold
            errors_flat = errors.reshape(errors.shape[0], -1)
            block_errors = errors_flat.any(dim=-1)

        # Apply reduction
        if self.reduction == "none":
            return block_errors.float()
        elif self.reduction == "sum":
            return block_errors.sum().float()
        else:  # default: 'mean'
            num_errors = block_errors.sum().item()
            total_blocks = block_errors.numel()
            # Ensure exact fraction for the test cases
            return torch.tensor(float(num_errors) / float(total_blocks) if total_blocks > 0 else 0.0)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the internal state with a batch of samples.

        Args:
            preds (Tensor): Predicted values
            targets (Tensor): Target values
        """
        if preds.numel() == 0 or targets.numel() == 0:
            return

        if self.block_size is not None:
            preds_blocks = self._reshape_into_blocks(preds)
            targets_blocks = self._reshape_into_blocks(targets)

            # Count blocks with errors
            errors = torch.abs(preds_blocks - targets_blocks) > self.threshold
            block_errors = errors.any(dim=-1)

            # Count total blocks and blocks with errors
            self.total_blocks += block_errors.numel()
            self.error_blocks += block_errors.sum().item()
        else:
            # Each row is a block
            errors = torch.abs(preds - targets) > self.threshold
            errors_flat = errors.reshape(errors.shape[0], -1)
            block_errors = errors_flat.any(dim=-1)

            self.total_blocks += block_errors.numel()
            self.error_blocks += block_errors.sum().item()

    def compute(self) -> Tensor:
        """Compute accumulated block error rate.

        Returns:
            Tensor: Block error rate value
        """
        # Return exact fraction to avoid floating-point issues in tests
        return torch.tensor(float(self.error_blocks) / float(max(self.total_blocks, 1)), dtype=torch.float32)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_blocks.zero_()
        self.error_blocks.zero_()


# Alias for backward compatibility and convenience
BLER = BlockErrorRate
SymbolErrorRate = BlockErrorRate
FrameErrorRate = BlockErrorRate
FER = BlockErrorRate
SER = BlockErrorRate
MetricRegistry.register_metric("fer")(BlockErrorRate)  # Register FER (Frame Error Rate) as another alias
MetricRegistry.register_metric("ser")(BlockErrorRate)  # Register SER (Frame Error Rate) as another alias
