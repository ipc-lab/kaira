"""Block Error Rate (BLER) metric for communication systems.

BLER is a key performance indicator for block-based transmission schemes like those used
in modern wireless systems :cite:`lin2004error` :cite:`moon2005error`.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("bler")
class BlockErrorRate(BaseMetric):
    """Block Error Rate (BLER) metric.

    BLER measures the number of blocks containing at least one error divided by the total number of
    blocks transmitted. Lower values indicate better performance. It's commonly used in systems
    employing block codes :cite:`lin2004error`.

    This metric can also be used for Frame Error Rate (FER) or Symbol Error Rate (SER) by setting
    the `block_size` appropriately or leaving it as None to treat each row as a block/frame/symbol.

    Attributes:
        block_size (Optional[int]): Size of each block. If None, each row is a block.
        threshold (float): Threshold for considering values as different.
        reduction (str): Reduction method ('mean', 'sum', 'none').
        total_blocks (Tensor): Accumulated total number of blocks processed.
        error_blocks (Tensor): Accumulated number of blocks with errors.
    """

    is_differentiable = False
    higher_is_better = False

    def __init__(
        self,
        block_size: Optional[int] = None,
        threshold: float = 0.0,
        reduction: str = "mean",
        name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the BlockErrorRate module.

        Args:
            block_size (Optional[int]): Size of each block in the input.
                If None, each row of the input is treated as a separate block.
            threshold (float): Threshold for considering values as different.
                Useful for floating-point comparisons.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
            name (Optional[str]): Name for the metric.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "BLER")  # Pass only name
        if block_size is not None and block_size <= 0:
            raise ValueError("block_size must be a positive integer or None")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.block_size = block_size
        self.threshold = threshold
        self.reduction = reduction

        self.register_buffer("total_blocks", torch.tensor(0))
        self.register_buffer("error_blocks", torch.tensor(0))

    def _reshape_into_blocks(self, data: Tensor) -> Tensor:
        """Reshape input tensor into blocks based on block_size."""
        if data.numel() == 0:
            batch_size = data.shape[0]
            if self.block_size is None:
                # Shape: [batch_size, 1, elements_per_row (0 if original cols=0)]
                return data.reshape(batch_size, 1, -1)
            else:
                # Shape: [batch_size, num_blocks (0), block_size]
                # Explicitly set the middle dimension to 0 for empty tensors
                return data.reshape(batch_size, 0, self.block_size)

        batch_size = data.shape[0]
        remainder_dims = data.shape[1:]

        if self.block_size is None:
            # Treat each row as a block
            return data.reshape(batch_size, 1, -1)  # Shape: [batch_size, 1, elements_per_row]

        elements_per_batch_item = torch.prod(torch.tensor(remainder_dims)).item()
        if elements_per_batch_item % self.block_size != 0:
            raise ValueError(f"Total elements per batch item ({elements_per_batch_item}) must be divisible by block_size ({self.block_size})")

        num_blocks = elements_per_batch_item // self.block_size

        # Flatten the non-batch dimensions first
        data_flat = data.reshape(batch_size, -1)
        # Reshape into blocks
        return data_flat.reshape(batch_size, num_blocks, self.block_size)

    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the Block Error Rate for the current batch.

        Args:
            x (Tensor): The predicted tensor.
            y (Tensor): The target tensor.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tensor: Block error rate for the batch, potentially reduced.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Reshape inputs into blocks
        x_blocks = self._reshape_into_blocks(x)
        y_blocks = self._reshape_into_blocks(y)

        # Check for errors within each block
        errors_in_block = torch.abs(x_blocks - y_blocks) > self.threshold
        # Determine if any error exists in each block
        block_has_error = errors_in_block.any(dim=-1)  # Check along the block_size dimension

        if self.reduction == "none":
            # Return error status for each block (flattened across batches)
            return block_has_error.float().flatten()
        elif self.reduction == "sum":
            # Return total number of error blocks
            return block_has_error.sum().float()
        else:  # reduction == "mean"
            # Return the mean error rate
            num_errors = block_has_error.sum().item()
            total_blocks = block_has_error.numel()
            # Ensure exact fraction for the test cases
            return torch.tensor(float(num_errors) / float(total_blocks) if total_blocks > 0 else 0.0)

    def update(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> None:
        """Update accumulated statistics with results from a new batch.

        Args:
            x (Tensor): The predicted tensor for the current batch.
            y (Tensor): The target tensor for the current batch.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        x_blocks = self._reshape_into_blocks(x)
        y_blocks = self._reshape_into_blocks(y)

        errors_in_block = torch.abs(x_blocks - y_blocks) > self.threshold
        block_has_error = errors_in_block.any(dim=-1)

        self.total_blocks += block_has_error.numel()
        self.error_blocks += block_has_error.sum().item()

    def compute(self) -> Tensor:
        """Compute accumulated block error rate.

        Returns:
            Tensor: Block error rate value
        """
        # Return exact fraction to avoid floating-point issues in tests
        return torch.tensor(float(self.error_blocks) / float(max(self.total_blocks.item(), 1)), dtype=torch.float32)

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
