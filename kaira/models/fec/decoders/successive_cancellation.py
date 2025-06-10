"""
SuccessiveCancellationDecoder: Decoder for Polar codes using the Successive Cancellation (SC) method.

This class implements the Successive Cancellation algorithm for decoding Polar codes. It processes the received codeword
and estimates the transmitted message bits. The SC method is a recursive decoding algorithm that leverages the structure
of Polar codes for efficient decoding.

Attributes:
    encoder (PolarCodeEncoder): The Polar code encoder used for encoding messages.
    info_indices (torch.Tensor): Indices of information bits in the Polar code.
    device (torch.device): Device on which the decoder operates (e.g., CPU or GPU).
    dtype (torch.dtype): Data type used for computations.
    polar_i (bool): Indicates whether polar_i is enabled in the encoder.
    frozen_zeros (bool): Indicates whether frozen bits are initialized to zeros.
    m (int): Number of stages in the Polar code.
    regime (str): Decoding regime ('sum_product' or 'min_sum').
    clip (float): Clipping value for numerical stability.

References:
    :cite:`arikan2009channel`
"""

from typing import Any, Tuple

import torch

from kaira.models.fec.encoders.polar_code import PolarCodeEncoder

from ..utils import apply_blockwise, min_sum, sign_to_bin, sum_product
from .base import BaseBlockDecoder


class SuccessiveCancellationDecoder(BaseBlockDecoder[PolarCodeEncoder]):
    """Decoder for Polar code using Successive Cancellation (SC) method :cite:`arikan2009channel`.

    This class implements the Successive Cancellation algorithm for decoding Polar codes. It processes the received codeword
    and estimates the transmitted message bits. The SC method is a recursive decoding algorithm that leverages the structure
    of Polar codes for efficient decoding.

    Attributes:
        encoder (PolarCodeEncoder): The Polar code encoder used for encoding messages.
        info_indices (torch.Tensor): Indices of information bits in the Polar code.
        device (torch.device): Device on which the decoder operates (e.g., CPU or GPU).
        dtype (torch.dtype): Data type used for computations.
        polar_i (bool): Indicates whether polar_i is enabled in the encoder.
        frozen_zeros (bool): Indicates whether frozen bits are initialized to zeros.
        m (int): Number of stages in the Polar code.
        regime (str): Decoding regime ('sum_product' or 'min_sum').
        clip (float): Clipping value for numerical stability.
    """

    def __init__(self, encoder: PolarCodeEncoder, *args: Any, **kwargs: Any):
        super().__init__(encoder, *args, **kwargs)
        self.info_indices = encoder.info_indices
        self.device = encoder.device
        self.dtype = encoder.dtype
        self.polar_i = encoder.polar_i
        self.frozen_zeros = encoder.frozen_zeros
        self.m = encoder.m  # Number of stages in the polar code: m = log2(code_length)
        self.regime = kwargs.get("regime", "sum_product")  # Decoding regime: 'sum_product' or 'min_sum'
        if self.regime not in ["sum_product", "min_sum"]:
            raise ValueError("Invalid regime. Choose either 'sum_product' or 'min_sum'.")

        self.clip = kwargs.get("clip", 1000.0)  # Default clip value for numerical stability

    def f2(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Combine two binary vectors using XOR operation.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): Tuple of two binary tensors of shape (batch_size, n).

        Returns:
            torch.Tensor: Combined binary tensor of shape (batch_size, n).
        """
        x1, x2 = x
        return torch.cat([torch.remainder(x1 + x2, 2), x2], dim=1)

    def checknode(self, y: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Check node operation (sum-product or min-sum) for Successive Cancellation decoding.

        Args:
            y (Tuple[torch.Tensor, torch.Tensor]): Tuple of two tensors representing the received codeword.

        Returns:
            torch.Tensor: Processed tensor after check node operation.
        """
        y1, y2 = y
        if self.regime == "sum_product":
            return (sum_product(y1, y2)).clip(-self.clip, self.clip)
        if self.regime == "min_sum":
            return (min_sum(y1, y2)).clip(-self.clip, self.clip)

    def bitnode(self, y: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Bit node operation for Successive Cancellation decoding.

        Args:
            y (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing:
                - y_even: Tensor of shape (batch_size, n/2) for even positions.
                - y_odd: Tensor of shape (batch_size, n/2) for odd positions.
                - x: Tensor of shape (batch_size, n) representing the current estimate.

        Returns:
            torch.Tensor: Processed tensor after bit node operation.
        """
        y1, y2, x = y
        return y2 + (1 - 2 * x) * y1

    def decode_recursive(self, y: torch.Tensor, info_indices: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decodes the received codeword using the Successive Cancellation algorithm.

        This method recursively processes the received codeword to estimate the transmitted message bits. It splits the input
        into even and odd positions, applies check node and bit node operations, and combines the results using the Polar code structure.

        Args:
            y (torch.Tensor): Received codeword tensor of shape (batch_size, code_length).
            info_indices (torch.Tensor): Boolean array indicating positions of information bits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Estimated message bits tensor of shape (batch_size, code_dimension).
                - Estimated codeword bits tensor of shape (batch_size, code_length).
                - Log-likelihood ratio (LLR) tensor of shape (batch_size, code_length).
        """
        N = len(info_indices)
        if N == 1:
            llr = y.reshape(-1, 1)
            if info_indices[0]:
                x = sign_to_bin(torch.sign(llr))
            else:
                if self.frozen_zeros:
                    frozen = torch.zeros_like(llr).to(y.device)
                else:
                    frozen = torch.ones_like(llr).to(y.device)
                x = frozen.reshape(-1, 1)
            x = x.reshape(-1, 1)
            return x, x.clone(), llr.reshape(-1, 1).clone()

        if not self.polar_i:
            even_pos = torch.arange(0, N // 2).reshape(-1).to(self.device)
            odd_pos = torch.arange(N // 2, N).reshape(-1).to(self.device)
        else:
            even_pos = torch.arange(0, N, 2).reshape(-1).to(self.device)
            odd_pos = torch.arange(1, N, 2).reshape(-1).to(self.device)

        y_even, y_odd = y[:, even_pos], y[:, odd_pos]

        y1 = self.checknode((y_even, y_odd))
        u1, x1, _ = self.decode_recursive(y1, info_indices[: N // 2])

        y2 = self.bitnode((y_even, y_odd, x1))
        u2, x2, _ = self.decode_recursive(y2, info_indices[N // 2 :])

        u = torch.cat([u1, u2], dim=1)
        y_final = torch.cat([y1, y2], dim=1)
        x = self.f2((x1.clone(), x2.clone()))
        if self.polar_i:
            perm = torch.arange(N).reshape(2, -1).T.reshape(-1)
            x = x[:, perm]
        return u, x, y_final

    def forward(self, received: torch.Tensor, return_for_loss=False, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the received codeword using Successive Cancellation algorithm.

        Args:
            received (torch.Tensor): Received codeword tensor of shape (batch_size, n).
            return_for_loss (bool): If True, returns the log-likelihood ratio (LLR) values for loss calculation.
                                    If False, returns the estimated message bits.

        Returns:
            torch.Tensor: Estimated message bits of shape (batch_size, k) if `return_for_loss` is False,
                        or LLR values of shape (batch_size, n) if `return_for_loss` is True.
        """
        batch_size, n = received.shape
        self.info_indices = self.encoder.info_indices

        # Ensure the received tensor is on the correct device
        received = received.to(self.device)

        # Reshape codeword to match the expected input shape
        received = received.reshape(batch_size, n)

        # Perform decoding
        def decode_block(received_block: torch.Tensor) -> torch.Tensor:
            """Decode a single block of received codeword."""
            # Ensure the received tensor is on the correct device
            B, _, N = received_block.size()
            assert N == self.code_length, f"Received block size {N} does not match codeword size {self.code_length}"
            # Reshape the received block to match the expected input shape
            received_block = received_block.view(-1, N)
            # Decode the block using the recursive decoding method
            u, _, y = self.decode_recursive(received_block, self.info_indices)
            if return_for_loss:
                # Return the LLR values for loss calculation
                llr = y
                return llr
            return u.reshape(-1, N)[:, self.info_indices]

        # Return the estimated message bits
        return apply_blockwise(received, self.code_length, decode_block)
