"""Polar Code module for forward error correction.

This module provides an implementation of Polar codes for binary data transmission,
a class of linear block codes widely used in error correction for digital communication.
Polar codes are known for their channel polarization property, which enables efficient encoding and decoding.

The implementation follows common conventions in coding theory with particular focus
on channels polarization as introduced by Arikan.

References:
- E. Arikan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels," IEEE Transactions on Information Theory, 2008.
"""

from typing import Any, Optional

import numpy as np
import torch

from kaira.models.registry import ModelRegistry

from ..encoders.base import BaseBlockCodeEncoder
from ..utils import apply_blockwise


def _index_matrix(N: int) -> np.ndarray:
    """Returns the index matrix for polar code construction, indicating the bit indices involved in
    each stage of the polarization process.

    Args:
        N (int): Codeword length (must be a power of 2).

    Returns:
        np.ndarray: Index matrix of shape (N // 2, n), where n = log2(N).
    """
    x = np.arange(1, N + 1)
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"
    M = np.zeros((N - 1, n), dtype=int)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2**k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def calculate_gm(code_length: int, device: torch.device) -> torch.Tensor:
    """Return the generator matrix of the polar code (without interleaving).

    Args:
        code_length: Length of the polar code (must be a power of 2)
        device: Device to place the tensor on (CPU or GPU)
    Returns:
        torch.Tensor: Generator matrix of the polar code of shape (N, N)
    """
    factor_graph = np.array([[1, 0], [1, 1]])
    n_factor = factor_graph.copy()
    for _ in range(int(np.log2(code_length)) - 1):
        n_factor = np.kron(n_factor, factor_graph)
    return torch.tensor(n_factor).to(device=device).float()


@ModelRegistry.register_model("polar_code_encoder")
class PolarCodeEncoder(BaseBlockCodeEncoder):
    """Encoder for Polar code.

    This class implements the encoding process for Polar codes, a type of linear block code used in error correction.
    Polar codes leverage the channel polarization property to achieve efficient encoding and decoding.

    The encoder transforms binary input messages into codewords using the Polar transformation.
    It supports customization of frozen bits, device selection, and data type configuration.

    Attributes:
        device (str): Device on which the encoder operates (e.g., 'cpu' or 'cuda').
        m (int): Number of stages in the Polar code (log2 of the code length).
        polar_i (bool): Indicates whether to apply permutation during the Polar transform.
        frozen_zeros (bool): Specifies whether frozen bits are initialized to zeros.
        dtype (torch.dtype): Data type used for computations (e.g., torch.float32).
        load_rank (bool): Indicates whether to load rank-based polar indices as defined in the 5G standard.
        rank (np.ndarray): Rank-based indices for frozen bits (loaded if `load_rank` is True).
        info_indices (np.ndarray): Boolean array indicating positions of information bits.
        mask_dict (np.ndarray): Mask dictionary for the Polar code structure.
    """

    def __init__(self, code_dimension: int, code_length: int, *args: Any, **kwargs: Any):
        """Initializes the PolarCodeEncoder.

        Args:
            code_dimension (int): Number of information bits in the Polar code.
            code_length (int): Total length of the Polar codeword (must be a power of 2).
            *args (Any): Variable positional arguments passed to the base class.
            **kwargs (Any): Variable keyword arguments for additional configuration, including:

                - device (str): Device on which the encoder operates (default: 'cpu').
                - polar_i (bool): Whether to apply permutation during the Polar transform (default: False).
                - frozen_zeros (bool): Whether frozen bits are initialized to zeros (default: False).
                - dtype (torch.dtype): Data type used for computations (default: torch.float32).
                - load_rank (bool): Whether to load rank-based polar indices as defined in the 5G standard (default: True).
                - info_indices (np.ndarray): Boolean array indicating positions of information bits.
                  Required when load_rank=False. Must have length equal to code_length and exactly
                  code_dimension True values.
        """
        super().__init__(code_length, code_dimension, *args, **kwargs)
        self.device = kwargs.get("device", "cpu")
        self.m = int(np.log2(code_length))
        assert 2**self.m == code_length, "n must be a power of 2"
        self.polar_i = kwargs.get("polar_i", False)
        self.frozen_zeros = kwargs.get("frozen_zeros", False)
        self.dtype = kwargs.get("dtype", torch.float32)
        self.load_rank = kwargs.get("load_rank", True)
        if self.load_rank:
            print("Loading rank polar indices as defined in 5G standard...")
            import os

            import pandas as pd

            # Get the directory of this module file
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path to the CSV file relative to this module
            csv_path = os.path.join(module_dir, "..", "rank_polar.csv")
            rank = pd.read_csv(csv_path, sep=" ", index_col=0)
            self.rank = rank.Q.values
            F = np.zeros(self.code_length)
            F[self.rank[self.rank < self.code_length][: self.code_length - self.code_dimension]] = 1
            info_ind = np.where(F == 0)[0]
            self.info_indices = np.zeros(self.code_length)
            self.info_indices[info_ind] = 1
            self.info_indices = self.info_indices.astype(bool)
        else:
            # When load_rank=False, info_indices must be provided
            info_indices = kwargs.get("info_indices", None)
            if info_indices is None:
                raise ValueError("When load_rank=False, info_indices must be provided as a boolean array " "indicating the positions of information bits. The array should have length " f"equal to code_length ({self.code_length}) and exactly {self.code_dimension} " "True values.")

            # Validate info_indices
            info_indices = np.asarray(info_indices, dtype=bool)
            if len(info_indices) != self.code_length:
                raise ValueError(f"info_indices must have length {self.code_length}, got {len(info_indices)}")
            if np.sum(info_indices) != self.code_dimension:
                raise ValueError(f"info_indices must have exactly {self.code_dimension} True values, " f"got {np.sum(info_indices)}")

            self.info_indices = info_indices

        self.mask_dict: Optional[np.ndarray] = None

    def get_generator_matrix(self) -> torch.Tensor:
        """Returns the generator matrix of the Polar code (without interleaving).

        The generator matrix is used to encode information bits into codewords.
        It is constructed based on the structure of Polar codes.

        Returns:
            torch.Tensor: Generator matrix of shape (N, N), where N is the code length.
        """
        return calculate_gm(self.code_length, self.device)

    def polar_transform(self, u: torch.Tensor, return_arr: bool = False) -> torch.Tensor:
        """Applies the Polar transform to the input tensor.

        The Polar transform is a recursive process that combines and splits bits to achieve channel polarization.
        This method performs the transformation based on the mask dictionary and supports optional permutation.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, code_length).
            return_arr (bool): If True, returns intermediate results of the transformation as a list. Default is False.

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, code_length) if `return_arr` is False.
            List[torch.Tensor]: List of intermediate tensors during the transformation if `return_arr` is True.
        """
        N = u.shape[1]
        assert N == self.code_length, "Input tensor must have shape (batch_size, n)"
        bs = u.shape[0]

        if self.mask_dict is None or self.mask_dict.shape[0] != self.m:
            mask_dict = _index_matrix(self.code_length).T.astype(int) - 1
            self.mask_dict = mask_dict[np.flip(np.arange(self.m))]

        # Ensure mask_dict is properly initialized
        assert self.mask_dict is not None, "mask_dict should be initialized"

        x = u.clone().to(int)
        if return_arr:
            arr_x = [x.clone().reshape(bs, N, 1)]
        for i in range(self.m):
            i_back = self.m - i - 1
            add_k = N // (2 ** (i_back + 1))
            perm_ind = torch.arange(N).reshape(N // 2 ** (i + 1), 2, -1).permute(0, 2, 1).reshape(-1).to(u.device)
            x[:, self.mask_dict[i]] = torch.bitwise_xor(x[:, self.mask_dict[i]], x[:, self.mask_dict[i] + add_k])
            if self.polar_i:
                x = x[:, perm_ind]
            if return_arr:
                arr_x.append(x.clone().reshape(bs, N, 1))

        if return_arr:
            return arr_x
        return x.reshape(bs, N).to(self.dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encodes the input message using the Polar transformation.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, code_dimension).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Encoded codeword of shape (batch_size, code_length).
        """
        # Ensure input is on the correct device
        input = x.to(self.device)
        # Check input shape
        k = x.shape[1]
        assert k == self.code_dimension, f"Input shape mismatch: expected {self.code_dimension}, got {k}"

        def encode_fn(x):
            """Function to encode a single block of input."""
            # Initialize the codeword tensor
            bs = x.shape[0]
            N = self.code_length
            if self.frozen_zeros:
                codeword = torch.zeros((bs, N), dtype=self.dtype, device=self.device)
            else:
                codeword = torch.ones((bs, N), dtype=self.dtype, device=self.device)
            # Set the information bits in the codeword
            codeword[:, self.info_indices] = x.view(bs, self.code_dimension)
            # Perform the Polar transform to generate the codeword
            codeword = self.polar_transform(codeword, return_arr=False)
            return codeword

        return apply_blockwise(input, self.code_dimension, encode_fn)
