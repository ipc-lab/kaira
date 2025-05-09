"""Low-Density Parity-Check (LDPC) Code module for forward error correction.

This module provides an implementation of Low-Density Parity-Check (LDPC) codes for binary data transmission,
a class of linear block codes widely used in error correction for digital communication. LDPC codes are known for
their sparse parity-check matrices, which enable efficient encoding and decoding using iterative algorithms.

The implementation follows common conventions in coding theory with particular focus
on LDPC codes which are defined by a sparse parity-check matrix H.

References:
- R. G. Gallager, "Low-Density Parity-Check Codes," 1962.
- T. J. Richardson and R. L. Urbanke, "Modern Coding Theory," 2008.
"""

from typing import Any

import torch

from kaira.models.registry import ModelRegistry

from ..encoders.linear_block_code import LinearBlockCodeEncoder
from ..utils import row_reduction


@ModelRegistry.register_model("ldpc_code_encoder")
class LDPCCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for LDPC code.

    This encoder follows conventional approach of linear block codes and
    transforms binary input messages into codewords according to
    the calculated generator matrix. It serves as the encoding component of
    a linear block code system.

    The encoder applies the formula: c = mG, where:
    - c is the codeword
    - m is the message
    - G is the generator matrix

    This implementation follows the standard approach to linear block coding described in the
    error control coding literature :cite:`lin2004error,moon2005error,sklar2001digital`.

    Attributes:
        generator_matrix (torch.Tensor): The generator matrix G of the code
        check_matrix (torch.Tensor): The parity check matrix H

    Args:
        check_matrix (torch.Tensor): The parity check matrix to define LDPC code.
            Must be a binary matrix of shape (n - k, n) where k is the message length
            and n is the codeword length.
        *args: Variable positional arguments passed to the base class.
        **kwargs: Variable keyword arguments passed to the base class.
    """

    def __init__(self, check_matrix: torch.Tensor, *args: Any, **kwargs: Any):
        """Initialize the linear block encoder.

        Args:
            check_matrix (torch.Tensor): The parity check matrix for encoding.
                Must be a binary matrix of shape (n - k, n) where k is the message length
                and n is the codeword length.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        self.device = kwargs.get("device", "cpu")
        # Ensure generator matrix is a torch tensor
        if not isinstance(check_matrix, torch.Tensor):
            check_matrix = torch.tensor(check_matrix).to(self.device)
        if check_matrix.device != self.device:
            check_matrix = check_matrix.to(self.device)

        generator_matrix = self.get_generator_matrix(check_matrix)

        # Initialize the base class with dimensions
        super().__init__(generator_matrix=generator_matrix, check_matrix=check_matrix)

    def get_generator_matrix(self, check_matrix_: torch.Tensor) -> torch.Tensor:
        """Derive the generator matrix from a parity check matrix.

        This method computes the generator matrix for an LDPC code by:
        1. Transposing the parity check matrix
        2. Appending an identity matrix to obtain [H | I]
        3. Performing Gaussian elimination (row reduction) to obtain [A | B]
        4. Extracting the generator matrix from the result

        The process ensures that G·Hᵀ = 0, which is the defining property of a valid
        generator matrix for the code.

        Args:
            check_matrix_: The parity check matrix of the LDPC code

        Returns:
            The generator matrix for the LDPC code
        """
        check_matrix = check_matrix_.clone().to(torch.int64).t()
        check_matrix_eye = torch.cat((check_matrix, torch.eye(check_matrix.shape[0]).to(bool).to(check_matrix.device)), dim=1)
        check_matrix_eye, rank = row_reduction(check_matrix_eye, num_cols=check_matrix.shape[1])
        generator_matrix = row_reduction(check_matrix_eye[rank:, check_matrix.shape[1] :])[0]
        return generator_matrix
