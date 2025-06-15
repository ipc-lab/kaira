"""Low-Density Parity-Check (LDPC) Code module for forward error correction.

This module provides an implementation of Low-Density Parity-Check (LDPC) codes for binary data transmission,
a class of linear block codes widely used in error correction for digital communication. LDPC codes are known for
their sparse parity-check matrices, which enable efficient encoding and decoding using iterative algorithms.

The implementation follows common conventions in coding theory with particular focus
on LDPC codes which are defined by a sparse parity-check matrix H.

References:
    :cite:`gallager1962low`, :cite:`gallager1963low`, :cite:`richardson2008modern`
"""

from typing import Any

import torch

from kaira.models.registry import ModelRegistry

from ..encoders.linear_block_code import LinearBlockCodeEncoder
from ..rptu_database import CITATION, EXISTING_CODES, get_code_from_database, parse_alist
from ..utils import row_reduction


@ModelRegistry.register_model("ldpc_code_encoder")
class LDPCCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for LDPC code :cite:`gallager1962low`, :cite:`gallager1963low`.

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
    """

    def __init__(self, check_matrix: torch.Tensor = None, rptu_database: bool = False, *args: Any, **kwargs: Any):
        """Initializes the linear block encoder for LDPC codes.

        Args:
            check_matrix (torch.Tensor, optional): The parity check matrix for encoding.
                Should be a binary matrix of shape (code_length - code_dimension, code_length), where
                code_dimension is the message length and code_length is the codeword length.
                If None and `rptu_database` is True, the matrix
                will be loaded from the RPTU database.
            rptu_database (bool, optional): If True, loads the check matrix from the RPTU
                code database using parameters provided in `kwargs`. Default is False.
            *args: Additional positional arguments passed to the base class.
            **kwargs: Additional keyword arguments. Expected keys when `rptu_database` is True:
                - code_length (int): Codeword length.
                - code_dimension (int): Message length.
                - rptu_standart (str, optional): Standard name for the LDPC code. If not provided,
                the first available standard is used.
                - device (str, optional): Device to place the tensors on (e.g., "cpu" or "cuda").

        Raises:
            ValueError: If the requested (code_length, code_dimension) code or standard is not found in the RPTU database.
        """
        # Validate input parameters
        if not rptu_database and check_matrix is None:
            raise ValueError("Either a valid `check_matrix` must be provided or `rptu_database` must be set to True.")
        # Initialize the base class from rptu_database or provided check_matrix
        if rptu_database:
            print("Loading LDPC code from RPTU database...")
            print(CITATION)
            print("------------------------------------")
            code_length = kwargs.get("code_length", None)
            code_dimension = kwargs.get("code_dimension", None)
            rptu_standart = kwargs.get("rptu_standart", None)

            if code_length is None or code_dimension is None:
                raise ValueError("code_length and code_dimension must be provided when using rptu_database.")

            code_key = (code_length, code_dimension)
            if code_key in EXISTING_CODES.keys():
                if rptu_standart is not None:
                    if rptu_standart not in EXISTING_CODES[code_key].keys():
                        raise ValueError(f"LDPC code with (code_length={code_length}, code_dimension={code_dimension}) and rptu_standart='{rptu_standart}' not found in rptu_database.")
                else:
                    rptu_standart = list(EXISTING_CODES[code_key].keys())[0]  # Default to first available standard
                    print(f"Using default rptu_standart='{rptu_standart}' for (code_length={code_length}, code_dimension={code_dimension}).")
                content = get_code_from_database(EXISTING_CODES[code_key][rptu_standart])
            else:
                print(f"Available LDPC codes from rptu database: {EXISTING_CODES.keys()}")
                raise ValueError(f"LDPC code with (code_length={code_length}, code_dimension={code_dimension}) not found in rptu_database.")
            check_matrix = parse_alist(content)
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
