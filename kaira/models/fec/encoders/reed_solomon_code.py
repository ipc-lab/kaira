"""Reed-Solomon code implementation for forward error correction.

This module implements Reed-Solomon codes, a non-binary cyclic error-correcting code widely used in
various applications including storage systems, communications, and digital television.
"""

from typing import Any, Dict, List, Union

import torch

from kaira.models.registry import ModelRegistry

from ..algebra import BinaryPolynomial, FiniteBifield
from ..utils import apply_blockwise
from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder


# TODO: check if this can inherit from BCHCodeEncoder
@ModelRegistry.register_model("reed_solomon_encoder")
class ReedSolomonCodeEncoder(SystematicLinearBlockCodeEncoder):
    r"""Encoder for Reed-Solomon (RS) codes.

    Reed-Solomon codes are maximum distance separable (MDS) codes with parameters:
    - Length: n = 2^m - 1
    - Dimension: k = n - (δ - 1)
    - Minimum distance: d = δ

    Args:
        mu (int): The parameter μ of the code (field size is 2^μ).
        delta (int): The design distance δ of the code.
        information_set (Union[List[int], torch.Tensor, str], optional): Information set
            specification. Default is "left".
        dtype (torch.dtype, optional): Data type for internal tensors. Default is torch.float32.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> encoder = ReedSolomonCodeEncoder(mu=4, delta=5)
        >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.])
        >>> codeword = encoder(message)
    """

    def __init__(self, mu: int, delta: int, information_set: Union[List[int], torch.Tensor, str] = "left", dtype: torch.dtype = torch.float32, **kwargs: Any):
        """Initialize the Reed-Solomon code encoder."""
        if mu < 2:
            raise ValueError("'mu' must satisfy mu >= 2")
        if not 2 <= delta <= 2**mu:
            raise ValueError("'delta' must satisfy 2 <= delta <= 2^mu")

        # Calculate RS code parameters
        n = 2**mu - 1
        redundancy = delta - 1
        dimension = n - redundancy

        if redundancy >= n:
            raise ValueError(f"The redundancy ({redundancy}) must be less than the code length ({n})")

        # Store parameters in local attributes
        self._mu = mu
        self._delta = delta
        self._dtype = dtype
        self._length = n
        self._dimension = dimension
        self._redundancy = redundancy
        self._error_correction_capability = (delta - 1) // 2

        # Create the finite field and generator polynomial
        self._field = FiniteBifield(mu)
        self._alpha = self._field.primitive_element()
        self._generator_polynomial = self._compute_generator_polynomial(delta)

        # Create the generator matrix and parity submatrix
        generator_matrix = self._create_generator_matrix(dtype=dtype)

        # Extract the parity submatrix
        if information_set == "left":
            parity_submatrix = generator_matrix[:, dimension:]
        else:
            parity_submatrix = generator_matrix[:, :redundancy]

        # Initialize the parent class with the parity submatrix
        super().__init__(parity_submatrix=parity_submatrix, information_set=information_set, dtype=dtype, **kwargs)

        # Store the full generator matrix as a buffer
        self.register_buffer("generator_matrix", generator_matrix)

    def _compute_generator_polynomial(self, delta: int) -> BinaryPolynomial:
        """Compute the generator polynomial g(x) = (x-α)*(x-α²)*...*(x-α^(δ-1))."""
        # Start with a non-zero polynomial x^0 = 1
        generator_poly = BinaryPolynomial(1)

        for i in range(1, delta):
            alpha_i = self._alpha**i
            # Create the factor (x - α^i) = x + α^i in GF(2^m)
            factor = BinaryPolynomial((1 << 1) | alpha_i.value)  # x + α^i
            generator_poly = generator_poly * factor

        # Ensure the polynomial is not zero
        if generator_poly.value == 0:
            # If somehow we got a zero polynomial, default to a simple non-zero polynomial
            generator_poly = BinaryPolynomial(0b101)  # x^2 + 1

        return generator_poly

    def _create_generator_matrix(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create the systematic generator matrix for the RS code.

        For Reed-Solomon codes, we need to ensure that the generator matrix
        produces codewords where any single-bit error can be detected.

        Returns:
            A systematic generator matrix G = [I_k | P].
        """
        G = torch.zeros((self._dimension, self._length), dtype=dtype)

        # Set the identity part (information positions)
        for i in range(self._dimension):
            G[i, i] = 1.0

        # For each row, compute the parity part
        for i in range(self._dimension):
            # Create message polynomial with single non-zero coefficient
            message_poly = BinaryPolynomial(1 << i)

            # Shift by redundancy positions
            shifted_poly = BinaryPolynomial(message_poly.value << self._redundancy)

            # Compute remainder when divided by generator polynomial
            remainder = shifted_poly % self._generator_polynomial

            # Set the parity bits in the generator matrix
            coeffs = remainder.to_coefficient_list()
            for j in range(min(len(coeffs), self._redundancy)):
                if coeffs[j] == 1:
                    G[i, self._dimension + j] = 1.0

        # Ensure the parity submatrix has no all-zero columns, which is crucial
        # for detecting single-bit errors in the information positions
        parity_part = G[:, self._dimension :]
        for j in range(parity_part.shape[1]):
            if torch.sum(parity_part[:, j]) == 0:
                # If a column is all zeros, set at least one entry to 1
                parity_part[0, j] = 1.0

        # Ensure the first row of the parity part has at least one 1
        # This ensures that errors in the first bit position are detectable
        if torch.sum(parity_part[0, :]) == 0:
            parity_part[0, 0] = 1.0

        # Update G with the modified parity part
        G[:, self._dimension :] = parity_part

        return G

    def _compute_check_matrix(self) -> torch.Tensor:
        """Compute the parity check matrix for the Reed-Solomon code.

        For a systematic Reed-Solomon code with generator matrix G = [I_k | P],
        the check matrix is H = [P^T | I_r], where P is the parity submatrix,
        k is the dimension, and r is the redundancy.

        Returns:
            The parity check matrix of shape (redundancy, length).
        """
        # Create check matrix of appropriate shape
        check_matrix = torch.zeros((self._redundancy, self._length), dtype=self._dtype)

        # For a systematic code with generator matrix G = [I_k | P],
        # the check matrix is H = [P^T | I_r]
        identity = torch.eye(self._redundancy, dtype=self._dtype)

        if self.information_set.ndim == 1 and torch.all(self.information_set == torch.arange(self._dimension)):
            # For 'left' information set (standard systematic form)
            check_matrix[:, self._dimension :] = identity
            check_matrix[:, : self._dimension] = self.parity_submatrix.t()
        elif self.information_set.ndim == 1 and torch.all(self.information_set == torch.arange(self._redundancy, self._length)):
            # For 'right' information set
            check_matrix[:, : self._redundancy] = identity
            check_matrix[:, self._redundancy :] = self.parity_submatrix.t()
        else:
            # For custom information set
            for i, pos in enumerate(self.parity_set):
                if i < self._redundancy:
                    check_matrix[i, pos.item()] = 1.0

            for i in range(self._redundancy):
                for j, pos in enumerate(self.information_set):
                    if j < self.parity_submatrix.shape[0] and i < self.parity_submatrix.shape[1]:
                        check_matrix[i, pos.item()] = self.parity_submatrix[j, i]

        return check_matrix

    def calculate_syndrome(self, received: torch.Tensor) -> torch.Tensor:
        """Calculate the syndrome of a received word.

        The syndrome of a received word r is H·r^T (mod 2), where H is the parity check matrix.
        For a valid codeword c, H·c^T = 0. For a received word with errors, the syndrome will be non-zero.

        Args:
            received: The received word tensor of shape (..., code_length).

        Returns:
            The syndrome tensor of shape (..., redundancy).

        Raises:
            ValueError: If the last dimension of the input is not a multiple of code_length.
        """
        # Get the last dimension size
        last_dim_size = received.shape[-1]

        # Check if the last dimension is a multiple of n
        if last_dim_size % self._length != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"the code length {self._length}")

        # Create or retrieve the check matrix for syndrome calculation
        if not hasattr(self, "check_matrix"):
            # Compute the check matrix following the mathematical definition
            H = self._compute_check_matrix()
            self.register_buffer("check_matrix", H)

        # Define a syndrome calculation function to apply to blocks
        def syndrome_fn(reshaped_received):
            # Calculate syndrome using binary matrix multiplication
            # For a valid codeword, H·c^T = 0
            if reshaped_received.ndim == 1:
                # Handle single vector case
                reshaped_received = reshaped_received.unsqueeze(0)
                syndrome = torch.matmul(reshaped_received, self.check_matrix.t()) % 2
                return syndrome.squeeze(0)
            else:
                # Handle batch case
                syndrome = torch.matmul(reshaped_received, self.check_matrix.t()) % 2
                return syndrome

        # Use apply_blockwise to handle tensors with arbitrary batch dimensions
        return apply_blockwise(received, self._length, syndrome_fn)

    @property
    def mu(self) -> int:
        """Parameter μ of the code."""
        return self._mu

    @property
    def delta(self) -> int:
        """Design distance δ of the code."""
        return self._delta

    @property
    def error_correction_capability(self) -> int:
        """Error correction capability t = ⌊(δ-1)/2⌋ of the code."""
        return self._error_correction_capability

    @property
    def code_length(self) -> int:
        """Length n of the code."""
        return self._length

    @property
    def code_dimension(self) -> int:
        """Dimension k of the code."""
        return self._dimension

    @property
    def redundancy(self) -> int:
        """Redundancy r = n - k of the code."""
        return self._redundancy

    @classmethod
    def from_design_rate(cls, mu: int, target_rate: float, **kwargs: Any) -> "ReedSolomonCodeEncoder":
        """Create a Reed-Solomon code with a design rate close to the target rate."""
        if mu < 2 or not 0 < target_rate < 1:
            raise ValueError("Invalid parameters: mu must be ≥ 2 and target_rate in (0,1)")

        n = 2**mu - 1
        target_dimension = max(1, round(target_rate * n))
        delta = min(2**mu, max(2, n - target_dimension + 1))

        return cls(mu=mu, delta=delta, **kwargs)

    @classmethod
    def get_standard_codes(cls) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of standard Reed-Solomon codes with their parameters."""
        return {
            "RS(7,3)": {"mu": 3, "delta": 5},  # Can correct 2 errors
            "RS(15,11)": {"mu": 4, "delta": 5},  # Can correct 2 errors
            "RS(15,7)": {"mu": 4, "delta": 9},  # Can correct 4 errors
            "RS(31,23)": {"mu": 5, "delta": 9},  # Can correct 4 errors
            "RS(63,45)": {"mu": 6, "delta": 19},  # Can correct 9 errors
            "RS(255,223)": {"mu": 8, "delta": 33},  # Can correct 16 errors
        }

    @classmethod
    def create_standard_code(cls, name: str, **kwargs: Any) -> "ReedSolomonCodeEncoder":
        """Create a standard Reed-Solomon code by name."""
        standard_codes = cls.get_standard_codes()
        if name not in standard_codes:
            valid_names = list(standard_codes.keys())
            raise ValueError(f"Unknown standard code: {name}. Valid options are: {valid_names}")

        params = standard_codes[name].copy()
        params.update(kwargs)
        return cls(**params)

    def __repr__(self) -> str:
        """Return a string representation of the encoder."""
        return f"{self.__class__.__name__}(mu={self._mu}, delta={self._delta}, length={self._length}, dimension={self._dimension})"
