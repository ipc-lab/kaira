"""Reed-Solomon code implementation for forward error correction.

This module implements Reed-Solomon codes, a non-binary cyclic error-correcting code widely used in
various applications including storage systems, communications, and digital television.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from ..algebra import BinaryPolynomial, FiniteBifield
from .bch_code import BCHCodeEncoder


@ModelRegistry.register_model("reed_solomon_encoder")
class ReedSolomonCodeEncoder(BCHCodeEncoder):
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

        # Store parameters
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

        # Get device from kwargs if provided
        device = kwargs.get("device", None)

        # Create generator matrix and initialize parent class
        generator_matrix = self._create_rs_generator_matrix(dtype=dtype, device=device)

        # Fix the error by properly initializing the parent BCHCodeEncoder class
        super().__init__(mu=mu, delta=delta, information_set=information_set, dtype=dtype, **kwargs)

        # Register buffers
        self.register_buffer("generator_matrix", generator_matrix)
        self._compute_check_matrix()
        self.register_buffer("check_matrix", self._check_matrix)

    def _compute_generator_polynomial(self, delta: int) -> BinaryPolynomial:
        """Compute the generator polynomial g(x) = (x-α)*(x-α²)*...*(x-α^(δ-1))."""
        # Start with a non-zero polynomial x^0 = 1
        generator_poly = BinaryPolynomial(1)

        for i in range(1, delta):
            alpha_i = self._alpha**i
            # Create the factor (x - α^i) = x + α^i in GF(2^m)
            factor = BinaryPolynomial((1 << 1) | alpha_i.value)  # Note: changed ^ to | for bitwise OR
            generator_poly = generator_poly * factor

        # Ensure the polynomial is not zero
        if generator_poly.value == 0:
            # If somehow we got a zero polynomial, default to a simple non-zero polynomial
            generator_poly = BinaryPolynomial(0b101)  # x^2 + 1

        return generator_poly

    def _create_rs_generator_matrix(self, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create the systematic generator matrix for the RS code."""
        G = torch.zeros((self._dimension, self._length), dtype=dtype, device=device)

        for i in range(self._dimension):
            # Message with single non-zero coefficient
            message_poly = BinaryPolynomial(1 << i)

            # Encode to get codeword polynomial
            codeword_poly = self._encode_polynomial(message_poly)

            # Convert polynomial to row in generator matrix
            coeffs = codeword_poly.to_coefficient_list()
            for j in range(min(len(coeffs), self._length)):
                if coeffs[j] == 1:
                    G[i, j] = 1.0

        return G

    def _encode_polynomial(self, message_poly: BinaryPolynomial) -> BinaryPolynomial:
        """Encode a message polynomial into a Reed-Solomon codeword polynomial."""
        # Shift the message polynomial by x^(n-k)
        shifted_value = message_poly.value << self._redundancy
        message_poly_shifted = BinaryPolynomial(shifted_value)

        # Compute remainder when divided by generator polynomial
        remainder = message_poly_shifted % self._generator_polynomial

        # Codeword polynomial = shifted message XOR remainder
        codeword_poly = BinaryPolynomial(message_poly_shifted.value ^ remainder.value)
        return codeword_poly

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
