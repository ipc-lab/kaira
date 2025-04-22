"""Cyclic code implementation for forward error correction.

This module implements cyclic codes, an important class of linear block codes with additional
structural properties. In a cyclic code, if c is a codeword, then every cyclic shift of c is
also a codeword. Cyclic codes are characterized by their generator polynomial g(X) and check
polynomial h(X), which are related by g(X)h(X) = X^n + 1.

Cyclic codes include many important error-correcting codes like BCH codes, Reed-Solomon codes,
and others. They are widely used in digital communications and storage systems because of their
efficient encoding and decoding algorithms :cite:`lin2004error,moon2005error,richardson2008modern`.
"""

from functools import lru_cache
from typing import Any, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from ..algebra import BinaryPolynomial
from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder


@ModelRegistry.register_model("cyclic_code_encoder")
class CyclicCodeEncoder(SystematicLinearBlockCodeEncoder):
    r"""Encoder for cyclic codes.

    A cyclic code is a linear block code with the additional property that any cyclic shift of a
    codeword is also a codeword. A cyclic code is characterized by its generator polynomial g(X)
    of degree m (the redundancy of the code), and by its check polynomial h(X) of degree k
    (the dimension of the code). These polynomials are related by g(X)h(X) = X^n + 1, where
    n = k + m is the length of the code.

    Currently, the implementation only supports systematic encoding, where the first k bits of
    the codeword are the information bits, and the last m bits are the parity bits.

    Examples of generator polynomials for common codes:

    | Code (n, k, d)    | Generator polynomial g(X)              | Integer representation          |
    | ----------------- | -------------------------------------- | ------------------------------- |
    | Hamming (7,4,3)   | X^3 + X + 1                            | 0b1011 = 11                     |
    | Simplex (7,3,4)   | X^4 + X^2 + X + 1                      | 0b10111 = 23                    |
    | BCH (15,5,7)      | X^10 + X^8 + X^5 + X^4 + X^2 + X + 1   | 0b10100110111 = 1335            |
    | Golay (23,12,7)   | X^11 + X^9 + X^7 + X^6 + X^5 + X + 1   | 0b101011100011 = 2787           |

    For more details, see :cite:`lin2004error,moon2005error`.

    Args:
        code_length (int): The length n of the code
        generator_polynomial (int, optional): The generator polynomial g(X) of the code
        check_polynomial (int, optional): The check polynomial h(X) of the code
        information_set (Union[List[int], torch.Tensor, str], optional): Information set
            specification. Default is "left".
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> encoder = CyclicCodeEncoder(code_length=23, generator_polynomial=0b101011100011)  # Golay (23, 12)
        >>> encoder.code_length, encoder.code_dimension, encoder.redundancy
        (23, 12, 11)
        >>> encoder.generator_poly
        BinaryPolynomial(0b101011100011)
    """

    def __init__(self, code_length: int, generator_polynomial: Optional[int] = None, check_polynomial: Optional[int] = None, information_set: Union[List[int], torch.Tensor, str] = "left", **kwargs: Any):
        """Initialize the cyclic code encoder.

        Args:
            code_length: The length n of the code
            generator_polynomial: The generator polynomial g(X) of the code, specified as an integer
            check_polynomial: The check polynomial h(X) of the code, specified as an integer
            information_set: Either indices of information positions, which must be a k-sublist
                of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If neither generator_polynomial nor check_polynomial is provided,
                or if the provided polynomial is not a factor of X^n + 1.
        """
        self._length = code_length

        # Compute the modulus polynomial X^n + 1
        self._modulus_value = BinaryPolynomial(1 << code_length).value | 0b1
        self._modulus_poly = BinaryPolynomial(self._modulus_value)

        # Validate and set up the generator and check polynomials
        if generator_polynomial is None and check_polynomial is None:
            raise ValueError("Either 'generator_polynomial' or 'check_polynomial' must be provided")

        if generator_polynomial is not None and check_polynomial is None:
            self._generator_poly = BinaryPolynomial(generator_polynomial)
            # Use custom division implementation to avoid divmod error
            quotient, remainder = self._custom_div_with_remainder(self._modulus_poly, self._generator_poly)
            if remainder.value != 0:
                raise ValueError("'generator_polynomial' must be a factor of X^n + 1")
            self._check_poly = quotient
            self._polynomial_source = "generator_polynomial"

        elif generator_polynomial is None and check_polynomial is not None:
            self._check_poly = BinaryPolynomial(check_polynomial)
            # Use custom division implementation to avoid divmod error
            quotient, remainder = self._custom_div_with_remainder(self._modulus_poly, self._check_poly)
            if remainder.value != 0:
                raise ValueError("'check_polynomial' must be a factor of X^n + 1")
            self._generator_poly = quotient
            self._polynomial_source = "check_polynomial"

        else:  # Both provided, verify they're correct
            # Ensure we have valid integers (not None) before creating BinaryPolynomials
            if generator_polynomial is None or check_polynomial is None:
                raise ValueError("Both generator_polynomial and check_polynomial must not be None")

            self._generator_poly = BinaryPolynomial(generator_polynomial)
            self._check_poly = BinaryPolynomial(check_polynomial)
            product = self._generator_poly * self._check_poly
            if (product % self._modulus_poly).value != 0:
                raise ValueError("g(X)h(X) must equal X^n + 1")
            self._polynomial_source = "both_polynomials"

        # Calculate redundancy and dimension
        self._redundancy = self._generator_poly.degree
        self._dimension = code_length - self._redundancy

        # Create the generator matrix for systematic coding
        generator_matrix = self._generate_systematic_matrix()

        # Extract the parity submatrix for systematic encoding
        k, n = self._dimension, self._length
        parity_submatrix = generator_matrix[:, k:n] if information_set == "left" else generator_matrix[:, 0 : n - k]
        super().__init__(parity_submatrix=parity_submatrix, information_set=information_set, **kwargs)

        # Register additional buffers specific to cyclic codes
        gen_poly_tensor = torch.tensor(self._generator_poly.to_coefficient_list(), dtype=torch.float32)
        check_poly_tensor = torch.tensor(self._check_poly.to_coefficient_list(), dtype=torch.float32)
        self.register_buffer("generator_poly_coeffs", gen_poly_tensor)
        self.register_buffer("check_poly_coeffs", check_poly_tensor)

        # Compute the check matrix
        self._compute_check_matrix()

        # Register the check matrix buffer
        self.register_buffer("check_matrix", self._check_matrix)

    def _generate_systematic_matrix(self) -> torch.Tensor:
        """Generate the systematic generator matrix for the code.

        Returns:
            The systematic generator matrix
        """
        n, k, m = self._length, self._dimension, self._redundancy
        generator_matrix = torch.zeros((k, n), dtype=torch.float32)

        # Create a monomial X^i
        X = BinaryPolynomial(0b10)  # The polynomial X

        # For each row i of the generator matrix
        for i in range(k):
            # Start with the monomial X^(m+i)
            shifted_poly = self._custom_pow(X, m + i)

            # Compute the remainder when divided by the generator polynomial
            remainder_poly = shifted_poly % self._generator_poly

            # The codeword polynomial is X^(m+i) - remainder
            # In GF(2), subtraction is the same as addition (XOR)
            codeword_poly = BinaryPolynomial(shifted_poly.value ^ remainder_poly.value)

            # Convert the polynomial to a row in the generator matrix
            coeffs = codeword_poly.to_coefficient_list()
            for j in range(min(len(coeffs), n)):
                if coeffs[j] == 1:
                    generator_matrix[i, j] = 1.0

        return generator_matrix

    def _custom_div_with_remainder(self, dividend: BinaryPolynomial, divisor: BinaryPolynomial) -> tuple[BinaryPolynomial, BinaryPolynomial]:
        """Custom polynomial division implementation that returns both quotient and remainder.

        Args:
            dividend: The dividend polynomial
            divisor: The divisor polynomial

        Returns:
            A tuple of (quotient, remainder) polynomials
        """
        # Check if divisor is zero - handle case when value is a tensor
        if isinstance(divisor.value, torch.Tensor):
            if torch.all(divisor.value == 0):
                raise ValueError("Division by zero polynomial")
        elif divisor.value == 0:
            raise ValueError("Division by zero polynomial")

        # Optimizations
        if isinstance(dividend.value, torch.Tensor):
            if torch.all(dividend.value == 0):
                return BinaryPolynomial(0), BinaryPolynomial(0)
        elif dividend.value == 0:
            return BinaryPolynomial(0), BinaryPolynomial(0)

        if dividend == divisor:
            return BinaryPolynomial(1), BinaryPolynomial(0)
        if dividend.degree < divisor.degree:
            return BinaryPolynomial(0), dividend

        quotient = 0
        remainder = dividend.value
        divisor_degree = divisor.degree
        divisor_value = divisor.value

        # Implement polynomial division in GF(2)
        while True:
            remainder_poly = BinaryPolynomial(remainder)
            remainder_degree = remainder_poly.degree
            if remainder_degree < divisor_degree:
                break

            # Calculate the shift for the current division step
            shift = remainder_degree - divisor_degree

            # Update the quotient (set the bit at position 'shift')
            quotient |= 1 << shift

            # Subtract (XOR) divisor * x^shift from the remainder
            remainder ^= divisor_value << shift

        return BinaryPolynomial(quotient), BinaryPolynomial(remainder)

    def _custom_div(self, dividend: BinaryPolynomial, divisor: BinaryPolynomial) -> BinaryPolynomial:
        """Custom polynomial division implementation to avoid using divmod operator.

        Args:
            dividend: The dividend polynomial
            divisor: The divisor polynomial

        Returns:
            The quotient polynomial
        """
        quotient, _ = self._custom_div_with_remainder(dividend, divisor)
        return quotient

    def _custom_pow(self, base: BinaryPolynomial, exponent: int) -> BinaryPolynomial:
        """Custom polynomial exponentiation implementation to avoid using ** operator.

        Args:
            base: The base polynomial
            exponent: The exponent

        Returns:
            The polynomial raised to the given power
        """
        if exponent == 0:
            return BinaryPolynomial(1)
        if exponent == 1:
            return base

        # Use binary exponentiation (square-and-multiply algorithm)
        result = BinaryPolynomial(1)
        current_power = base

        while exponent > 0:
            if exponent & 1:  # If current bit is 1
                result = result * current_power
            current_power = current_power * current_power  # Square
            exponent >>= 1

        return result

    @property
    def generator_poly(self) -> BinaryPolynomial:
        """Generator polynomial g(X) of the code."""
        return self._generator_poly

    @property
    def check_poly(self) -> BinaryPolynomial:
        """Check polynomial h(X) of the code."""
        return self._check_poly

    @property
    def modulus_poly(self) -> BinaryPolynomial:
        """Modulus polynomial X^n + 1 of the code."""
        return self._modulus_poly

    def encode_message_polynomial(self, message_poly: BinaryPolynomial) -> BinaryPolynomial:
        """Encode a message polynomial into a codeword polynomial using systematic encoding.

        Args:
            message_poly: The message polynomial to encode

        Returns:
            The systematically encoded codeword polynomial
        """
        # For systematic encoding, we shift the message by X^m and then
        # subtract the remainder when divided by the generator polynomial
        # We need to implement the shift manually since << operator is not supported
        shifted_value = message_poly.value << self._redundancy
        message_poly_shifted = BinaryPolynomial(shifted_value)
        remainder = message_poly_shifted % self._generator_poly

        # In GF(2), subtraction is the same as addition (XOR)
        codeword_poly = BinaryPolynomial(message_poly_shifted.value ^ remainder.value)

        return codeword_poly

    def extract_message_polynomial(self, codeword_poly: BinaryPolynomial) -> BinaryPolynomial:
        """Extract a message polynomial from a codeword polynomial.

        Args:
            codeword_poly: The codeword polynomial to extract from

        Returns:
            The message polynomial
        """
        # For systematic codes, the message is in the higher-order terms
        # Simply right-shift by m positions
        # We need to implement the shift manually since >> operator is not supported
        shifted_value = codeword_poly.value >> self._redundancy
        return BinaryPolynomial(shifted_value)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode the input tensor using polynomial encoding.

        For cyclic codes, encoding can be done using polynomial operations.
        This implementation delegates to the parent class for efficiency.

        Args:
            x: The input tensor of shape (..., message_length) or (..., b*message_length)
               where b is a positive integer.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Encoded tensor of shape (..., codeword_length) or (..., b*codeword_length)
        """
        return super().forward(x, *args, kwargs)

    @classmethod
    def create_standard_code(cls, name: str, **kwargs: Any) -> "CyclicCodeEncoder":
        """Create a standard cyclic code by name.

        Args:
            name: Name of the standard code from the list of standard codes.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A cyclic code encoder for the requested standard code.

        Raises:
            ValueError: If the requested code is not recognized.
        """
        standard_codes = {
            "Hamming(7,4)": {"code_length": 7, "generator_polynomial": 0b1011},
            "Simplex(7,3)": {"code_length": 7, "generator_polynomial": 0b10111},
            "BCH(15,7)": {"code_length": 15, "generator_polynomial": 0b10011},
            "BCH(15,5)": {"code_length": 15, "generator_polynomial": 0b10100110111},
            "Golay(23,12)": {"code_length": 23, "generator_polynomial": 0b101011100011},
        }

        if name not in standard_codes:
            valid_names = list(standard_codes.keys())
            raise ValueError(f"Unknown standard code: {name}. Valid options are: {valid_names}")

        params = standard_codes[name].copy()
        params.update(kwargs)

        return cls(**params)

    @lru_cache(maxsize=8)
    def minimum_distance(self) -> int:
        """Calculate the minimum distance of the code.

        For cyclic codes, the minimum distance is the minimum weight of any non-zero codeword.
        This implements an optimized approach for small to medium-sized codes.

        Returns:
            The minimum distance of the code
        """
        # The minimum distance is at least the minimum weight of the generator polynomial
        min_dist = bin(self._generator_poly.value).count("1")

        # For small codes, we can enumerate all codewords and find the minimum weight
        if self._dimension <= 12:  # Practical limit for enumeration
            min_weight = float("inf")

            # Start from message 1 (skip the zero message which gives zero codeword)
            for i in range(1, 2**self._dimension):
                msg_poly = BinaryPolynomial(i)
                code_poly = self.encode_message_polynomial(msg_poly)
                weight = bin(code_poly.value).count("1")
                min_weight = min(min_weight, weight)

            # Convert min_weight from float to int to satisfy type checker
            return int(min_weight)

        # For larger codes, we return a lower bound
        return min_dist

    def _tensor_to_int(self, tensor: torch.Tensor, bit_length: int) -> int:
        """Convert a binary tensor to an integer.

        Args:
            tensor: The binary tensor to convert
            bit_length: The number of bits to consider

        Returns:
            The integer representation
        """
        result = 0
        for i in range(bit_length):
            if tensor[..., i] > 0.5:  # Binarize
                result |= 1 << i
        return result

    def _polynomial_to_tensor(self, poly: BinaryPolynomial, tensor: torch.Tensor, idx: int, batch_shape: tuple, max_degree: Optional[int] = None) -> None:
        """Convert a polynomial to a tensor in-place.

        Args:
            poly: The polynomial to convert
            tensor: The tensor to fill
            idx: The batch index
            batch_shape: The shape of the batch
            max_degree: Maximum degree to consider (optional)
        """
        coeffs = poly.to_coefficient_list()
        max_idx = len(coeffs) if max_degree is None else min(len(coeffs), max_degree + 1)

        for i in range(max_idx):
            if coeffs[i] > 0.5:
                if batch_shape:
                    tensor[..., idx, i] = 1.0
                else:
                    tensor[..., i] = 1.0

    def __repr__(self) -> str:
        """Return a string representation of the encoder.

        Returns:
            A string representation with key parameters
        """
        constructor_param = ""
        if hasattr(self, "_polynomial_source"):
            if self._polynomial_source == "generator_polynomial":
                constructor_param = f"generator_polynomial={self._generator_poly.value}"
            elif self._polynomial_source == "check_polynomial":
                constructor_param = f"check_polynomial={self._check_poly.value}"
            else:
                constructor_param = f"generator_polynomial={self._generator_poly.value}, " f"check_polynomial={self._check_poly.value}"

        return f"{self.__class__.__name__}(" f"code_length={self._length}, " f"{constructor_param}, " f"dimension={self._dimension}, " f"redundancy={self._redundancy}" f")"

    # TODO: Move this to the parent class
    def _compute_check_matrix(self) -> None:
        """Compute the parity check matrix from the generator matrix.

        For a cyclic code in systematic form, the check matrix should have a structure that detects
        errors in all positions. We construct it such that: H = [P^T | I_m] where P is the parity
        submatrix of G.
        """
        # For a systematic (n,k) code with generator matrix G = [I_k | P],
        # the check matrix is H = [P^T | I_(n-k)]
        identity_part = torch.eye(self._redundancy, dtype=torch.float32, device=self.generator_matrix.device)

        if self.information_set == "left":
            # For 'left' information set, G = [I_k | P]
            parity_part = self.generator_matrix[:, self._dimension :].T
            # H = [P^T | I_m]
            self._check_matrix = torch.cat([parity_part, identity_part], dim=1)
        else:
            # For 'right' information set, G = [P | I_k]
            parity_part = self.generator_matrix[:, : self._redundancy].T
            # H = [I_m | P^T]
            self._check_matrix = torch.cat([identity_part, parity_part], dim=1)
