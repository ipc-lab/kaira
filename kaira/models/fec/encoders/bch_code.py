"""BCH code implementation for forward error correction.

This module implements Bose-Chaudhuri-Hocquenghem (BCH) codes, a class of cyclic error-correcting
codes that are constructed using polynomials over finite fields. BCH codes are powerful and
versatile, providing the ability to control the trade-off between redundancy and error-correcting
capability.

For given parameters μ ≥ 2 and δ satisfying 2 ≤ δ ≤ 2^μ - 1, a binary BCH code has
the following parameters, where δ = 2τ + 1:

- Length: n = 2^μ - 1
- Dimension: k ≥ n - μτ
- Redundancy: m ≤ μτ
- Minimum distance: d ≥ δ

This implementation handles narrow-sense, primitive BCH codes, which are optimal
for many applications requiring reliable transmission over noisy channels.

    :cite:`lin2004error`
    :cite:`moon2005error`
    :cite:`richardson2008modern`
"""

from functools import cache, lru_cache
from typing import Any, Dict, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from ..algebra import BinaryPolynomial, FiniteBifield
from .cyclic_code import CyclicCodeEncoder


@cache
def compute_bch_generator_polynomial(mu: int, delta: int) -> BinaryPolynomial:
    """Compute the generator polynomial for a BCH code.

    Args:
        mu: The parameter μ of the BCH code.
        delta: The design distance δ of the BCH code.

    Returns:
        The generator polynomial.
    """
    # Create the finite field
    field = FiniteBifield(mu)

    # Get the primitive element
    alpha = field.primitive_element()

    # Compute the minimal polynomials of alpha^1, alpha^2, ..., alpha^(delta-1)
    minimal_polys = set()
    for i in range(1, delta):
        minimal_poly = (alpha**i).minimal_polynomial()
        minimal_polys.add(minimal_poly)

    # Compute the LCM of the minimal polynomials
    if not minimal_polys:
        raise ValueError("No minimal polynomials found")

    # Convert the set to a list for consistent ordering
    minimal_polys_list = sorted(list(minimal_polys), key=lambda p: p.value)

    # Compute the LCM
    generator_poly = minimal_polys_list[0]
    for poly in minimal_polys_list[1:]:
        generator_poly = generator_poly.lcm(poly)

    return generator_poly


@cache
def get_valid_bose_distances(mu: int) -> List[int]:
    """Get all valid Bose distances for a given mu.

    Args:
        mu: The parameter μ of the BCH code.

    Returns:
        List of all valid Bose distances for the given mu.
    """

    valid_distances = []
    for delta in range(2, 2**mu):
        if is_bose_distance(mu, delta):
            valid_distances.append(delta)

    return valid_distances


@lru_cache(maxsize=64)
def is_bose_distance(mu: int, delta: int) -> bool:
    """Check if delta is a Bose distance for the given mu.

    A Bose distance is a value δ such that the BCH code with parameters μ and δ
    has a different generator polynomial than the BCH code with parameters μ and δ-1.

    Args:
        mu: The parameter μ of the BCH code.
        delta: The potential Bose distance δ.

    Returns:
        True if delta is a Bose distance, False otherwise.
    """
    # Simple checks first
    if delta < 2 or delta > 2**mu - 1:
        return False

    if delta == 2:
        return True  # δ=2 is always a Bose distance

    # Special cases for efficiency
    if delta == 3:
        return True  # δ=3 is always a Bose distance
    if delta == 5 and mu >= 3:
        return True  # δ=5 is a Bose distance for mu >= 3
    if delta == 2**mu - 1:
        return True  # Maximum possible δ is always a Bose distance

    # Check if the minimal polynomial of alpha^delta is already in the LCM set
    field = FiniteBifield(mu)
    alpha = field.primitive_element()

    # Get minimal polynomials for powers 1 to delta-1
    minimal_polys = set()
    for i in range(1, delta):
        minimal_poly = (alpha**i).minimal_polynomial()
        minimal_polys.add(minimal_poly.value)  # Store the value for easier comparison

    # Check if the minimal polynomial of alpha^delta is already included
    delta_poly = (alpha**delta).minimal_polynomial()
    return delta_poly.value not in minimal_polys


def create_bch_generator_matrix(length: int, generator_poly: BinaryPolynomial, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create the generator matrix for a BCH code.

    This function creates a systematic generator matrix for a BCH code.

    Args:
        length: The length of the code.
        generator_poly: The generator polynomial.
        dtype: The data type for the resulting tensor. Default is torch.float32.
        device: The device to place the tensor on. Default is None (uses current device).

    Returns:
        The generator matrix.
    """
    # Compute dimensions
    n = length
    redundancy = generator_poly.degree
    dimension = n - redundancy

    # Create the generator matrix
    G = torch.zeros((dimension, n), dtype=dtype, device=device)

    # First, set the identity matrix in the first k columns (for systematic form)
    for i in range(dimension):
        G[i, i] = 1.0

    # For each row, compute the parity part
    for i in range(dimension):
        # Multiply the message polynomial x^i by x^(n-k)
        message_poly = BinaryPolynomial(1 << i)
        shifted_poly = BinaryPolynomial(message_poly.value << redundancy)

        # Find the remainder when divided by the generator polynomial
        remainder = shifted_poly % generator_poly

        # Set the parity bits in the generator matrix
        # The remainder corresponds to the parity bits
        coeffs = remainder.to_coefficient_list()
        for j in range(min(len(coeffs), redundancy)):
            if coeffs[j] == 1:
                G[i, dimension + j] = 1.0

    return G


@ModelRegistry.register_model("bch_code_encoder")
class BCHCodeEncoder(CyclicCodeEncoder):
    r"""Encoder for BCH (Bose–Chaudhuri–Hocquenghem) codes.

    BCH codes are a class of powerful cyclic error-correcting codes that can be designed
    to correct multiple errors. They are constructed using polynomials over finite fields
    and provide great flexibility in the trade-off between redundancy and error-correcting
    capability :cite:`lin2004error,richardson2008modern`.

    For given parameters μ ≥ 2 and δ satisfying 2 ≤ δ ≤ 2^μ - 1, a binary BCH code has
    the following parameters, where δ = 2τ + 1:

    - Length: n = 2^μ - 1
    - Dimension: k ≥ n - μτ
    - Redundancy: m ≤ μτ
    - Minimum distance: d ≥ δ

    This implementation handles narrow-sense, primitive BCH codes
    :cite:`lin2004error,moon2005error,sklar2001digital`.

    Args:
        mu (int): The parameter μ of the code. Must satisfy μ ≥ 2.
        delta (int): The design distance δ of the code. Must satisfy 2 ≤ δ ≤ 2^μ - 1
            and be a valid Bose distance.
        information_set (Union[List[int], torch.Tensor, str], optional): Information set
            specification. Default is "left".
        dtype (torch.dtype, optional): Data type for internal tensors. Default is torch.float32.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> encoder = BCHCodeEncoder(mu=4, delta=5)
        >>> print(f"Length: {encoder.length, Dimension: {encoder.dimension}, Redundancy: {encoder.redundancy}")
        Length: 15, Dimension: 7, Redundancy: 8
        >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0.])
        >>> codeword = encoder(message)
        >>> print(codeword)
        tensor([1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1.])
    """

    def __init__(self, mu: int, delta: int, information_set: Union[List[int], torch.Tensor, str] = "left", dtype: torch.dtype = torch.float32, **kwargs: Any):
        """Initialize the BCH code encoder.

        Args:
            mu: The parameter μ of the code. Must satisfy μ ≥ 2.
            delta: The design distance δ of the code. Must satisfy 2 ≤ δ ≤ 2^μ - 1
                and be a valid Bose distance.
            information_set: Either indices of information positions, which must be a k-sublist
                of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
            dtype: Data type for internal tensors. Default is torch.float32.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If μ < 2 or if δ is not a valid Bose distance.
        """
        if mu < 2:
            raise ValueError("'mu' must satisfy mu >= 2")
        if not 2 <= delta <= 2**mu - 1:
            raise ValueError("'delta' must satisfy 2 <= delta <= 2**mu - 1")

        # Store parameters
        self._mu = mu
        self._delta = delta
        self._dtype = dtype

        # Check if delta is a valid Bose distance
        if not is_bose_distance(mu, delta):
            # Find the next valid Bose distance
            valid_distances = get_valid_bose_distances(mu)
            next_deltas = [d for d in valid_distances if d > delta]

            if next_deltas:
                next_delta = next_deltas[0]
                raise ValueError(f"'delta' must be a Bose distance (the next one is {next_delta})")
            else:
                raise ValueError("'delta' must be a Bose distance")

        # Compute the generator polynomial
        self._generator_polynomial = compute_bch_generator_polynomial(mu, delta)

        # Compute code parameters
        n = 2**mu - 1
        m = self._generator_polynomial.degree
        k = n - m

        # Calculate error correction capability
        self._error_correction_capability = (delta - 1) // 2

        # Get device from kwargs if provided
        # device = kwargs.get("device", None)

        # Create generator matrix
        # generator_matrix = create_bch_generator_matrix(length=n, generator_poly=self._generator_polynomial, dtype=dtype, device=device)

        # Initialize the parent class with proper parameters
        super().__init__(code_length=n, generator_polynomial=self._generator_polynomial.value, information_set=information_set, dtype=dtype, **kwargs)

        # Store dimensions
        self._length = n
        self._dimension = k
        self._redundancy = m

        # Create the finite field (used for decoding)
        self._field = FiniteBifield(mu)
        self._alpha = self._field.primitive_element()

        # Compute the check matrix
        self._compute_check_matrix()

        # Register the check matrix buffer
        self.register_buffer("check_matrix", self._check_matrix)

    def _compute_check_matrix(self) -> None:
        """Compute the parity check matrix from the generator matrix."""
        # For a systematic code, the check matrix H can be derived from the generator matrix G.
        # If G = [I_k | P], then H = [P^T | I_(n-k)]
        identity_part = torch.eye(self._redundancy, dtype=self._dtype, device=self.generator_matrix.device)
        parity_part = self.generator_matrix[:, self._dimension :].T

        # Construct H = [P^T | I_m]
        self._check_matrix = torch.cat([parity_part, identity_part], dim=1)

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
        """Error correction capability of the code (t = ⌊(δ-1)/2⌋)."""
        return self._error_correction_capability

    @lru_cache(maxsize=None)
    def minimum_distance(self) -> int:
        """Get the minimum distance of the code.

        For BCH codes, the minimum distance is at least the design distance.

        Returns:
            The minimum distance of the code, which is at least δ.
        """
        return self._delta

    @classmethod
    def from_design_rate(cls, mu: int, target_rate: float, **kwargs: Any) -> "BCHCodeEncoder":
        """Create a BCH code with a design rate close to the target rate.

        Args:
            mu: The parameter μ of the BCH code.
            target_rate: The target rate (k/n) of the code.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A BCH code encoder with rate close to the target rate.

        Raises:
            ValueError: If no suitable code can be found.
        """
        if mu < 2:
            raise ValueError("'mu' must satisfy mu >= 2")
        if not 0 < target_rate < 1:
            raise ValueError("'target_rate' must be between 0 and 1")

        # Get all valid Bose distances for this mu
        valid_distances = get_valid_bose_distances(mu)

        # Calculate the code length
        n = 2**mu - 1

        # Find the delta that gives the closest rate to the target
        best_delta = None
        best_diff = float("inf")

        for delta in valid_distances:
            # Compute the generator polynomial to get the dimension
            generator_poly = compute_bch_generator_polynomial(mu, delta)
            k = n - generator_poly.degree
            rate = k / n

            diff = abs(rate - target_rate)
            if diff < best_diff:
                best_diff = diff
                best_delta = delta

        if best_delta is None:
            raise ValueError(f"Could not find a suitable BCH code for mu={mu} and rate={target_rate}")

        return cls(mu=mu, delta=best_delta, **kwargs)

    @classmethod
    def get_standard_codes(cls) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of standard BCH codes with their parameters.

        Returns:
            Dictionary mapping code names to their parameters.
        """
        return {
            "BCH(7,4)": {"mu": 3, "delta": 3},  # Equivalent to Hamming(7,4)
            "BCH(15,7)": {"mu": 4, "delta": 5},  # Can correct 2 errors
            "BCH(15,5)": {"mu": 4, "delta": 7},  # Can correct 3 errors
            "BCH(31,16)": {"mu": 5, "delta": 7},  # Can correct 3 errors
            "BCH(31,11)": {"mu": 5, "delta": 11},  # Can correct 5 errors
            "BCH(63,36)": {"mu": 6, "delta": 11},  # Can correct 5 errors
            "BCH(63,24)": {"mu": 6, "delta": 15},  # Can correct 7 errors
            "BCH(127,64)": {"mu": 7, "delta": 21},  # Can correct 10 errors
            "BCH(127,36)": {"mu": 7, "delta": 31},  # Can correct 15 errors
            "BCH(255,123)": {"mu": 8, "delta": 39},  # Can correct 19 errors
            "BCH(255,71)": {"mu": 8, "delta": 59},  # Can correct 29 errors
        }

    @classmethod
    def create_standard_code(cls, name: str, **kwargs: Any) -> "BCHCodeEncoder":
        """Create a standard BCH code by name.

        Args:
            name: Name of the standard code from get_standard_codes().
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A BCH code encoder for the requested standard code.

        Raises:
            ValueError: If the requested code is not recognized.
        """
        standard_codes = cls.get_standard_codes()
        if name not in standard_codes:
            valid_names = list(standard_codes.keys())
            raise ValueError(f"Unknown standard code: {name}. Valid options are: {valid_names}")

        params = standard_codes[name].copy()
        params.update(kwargs)

        return cls(**params)

    def __repr__(self) -> str:
        """Return a string representation of the encoder.

        Returns:
            A string representation with key parameters
        """
        return f"{self.__class__.__name__}(" f"mu={self._mu}, " f"delta={self._delta}, " f"length={self._length}, " f"dimension={self._dimension}, " f"redundancy={self._redundancy}, " f"t={self._error_correction_capability}, " f"dtype={self._dtype.__repr__()}" f")"

    def calculate_syndrome_polynomial(self, received: List[Any]) -> List[Any]:
        """Calculate the syndrome polynomial for a received word.

        This method computes the syndrome polynomial S(x) for a received codeword by evaluating
        the received polynomial at powers of alpha, which are the roots of the generator polynomial.

        Args:
            received: List of field elements representing the received word

        Returns:
            List of syndrome values in the field, S = [S_0, S_1, ..., S_{2t-1}]
        """
        syndrome = []
        for i in range(1, 2 * self._error_correction_capability + 1):
            # Evaluate the received polynomial at alpha^i
            alpha_i = self._alpha**i
            eval_result = self._field(0)  # Initialize with field zero element
            for j, bit in enumerate(received):
                if bit != self._field.zero:
                    # For each non-zero bit, add alpha^(j*i) to the result
                    eval_result = eval_result + (alpha_i**j)
            syndrome.append(eval_result)

        return syndrome
