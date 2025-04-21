"""Finite field algebra utilities for forward error correction.

This module provides mathematical utilities for working with binary polynomials and finite fields,
which are essential for algebraic error correction codes such as BCH, Reed-Solomon, and others.

The module implements:
- Binary polynomials over GF(2) with efficient arithmetic operations
- Finite fields GF(2^m) with complete field arithmetic
- Field element operations including inverses, minimal polynomials, and traces

These implementations form the mathematical foundation for more complex error correction codes
and are optimized for both correctness and computational efficiency. The module supports both
symbolic manipulations and numeric computations compatible with PyTorch tensors.

    :cite:`lin2004error`
    :cite:`blahut2003algebraic`
"""

from typing import Any, Dict, List

import torch


class BinaryPolynomial:
    """Class representing a binary polynomial.

    This implements polynomials over the binary field GF(2), where coefficients are either 0 or 1.
    The polynomial is represented using a binary integer, where each bit represents a coefficient.
    """

    def __init__(self, value: int = 0):
        """Initialize the binary polynomial.

        Args:
            value: Integer representation of the polynomial, where each bit
                  represents a coefficient. Default is 0.
        """
        self.value = value

    @property
    def degree(self) -> int:
        """Get the degree of the polynomial.

        Returns:
            The degree of the polynomial, or -1 for the zero polynomial.
        """
        if self.value == 0:
            return -1  # Zero polynomial has degree -1 by convention
        return self.value.bit_length() - 1

    def __mul__(self, other: "BinaryPolynomial") -> "BinaryPolynomial":
        """Multiply two binary polynomials.

        Args:
            other: The polynomial to multiply with.

        Returns:
            The product polynomial.
        """
        if not isinstance(other, BinaryPolynomial):
            return NotImplemented

        # Optimization for small polynomials
        if self.value == 0 or other.value == 0:
            return BinaryPolynomial(0)

        result = 0
        a, b = self.value, other.value

        # Implement polynomial multiplication in GF(2)
        while b > 0:
            if b & 1:  # If the current bit of b is 1
                result ^= a  # XOR (equivalent to addition in GF(2))
            a <<= 1  # Multiply a by x (shift left)
            b >>= 1  # Move to the next bit of b

        return BinaryPolynomial(result)

    def __mod__(self, modulus: "BinaryPolynomial") -> "BinaryPolynomial":
        """Compute polynomial modulo another polynomial.

        Args:
            modulus: The modulus polynomial.

        Returns:
            The remainder polynomial.
        """
        if not isinstance(modulus, BinaryPolynomial):
            return NotImplemented

        if modulus.value == 0:
            raise ValueError("Cannot compute modulo the zero polynomial")

        # Optimization for common cases
        if self.value == 0:
            return BinaryPolynomial(0)
        if self.degree < modulus.degree:
            return self

        remainder = self.value
        modulus_degree = modulus.degree
        modulus_value = modulus.value

        # Implement polynomial division in GF(2)
        while True:
            remainder_degree = BinaryPolynomial(remainder).degree
            if remainder_degree < modulus_degree:
                break

            # Subtract (XOR) modulus * x^(remainder_degree - modulus_degree)
            shift = remainder_degree - modulus_degree
            remainder ^= modulus_value << shift

        return BinaryPolynomial(remainder)

    def evaluate(self, x: Any) -> Any:
        """Evaluate the polynomial at a given point.

        Args:
            x: The point at which to evaluate the polynomial.

        Returns:
            The value of the polynomial at x.
        """
        if self.value == 0:
            # Optimization for zero polynomial
            if hasattr(x, "field"):  # If x is a field element
                return x.__class__(x.field, 0)
            return 0

        value = self.value
        # Handle field element evaluation
        if hasattr(x, "field"):  # If x is a field element
            result = x.__class__(x.field, 0)
            power = x.__class__(x.field, 1)
            while value > 0:
                if value & 1:
                    result = result + power
                power = power * x
                value >>= 1
            return result
        # Handle integer evaluation
        result = 0
        power = 1
        while value > 0:
            if value & 1:
                result ^= power
            power *= x
            value >>= 1
        return result

    def lcm(self, other: "BinaryPolynomial") -> "BinaryPolynomial":
        """Compute the least common multiple of two polynomials.

        Args:
            other: The other polynomial.

        Returns:
            The least common multiple polynomial.
        """
        if not isinstance(other, BinaryPolynomial):
            raise TypeError("Operand must be a BinaryPolynomial")

        # Optimizations
        if self.value == 0 or other.value == 0:
            return BinaryPolynomial(0)
        if self == other:
            return self

        # Use the formula: lcm(a, b) = a * b / gcd(a, b)
        gcd = self.gcd(other)
        if gcd.value == 0:
            return BinaryPolynomial(0)

        product = self * other
        quotient = product.div(gcd)
        return quotient

    def gcd(self, other: "BinaryPolynomial") -> "BinaryPolynomial":
        """Compute the greatest common divisor of two polynomials.

        Args:
            other: The other polynomial.

        Returns:
            The greatest common divisor polynomial.
        """
        if not isinstance(other, BinaryPolynomial):
            raise TypeError("Operand must be a BinaryPolynomial")

        a, b = self, other

        # Handle special cases
        if a.value == 0:
            return b
        if b.value == 0:
            return a
        if a == b:
            return a

        # Euclidean algorithm
        while b.value != 0:
            a, b = b, a % b

        return a

    def div(self, divisor: "BinaryPolynomial") -> "BinaryPolynomial":
        """Compute polynomial division (quotient).

        Args:
            divisor: The divisor polynomial.

        Returns:
            The quotient polynomial.
        """
        if not isinstance(divisor, BinaryPolynomial):
            raise TypeError("Operand must be a BinaryPolynomial")

        if divisor.value == 0:
            raise ValueError("Division by zero polynomial")

        # Optimizations
        if self.value == 0:
            return BinaryPolynomial(0)
        if self == divisor:
            return BinaryPolynomial(1)
        if self.degree < divisor.degree:
            return BinaryPolynomial(0)

        quotient = 0
        remainder = self.value
        divisor_degree = divisor.degree
        divisor_value = divisor.value

        # Implement polynomial division in GF(2)
        while True:
            remainder_degree = BinaryPolynomial(remainder).degree
            if remainder_degree < divisor_degree:
                break

            # Calculate the shift for the current division step
            shift = remainder_degree - divisor_degree

            # Update the quotient (set the bit at position 'shift')
            quotient |= 1 << shift

            # Subtract (XOR) divisor * x^shift from the remainder
            remainder ^= divisor_value << shift

        return BinaryPolynomial(quotient)

    def derivative(self) -> "BinaryPolynomial":
        """Compute the formal derivative of the polynomial.

        In GF(2), the derivative is the sum of the terms with odd powers.

        Returns:
            The derivative polynomial.
        """
        result = 0
        value = self.value
        power = 0

        while value > 0:
            if power % 2 == 1 and (value & 1):
                result |= 1 << (power - 1)
            power += 1
            value >>= 1

        return BinaryPolynomial(result)

    def to_coefficient_list(self) -> List[int]:
        """Convert the polynomial to a list of coefficients.

        Returns:
            List of coefficients, from lowest to highest degree.
        """
        if self.value == 0:
            return [0]

        coeffs = []
        value = self.value

        while value > 0:
            coeffs.append(value & 1)
            value >>= 1

        return coeffs

    def to_torch_tensor(self, dtype=None, device=None):
        """Convert the polynomial to a PyTorch tensor.

        Args:
            dtype: The data type for the tensor. Default is torch.float32.
            device: The device to place the tensor on. Default is None (uses current device).

        Returns:
            Tensor of coefficients, from lowest to highest degree.

        Raises:
            ImportError: If torch is not installed.
        """
        if torch is None:
            raise ImportError("PyTorch is not installed. Please install it to use this feature.")

        coeffs = self.to_coefficient_list()
        dtype = dtype or torch.float32
        return torch.tensor(coeffs, dtype=dtype, device=device)

    def __str__(self) -> str:
        """Convert to string representation.

        Returns:
            The polynomial as a string.
        """
        if self.value == 0:
            return "0"

        terms = []
        value = self.value
        degree = 0

        while value > 0:
            if value & 1:  # If the current bit is 1
                if degree == 0:
                    terms.append("1")
                elif degree == 1:
                    terms.append("x")
                else:
                    terms.append(f"x^{degree}")
            degree += 1
            value >>= 1

        return " + ".join(reversed(terms))

    def __repr__(self) -> str:
        """Convert to formal string representation.

        Returns:
            The formal representation of the polynomial.
        """
        return f"BinaryPolynomial(0b{bin(self.value)[2:]})"

    def __eq__(self, other: object) -> bool:
        """Check if two polynomials are equal.

        Args:
            other: The polynomial to compare with.

        Returns:
            True if the polynomials are equal, False otherwise.
        """
        if not isinstance(other, BinaryPolynomial):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        """Compute a hash for the polynomial.

        Returns:
            Hash of the polynomial.
        """
        return hash(self.value)


class FiniteBifield:
    """Class representing a finite field GF(2^m).

    This implements the Galois Field GF(2^m) for a given m.
    """

    # Cache for field instances
    _instances: Dict[int, "FiniteBifield"] = {}

    def __new__(cls, m: int):
        """Create a new field instance or return a cached one.

        Args:
            m: The power of 2 determining the field size (2^m).

        Returns:
            A field instance.
        """
        if m in cls._instances:
            return cls._instances[m]
        instance = super().__new__(cls)
        cls._instances[m] = instance
        return instance

    def __init__(self, m: int):
        """Initialize the finite field.

        Args:
            m: The power of 2 determining the field size (2^m).
        """
        # Skip initialization if already initialized
        if hasattr(self, "m") and self.m == m:
            return

        if m <= 0:
            raise ValueError("m must be positive")

        self.m: int = m
        self.size: int = 2**m
        self._element_cache: Dict[int, "FiniteBifieldElement"] = {}

        # Use common primitive polynomials
        if m <= 10:
            primitive_polys = {
                1: 0b11,  # x + 1
                2: 0b111,  # x^2 + x + 1
                3: 0b1011,  # x^3 + x + 1
                4: 0b10011,  # x^4 + x + 1
                5: 0b100101,  # x^5 + x^2 + 1
                6: 0b1000011,  # x^6 + x + 1
                7: 0b10000011,  # x^7 + x + 1
                8: 0b100011101,  # x^8 + x^4 + x^3 + x^2 + 1
                9: 0b1000010001,  # x^9 + x^4 + 1
                10: 0b10000001001,  # x^10 + x^3 + 1
            }
            self.modulus = BinaryPolynomial(primitive_polys[m])
        elif m <= 16:
            # Additional primitive polynomials for larger fields
            primitive_polys = {
                11: 0b100000000101,  # x^11 + x^2 + 1
                12: 0b1000000001101,  # x^12 + x^3 + x^2 + 1
                13: 0b10000000011011,  # x^13 + x^4 + x^3 + x + 1
                14: 0b100000000010001,  # x^14 + x^5 + 1
                15: 0b1000000000001011,  # x^15 + x + 1
                16: 0b10000000000001011,  # x^16 + x^3 + x + 1
            }
            self.modulus = BinaryPolynomial(primitive_polys[m])
        else:
            # For very large fields, support could be extended
            raise NotImplementedError("Fields larger than GF(2^16) are not implemented")

        # Cache for log and exp tables to speed up arithmetic operations
        self._exp_table = [0] * self.size
        self._log_table = [0] * self.size
        self._init_log_exp_tables()

    @property
    def zero(self) -> "FiniteBifieldElement":
        """Get the zero element of the field.

        Returns:
            The zero element (additive identity) of the field.
        """
        return self(0)

    @property
    def one(self) -> "FiniteBifieldElement":
        """Get the one element of the field.

        Returns:
            The one element (multiplicative identity) of the field.
        """
        return self(1)

    def _init_log_exp_tables(self) -> None:
        """Initialize log and exponential tables for fast field arithmetic."""
        # Initialize exp and log tables
        primitive_element = 2  # The element "x" is 2 in our representation
        self._exp_table[0] = 1
        self._log_table[0] = 0  # log(0) is undefined, but we set it to 0 for convenience

        value = 1
        for i in range(1, self.size):
            value = (value * primitive_element) % self.size
            if value == 0:
                # This should not happen with a primitive polynomial
                break

            self._exp_table[i] = value
            self._log_table[value] = i

    def __call__(self, value: int) -> "FiniteBifieldElement":
        """Create an element of the field.

        Args:
            value: The integer representation of the field element.

        Returns:
            A field element.
        """
        # Use caching to reduce object creation
        value = value % self.size
        if value in self._element_cache:
            return self._element_cache[value]

        element = FiniteBifieldElement(self, value)
        self._element_cache[value] = element
        return element

    def __eq__(self, other: object) -> bool:
        """Check if two fields are equal.

        Args:
            other: The field to compare with.

        Returns:
            True if the fields are equal, False otherwise.
        """
        if not isinstance(other, FiniteBifield):
            return NotImplemented
        return self.m == other.m

    def primitive_element(self) -> "FiniteBifieldElement":
        """Get a primitive element of the field.

        Returns:
            A primitive element (generator) of the field.
        """
        # For our implementation, the element 'x' (represented by value 2 or 0b10)
        # is primitive when using the standard primitive polynomials
        return self(0b10)

    def get_all_elements(self) -> List["FiniteBifieldElement"]:
        """Get all elements of the field.

        Returns:
            A list of all field elements.
        """
        return [self(i) for i in range(self.size)]

    def get_minimal_polynomials(self) -> Dict[int, BinaryPolynomial]:
        """Get all minimal polynomials in the field.

        Returns:
            Dictionary mapping element values to their minimal polynomials.
        """
        # Cache for minimal polynomials
        if hasattr(self, "_minimal_polys"):
            return self._minimal_polys

        self._minimal_polys: Dict[int, BinaryPolynomial] = {}
        for i in range(1, self.size):
            element = self(i)
            minimal_poly = element.minimal_polynomial()
            self._minimal_polys[i] = minimal_poly

        return self._minimal_polys

    def __repr__(self) -> str:
        """Convert to formal string representation.

        Returns:
            The formal representation of the field.
        """
        return f"FiniteBifield(m={self.m})"


class FiniteBifieldElement:
    """Class representing an element of a finite field GF(2^m)."""

    # Add class-level type annotation
    _minimal_poly: BinaryPolynomial

    def __init__(self, field: FiniteBifield, value: int):
        """Initialize the field element.

        Args:
            field: The finite field this element belongs to.
            value: The integer representation of the element.
        """
        self.field = field
        self.value = value % (2**field.m)  # Ensure the value is within field range
        # _minimal_poly will be initialized in the minimal_polynomial method

    def __add__(self, other: "FiniteBifieldElement") -> "FiniteBifieldElement":
        """Add two field elements.

        Addition in GF(2^m) is just bitwise XOR.

        Args:
            other: The element to add.

        Returns:
            The sum of the elements.
        """
        if not isinstance(other, FiniteBifieldElement):
            return NotImplemented
        if self.field != other.field:
            raise ValueError("Elements must be from the same field")

        return FiniteBifieldElement(self.field, self.value ^ other.value)

    def __mul__(self, other: "FiniteBifieldElement") -> "FiniteBifieldElement":
        """Multiply two field elements.

        Multiplication in GF(2^m) is polynomial multiplication modulo the field's modulus.

        Args:
            other: The element to multiply with.

        Returns:
            The product of the elements.
        """
        if not isinstance(other, FiniteBifieldElement):
            return NotImplemented
        if self.field != other.field:
            raise ValueError("Elements must be from the same field")

        # Optimizations for common cases
        if self.value == 0 or other.value == 0:
            return self.field(0)
        if self.value == 1:
            return other
        if other.value == 1:
            return self

        # Convert to polynomials
        a_poly = BinaryPolynomial(self.value)
        b_poly = BinaryPolynomial(other.value)

        # Multiply and reduce modulo the field's modulus
        result_poly = (a_poly * b_poly) % self.field.modulus

        return FiniteBifieldElement(self.field, result_poly.value)

    def __pow__(self, exponent: int) -> "FiniteBifieldElement":
        """Raise the element to a power.

        Args:
            exponent: The exponent to raise to.

        Returns:
            The element raised to the power.
        """
        if exponent < 0:
            raise ValueError("Exponent must be non-negative")

        # Handle special cases
        if exponent == 0:
            return FiniteBifieldElement(self.field, 1)
        if exponent == 1:
            return self
        if self.value == 0:
            return self
        if self.value == 1:
            return self

        # Use square-and-multiply algorithm for efficiency
        result = FiniteBifieldElement(self.field, 1)
        base = self
        while exponent > 0:
            if exponent & 1:  # If the current bit is 1
                result = result * base
            base = base * base
            exponent >>= 1

        return result

    def minimal_polynomial(self) -> BinaryPolynomial:
        """Compute the minimal polynomial of this field element.

        Returns:
            The minimal polynomial of the field element.
        """
        # Cache computation
        if hasattr(self, "_minimal_poly"):
            return self._minimal_poly

        conjugates = self.conjugates()
        d = len(conjugates)

        # Brute-force search for monic polynomial of degree d with all conjugates as roots
        for mask in range(1 << d):
            # Build integer representation p_value for polynomial X^d + sum_{i:mask bit i} X^i
            p_value = 1 << d
            for i in range(d):
                if (mask >> i) & 1:
                    p_value ^= 1 << i
            p = BinaryPolynomial(p_value)

            # Check p vanishes on all conjugates
            all_zero = True
            for conj in conjugates:
                eval_result = p.evaluate(conj)
                val = eval_result.value if hasattr(eval_result, "value") else eval_result
                if val != 0:
                    all_zero = False
                    break
            if all_zero:
                self._minimal_poly = p
                return p

        # If execution reaches here, create a default polynomial to satisfy type checking
        # This should never happen in practice
        # default_poly = BinaryPolynomial(1)
        raise RuntimeError(f"Failed to find minimal polynomial for element {self}")  # Will not return

    def trace(self) -> int:
        """Compute the trace of the field element.

        The trace is the sum of the element's conjugates.
        In GF(2^m), it's the sum of a, a^2, a^4, ..., a^(2^(m-1)).

        Returns:
            The trace (0 or 1).
        """
        result = self.value
        element = self

        for _ in range(1, self.field.m):
            element = element * element  # Square
            result ^= element.value

        return result & 1

    def inverse(self) -> "FiniteBifieldElement":
        """Compute the multiplicative inverse of the field element.

        Returns:
            The multiplicative inverse.

        Raises:
            ValueError: If the element is zero.
        """
        if self.value == 0:
            raise ValueError("Cannot compute inverse of zero")
        if self.value == 1:
            return self

        # Use Fermat's Little Theorem: a^(2^m - 2) is the inverse of a
        return self ** (self.field.size - 2)

    def to_polynomial(self) -> BinaryPolynomial:
        """Convert the field element to a binary polynomial.

        Returns:
            The binary polynomial representation.
        """
        return BinaryPolynomial(self.value)

    def conjugates(self) -> List["FiniteBifieldElement"]:
        """Get all conjugates of the field element.

        The conjugates are a, a^2, a^4, ..., a^(2^(m-1))

        Returns:
            List of conjugate elements.
        """
        conjugates = [self]
        element = self

        for _ in range(1, self.field.m):
            element = element * element  # Square
            if element.value == self.value:
                break
            conjugates.append(element)

        return conjugates

    def __eq__(self, other: object) -> bool:
        """Check if two field elements are equal.

        Args:
            other: The element to compare with.

        Returns:
            True if the elements are equal, False otherwise.
        """
        if isinstance(other, FiniteBifieldElement):
            return self.field == other.field and self.value == other.value
        elif isinstance(other, int) and other == 0:
            # Special case for comparing with integer 0 for polynomial evaluation tests
            return self.value == 0
        return NotImplemented

    def __hash__(self) -> int:
        """Compute a hash of the field element.

        Returns:
            Hash of the field element.
        """
        return hash((self.field.m, self.value))

    def __str__(self) -> str:
        """Convert to string representation.

        Returns:
            The string representation of the field element.
        """
        return f"{self.value}"

    def __repr__(self) -> str:
        """Convert to formal string representation.

        Returns:
            The formal representation of the field element.
        """
        return f"FiniteBifieldElement({self.field}, {self.value})"
