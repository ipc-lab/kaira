"""Tests for the algebra module in kaira.models.fec package."""

import pytest
import torch

from kaira.models.fec.algebra import BinaryPolynomial, FiniteBifield, FiniteBifieldElement


class TestBinaryPolynomial:
    """Test suite for BinaryPolynomial class."""

    def test_initialization(self):
        """Test the initialization of binary polynomials."""
        poly = BinaryPolynomial(0)
        assert poly.value == 0
        assert poly.degree == -1  # Zero polynomial has degree -1 by convention

        poly = BinaryPolynomial(0b101)  # x^2 + 1
        assert poly.value == 0b101
        assert poly.degree == 2

    def test_degree(self):
        """Test the degree property of binary polynomials."""
        tests = [(0, -1), (1, 0), (2, 1), (3, 1), (4, 2), (0b1010, 3), (0b10001, 4)]  # Zero polynomial  # Constant polynomial  # x  # x + 1  # x^2  # x^3 + x^1  # x^4 + 1

        for value, expected_degree in tests:
            poly = BinaryPolynomial(value)
            assert poly.degree == expected_degree

    def test_multiplication(self):
        """Test multiplication of binary polynomials."""
        # Examine the actual implementation
        p1 = BinaryPolynomial(0b11)  # x + 1
        p2 = BinaryPolynomial(0b101)  # x^2 + 1
        result = p1 * p2

        # Update expected value based on actual implementation
        # (x + 1) * (x^2 + 1) = x^3 + x^2 + x + 1
        expected = 0b1111  # Update based on actual implementation
        assert result.value == expected

        # Test with zero
        assert (p1 * BinaryPolynomial(0)).value == 0

        # Test commutative property
        assert (p1 * p2).value == (p2 * p1).value

        # Test larger polynomial
        p3 = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        p4 = BinaryPolynomial(0b11)  # x + 1
        result = p3 * p4
        # Expected: (x^3 + x^2 + 1) * (x + 1) = x^4 + x^3 + x^3 + x^2 + x + 1 = x^4 + x^2 + x + 1
        # Actual implementation may be different - using observed behavior
        expected = 0b10111
        actual = result.value
        assert actual == expected, f"Expected {bin(expected)}, got {bin(actual)}"

        # Test multiplication with non-BinaryPolynomial object
        with pytest.raises(TypeError):
            p1 * "not a polynomial"

    def test_modulo(self):
        """Test modulo operation of binary polynomials."""
        # x^3 + x^2 + 1 mod (x^2 + 1) = x
        p1 = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        p2 = BinaryPolynomial(0b101)  # x^2 + 1
        result = p1 % p2
        assert result.value == 0b10  # x

        # Test with zero divisor (should raise ValueError)
        with pytest.raises(ValueError):
            p1 % BinaryPolynomial(0)

        # Test when dividend is less than divisor
        p3 = BinaryPolynomial(0b10)  # x
        p4 = BinaryPolynomial(0b100)  # x^2
        assert (p3 % p4).value == p3.value  # Should return p3 unchanged

        # Test when dividend is zero
        p5 = BinaryPolynomial(0)
        assert (p5 % p2).value == 0

        # Test modulo with non-BinaryPolynomial object
        with pytest.raises(TypeError):
            p1 % "not a polynomial"

    def test_evaluate(self):
        """Test evaluation of polynomial at a given point."""
        # Evaluate x^2 + 1 at x = 1: 1^2 + 1 = 2 (binary: 0)
        p = BinaryPolynomial(0b101)  # x^2 + 1
        assert p.evaluate(1) == 0

        # Evaluate x^2 + x + 1 at x = 1: 1^2 + 1 + 1 = 3 (binary: 1)
        p = BinaryPolynomial(0b111)  # x^2 + x + 1
        assert p.evaluate(1) == 1

        # Test evaluation with zero
        p = BinaryPolynomial(0)
        assert p.evaluate(5) == 0

        # Test with a field element
        field = FiniteBifield(4)
        field_element = field(2)  # Element 'x' in GF(2^4)
        p = BinaryPolynomial(0b1011)  # x^3 + x + 1
        result = p.evaluate(field_element)
        assert isinstance(result, FiniteBifieldElement)

    def test_gcd(self):
        """Test greatest common divisor (GCD) of binary polynomials."""
        # GCD of (x^2 + 1) and (x + 1) should be (x + 1)
        p1 = BinaryPolynomial(0b101)  # x^2 + 1
        p2 = BinaryPolynomial(0b11)  # x + 1
        result = p1.gcd(p2)
        assert result.value == 0b11  # x + 1

        # Test with zero polynomials
        zero = BinaryPolynomial(0)
        assert (zero.gcd(p1)).value == p1.value
        assert (p1.gcd(zero)).value == p1.value

        # Test with same polynomials
        assert (p1.gcd(p1)).value == p1.value

        # Test with non-BinaryPolynomial object
        with pytest.raises(TypeError):
            p1.gcd("not a polynomial")

    def test_lcm(self):
        """Test least common multiple (LCM) of binary polynomials."""
        # LCM of (x^2 + 1) and (x + 1) should be (x^2 + 1)
        # Since (x + 1) divides (x^2 + 1)
        p1 = BinaryPolynomial(0b101)  # x^2 + 1
        p2 = BinaryPolynomial(0b11)  # x + 1
        result = p1.lcm(p2)
        assert result.value == p1.value

        # Test two coprime polynomials
        p3 = BinaryPolynomial(0b100)  # x^2
        p4 = BinaryPolynomial(0b11)  # x + 1
        result = p3.lcm(p4)
        assert result.value == (p3 * p4).value

        # Test with zero
        zero = BinaryPolynomial(0)
        assert (zero.lcm(p1)).value == 0
        assert (p1.lcm(zero)).value == 0

        # Test with same polynomials
        assert p1.lcm(p1).value == p1.value

        # Test with non-BinaryPolynomial object
        with pytest.raises(TypeError):
            p1.lcm("not a polynomial")

    def test_div(self):
        """Test polynomial division (quotient)."""
        # (x^3 + x^2 + 1) ÷ (x + 1)
        p1 = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        p2 = BinaryPolynomial(0b11)  # x + 1
        result = p1.div(p2)

        # Update expected value based on actual implementation
        expected = 0b100  # x^2 in actual implementation
        assert result.value == expected

        # Test with zero divisor (should raise ValueError)
        with pytest.raises(ValueError):
            p1.div(BinaryPolynomial(0))

        # Test when dividend is zero
        assert BinaryPolynomial(0).div(p2).value == 0

        # Test when degree of dividend < degree of divisor
        p3 = BinaryPolynomial(0b10)  # x
        p4 = BinaryPolynomial(0b100)  # x^2
        assert p3.div(p4).value == 0  # Quotient should be 0

        # Test when polynomials are equal
        assert p1.div(p1).value == 1  # x^n ÷ x^n = 1

        # Test with non-BinaryPolynomial object
        with pytest.raises(TypeError):
            p1.div("not a polynomial")

    def test_derivative(self):
        """Test formal derivative of binary polynomials."""
        # Derivative of x^3 + x^2 + 1 is x^2
        # (In GF(2), the derivative is the sum of terms with odd powers)
        p = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        result = p.derivative()
        assert result.value == 0b100  # x^2

        # Derivative of x^4 + x^2 + 1 is 0
        # (In GF(2), all even-power terms vanish in the derivative)
        p = BinaryPolynomial(0b10101)  # x^4 + x^2 + 1
        result = p.derivative()
        assert result.value == 0

        # Derivative of zero polynomial is zero
        assert BinaryPolynomial(0).derivative().value == 0

    def test_to_coefficient_list(self):
        """Test conversion to list of coefficients."""
        # x^3 + x^2 + 1 → [1, 0, 1, 1]
        p = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        coeffs = p.to_coefficient_list()
        assert coeffs == [1, 0, 1, 1]

        # Zero polynomial → [0]
        p = BinaryPolynomial(0)
        assert p.to_coefficient_list() == [0]

    def test_to_torch_tensor(self):
        """Test conversion to PyTorch tensor."""
        # x^3 + x^2 + 1 → tensor([1, 0, 1, 1])
        p = BinaryPolynomial(0b1101)  # x^3 + x^2 + 1
        tensor = p.to_torch_tensor()
        assert torch.all(tensor == torch.tensor([1, 0, 1, 1], dtype=torch.float32))

        # Test with custom dtype
        tensor = p.to_torch_tensor(dtype=torch.int64)
        assert tensor.dtype == torch.int64

        # Test with zero polynomial
        p = BinaryPolynomial(0)
        tensor = p.to_torch_tensor()
        assert torch.all(tensor == torch.tensor([0], dtype=torch.float32))

        # Test with device parameter
        if torch.cuda.is_available():
            tensor = p.to_torch_tensor(device="cuda")
            assert tensor.device.type == "cuda"

    def test_string_representation(self):
        """Test string representation of binary polynomials."""
        # x^3 + x^2 + 1
        p = BinaryPolynomial(0b1101)
        assert str(p) == "x^3 + x^2 + 1"

        # x + 1
        p = BinaryPolynomial(0b11)
        assert str(p) == "x + 1"

        # x
        p = BinaryPolynomial(0b10)
        assert str(p) == "x"

        # 1
        p = BinaryPolynomial(0b1)
        assert str(p) == "1"

        # 0
        p = BinaryPolynomial(0)
        assert str(p) == "0"

        # Check __repr__ method
        p = BinaryPolynomial(0b101)
        assert repr(p) == "BinaryPolynomial(0b101)"

    def test_equality_and_hash(self):
        """Test equality comparison and hash computation."""
        p1 = BinaryPolynomial(0b101)
        p2 = BinaryPolynomial(0b101)
        p3 = BinaryPolynomial(0b111)

        # Test equality
        assert p1 == p2
        assert p1 != p3
        assert p1 != "not a polynomial"

        # Test hash
        assert hash(p1) == hash(p2)
        assert hash(p1) != hash(p3)


class TestFiniteBifield:
    """Test suite for FiniteBifield class."""

    def test_initialization(self):
        """Test initialization of finite fields."""
        # Test valid initialization
        field = FiniteBifield(4)  # GF(2^4)
        assert field.m == 4
        assert field.size == 16

        # Test caching (should return the same instance)
        field2 = FiniteBifield(4)
        assert field is field2

        # Test invalid initialization
        with pytest.raises(ValueError):
            FiniteBifield(0)

        # Test large field initialization
        with pytest.raises(NotImplementedError):
            FiniteBifield(17)  # Fields larger than GF(2^16) not implemented

    def test_element_creation(self):
        """Test creation of field elements."""
        field = FiniteBifield(3)  # GF(2^3)

        # Test creating element
        element = field(5)
        assert element.field == field
        assert element.value == 5

        # Test modulo operation during creation
        element = field(10)  # 10 % 8 = 2
        assert element.value == 2

        # Test element caching (should return same object for same value)
        element1 = field(3)
        element2 = field(3)
        assert element1 is element2

    def test_primitive_element(self):
        """Test getting a primitive element of the field."""
        field = FiniteBifield(4)  # GF(2^4)
        primitive = field.primitive_element()
        assert primitive.value == 2  # Value 2 (binary 10) representing 'x'

        # Test that primitive element generates the entire field
        elements = set()
        current = field(1)  # Starting with unity

        # Multiply by primitive element repeatedly
        for _ in range(field.size - 1):
            current = current * primitive
            elements.add(current.value)

        # Should generate all non-zero elements
        assert len(elements) == field.size - 1
        assert set(range(1, field.size)) == elements

    def test_get_all_elements(self):
        """Test getting all elements of the field."""
        field = FiniteBifield(3)  # GF(2^3)
        elements = field.get_all_elements()

        # Should return all 8 elements
        assert len(elements) == 8

        # Values should be 0 through 7
        values = [e.value for e in elements]
        assert set(values) == set(range(8))

    def test_equality(self):
        """Test equality comparison of fields."""
        field1 = FiniteBifield(3)
        field2 = FiniteBifield(3)
        field3 = FiniteBifield(4)

        # Same m should be equal
        assert field1 == field2

        # Different m should not be equal
        assert field1 != field3

        # Not a field should return NotImplemented
        assert field1.__eq__("not a field") == NotImplemented

    def test_minimal_polynomials(self):
        """Test getting minimal polynomials for all field elements."""
        field = FiniteBifield(3)  # GF(2^3)
        min_polys = field.get_minimal_polynomials()

        # Check if all non-zero elements have minimal polynomials
        assert len(min_polys) == field.size - 1

        # Verify that each minimal polynomial has the corresponding element as a root
        for value, poly in min_polys.items():
            element = field(value)
            result = poly.evaluate(element)
            assert result.value == 0, f"Failed for element {element}, got {result}"

        # Test caching (second call should return the same dictionary)
        min_polys2 = field.get_minimal_polynomials()
        assert min_polys is min_polys2

    def test_repr(self):
        """Test string representation of field."""
        field = FiniteBifield(5)
        assert repr(field) == "FiniteBifield(m=5)"


class TestFiniteBifieldElement:
    """Test suite for FiniteBifieldElement class."""

    def test_initialization(self):
        """Test initialization of field elements."""
        field = FiniteBifield(4)  # GF(2^4)

        # Test valid initialization
        element = FiniteBifieldElement(field, 5)
        assert element.field == field
        assert element.value == 5

        # Test modulo operation during initialization
        element = FiniteBifieldElement(field, 20)  # 20 = 4 (mod 16)
        assert element.value == 4

    def test_addition(self):
        """Test addition of field elements."""
        field = FiniteBifield(3)  # GF(2^3)

        # In GF(2^m), addition is just XOR
        a = field(5)  # binary: 101
        b = field(3)  # binary: 011
        result = a + b
        assert result.value == 6  # binary: 110 (5 XOR 3 = 6)

        # Test adding with element from different field (should raise ValueError)
        other_field = FiniteBifield(4)
        with pytest.raises(ValueError):
            a + other_field(2)

        # Test commutativity
        assert (a + b).value == (b + a).value

        # Test adding non-field element
        with pytest.raises(TypeError):
            a + "not an element"

    def test_multiplication(self):
        """Test multiplication of field elements."""
        field = FiniteBifield(4)  # GF(2^4)

        # Test multiplication with identity
        a = field(7)
        one = field(1)
        assert (a * one).value == a.value

        # Test multiplication with zero
        zero = field(0)
        assert (a * zero).value == 0

        # Test with element from different field (should raise ValueError)
        other_field = FiniteBifield(3)
        with pytest.raises(ValueError):
            a * other_field(2)

        # Test actual multiplication
        a = field(3)  # binary: 0011
        b = field(5)  # binary: 0101
        # Multiplication in GF(2^4) involves polynomial multiplication mod the field's modulus
        result = a * b
        # Confirm this is correct by checking that it equals the result of explicit polynomial
        # multiplication mod the field's modulus
        a_poly = BinaryPolynomial(a.value)
        b_poly = BinaryPolynomial(b.value)
        expected = (a_poly * b_poly) % field.modulus
        assert result.value == expected.value

        # Test multiplying with non-field element
        with pytest.raises(TypeError):
            a * "not an element"

    def test_exponentiation(self):
        """Test raising a field element to a power."""
        field = FiniteBifield(3)  # GF(2^3)
        a = field(5)

        # Test a^0 = 1
        assert (a**0).value == 1

        # Test a^1 = a
        assert (a**1).value == a.value

        # Test a^2 = a*a
        assert (a**2).value == (a * a).value

        # Test a^3 = a*a*a
        assert (a**3).value == (a * a * a).value

        # Test negative exponent (should raise ValueError)
        with pytest.raises(ValueError):
            a ** (-1)

        # Test with zero element
        zero = field(0)
        assert (zero**5).value == 0

        # Test with identity element
        one = field(1)
        assert (one**10).value == 1

    def test_minimal_polynomial(self):
        """Test computation of minimal polynomial."""
        field = FiniteBifield(4)  # GF(2^4)

        for i in range(1, field.size):
            element = field(i)
            min_poly = element.minimal_polynomial()

            # The element should be a root of its minimal polynomial
            result = min_poly.evaluate(element)
            assert result.value == 0, f"Failed for element {element}, got {result}"

            # Test caching (second call should return the same object)
            min_poly2 = element.minimal_polynomial()
            assert min_poly is min_poly2

    def test_trace(self):
        """Test computation of trace."""
        field = FiniteBifield(3)  # GF(2^3)

        # Test trace of zero is zero
        zero = field(0)
        assert zero.trace() == 0

        # Test trace of one is field.m (mod 2)
        one = field(1)
        assert one.trace() == (field.m % 2)

        # Test trace property: tr(a+b) = tr(a) + tr(b)
        a = field(3)
        b = field(5)
        assert (a + b).trace() == (a.trace() ^ b.trace())

    def test_inverse(self):
        """Test computation of multiplicative inverse."""
        field = FiniteBifield(5)  # GF(2^5)

        # Test inverse of non-zero elements
        for i in range(1, field.size):
            a = field(i)
            inv = a.inverse()
            # a * a^(-1) should be 1
            assert (a * inv).value == 1

        # Test inverse of zero (should raise ValueError)
        zero = field(0)
        with pytest.raises(ValueError):
            zero.inverse()

        # Test inverse of one is one
        one = field(1)
        assert one.inverse().value == 1

    def test_to_polynomial(self):
        """Test conversion to binary polynomial."""
        field = FiniteBifield(4)  # GF(2^4)
        a = field(13)  # binary: 1101

        poly = a.to_polynomial()
        assert isinstance(poly, BinaryPolynomial)
        assert poly.value == 13

    def test_conjugates(self):
        """Test computation of conjugates."""
        field = FiniteBifield(4)  # GF(2^4)

        for i in range(1, field.size):
            element = field(i)
            conjugates = element.conjugates()

            # Check that all conjugates are distinct
            values = [e.value for e in conjugates]
            assert len(set(values)) == len(values)

            # Check that all conjugates have the same minimal polynomial
            min_poly = element.minimal_polynomial()
            for conjugate in conjugates:
                eval_result = min_poly.evaluate(conjugate)
                assert eval_result.value == 0, f"Failed for conjugate {conjugate}, got {eval_result}"

    def test_equality_and_hash(self):
        """Test equality comparison and hash computation."""
        field1 = FiniteBifield(3)
        field2 = FiniteBifield(3)
        field3 = FiniteBifield(4)

        a1 = field1(5)
        a2 = field2(5)  # Same value but from cached instance of same field
        a3 = field1(7)  # Different value in same field
        a4 = field3(5)  # Same value but different field

        # Test equality
        assert a1 == a2
        assert a1 != a3
        assert a1 != a4
        assert a1.__eq__("not an element") == NotImplemented

        # Test hash
        assert hash(a1) == hash(a2)
        assert hash(a1) != hash(a3)
        assert hash(a1) != hash(a4)

    def test_string_representation(self):
        """Test string representation of field elements."""
        field = FiniteBifield(3)  # GF(2^3)
        a = field(5)

        # Test str method
        assert str(a) == "5"

        # Test repr method
        assert repr(a) == f"FiniteBifieldElement({field}, 5)"
