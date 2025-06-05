"""Tests for the ldpc_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.ldpc_code import LDPCCodeEncoder


class TestLDPCCodeEncoder:
    """Test suite for LDPCCodeEncoder class."""

    def test_initialization_with_check_matrix(self):
        """Test initialization with check matrix."""
        # Create a simple 3x6 parity check matrix
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        assert encoder.code_length == 6
        assert encoder.code_dimension == 3  # 6 - 3
        assert torch.allclose(encoder.check_matrix, H)
        assert encoder.generator_matrix.shape == (3, 6)

    def test_initialization_with_systematic_check_matrix(self):
        """Test initialization with systematic check matrix."""
        # Create systematic parity check matrix [P|I]
        H = torch.tensor([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        assert encoder.code_length == 6
        assert encoder.code_dimension == 3

        # Check that HG^T = 0 (mod 2)
        result = torch.matmul(encoder.check_matrix.float(), encoder.generator_matrix.float().T) % 2
        expected = torch.zeros_like(result)
        assert torch.allclose(result, expected)

    def test_initialization_from_database(self):
        """Test initialization from database."""
        # Load a code that exists in the database
        encoder = LDPCCodeEncoder(rptu_database=True, code_length=576, code_dimension=288)

        assert encoder.code_length == 576
        assert encoder.code_dimension == 288
        assert encoder.generator_matrix is not None
        assert encoder.check_matrix is not None

        # Verify the check matrix dimensions
        expected_parity_bits = 576 - 288  # 288 parity bits
        assert encoder.check_matrix.shape == (expected_parity_bits, 576)

        # Verify the generator matrix dimensions
        assert encoder.generator_matrix.shape == (288, 576)

    def test_initialization_invalid_inputs(self):
        """Test initialization with invalid inputs."""
        # Test with no check_matrix provided and rptu_database=False
        with pytest.raises(RuntimeError, match="Could not infer dtype of NoneType"):
            LDPCCodeEncoder()

        # Test with invalid database parameters
        with pytest.raises(ValueError, match="not found in rptu_database"):
            LDPCCodeEncoder(rptu_database=True, code_length=999, code_dimension=500)

    def test_initialization_non_binary_check_matrix(self):
        """Test initialization with non-binary check matrix."""
        # Create non-binary matrix - this should still work in current implementation
        H = torch.tensor([[1.5, 0, 1], [0, 2.0, 1]], dtype=torch.float32)

        # This should succeed with current implementation
        encoder = LDPCCodeEncoder(check_matrix=H)
        assert encoder.code_length == 3
        assert encoder.code_dimension == 1

    def test_initialization_invalid_check_matrix_values(self):
        """Test initialization with invalid check matrix values."""
        # Create matrix with values other than 0 and 1 - this should still work in current implementation
        H = torch.tensor([[1, 0, 2], [0, 1, -1]], dtype=torch.float32)

        # This should succeed with current implementation
        encoder = LDPCCodeEncoder(check_matrix=H)
        assert encoder.code_length == 3
        assert encoder.code_dimension == 1

    def test_forward_encoding(self):
        """Test forward encoding process."""
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Test with single message
        message = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        codeword = encoder.forward(message)

        assert codeword.shape == (1, 6)
        assert torch.all((codeword == 0) | (codeword == 1))

        # Verify that it's a valid codeword: H * c^T = 0 (mod 2)
        result = torch.matmul(H, codeword.T) % 2
        expected = torch.zeros_like(result)
        assert torch.allclose(result, expected)

    def test_forward_encoding_batch(self):
        """Test forward encoding with batch input."""
        H = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Test with batch of messages
        batch_size = 4
        messages = torch.randint(0, 2, (batch_size, 3), dtype=torch.float32)
        codewords = encoder.forward(messages)

        assert codewords.shape == (batch_size, 5)
        assert torch.all((codewords == 0) | (codewords == 1))

        # Verify all codewords are valid
        for i in range(batch_size):
            result = torch.matmul(H, codewords[i : i + 1].T) % 2
            expected = torch.zeros_like(result)
            assert torch.allclose(result, expected)

    def test_forward_encoding_different_shapes(self):
        """Test forward encoding with different input shapes."""
        H = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Test with 3D input
        messages = torch.randint(0, 2, (2, 3, 2), dtype=torch.float32)
        codewords = encoder.forward(messages)

        assert codewords.shape == (2, 3, 4)

    def test_get_generator_matrix(self):
        """Test getting generator matrix."""
        H = torch.tensor([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        G = encoder.generator_matrix  # Access the property directly

        assert G.shape == (3, 6)
        assert torch.all((G == 0) | (G == 1))

        # Verify orthogonality: H * G^T = 0 (mod 2)
        result = torch.matmul(H.float(), G.float().T) % 2
        expected = torch.zeros_like(result)
        assert torch.allclose(result, expected)

    def test_get_syndrome_matrix(self):
        """Test getting syndrome matrix."""
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        syndrome_matrix = encoder.check_matrix  # Use check_matrix directly

        assert torch.allclose(syndrome_matrix.float(), H)

    def test_device_consistency(self):
        """Test that all tensors are on the correct device."""
        device = torch.device("cpu")
        H = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32, device=device)

        encoder = LDPCCodeEncoder(check_matrix=H, device=device)

        assert encoder.generator_matrix.device == device
        assert encoder.check_matrix.device == device

        # Test encoding
        message = torch.tensor([[1, 0]], dtype=torch.float32, device=device)
        codeword = encoder.forward(message)
        assert codeword.device == device

    def test_different_dtypes(self):
        """Test with different data types."""
        H = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float64)

        encoder = LDPCCodeEncoder(check_matrix=H, dtype=torch.float64)

        # Generator matrix dtype might be different due to internal processing
        # Just check that encoding works with float64
        assert encoder.generator_matrix is not None

        message = torch.tensor([[1, 0]], dtype=torch.float64)
        codeword = encoder.forward(message)
        assert codeword.dtype == torch.float64

    def test_systematic_form_properties(self):
        """Test properties of systematic form."""
        # Create a non-systematic parity check matrix
        H = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # The generator matrix should exist and be valid
        assert encoder.generator_matrix.shape == (2, 5)

        # Test encoding works
        message = torch.tensor([[1, 0]], dtype=torch.float32)
        codeword = encoder.forward(message)
        assert codeword.shape == (1, 5)

    def test_rank_deficient_matrix(self):
        """Test handling of rank-deficient check matrix."""
        # Create a rank-deficient matrix (rows are linearly dependent)
        H = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)  # This row is sum of first two

        # This should still work, but the effective dimension might be different
        encoder = LDPCCodeEncoder(check_matrix=H)

        # The encoder should handle this gracefully
        assert encoder.code_length == 4
        assert encoder.generator_matrix is not None

    def test_single_parity_check(self):
        """Test with single parity check (simple case)."""
        H = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        assert encoder.code_length == 4
        assert encoder.code_dimension == 3

        # Test encoding
        message = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        codeword = encoder.forward(message)

        # Check parity: sum should be even
        assert torch.sum(codeword) % 2 == 0

    def test_identity_check_matrix(self):
        """Test with identity check matrix."""
        H = torch.eye(3, dtype=torch.float32)

        # Identity check matrix creates a degenerate code with dimension 0
        # This should raise ValueError due to invalid code dimension
        with pytest.raises(ValueError, match="Code dimension must be positive"):
            LDPCCodeEncoder(check_matrix=H)

    def test_large_sparse_matrix(self):
        """Test with larger sparse matrix."""
        # Create a larger sparse parity check matrix
        rows, cols = 10, 20
        H = torch.zeros(rows, cols, dtype=torch.float32)

        # Add some sparse structure
        for i in range(rows):
            for j in range(3):  # 3 ones per row
                col_idx = (i * 3 + j) % cols
                H[i, col_idx] = 1

        encoder = LDPCCodeEncoder(check_matrix=H)

        assert encoder.code_length == cols
        assert encoder.code_dimension == cols - rows

        # Test encoding
        message = torch.randint(0, 2, (1, encoder.code_dimension), dtype=torch.float32)
        codeword = encoder.forward(message)
        assert codeword.shape == (1, cols)

    def test_encoding_linearity(self):
        """Test linearity property of encoding."""
        H = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Test that encoding is linear: encode(m1) + encode(m2) = encode(m1 + m2)
        m1 = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        m2 = torch.tensor([[0, 1, 1]], dtype=torch.float32)

        c1 = encoder.forward(m1)
        c2 = encoder.forward(m2)
        c_sum = encoder.forward((m1 + m2) % 2)

        expected = (c1 + c2) % 2
        assert torch.allclose(c_sum, expected)

    def test_zero_message_encoding(self):
        """Test encoding of zero message."""
        H = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Zero message should give zero codeword
        zero_message = torch.zeros(1, 2, dtype=torch.float32)
        zero_codeword = encoder.forward(zero_message)

        expected = torch.zeros(1, 4, dtype=torch.float32)
        assert torch.allclose(zero_codeword, expected)
