"""Tests for the systematic_linear_block_code module in kaira.models.fec.encoders package."""

import numpy as np
import pytest
import torch

from kaira.models.fec.encoders.systematic_linear_block_code import (
    SystematicLinearBlockCodeEncoder,
    create_systematic_generator_matrix,
    get_information_and_parity_sets,
)
from kaira.models.registry import ModelRegistry


class TestSystematicLinearBlockCodeHelperFunctions:
    """Test suite for helper functions in systematic linear block code encoder module."""

    def test_get_information_and_parity_sets_left(self):
        """Test determining information and parity sets for 'left' configuration."""
        # Test 'left' configuration
        k, n = 3, 7
        info_indices, parity_indices = get_information_and_parity_sets(k, n, "left")

        assert torch.equal(info_indices, torch.tensor([0, 1, 2]))
        assert torch.equal(parity_indices, torch.tensor([3, 4, 5, 6]))

    def test_get_information_and_parity_sets_right(self):
        """Test determining information and parity sets for 'right' configuration."""
        # Test 'right' configuration
        k, n = 3, 7
        info_indices, parity_indices = get_information_and_parity_sets(k, n, "right")

        assert torch.equal(info_indices, torch.tensor([4, 5, 6]))
        assert torch.equal(parity_indices, torch.tensor([0, 1, 2, 3]))

    def test_get_information_and_parity_sets_custom(self):
        """Test determining information and parity sets for custom configuration."""
        # Test custom configuration with list
        k, n = 3, 7
        custom_info_set = [0, 2, 4]
        info_indices, parity_indices = get_information_and_parity_sets(k, n, custom_info_set)

        assert torch.equal(info_indices, torch.tensor([0, 2, 4]))
        assert torch.all(torch.sort(parity_indices)[0] == torch.tensor([1, 3, 5, 6]))

        # Test custom configuration with tensor
        custom_info_set = torch.tensor([1, 3, 5])
        info_indices, parity_indices = get_information_and_parity_sets(k, n, custom_info_set)

        assert torch.equal(info_indices, torch.tensor([1, 3, 5]))
        assert torch.all(torch.sort(parity_indices)[0] == torch.tensor([0, 2, 4, 6]))

    def test_get_information_and_parity_sets_invalid(self):
        """Test error cases for information and parity sets."""
        k, n = 3, 7

        # Test invalid string
        with pytest.raises(ValueError):
            get_information_and_parity_sets(k, n, "center")

        # Test invalid index size
        with pytest.raises(ValueError):
            get_information_and_parity_sets(k, n, [0, 1])  # too few indices

        # Test invalid index values (out of range)
        with pytest.raises(ValueError):
            get_information_and_parity_sets(k, n, [0, 1, 8])  # index 8 out of range

    def test_create_systematic_generator_matrix_left(self):
        """Test creating systematic generator matrix with 'left' information set."""
        # Define a simple parity submatrix for a (7,3) code
        parity_submatrix = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float)

        # Create generator matrix with 'left' information set (default)
        generator = create_systematic_generator_matrix(parity_submatrix)

        # Verify dimensions
        assert generator.shape == (3, 7)

        # Verify structure for 'left' configuration: G = [I_k | P]
        assert torch.equal(generator[:, :3], torch.eye(3))
        assert torch.equal(generator[:, 3:], parity_submatrix)

    def test_create_systematic_generator_matrix_right(self):
        """Test creating systematic generator matrix with 'right' information set."""
        # Define a simple parity submatrix for a (7,3) code
        parity_submatrix = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float)

        # Create generator matrix with 'right' information set
        generator = create_systematic_generator_matrix(parity_submatrix, "right")

        # Verify dimensions
        assert generator.shape == (3, 7)

        # Verify structure for 'right' configuration: G = [P | I_k]
        assert torch.equal(generator[:, 4:], torch.eye(3))
        assert torch.equal(generator[:, :4], parity_submatrix)

    def test_create_systematic_generator_matrix_custom(self):
        """Test creating systematic generator matrix with custom information set."""
        # Define a simple parity submatrix for a (7,3) code
        parity_submatrix = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float)

        # Create generator matrix with custom information set
        info_set = [0, 2, 5]
        generator = create_systematic_generator_matrix(parity_submatrix, info_set)

        # Verify dimensions
        assert generator.shape == (3, 7)

        # Verify identity matrix at specified positions
        for i, pos in enumerate(info_set):
            assert generator[i, pos] == 1
            assert sum(generator[:, pos]) == 1  # Only one 1 in the column

        # Verify parity submatrix at remaining positions
        parity_set = [1, 3, 4, 6]  # Complement of info_set
        for i, pos in enumerate(parity_set):
            assert torch.equal(generator[:, pos], parity_submatrix[:, i])


class TestSystematicLinearBlockCodeEncoder:
    """Test suite for SystematicLinearBlockCodeEncoder class."""

    def setup_method(self):
        """Set up the encoder for testing."""
        # Define a parity submatrix for a (7,3) code
        self.parity_submatrix = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float)

        # Store expected information and parity sets for test validation
        self.expected_info_set_left = torch.tensor([0, 1, 2])
        self.expected_parity_set_left = torch.tensor([3, 4, 5, 6])

        self.expected_info_set_right = torch.tensor([4, 5, 6])
        self.expected_parity_set_right = torch.tensor([0, 1, 2, 3])

        # Create an encoder instance with 'left' information set (default)
        self.encoder = SystematicLinearBlockCodeEncoder(self.parity_submatrix)

        # Create an encoder instance with 'right' information set
        self.encoder_right = SystematicLinearBlockCodeEncoder(self.parity_submatrix, "right")

        # Create an encoder instance with custom information set
        self.custom_info_set = [0, 2, 5]
        self.encoder_custom = SystematicLinearBlockCodeEncoder(self.parity_submatrix, self.custom_info_set)

    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Verify properties of the default 'left' encoder
        assert self.encoder.code_length == 7
        assert self.encoder.code_dimension == 3
        assert self.encoder.redundancy == 4
        assert self.encoder.code_rate == 3 / 7
        assert torch.equal(self.encoder.parity_submatrix, self.parity_submatrix)
        assert torch.equal(self.encoder.information_set, self.expected_info_set_left)

        # Access parity_set through the property
        assert torch.equal(self.encoder.parity_set, self.expected_parity_set_left)

        # Verify properties of the 'right' encoder
        assert torch.equal(self.encoder_right.information_set, self.expected_info_set_right)
        assert torch.equal(self.encoder_right.parity_set, self.expected_parity_set_right)

        # Verify properties of the custom encoder
        assert torch.equal(self.encoder_custom.information_set, torch.tensor(self.custom_info_set))

        # Test initialization with numpy array instead of torch tensor
        parity_np = np.array([[1, 1, 0], [0, 1, 1]])
        encoder_np = SystematicLinearBlockCodeEncoder(parity_np)
        assert encoder_np.code_length == 5
        assert encoder_np.code_dimension == 2
        assert encoder_np.redundancy == 3

        # Test with invalid information set (wrong size)
        with pytest.raises(ValueError):
            SystematicLinearBlockCodeEncoder(self.parity_submatrix, [0, 1])

    def test_generator_matrix_structure(self):
        """Test the structure of the generated systematic generator matrix."""
        # For 'left' configuration
        G_left = self.encoder.generator_matrix
        assert torch.equal(G_left[:, :3], torch.eye(3))  # Identity in first k columns
        assert torch.equal(G_left[:, 3:], self.parity_submatrix)  # Parity submatrix in last n-k columns

        # For 'right' configuration
        G_right = self.encoder_right.generator_matrix
        assert torch.equal(G_right[:, 4:], torch.eye(3))  # Identity in last k columns
        assert torch.equal(G_right[:, :4], self.parity_submatrix)  # Parity submatrix in first n-k columns

        # For custom configuration
        G_custom = self.encoder_custom.generator_matrix
        # Verify identity at specified positions
        for i, pos in enumerate(self.custom_info_set):
            assert G_custom[i, pos] == 1
            assert sum(G_custom[:, pos]) == 1

    def test_check_matrix_structure(self):
        """Test the structure of the parity check matrix."""
        # For 'left' configuration: H = [P^T | I_(n-k)]
        H_left = self.encoder.check_matrix
        assert H_left.shape == (4, 7)  # (n-k) × n
        assert torch.equal(H_left[:, :3], self.parity_submatrix.T)  # P^T in first k columns
        assert torch.equal(H_left[:, 3:], torch.eye(4))  # Identity in last n-k columns

        # Check orthogonality: G·H^T = 0
        product = torch.matmul(self.encoder.generator_matrix, H_left.T) % 2
        assert torch.all(product == 0)

        # For 'right' configuration: H = [I_(n-k) | P^T]
        H_right = self.encoder_right.check_matrix
        assert torch.equal(H_right[:, :4], torch.eye(4))  # Identity in first n-k columns
        assert torch.equal(H_right[:, 4:], self.parity_submatrix.T)  # P^T in last k columns

        # Check orthogonality again
        product_right = torch.matmul(self.encoder_right.generator_matrix, H_right.T) % 2
        assert torch.all(product_right == 0)

    def test_encoding(self):
        """Test encoding functionality."""
        # Single message for 'left' configuration
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = self.encoder(message)

        # For a systematic code with 'left' configuration:
        # - The first k bits should be identical to the message
        # - The last n-k bits should be the parity bits
        assert torch.equal(codeword[:3], message)
        expected_parity = torch.matmul(message, self.parity_submatrix) % 2
        assert torch.equal(codeword[3:], expected_parity)

        # Single message for 'right' configuration
        codeword_right = self.encoder_right(message)

        # For a systematic code with 'right' configuration:
        # - The last k bits should be identical to the message
        # - The first n-k bits should be the parity bits
        assert torch.equal(codeword_right[4:], message)
        expected_parity_right = torch.matmul(message, self.parity_submatrix) % 2
        assert torch.equal(codeword_right[:4], expected_parity_right)

        # Batch of messages
        batch = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        codewords = self.encoder(batch)

        # Verify systematic property for each message in batch
        for i, msg in enumerate(batch):
            assert torch.equal(codewords[i, :3], msg)
            expected_batch_parity = torch.matmul(msg, self.parity_submatrix) % 2
            assert torch.equal(codewords[i, 3:], expected_batch_parity)

        # Test with invalid message length
        with pytest.raises(ValueError):
            invalid_message = torch.tensor([1.0, 0.0, 1.0, 0.0])
            self.encoder(invalid_message)

    def test_project_word(self):
        """Test projecting a codeword onto the information set."""
        # Encode a message for 'left' configuration
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = self.encoder(message)

        # Project the codeword back to get the message
        projected = self.encoder.project_word(codeword)
        assert torch.equal(projected, message)

        # Encode and project for 'right' configuration
        codeword_right = self.encoder_right(message)
        projected_right = self.encoder_right.project_word(codeword_right)
        assert torch.equal(projected_right, message)

        # Test with batch of codewords
        batch = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        codewords = self.encoder(batch)
        projected_batch = self.encoder.project_word(codewords)
        assert torch.equal(projected_batch, batch)

        # Test with invalid codeword length
        with pytest.raises(ValueError):
            invalid_codeword = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            self.encoder.project_word(invalid_codeword)

    def test_syndrome_calculation(self):
        """Test syndrome calculation for valid and invalid codewords."""
        # Generate a valid codeword for 'left' configuration
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = self.encoder(message)

        # Syndrome should be zero for a valid codeword
        syndrome = self.encoder.calculate_syndrome(codeword)
        assert torch.all(syndrome == 0)

        # Create an invalid codeword by flipping a bit
        invalid_codeword = codeword.clone()
        invalid_codeword[0] = 1 - invalid_codeword[0]  # Flip the first bit

        # Syndrome should be non-zero for invalid codeword
        syndrome_invalid = self.encoder.calculate_syndrome(invalid_codeword)
        assert not torch.all(syndrome_invalid == 0)

        # For systematic codes, the syndrome should match the first column of H if first bit is flipped
        expected_syndrome = self.encoder.check_matrix[:, 0]
        assert torch.equal(syndrome_invalid, expected_syndrome)

        # Test for 'right' configuration
        codeword_right = self.encoder_right(message)
        syndrome_right = self.encoder_right.calculate_syndrome(codeword_right)
        assert torch.all(syndrome_right == 0)

        # Test with invalid syndrome length
        with pytest.raises(ValueError):
            invalid_codeword_length = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # Length 6, not 7
            self.encoder.calculate_syndrome(invalid_codeword_length)

    def test_inverse_encode(self):
        """Test decoding functionality."""
        # Encode a message
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = self.encoder(message)

        # Decode the codeword
        decoded, syndrome = self.encoder.inverse_encode(codeword)

        # Check that we get back the original message and zero syndrome
        assert torch.equal(decoded, message)
        assert torch.all(syndrome == 0)

        # Test with an error in the codeword
        codeword_with_error = codeword.clone()
        codeword_with_error[3] = 1 - codeword_with_error[3]  # Flip a parity bit

        decoded_with_error, syndrome_with_error = self.encoder.inverse_encode(codeword_with_error)

        # For an error in a parity bit, the message should still be correct
        assert torch.equal(decoded_with_error, message)
        # But the syndrome should detect the error
        assert not torch.all(syndrome_with_error == 0)

        # Now test with an error in an information bit
        codeword_with_info_error = codeword.clone()
        codeword_with_info_error[0] = 1 - codeword_with_info_error[0]  # Flip an info bit

        decoded_with_info_error, syndrome_with_info_error = self.encoder.inverse_encode(codeword_with_info_error)

        # For an error in an information bit, the decoded message will be wrong
        assert not torch.equal(decoded_with_info_error, message)
        # And the syndrome should detect the error
        assert not torch.all(syndrome_with_info_error == 0)

        # Test with batch decoding
        batch = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        codewords = self.encoder(batch)
        decoded_batch, syndromes_batch = self.encoder.inverse_encode(codewords)

        assert torch.equal(decoded_batch, batch)
        assert torch.all(syndromes_batch == 0)

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("systematic_linear_block_code_encoder")

        # Verify it's the correct class
        assert model_class is SystematicLinearBlockCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("systematic_linear_block_code_encoder", parity_submatrix=self.parity_submatrix)

        assert isinstance(model, SystematicLinearBlockCodeEncoder)

    def test_representation(self):
        """Test string representation."""
        # Simple test to make sure __repr__ works without errors
        repr_str = str(self.encoder)

        # Check that it contains key information
        assert "SystematicLinearBlockCodeEncoder" in repr_str
        assert "parity_submatrix=tensor" in repr_str
        assert "information_set=tensor" in repr_str
        assert f"dimension={self.encoder.code_dimension}" in repr_str
        assert f"length={self.encoder.code_length}" in repr_str
        assert f"redundancy={self.encoder.redundancy}" in repr_str
