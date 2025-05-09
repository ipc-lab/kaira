"""Tests for the Belief Propagation decoder in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.channels.analog import AWGNChannel
from kaira.models.fec.decoders.belief_propagation import BeliefPropagationDecoder
from kaira.models.fec.encoders.ldpc_code import LDPCCodeEncoder
from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder


class TestBeliefPropagationDecoder:
    """Test suite for BeliefPropagationDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)

        # Initialize the decoder with this encoder
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=10)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify properties are correctly set
        assert decoder.bp_iters == 10
        assert decoder.n == 6
        assert decoder.n_v == 6
        assert decoder.n_c == 3
        assert torch.all(decoder.H == H)
        # The decoder classifies LDPCCodeEncoder as not_ldpc=True in current implementation
        assert decoder.not_ldpc

        # Test with a linear block code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        linear_encoder = LinearBlockCodeEncoder(generator_matrix=G)

        linear_decoder = BeliefPropagationDecoder(encoder=linear_encoder, bp_iters=5)
        assert linear_decoder.encoder is linear_encoder
        assert linear_decoder.bp_iters == 5
        assert linear_decoder.not_ldpc

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Create a mock encoder that isn't a LinearBlockCodeEncoder or LDPCCodeEncoder
        from unittest.mock import Mock

        # Create a mock object that pretends to be an encoder but isn't of the expected types
        mock_encoder = Mock()

        # The decoder should check encoder type and raise a TypeError
        with pytest.raises(TypeError, match="Encoder must be a LinearBlockCodeEncoder or LDPCCodeEncoder"):
            BeliefPropagationDecoder(encoder=mock_encoder)

    def test_calc_code_metrics(self):
        """Test calculation of code metrics."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=10)

        # Verify code metrics
        assert decoder.num_edges == 9  # Total number of 1s in H
        assert torch.all(decoder.var_degree == torch.tensor([1, 1, 2, 2, 2, 1]))  # Sum of columns
        assert torch.all(decoder.check_degree == torch.tensor([3, 3, 3]))  # Sum of rows

    def test_prep_edge_ind(self):
        """Test preparation of edge indices."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=10)

        # Verify edge indices are prepared
        assert hasattr(decoder, "lv_ind")
        assert hasattr(decoder, "edge_map")
        assert hasattr(decoder, "cv_map")
        assert hasattr(decoder, "marg_ec")
        assert hasattr(decoder, "ext_ec")
        assert hasattr(decoder, "ext_ce")
        assert hasattr(decoder, "cv_order")
        assert hasattr(decoder, "vc_group")
        assert hasattr(decoder, "cv_group")

        # Verify lv_ind has the correct length
        assert len(decoder.lv_ind) == decoder.num_edges

        # Verify edge_map has one entry per variable node
        assert len(decoder.edge_map) == decoder.n_v

        # Verify cv_map has one entry per check node
        assert len(decoder.cv_map) == decoder.n_c

    def test_decoding_no_errors(self):
        """Test decoding a codeword with no errors."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=20)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = encoder(message)

        # Convert to bipolar form and back (no errors)
        bipolar_codeword = 1 - 2.0 * codeword
        received_soft = bipolar_codeword.unsqueeze(0)  # Add batch dimension for BP

        # Decode the received word
        decoded = decoder(received_soft)

        # Verify that the decoded message matches the original
        assert torch.all(decoded.squeeze() == message)

    def test_decoding_with_noise(self):
        """Test decoding a codeword with AWGN channel noise."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=50)
        channel = AWGNChannel(snr_db=10.0)  # High SNR to ensure correct decoding

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = encoder(message)

        # Convert to bipolar form and simulate AWGN channel
        bipolar_codeword = 1 - 2.0 * codeword
        received_noisy = channel(bipolar_codeword)
        received_soft = received_noisy.unsqueeze(0)  # Add batch dimension for BP

        # Decode the received word
        decoded = decoder(received_soft)

        # With high SNR, decoded message should match the original
        assert torch.all(decoded.squeeze() == message)

    def test_decoding_with_batch_dimension(self):
        """Test decoding with batch dimension."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=20)
        channel = AWGNChannel(snr_db=10.0)  # High SNR to ensure correct decoding

        # Create messages and encode them
        messages = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        codewords = encoder(messages)

        # Convert to bipolar form and simulate AWGN channel
        bipolar_codewords = 1 - 2.0 * codewords
        received_noisy = channel(bipolar_codewords)
        received_soft = received_noisy  # Already has batch dimension

        # Decode the received words
        decoded = decoder(received_soft)

        # With high SNR, decoded messages should match the originals
        assert torch.all(decoded == messages)

    def test_return_soft_output(self):
        """Test returning soft outputs along with decoded messages."""
        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=20)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        codeword = encoder(message)

        # Convert to bipolar form (no errors)
        bipolar_codeword = 1 - 2.0 * codeword
        received_soft = bipolar_codeword.unsqueeze(0)  # Add batch dimension for BP

        # Decode with return_soft=True
        decoded, soft_output = decoder(received_soft, return_soft=True)

        # Verify that the decoded message matches the original
        assert torch.all(decoded.squeeze() == message)

        # Verify soft output has the correct shape
        assert len(soft_output.shape) == 2  # Should have batch and code length dimensions
        assert soft_output.size(-1) == bipolar_codeword.size(-1)

        # Verify hard decisions from soft outputs match codeword
        hard_decisions = (soft_output.squeeze() < 0).float()  # Convert LLRs to binary (0/1)
        assert torch.all(hard_decisions == codeword)

    def test_device_transfer(self):
        """Test decoder works correctly after transferring to a different device."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create a simple parity check matrix for an LDPC code
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder=encoder, bp_iters=20, device="cpu")

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0], device="cpu")
        codeword = encoder(message)

        # Convert to bipolar form (no errors)
        bipolar_codeword = 1 - 2.0 * codeword
        received_soft = bipolar_codeword.unsqueeze(0)  # Add batch dimension for BP

        # Move to CUDA
        cuda_received = received_soft.cuda()

        # Decode on CUDA
        decoded = decoder(cuda_received)

        # Verify that the decoded message matches the original
        assert decoded.device.type == "cuda"
        assert torch.all(decoded.cpu().squeeze() == message)

        # Move back to CPU
        cpu_received = cuda_received.cpu()
        decoded = decoder(cpu_received)

        # Verify that the decoded message matches the original
        assert decoded.device.type == "cpu"
        assert torch.all(decoded.squeeze() == message)
