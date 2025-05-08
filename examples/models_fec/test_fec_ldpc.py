import torch

from kaira.models.fec.encoders import LDPCCodeEncoder
from kaira.models.fec.decoders import BeliefPropagationDecoder
from kaira.channels.analog import AWGNChannel


parity_check_matrix = torch.tensor([
    [1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1]
], dtype=torch.float32)

# Initialize the encoder, decoder, and channel
encoder = LDPCCodeEncoder(parity_check_matrix)
decoder = BeliefPropagationDecoder(encoder, bp_iters=10)
channel = AWGNChannel(snr_db=5.0)

# Generate a random message
message = torch.randint(0, 2, (100, 3), dtype=torch.float32)

# Encode the message
codeword = encoder(message)

# Simulate transmission over AWGN channel
bipolar_codeword = 1 - 2.0 * codeword  # Convert to bipolar format
received_soft = channel(bipolar_codeword)

# Decode the received codeword
decoded_message = decoder(received_soft)
error = (message != decoded_message).to(torch.float32)

# Print the error rate
print("Bit error rate: ", torch.mean(error), ", Block error rate: ",
      torch.mean((torch.sum(error, dim=1) != 0).to(torch.float32)))


