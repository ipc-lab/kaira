"""Belief Propagation Decoder for Linear Block Codes and LDPC Codes.

This module implements a belief propagation (BP) decoder over a Tanner graph for linear block codes and LDPC codes
in particular. The belief propagation algorithm is an iterative message-passing algorithm used for decoding
error-correcting codes. It operates on the Tanner graph representation of the code and exchanges
messages between variable nodes and check nodes to iteratively refine the decoding.

The decoder is capable of handling both hard-decision and soft-decision decoding, making it
suitable for a wide range of applications, including LDPC codes and other linear block codes.

Key Features:
1. Iterative message-passing algorithm for decoding.
2. Supports both hard-decision and soft-decision decoding.
3. Efficient handling of sparse parity-check matrices.
4. Configurable number of iterations for decoding.

Attributes:
    encoder (Union[LinearBlockCodeEncoder, LDPCCodeEncoder]): The encoder instance providing code
        parameters and syndrome calculation methods.
    bp_iters (int): The number of belief propagation iterations to perform.
    G (torch.Tensor): The generator matrix of the code of size (k, n).
    H (torch.Tensor): The parity check matrix of the code of size (n - k, n).
    k (int): The dimension of the code (number of information bits).
    n (int): The length of the code (number of code bits).
    not_ldpc (bool): Boolean flag to indicate if the code is not LDPC.
    standard (bool): Boolean flag to indicate if the code is systematic.

Args:
    encoder (Union[LinearBlockCodeEncoder, LDPCCodeEncoder]): The encoder instance for the code being decoded.
    bp_iters (int): Number of belief propagation iterations to perform.
    arctanh (bool): Boolean flag to determine whether to use the arctanh function for message updates or
    its approximation
    return_soft (bool): Boolean flag to determine whether to return the soft output
    device (str): The device to use for computation (e.g., "cpu" or "cuda"). Defaults to "cpu".
    *args: Additional positional arguments passed to the base class.
    **kwargs: Additional keyword arguments passed to the base class.

Raises:
    TypeError: If the encoder is not an instance of `LinearBlockCodeEncoder` or `LDPCCodeEncoder`.

Examples:
    >>> from kaira.models.fec.encoders import LDPCCodeEncoder
    >>> from kaira.models.fec.decoders import BeliefPropagationDecoder
    >>> from kaira.channels.analog import AWGNChannel
    >>> import torch
    >>>
    >>> parity_check_matrix = torch.tensor([
    ...     [1, 0, 1, 1, 0, 0],
    ...     [0, 1, 1, 0, 1, 0],
    ...     [0, 0, 0, 1, 1, 1]
    ... ], dtype=torch.float32)
    >>> # Create an encoder for an LDPC code
    >>> encoder = LDPCCodeEncoder(parity_check_matrix)
    >>> decoder = BeliefPropagationDecoder(encoder, bp_iters=10)
    >>> channel = AWGNChannel(snr_db=5.0)
    >>>
    >>> # Encode a message
    >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0.])
    >>> codeword = encoder(message)
    >>>
    >>> # Simulate transmission over AWGN channel
    >>> bipolar_codeword = 1 - 2.0 * codeword
    >>> received_soft = channel(bipolar_codeword)
    >>>
    >>> # Decode and check if recovered correctly
    >>> decoded = decoder(received_soft)
    >>> print(torch.all(decoded == message))
    True
"""

from typing import Any, List, Tuple, Union

import torch

from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder
from kaira.models.fec.encoders.ldpc_code import LDPCCodeEncoder

from ..utils import apply_blockwise, Tailor_arctanh, sign_to_bin
from .base import BaseBlockDecoder
from itertools import combinations
from operator import itemgetter

class BeliefPropagationDecoder(BaseBlockDecoder[Union[LinearBlockCodeEncoder, LDPCCodeEncoder]]):
    """Belief propagation decoder for linear block codes and LDPC codes.

    This algorithm is an iterative message-passing technique used for decoding error-correcting codes
    by operating on the Tanner graph representation of the code. It exchanges messages between
    variable nodes and check nodes to iteratively refine the decoding process.
    It is particularly efficient for LDPC codes due to the sparsity of their parity-check matrices.

    The algorithm finds the shortest linear feedback shift register (LFSR) that generates the
    syndrome sequence, which corresponds to the error locator polynomial. The roots of this
    polynomial identify the positions of errors in the received word.

    The decoder works by:
    1. Initializing the decoder with the edge indices and mappings for the Tanner graph.
    2. Iteratively updating the messages passed between variable nodes and check nodes.
    3. Combining messages to compute soft outputs.
    4. Extracting the information bits from the decoded codeword.

    Attributes:
        encoder (Union[LinearBlockCodeEncoder, LDPCCodeEncoder]): The encoder instance
                providing code parameters.
        bp_iters (int): Number of belief propagation iterations to perform.
        G (torch.Tensor): The generator matrix of the code of size (k, n).
        H (torch.Tensor): The parity check matrix of the code of size (n - k, n).
        k (int): The dimension of the code (number of information bits).
        n (int): The length of the code (number of code bits).
        not_ldpc (bool): Boolean flag to indicate if the code is not LDPC.
        standard (bool): Boolean flag to indicate if the code is systematic.

    Args:
        encoder (Union[LinearBlockCodeEncoder, LDPCCodeEncoder]): The encoder for the code being decoded
        bp_iters (int): Number of belief propagation iterations to perform.
        arctanh (bool): Boolean flag to determine whether to use the arctanh function for message updates or
        its approximation
        return_soft (bool): Boolean flag to determine whether to return the soft output
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Raises:
        TypeError: If the encoder is not a LinearBlockCodeEncoder or LDPCCodeEncoder

    Examples:
        >>> from kaira.models.fec.encoders import LDPCCodeEncoder
        >>> from kaira.models.fec.decoders import BeliefPropagationDecoder
        >>> from kaira.channels.analog import AWGNChannel
        >>> import torch
        >>>
        >>> parity_check_matrix = torch.tensor([
        ...     [1, 0, 1, 1, 0, 0],
        ...     [0, 1, 1, 0, 1, 0],
        ...     [0, 0, 0, 1, 1, 1]
        ... ], dtype=torch.float32)
        >>> # Create an encoder for an LDPC code
        >>> encoder = LDPCCodeEncoder(parity_check_matrix)
        >>> decoder = BeliefPropagationDecoder(encoder, bp_iters=10)
        >>> channel = AWGNChannel(snr_db=5.0)
        >>>
        >>> # Encode a message
        >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0.])
        >>> codeword = encoder(message)
        >>>
        >>> # Simulate transmission over AWGN channel
        >>> bipolar_codeword = 1 - 2.0 * codeword
        >>> received_soft = channel(bipolar_codeword)
        >>>
        >>> # Decode and check if recovered correctly
        >>> decoded = decoder(received_soft)
        >>> print(torch.all(decoded == message))
        True
    """

    def __init__(self, encoder: Union[LinearBlockCodeEncoder, LDPCCodeEncoder], bp_iters: int = 10,
                 arctanh: bool = True, return_soft: bool = False, device: str = "cpu",
                 *args: Any, **kwargs: Any):
        """Initialize the Belief Propagation decoder.

        Sets up the decoder with an encoder instance and extracts relevant parameters
        needed for the decoding process.

        Args:
            encoder: The encoder instance for the code being decoded
            bp_iters: Number of belief propagation iterations to perform.
            arctanh: Boolean flag to determine whether to use the arctanh function or its approximation
            return_soft: Boolean flag to determine whether to return the soft output
            *args: Variable positional arguments passed to the base class
            **kwargs: Variable keyword arguments passed to the base class

        Raises:
            TypeError: If the encoder is not a LinearBlockCodeEncoder or LDPCCodeEncoder
        """
        super().__init__(encoder, *args, **kwargs)

        if not isinstance(encoder, (LinearBlockCodeEncoder, LDPCCodeEncoder)):
            raise TypeError(f"Encoder must be a LinearBlockCodeEncoder or LDPCCodeEncoder, got {type(encoder).__name__}")
        self.not_ldpc = False
        if isinstance(encoder, LinearBlockCodeEncoder):
            self.not_ldpc = True
        self.bp_iters = bp_iters
        self.arctanh = arctanh

        self.device = device
        self.G = encoder.generator_matrix.to(torch.float32)
        self.H = encoder.check_matrix.to(torch.int64)
        self.k = encoder._dimension
        self.n = encoder._length
        self.standard = False
        self.return_soft = return_soft

        self.calc_code_metrics()



    def calc_code_metrics(self):
        self.num_edges = torch.sum(self.H)
        self.var_degree = torch.sum(self.H, dim=0)
        self.check_degree = torch.sum(self.H, dim=1)
        self.n_v = self.H.size(1)
        self.n_c = self.H.size(0)
        self.prep_edge_ind()
        if not self.standard:
            self.idx_mess_t = torch.where(self.G.sum(0) == 1)[0]

    def prep_edge_ind(self):
        self.lv_ind = []
        self.edge_map = []
        self.cv_map = [[] for _ in range(self.n_c)]
        self.marg_ec = []
        self.ext_ec = []
        self.ext_ce = []
        self.cv_order = []
        self.vc_group = []
        self.cv_group = []
        ind = 0
        prev_vc_deg = -1
        for v_node in range(self.n_v):
            if self.var_degree[v_node].item() == prev_vc_deg:
                self.vc_group[-1].append(v_node)
            else:
                prev_vc_deg = self.var_degree[v_node].item()
                self.vc_group.append([v_node])

            self.edge_map.append(torch.arange(ind, ind + self.var_degree[v_node].item()))
            self.lv_ind.extend([v_node] * self.var_degree[v_node].item())
            ind += self.var_degree[v_node].item()

            c_nonzeros = torch.nonzero(self.H[:, v_node]).view(-1)
            for c_node, edge_ind in zip(c_nonzeros, self.edge_map[v_node]):
                self.cv_map[c_node].append(edge_ind.item())

            self.marg_ec.append(self.edge_map[v_node].to(self.device))

            if self.var_degree[v_node] > 1:
                node_ind = self.edge_map[v_node]
                node_ind = combinations(node_ind, r=self.var_degree[v_node].item() - 1)
                ext_ec = torch.tensor(list(node_ind))
                ext_ec = torch.flip(ext_ec, dims=(0,))
                self.ext_ec.append(ext_ec.to(self.device))
            else:
                self.ext_ec.append(torch.tensor([]).to(self.device))

        edge_order = []
        prev_cv_deg = -1
        for c_node in range(self.n_c):
            if self.check_degree[c_node].item() == prev_cv_deg:
                self.cv_group[-1].append(c_node)
            else:
                prev_cv_deg = self.check_degree[c_node].item()
                self.cv_group.append([c_node])

            edge_order.extend(self.cv_map[c_node])
            if self.check_degree[c_node] > 1:
                node_ind = self.cv_map[c_node]
                node_ind = combinations(node_ind, r=self.check_degree[c_node].item() - 1)
                ext_ce = torch.tensor(list(node_ind))
                ext_ce = torch.flip(ext_ce, dims=(0,))
                self.ext_ce.append(ext_ce.to(self.device))
            else:
                self.ext_ce.append(torch.tensor([]).to(self.device))

        self.lv_ind = torch.tensor(self.lv_ind).to(self.device)
        self.cv_order = torch.zeros(self.num_edges, dtype=torch.int64, device=self.device)
        self.cv_order[edge_order] = torch.arange(0, self.num_edges, device=self.device).to(torch.int64)

    def compute_vc(self, cv: torch.Tensor, soft_input: torch.Tensor) -> torch.Tensor:
        batch_size, _ = cv.size()
        lv_ind = self.lv_ind.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        reordered_soft_input = soft_input.gather(1, lv_ind)
        vc = reordered_soft_input - cv
        return vc

    def compute_cv(self, vc: torch.Tensor):
        batch_size, _ = vc.size()
        vc = vc.clamp(-500, 500)
        tanh_vc = torch.tanh(vc / 2.)
        cv = []
        for c_group in self.cv_group:
            deg = self.check_degree[c_group[0]].item()
            members = len(c_group)
            if deg > 1:
                ext_ce = list(itemgetter(*c_group)(self.ext_ce))
                if members == 1 and (self.not_ldpc):
                    len_ten = len(ext_ce)
                    ext_ce = torch.cat(ext_ce, dim=0).view(len_ten, -1)
                else:
                    ext_ce = torch.cat(ext_ce, dim=0)
                ext_ce = ext_ce.unsqueeze(0).repeat_interleave(batch_size, dim=0)

                vc_extended = tanh_vc.unsqueeze(1).repeat_interleave(deg*members, dim=1)
                vc_extended = vc_extended.gather(2, ext_ce)
                vc_extended_log2 = torch.log2(vc_extended.to(dtype=torch.complex64) + 1e-10)
                v_messages = torch.sum(vc_extended_log2, dim=2)
                v_messages_msg = torch.pow(2, v_messages).real

                if self.arctanh:
                    v_messages = v_messages_msg.clamp(-0.999, 0.999)
                    v_messages = 2 * torch.arctanh(v_messages)
                else:
                    v_messages = v_messages_msg.clamp(-1.001, 1.001)
                    v_messages = 2 * Tailor_arctanh(v_messages)
                v_messages = v_messages.clamp(-500, 500)
                v_messages = v_messages + 1
                v_messages = v_messages - 1

            else:
                v_messages = torch.zeros(B, members).to(self.device)
            cv.append(v_messages)

        cv = torch.cat(cv, dim=-1)
        new_order = self.cv_order.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        cv = cv.gather(1, new_order)
        return cv

    def marginalize(self, cv: torch.Tensor, soft_input: torch.Tensor) -> torch.Tensor:
        batch_size, _ = cv.size()

        soft_output = []
        for v_group in self.vc_group:
            members = len(v_group)
            edges = list(itemgetter(*v_group)(self.marg_ec))
            if members == 1:
                edges = torch.stack(edges, dim=0).view(1, -1)
            else:
                edges = torch.stack(edges, dim=0)
            edges = edges.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            cv_extended = cv.unsqueeze(1).repeat_interleave(members, dim=1)
            msg = cv_extended.gather(2, edges)
            msg = msg.sum(2)
            soft_output.append(msg)

        soft_output = torch.cat(soft_output, dim=-1)
        soft_output += soft_input
        return soft_output


    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using the Belief Propagation algorithm.

        This method implements the sum-product decoding algorithm for linear block codes:
        1. Input Validation: Ensures the input tensor's dimensions are valid.
        2. Initialization: Sets up messages and internal structures.
        3. Iterative Decoding: Updates variable-to-check and check-to-variable messages for a fixed number of iterations.
        4. Marginalization: Combines messages to compute soft outputs.
        5. Message Extraction: Extracts decoded messages and optionally returns soft outputs.

        Args:
            received: Received codeword tensor with shape (..., n) or (..., m*n)
                     where n is the code length and m is some multiple
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
                return_soft: If True, also return the estimated codeword

        Returns:
            Either:
            - Decoded tensor containing estimated messages with shape (..., k)
            - A tuple of (decoded tensor, soft codeword estimate tensor) if return_soft=True

        Raises:
            ValueError: If the last dimension of received is not a multiple of the code length

        """
        self.return_soft = kwargs.get("return_soft", False)
        if self.device != received.device:
            self.device = received.device
            self.lv_ind = self.lv_ind.to(self.device)
            self.edge_map = [edge_map.to(self.device) for edge_map in self.edge_map]
            self.marg_ec = [marg_ec.to(self.device) for marg_ec in self.marg_ec]
            self.ext_ec = [ext_ec.to(self.device) for ext_ec in self.ext_ec]
            self.ext_ce = [ext_ce.to(self.device) for ext_ce in self.ext_ce]
            self.cv_order = self.cv_order.to(self.device)
            if not self.standard:
                self.idx_mess_t = self.idx_mess_t.to(self.device)

        def decode_block(received_block: torch.Tensor) -> torch.Tensor:
            """Decode a single block of received codewords."""
            # Decode the block using the decoder's logic
            B, _, L = received_block.size()
            device = received_block.device
            messages = received_block.view(-1, L)
            cv = torch.zeros(messages.size(0), self.num_edges, device=device)
            for _ in range(self.bp_iters):
                vc = self.compute_vc(cv, messages)  # *= self.layers1[i % self.w_n]
                cv = self.compute_cv(vc)
                messages = self.marginalize(cv, received_block.view(-1, L))
            decoded_block = messages.view(B, L)
            idx_mess = self.idx_mess_t.unsqueeze(0).unsqueeze(0).repeat_interleave(B, dim=0).to(self.device)
            message_llr = decoded_block.view(B, 1, -1).gather(2, idx_mess).contiguous()

            decoded_llr = message_llr.view(B, -1)
            decoded_info = sign_to_bin(torch.sign(decoded_llr))
            if self.return_soft:
                return (decoded_info, decoded_block)
            return decoded_info

        # Check input dimensions
        *leading_dims, L = received.shape
        if L % self.code_length != 0:
            raise ValueError(f"Last dimension ({L}) must be divisible by code length ({self.code_length})")


        return apply_blockwise(received, self.code_length, decode_block)
