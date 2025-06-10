"""
BeliefPropagationPolarDecoder: A decoder for Polar codes using the Belief Propagation (BP) method over a factor graph.

This class implements the Belief Propagation algorithm for decoding Polar codes. It processes the received codeword
and estimates the transmitted message bits. The decoder supports two regimes: 'sum_product' and 'min_sum', and
provides options for early stopping and cyclic permutations.

References:
    :cite:`arikan2008channel`, :cite:`arikan2011systematic`
"""

from typing import Any

import torch

from kaira.models.fec.encoders.polar_code import PolarCodeEncoder, _index_matrix

from ..utils import (
    apply_blockwise,
    cyclic_perm,
    llr_to_bits,
    min_sum,
    stop_criterion,
    sum_product,
)
from .base import BaseBlockDecoder


class BeliefPropagationPolarDecoder(BaseBlockDecoder[PolarCodeEncoder]):
    """Decoder for Polar code using Belief Propagation (BP) method.

    This class implements the Belief Propagation algorithm for decoding Polar codes. It processes the received codeword
    and estimates the transmitted message bits. The decoder supports two regimes: 'sum_product' and 'min_sum', and
    provides options for early stopping and cyclic permutations.

    Attributes:
        encoder (PolarCodeEncoder): The Polar code encoder used for encoding messages.
        info_indices (torch.Tensor): Indices of information bits in the Polar code.
        frozen_ind (torch.Tensor): Boolean array indicating frozen bits.
        device (torch.device): Device on which the decoder operates (e.g., CPU or GPU).
        dtype (torch.dtype): Data type used for computations.
        polar_i (bool): Indicates whether polar_i is enabled in the encoder.
        frozen_zeros (bool): Indicates whether frozen bits are initialized to zeros.
        m (int): Number of stages in the Polar code.
        iteration_num (int): Number of iterations for decoding.
        mask_dict (torch.Tensor): Mask dictionary for the Polar code structure.
        early_stop (bool): Whether to use early stopping during decoding.
        regime (str): Decoding regime ('sum_product' or 'min_sum').
        clip (float): Clipping value for numerical stability.
        perm (str or None): Type of permutation of the factor graph used ('cycle' or None).
        permutations (torch.Tensor): Array of cyclic permutations.
        R_all (list): List of R matrices for each iteration.
        L_all (list): List of L matrices for each iteration.
    """

    def __init__(self, encoder: PolarCodeEncoder, *args: Any, **kwargs: Any):
        """Initializes the BeliefPropagationPolarDecoder.

        Args:
        encoder (PolarCodeEncoder): The Polar code encoder used for encoding messages.
        bp_iters (int): Number of iterations for decoding.
        early_stop (bool): Whether to use early stopping during decoding.
        regime (str): Decoding regime ('sum_product' or 'min_sum').
        clip (float): Clipping value for numerical stability.
        perm (str or None): Type of permutation of the factor graph used ('cycle' or None).
        """
        super().__init__(encoder, *args, **kwargs)
        self.info_indices = encoder.info_indices
        self.frozen_ind = (torch.ones(self.code_length) - self.info_indices.int()).bool()
        self.device = encoder.device
        self.dtype = encoder.dtype
        self.polar_i = encoder.polar_i
        if self.polar_i:
            raise ValueError("Belief Propagation decoder does not support polar_i=True. " "Please set polar_i=False in the PolarCodeEncoder.")
        self.frozen_zeros = encoder.frozen_zeros
        self.m = encoder.m  # Number of stages in the polar code
        self.iteration_num = kwargs.get("bp_iters", 10)  # Number of iterations for decoding

        self.early_stop = kwargs.get("early_stop", False)  # Whether to use early stopping
        if self.early_stop:
            self.generator_matrix = encoder.get_generator_matrix()

        self.regime = kwargs.get("regime", "sum_product")  # Decoding regime: 'sum_product' or 'min_sum'
        if self.regime not in ["sum_product", "min_sum"]:
            raise ValueError("Invalid regime. Choose either 'sum_product' or 'min_sum'.")

        self.mask_dict = encoder.mask_dict
        if self.mask_dict is None or self.mask_dict.shape[0] != self.m:
            mask_dict = _index_matrix(self.code_length).T.int() - 1
            self.mask_dict = mask_dict[torch.flip(torch.arange(self.m), [0])]
        self.clip = kwargs.get("clip", 1000000.0)  # Default clip value for numerical stability
        self.perm = kwargs.get("perm", None)
        if self.perm == "cycle" and not self.early_stop:
            print("Warning: Cyclic permutation is used, but early stopping is disabled. " "This may lead to suboptimal performance.")
        self.get_cyclic_permutations(perm=self.perm)
        self.print_decoder_type()

    def print_decoder_type(self):
        """Prints the type of decoder being used, along with its configuration."""
        print(
            f"Decoder type: {self.__class__.__name__}, "
            f"Polar Code Length: {self.code_length}, "
            f"Polar Code Dimension: {self.code_dimension}, "
            f"Number of iterations: {self.iteration_num}, "
            f"Function used during decoding: {self.regime}, "
            f"Early Stop: {self.early_stop}, "
            f"Permutations: {self.perm}"
        )

    def get_cyclic_permutations(self, perm=None):
        """Generates cyclic permutations for decoding.

        Args:
            perm (str or None): Type of permutation ('cycle' or None).
        """
        if perm is None:
            self.permutations = torch.arange(self.m).reshape(1, self.m)
        elif perm == "cycle":
            self.permutations = torch.tensor(cyclic_perm(list(torch.arange(self.m))))

    def checknode(self, y1, y2):
        """Check node operation for the Belief Propagation Polar Decoder.

        Args:
            y1 (torch.Tensor): Input message 1.
            y2 (torch.Tensor): Input message 2.

        Returns:
            torch.Tensor: Output message after applying the check node operation.
        """
        if self.regime == "sum_product":
            return sum_product(y1, y2)
        if self.regime == "min_sum":
            return min_sum(y1, y2)

    def update_right(self, R, L, perm):
        """Updates the right messages in the decoding graph.

        Args:
            R (torch.Tensor): Right message tensor.
            L (torch.Tensor): Left message tensor.
            perm (torch.Tensor): Permutation array.

        Returns:
            torch.Tensor: Updated right message tensor.
        """
        mask = self.mask_dict
        for i in perm:
            i_back = self.m - i - 1
            add_k = self.code_length // (2 ** (i_back + 1))
            if len(mask[i]) > 0:
                R[:, i + 1, mask[i]] = self.checknode(R[:, i, mask[i]], L[:, i + 1, mask[i] + add_k] + R[:, i, mask[i] + add_k])
                R[:, i + 1, mask[i] + add_k] = self.checknode(R[:, i, mask[i]], L[:, i + 1, mask[i]]) + R[:, i, mask[i] + add_k]
        R = R.clip(-self.clip, self.clip)
        self.R_all.append(R)
        return R

    def update_left(self, R, L, perm):
        """Updates the left messages in the decoding graph.

        Args:
            R (torch.Tensor): Right message tensor.
            L (torch.Tensor): Left message tensor.
            perm (torch.Tensor): Permutation array.

        Returns:
            torch.Tensor: Updated left message tensor.
        """
        mask = self.mask_dict
        for i in torch.flip(perm, [0]):
            i_back = self.m - i - 1
            add_k = self.code_length // (2 ** (i_back + 1))
            if len(mask[i]) > 0:
                L[:, i, mask[i]] = self.checknode(L[:, i + 1, mask[i]], L[:, i + 1, mask[i] + add_k] + R[:, i, mask[i] + add_k])
                L[:, i, mask[i] + add_k] = self.checknode(R[:, i, mask[i]], L[:, i + 1, mask[i]]) + L[:, i + 1, mask[i] + add_k]

        L = L.clip(-self.clip, self.clip)
        self.L_all.append(L.detach().clone())
        return L

    def _initialize_graph(self, llr):
        """Initializes the decoding graph.

        Args:
            llr (torch.Tensor): Log-likelihood ratio tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized R and L tensors.
        """
        R = torch.zeros([llr.shape[0], self.m + 1, self.code_length], device=self.device)
        if self.frozen_zeros:
            R[:, 0, self.frozen_ind] = self.clip
        else:
            R[:, 0, self.frozen_ind] = -self.clip
        L = torch.zeros_like(R).to(self.device)
        L[:, -1, :] = llr.view(llr.shape[0], -1)

        self.R_all = []
        self.R_all.append(R.detach().clone())
        self.L_all = []
        self.L_all.append(L.detach().clone())
        return R, L

    def decode_iterative(self, llr: torch.Tensor):
        """Performs iterative decoding using the Belief Propagation algorithm.

        Args:
            llr (torch.Tensor): Log-likelihood ratio tensor of shape (batch_size, N).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Decoded message bits and codeword bits.
        """

        not_satisfied_list = [0] * self.iteration_num
        bs = llr.size(0)
        not_satisfied = torch.arange(bs, dtype=torch.long, device=self.device)
        self.ans = []

        u_ans = torch.zeros_like(llr).to(self.device)
        x_ans = torch.zeros_like(llr).to(self.device)

        for i_p, p in enumerate(self.permutations):
            # initialize graph
            right, left = self._initialize_graph(llr)
            for i in range(self.iteration_num):
                left[not_satisfied] = self.update_left(right[not_satisfied], left[not_satisfied], p)
                right[not_satisfied] = self.update_right(right[not_satisfied], left[not_satisfied], p)

                self.ans.append((left[:, -1] + right[:, -1]).view(bs, 1, -1))

                u = left[not_satisfied, 0] + right[not_satisfied, 0]
                x = left[not_satisfied, -1] + right[not_satisfied, -1]

                not_satisfied_list[i] = not_satisfied.clone()
                u_ans[not_satisfied] = u.clone()
                x_ans[not_satisfied] = x.clone()
                if self.early_stop:
                    not_satisfied = stop_criterion(llr_to_bits(x), llr_to_bits(u), self.generator_matrix, not_satisfied)
                if not_satisfied.size(0) == 0:
                    break

        return llr_to_bits(torch.sign(u_ans)), llr_to_bits(x_ans)

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decodes the received codeword using the Belief Propagation algorithm.

        Args:
            received (torch.Tensor): Received codeword tensor of shape (batch_size, code_length).

        Returns:
            torch.Tensor: Estimated message bits of shape (batch_size, code_dimension).
        """
        batch_size, n = received.shape
        self.info_indices = self.encoder.info_indices

        # Ensure the received tensor is on the correct device
        received = received.to(self.device)

        # Reshape codeword to match the expected input shape
        received = received.reshape(batch_size, n)

        # Perform decoding
        def decode_block(received_block: torch.Tensor) -> torch.Tensor:
            """Decode a single block of received codeword."""
            # Ensure the received tensor is on the correct device
            B, _, N = received_block.size()
            assert N == self.code_length, f"Received block size {N} does not match codeword size {self.code_length}"
            # Reshape the received block to 2D as expected by decode_iterative
            received_block = received_block.view(B, N)
            # Decode the block using the iterative decoding method
            u, _ = self.decode_iterative(received_block)
            return u[:, self.info_indices]

        # Return the estimated message bits
        return apply_blockwise(received, self.code_length, decode_block)
