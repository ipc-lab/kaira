"""Min-Sum Decoder for LDPC Codes.

This module implements a Min-Sum decoder for LDPC codes, which is a simplified
version of the Belief Propagation algorithm. The Min-Sum algorithm approximates
the sum-product algorithm by replacing the computationally expensive check node
operations with simpler min and sign operations.

The Min-Sum decoder offers:
- Lower computational complexity compared to BP
- Reduced numerical precision requirements
- Suitable for hardware implementations
- Slight performance degradation compared to optimal BP

References:
    :cite:`kschischang2001factor`, :cite:`chen2005reduced`
"""

from typing import Any, Union

import torch

from kaira.models.fec.encoders.ldpc_code import LDPCCodeEncoder
from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder
from kaira.models.registry import ModelRegistry

from .belief_propagation import BeliefPropagationDecoder


@ModelRegistry.register_model("min_sum_ldpc_decoder")
class MinSumLDPCDecoder(BeliefPropagationDecoder):
    """Min-Sum decoder for LDPC codes  :cite:`chen2005reduced`.

    This decoder implements the Min-Sum algorithm, which is a simplified version
    of the Belief Propagation algorithm. It replaces the computationally expensive
    check node operations with simpler min and sign operations, reducing complexity
    while maintaining good performance.

    The Min-Sum algorithm performs the following operations:
    1. Variable node update: Same as standard BP
    2. Check node update: Use min-sum approximation instead of sum-product
    3. Message passing: Iterate between variable and check node updates

    Supports multiple variants:
    - Standard Min-Sum: scaling_factor=1.0, offset=0.0
    - Scaled Min-Sum: scaling_factor<1.0 (typically 0.7-0.9), offset=0.0
    - Normalized Min-Sum: scaling_factor<1.0, offset>0.0 (e.g., 0.75, 0.2)

    Args:
        encoder: The LDPC encoder instance providing code parameters
        bp_iters: Number of iterations to perform (default: 10)
        scaling_factor: Scaling factor to improve Min-Sum performance (default: 1.0)
        offset: Offset value for normalized Min-Sum variant (default: 0.0)
        normalized: If True, use optimized normalized parameters (default: False)
        return_soft: Whether to return soft outputs (default: False)
        device: Device for computation (default: "cpu")

    Attributes:
        scaling_factor: Multiplicative scaling factor applied to check node outputs
        offset: Additive offset for normalized Min-Sum variant
        normalized: Whether using normalized variant parameters
    """

    def __init__(self, encoder: Union[LinearBlockCodeEncoder, LDPCCodeEncoder], bp_iters: int = 10, scaling_factor: float = 1.0, offset: float = 0.0, normalized: bool = False, return_soft: bool = False, device: str = "cpu", *args: Any, **kwargs: Any):
        """Initialize the Min-Sum LDPC decoder.

        Args:
            encoder: The LDPC encoder instance
            bp_iters: Number of iterations to perform
            scaling_factor: Scaling factor to improve performance (typically 0.7-0.9)
            offset: Offset value for normalized Min-Sum (typically 0.1-0.5)
            normalized: If True, use optimized normalized parameters (overrides scaling_factor and offset)
            return_soft: Whether to return soft outputs
            device: Device for computation
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Initialize parent class without arctanh (not used in Min-Sum)
        super().__init__(encoder, bp_iters, arctanh=False, return_soft=return_soft, device=device)

        # Set parameters based on normalized flag
        if normalized:
            # Use optimized normalized Min-Sum parameters
            self.scaling_factor = 0.75
            self.offset = 0.2
            self.normalized = True
            self.algorithm_name = "Normalized Min-Sum"
        else:
            # Use provided parameters
            self.scaling_factor = scaling_factor
            self.offset = offset
            self.normalized = False
            if scaling_factor == 1.0 and offset == 0.0:
                self.algorithm_name = "Min-Sum"
            elif offset == 0.0:
                self.algorithm_name = "Scaled Min-Sum"
            else:
                self.algorithm_name = "Normalized Min-Sum"

        # Override the algorithm identifier
        if hasattr(self, "decoder_type"):
            self.decoder_type = "min_sum"

    def compute_cv_minsum(self, vc: torch.Tensor) -> torch.Tensor:
        """Compute check-to-variable messages using Min-Sum algorithm.

        The Min-Sum algorithm approximates the optimal sum-product check node
        operation with:
        1. Sign computation: XOR of input signs
        2. Magnitude computation: Minimum of input magnitudes
        3. Optional scaling and offset for improved performance

        Args:
            vc: Variable-to-check messages tensor of shape [batch_size, num_edges]

        Returns:
            Check-to-variable messages tensor of shape [batch_size, num_edges]
        """
        batch_size, _ = vc.size()
        vc = vc.clamp(-500, 500)  # Numerical stability
        cv = []

        for c_group in self.cv_group:
            deg = self.check_degree[c_group[0]].item()
            members = len(c_group)

            if deg > 1:
                # Get extrinsic message indices for this check group
                from operator import itemgetter

                ext_ce_list = list(itemgetter(*c_group)(self.ext_ce))

                if members == 1 and self.not_ldpc:
                    ext_ce = torch.cat(ext_ce_list, dim=0).view(len(ext_ce_list), -1)
                else:
                    ext_ce = torch.cat(ext_ce_list, dim=0)
                ext_ce = ext_ce.unsqueeze(0).repeat_interleave(batch_size, dim=0)

                # Gather variable-to-check messages for this group
                vc_extended = vc.unsqueeze(1).repeat_interleave(deg * members, dim=1)
                vc_group_messages = vc_extended.gather(2, ext_ce)

                # Min-Sum check node operation
                # 1. Extract signs and magnitudes
                signs = torch.sign(vc_group_messages)
                magnitudes = torch.abs(vc_group_messages)

                # 2. Compute output signs (XOR of input signs)
                sign_product = torch.prod(signs, dim=2, keepdim=True)
                output_signs = sign_product * signs  # Extrinsic sign

                # 3. Compute output magnitudes (min of input magnitudes)
                # For each output, take min over all other inputs (extrinsic minimum)
                min_magnitudes = torch.zeros_like(vc_group_messages)
                for i in range(vc_group_messages.size(2)):
                    # Create mask to exclude current position
                    mask = torch.ones_like(vc_group_messages, dtype=torch.bool)
                    mask[:, :, i] = False

                    # Find minimum over other positions
                    other_magnitudes = magnitudes.masked_select(mask).view(batch_size, deg * members, -1)
                    min_vals, _ = torch.min(other_magnitudes, dim=2)
                    min_magnitudes[:, :, i] = min_vals

                # 4. Combine signs and magnitudes
                v_messages = output_signs * min_magnitudes

                # 5. Apply scaling factor and offset (for improved Min-Sum variants)
                if self.scaling_factor != 1.0:
                    v_messages = v_messages * self.scaling_factor
                if self.offset != 0.0:
                    v_messages = v_messages - torch.sign(v_messages) * self.offset

                # Reshape to match expected output
                v_messages = v_messages.view(batch_size, -1)

            else:
                # Single connection case
                v_messages = torch.zeros(batch_size, members, device=self.device)

            cv.append(v_messages)

        # Concatenate and reorder messages
        cv_tensor = torch.cat(cv, dim=-1)
        new_order = self.cv_order.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        cv_tensor = cv_tensor.gather(1, new_order)

        return cv_tensor

    def compute_cv(self, vc: torch.Tensor) -> torch.Tensor:
        """Override parent's compute_cv to use Min-Sum algorithm."""
        return self.compute_cv_minsum(vc)

    def get_algorithm_info(self) -> dict:
        """Get information about the Min-Sum algorithm configuration.

        Returns:
            Dictionary containing algorithm parameters and characteristics
        """
        return {
            "algorithm": self.algorithm_name,
            "scaling_factor": self.scaling_factor,
            "offset": self.offset,
            "normalized": self.normalized,
            "iterations": self.bp_iters,
            "complexity": "O(E·I) where E=edges, I=iterations",
            "parameters": {"scaling_factor": self.scaling_factor, "offset": self.offset, "normalized": self.normalized},
            "advantages": ["Lower computational complexity than BP", "Simpler hardware implementation", "Reduced numerical precision requirements", "No transcendental functions required"],
            "disadvantages": ["Slight performance loss compared to optimal BP", "May require scaling/offset tuning for best performance"],
        }

    def __str__(self) -> str:
        """String representation of the decoder."""
        return f"MinSumLDPCDecoder(iterations={self.bp_iters}, " f"scaling={self.scaling_factor}, offset={self.offset}, " f"normalized={self.normalized})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
