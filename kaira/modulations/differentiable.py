"""Differentiable operations for modulation schemes.

This module provides differentiable alternatives to operations commonly used in digital modulation
that are not naturally differentiable, such as bit mapping and constellation symbol selection.
"""

import torch
import torch.nn.functional as F


def soft_symbol_mapping(soft_bits: torch.Tensor, constellation: torch.Tensor, bit_patterns: torch.Tensor) -> torch.Tensor:
    """Map soft bit probabilities to a weighted sum of constellation symbols.

    This function provides a differentiable path from soft bit probabilities to symbols
    by computing expectations over the constellation.

    Args:
        soft_bits: Soft bit probabilities with shape (..., K) where K is bits_per_symbol
                   Values should be in [0, 1] range, representing P(bit=1)
        constellation: Complex tensor of constellation points with shape (M,)
        bit_patterns: Binary tensor with shape (M, K) representing the bit patterns
                     for each constellation point

    Returns:
        Complex tensor with shape (...) representing the expected symbol value
    """
    # Reshape soft_bits for broadcasting with bit_patterns
    soft_bits = soft_bits.unsqueeze(-2)  # (..., 1, K)

    # Calculate probabilities of each bit pattern
    # For each bit position:
    #   - If bit pattern is 1, use soft_bit probability
    #   - If bit pattern is 0, use (1 - soft_bit) probability
    probs_when_bit_is_1 = soft_bits  # P(bit=1)
    probs_when_bit_is_0 = 1 - soft_bits  # P(bit=0)

    # Select probabilities based on the bit patterns
    # bit_patterns has shape (M, K)
    bit_probs = torch.where(bit_patterns.unsqueeze(0).bool(), probs_when_bit_is_1, probs_when_bit_is_0)  # (1, M, K)  # (..., 1, K)  # (..., 1, K)

    # Calculate the joint probability of each constellation point
    # by multiplying probabilities of individual bits
    symbol_probs = torch.prod(bit_probs, dim=-1)  # (..., M)

    # Calculate the expected symbol
    expected_symbol = torch.sum(symbol_probs * constellation, dim=-1)  # (...)

    return expected_symbol


def soft_bits_to_hard_symbols(soft_bits: torch.Tensor, constellation: torch.Tensor, bit_patterns: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Convert soft bits to hard symbols with a differentiable approximation.

    Uses a temperature-based softmax approach for approximating the hard decision
    while maintaining differentiability.

    Args:
        soft_bits: Soft bit probabilities with shape (..., K) where K is bits_per_symbol
                   Values should be in [0, 1] range, representing P(bit=1)
        constellation: Complex tensor of constellation points with shape (M,)
        bit_patterns: Binary tensor with shape (M, K) representing the bit patterns
                     for each constellation point
        temp: Temperature parameter for softmax (lower = harder decision)

    Returns:
        Complex tensor with shape (...) representing the selected symbol
    """
    # Reshape soft_bits for broadcasting
    soft_bits = soft_bits.unsqueeze(-2)  # (..., 1, K)

    # Calculate log probabilities for each bit pattern
    log_probs_when_bit_is_1 = torch.log(soft_bits + 1e-10)
    log_probs_when_bit_is_0 = torch.log(1 - soft_bits + 1e-10)

    log_bit_probs = torch.where(bit_patterns.unsqueeze(0).bool(), log_probs_when_bit_is_1, log_probs_when_bit_is_0)

    # Sum log probabilities to get joint log probability
    log_symbol_probs = torch.sum(log_bit_probs, dim=-1)  # (..., M)

    # Apply temperature scaling and softmax
    symbol_weights = F.softmax(log_symbol_probs / temp, dim=-1)  # (..., M)

    # Calculate the weighted sum of constellation points
    weighted_symbols = torch.sum(symbol_weights * constellation, dim=-1)

    return weighted_symbols


def hard_decisions_with_straight_through(soft_values: torch.Tensor) -> torch.Tensor:
    """Make hard 0/1 decisions while allowing backpropagation with straight-through estimator.

    Args:
        soft_values: Soft values typically in range [0, 1]

    Returns:
        Hard binary decisions (0 or 1) with gradients passed through unchanged
    """
    # Forward pass: hard thresholding
    hard_decisions = (soft_values > 0.5).float()

    # Straight-through estimator: pass gradients through unchanged
    return hard_decisions.detach() - soft_values.detach() + soft_values
