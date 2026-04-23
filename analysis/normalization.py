"""Soft-score normalization functions for the probabilistic analysis layer."""

from math import exp


def soft_score(value: float, mean: float, std: float) -> float:
    """sigmoid(z) where z = (value - mean) / std. Returns 0.5 when std == 0."""
    if std == 0:
        return 0.5
    z = (value - mean) / std
    # Clamp z to avoid overflow in exp(); sigmoid saturates to 0 or 1 outside this range.
    if z >= 709.0:
        return 1.0
    if z <= -709.0:
        return 0.0
    return 1.0 / (1.0 + exp(-z))


def inverted_soft_score(value: float, mean: float, std: float) -> float:
    """1 - soft_score(value, mean, std). For 'low is bad' features."""
    return 1.0 - soft_score(value, mean, std)
