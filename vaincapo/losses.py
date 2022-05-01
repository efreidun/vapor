"""Module that defines the loss functions."""

import torch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence to a standard Gaussian.

    Computes the mean KL-divergence for a set of Gaussian distributions with diagonal
    covariance matrices (M-variate).

    Args:
        mu: mean of the distribution, shape (N, M)
        logvar: log of variance of the distribution, shape (N, M)

    Returns:
        mean KL divergence
    """
    return torch.mean(
        torch.sum(0.5 * (torch.exp(logvar) + mu ** 2 - 1 - logvar), dim=1)
    )
