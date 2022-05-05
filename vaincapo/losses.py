"""Module that defines the loss functions."""

import math

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


def euclidean_dist(samples: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute euclidean distance between samples and groundtruth value.

    Args:
        samples: samples to be evaluated, shape (N, M, 3)
        gt: groundtruth values for the samples, shape (N, 3)

    Returns:
        Euclidean distance (L2-norm) of each sample to its gt value, shape (N, M)
    """
    return torch.norm(gt[:, None, :] - samples, dim=2)


def chordal_dist(samples: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute chordal distance between samples and groundtruth value.

    Args:
        samples: rotation matrices to be evaluated, shape (N, M, 3, 3)
        gt: groundtruth rotation matrices for the samples, shape (N, 3, 3)

    Returns:
        chordal distance (Frobenius norm) of each sample to its gt value, shape (N, M)
    """
    return torch.norm(gt[:, None, :, :] - samples, dim=(2, 3))


def geodesic_dist(
    samples: torch.Tensor, gt: torch.Tensor, deg: bool = False
) -> torch.Tensor:
    """Compute chordal distance between samples and groundtruth value.

    Args:
        samples: rotation matrices to be evaluated, shape (N, M, 3, 3)
        gt: groundtruth rotation matrices for the samples, shape (N, 3, 3)
        deg: if True, geodesic distance is returned in degrees, otherwise radians

    Returns:
        geodesic distance of each sample to its gt value, shape (N, M)
    """
    return chordal_to_geodesic(chordal_dist(samples, gt), deg)


def chordal_to_geodesic(dist: torch.Tensor, deg: bool = False) -> torch.Tensor:
    """Convert rotation chordal distance to geodesic distance.

    Args:
        dist: chordal distance, shape (N,)
        deg: if True, geodesic distance is returned in degrees, otherwise radians

    Returns:
        geodesic distance, shape (N,)
    """
    geo_dist = 2 * torch.asin(dist / (8 ** 0.5))
    if deg:
        geo_dist = geo_dist / math.pi * 180
    return geo_dist
