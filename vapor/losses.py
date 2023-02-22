"""Module that defines the loss functions."""

from typing import Tuple

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


def winners_take_all(
    tra_error: torch.Tensor,
    rot_error: torch.Tensor,
    tra_weight: float,
    rot_weight: float,
    num_winners: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the winners-take-all loss.

    Compute the mean loss for the num_winners of samples with smallest errors.

    Args:
        tra_error: translation errors, shape (N, M)
        rot_error: rotation errors, shape (N, M)
        tra_weight: weight used for mixing translation and rotation
        rot_weight: weight used for mixing translation and rotation
        num_winners: number of winners whose loss is computed

    Returns:
        mean loss of winning samples,
        mean translation loss of winning samples,
        mean rotation loss of winning samples
    """
    tot_error = tra_weight * tra_error + rot_weight * rot_error
    sorted_error, sort_indices = torch.sort(tot_error, dim=1)
    loss = torch.mean(sorted_error[:, :num_winners])
    tra_loss = torch.mean(
        torch.stack(
            [
                error[indices[:num_winners]]
                for error, indices in zip(tra_error, sort_indices)
            ]
        )
    )
    rot_loss = torch.mean(
        torch.stack(
            [
                error[indices[:num_winners]]
                for error, indices in zip(rot_error, sort_indices)
            ]
        )
    )
    return loss, tra_loss, rot_loss


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
