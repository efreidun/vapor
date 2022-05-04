"""Module that contains evaluation tools."""

from typing import Iterable, List

import torch

from vaincapo.utils import chordal_to_geodesic
from vaincapo.density_estimation import R3Gaussian, SO3Bingham


def evaluate_recall(
    tra_queries: torch.Tensor,
    tra_samples: torch.Tensor,
    rot_queries: torch.Tensor,
    rot_samples: torch.Tensor,
    threshold: Iterable[Iterable[float]],
    min_samples: int,
) -> List[float]:
    """Compute the recall percentage of predicitions.

    Args:
        tra_queries: query translations, shape (N, 3)
        tra_samples: translation samples drawn from distributions, shape (N, M, 3)
        rot_queries: query rotations, shape (N, 3, 3)
        rot_samples: rotation samples drawn from distributions, shape (N, M, 3, 3)
        threshold:
            translation & rotation thresholds (m, deg) defining true positives,
            shape (K, (2,))
        min_samples: minimum number of samples defining true positives

    Returns:
        recall percentage for each threshold, shape (K,)
    """
    tra_dists = torch.norm(tra_queries[:, None, :] - tra_samples, dim=2)
    rot_dists = chordal_to_geodesic(
        torch.norm(rot_queries[:, None, :, :] - rot_samples, dim=(2, 3)), deg=True
    )
    return [
        torch.mean(
            (
                torch.sum((tra_dists <= tra_th) & (rot_dists <= rot_th), dim=1)
                >= min_samples
            ).float()
        ).item()
        for tra_th, rot_th in threshold
    ]


def evaluate_tras_recall(
    queries: torch.Tensor,
    samples: torch.Tensor,
    threshold: Iterable[float],
    min_samples: int,
) -> List[float]:
    """Compute the recall percentage of predicitions.

    Args:
        queries: query translations, shape (N, 3)
        samples: samples drawn from distributions, shape (N, M, 3)
        threshold: threshold (m) defining true positives, shape (K,)
        min_samples: minimum number of samples defining true positives

    Returns:
        recall percentage for each threshold, shape (K,)
    """
    dists = torch.norm(queries[:, None, :] - samples, dim=2)
    return [
        torch.mean((torch.sum(dists <= th, dim=1) >= min_samples).float()).item()
        for th in threshold
    ]


def evaluate_rots_recall(
    queries: torch.Tensor,
    samples: torch.Tensor,
    threshold: Iterable[float],
    min_samples: int,
) -> List[float]:
    """Compute the recall percentage of predicitions.

    Args:
        queries: query rotations, shape (N, 3, 3)
        samples: samples drawn from distributions, shape (N, M, 3, 3)
        threshold: threshold (deg) defining true positives, shape (K,)
        min_samples: minimum number of samples defining true positives

    Returns:
        recall percentage for each threshold, shape (K,)
    """
    dists = chordal_to_geodesic(
        torch.norm(queries[:, None, :, :] - samples, dim=(2, 3)), deg=True
    )
    return [
        torch.mean((torch.sum(dists <= th, dim=1) >= min_samples).float()).item()
        for th in threshold
    ]


def evaluate_tras_likelihood(
    queries: torch.Tensor, samples: torch.Tensor, sigma: float
) -> float:
    """Compute the log probability densities of query points based on dist samples.

    Args:
        queries: query translations, shape (N, 3)
        samples: samples drawn from distributions, shape (N, M, 3)
        sigma: bandwidth parameter used for density estimation

    Returns:
        mean of log likelihoods of query points
    """
    return torch.mean(
        torch.tensor(
            [
                get_tra_log_likelihood(query, sample, sigma).item()
                for query, sample in zip(queries, samples)
            ]
        )
    )


def evaluate_rots_likelihood(
    queries: torch.Tensor, samples: torch.Tensor, sigma: float
) -> float:
    """Compute the log probability densities of query points based on dist samples.

    Args:
        queries: query rotations in quaternion param [w, x, y, z], shape (N, 4)
        samples: samples in quat param [w, x, y, z] drawn from dist, shape (N, M, 4)
        sigma: bandwidth parameter used for density estimation

    Returns:
        mean of log likelihoods of query points
    """
    return torch.mean(
        torch.tensor(
            [
                get_rot_log_likelihood(query, sample, sigma).item()
                for query, sample in zip(queries, samples)
            ]
        )
    )


def get_tra_log_likelihood(
    query: torch.Tensor, samples: torch.Tensor, sigma: float
) -> float:
    """Compute the log probability density of a query point based on dist samples.

    Args:
        query: query point, shape (3,)
        samples: samples drawn from distribution, shape (N, 3)
        sigma: bandwidth parameter used for density estimation

    Returns:
        log likelihood of query point
    """
    dist = R3Gaussian(samples, sigma)
    return dist.log_prob(query)


def get_rot_log_likelihood(
    query: torch.Tensor, samples: torch.Tensor, sigma: float
) -> float:
    """Compute the log probability density of a query point based on dist samples.

    Args:
        query: query point in quaternion parameterization [w, x, y, z], shape (4,)
        samples:
            samples in quaternion parameterization [w, x, y, z] drawn from distribution
            shape (N, 4)
        sigma: bandwidth parameter used for density estimation

    Returns:
        log likelihood of query point
    """
    dist = SO3Bingham(samples, sigma)
    return dist.log_prob(query[None, :])