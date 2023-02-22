"""Module that contains evaluation tools."""

from typing import Iterable, List

import torch

from vapor.density_estimation import R3Gaussian, SO3Bingham
from vapor.losses import euclidean_dist, geodesic_dist


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
    tra_dists = euclidean_dist(tra_samples, tra_queries)
    rot_dists = geodesic_dist(rot_samples, rot_queries, deg=True)
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
    dists = euclidean_dist(samples, queries)
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
    dists = geodesic_dist(samples, queries, deg=True)
    return [
        torch.mean((torch.sum(dists <= th, dim=1) >= min_samples).float()).item()
        for th in threshold
    ]


def evaluate_tras_likelihood(
    queries: torch.Tensor, samples: torch.Tensor, sigma: float
) -> torch.Tensor:
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
) -> torch.Tensor:
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
) -> torch.Tensor:
    """Compute the log probability density of a query point based on dist samples.

    Args:
        query: query point, shape (3,)
        samples: samples drawn from distribution, shape (N, 3)
        sigma: bandwidth parameter used for density estimation

    Returns:
        log likelihood of query point
    """
    dist = R3Gaussian(samples, sigma)
    return dist.log_prob(query.float())


def get_rot_log_likelihood(
    query: torch.Tensor, samples: torch.Tensor, sigma: float
) -> torch.Tensor:
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
