"""Module that contains evaluation tools."""

import torch

from vaincapo.utils import cont_to_rotmat
from vaincapo.density_estimation import R3Gaussian, SO3Bingham


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
    rotmat_queries = cont_to_rotmat(queries)
    rotmat_samples = cont_to_rotmat(samples.reshape(-1, 6))

    rotation_queries = Rotation.from_matrix(rotmat_queries.numpy())
    rotation_samples = Rotation.from_matrix(rotmat_samples.numpy())

    quat_queries = torch.tensor(rotation_queries.as_quat())
    quat_samples = torch.tensor(rotation_samples.as_quat())

    # change quaternion order from [x, y, z, w] to [w, x, y, z]
    quat_queries = torch.roll(quat_queries, 1, dim=1)
    quat_samples = torch.roll(quat_samples, 1, dim=1).reshape(*samples.shape[:2], 3, 3)

    return torch.mean(
        torch.tensor(
            [
                get_rot_log_likelihood(quat_query, quat_sample, sigma).item()
                for quat_query, quat_sample in zip(quat_queries, quat_samples)
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
