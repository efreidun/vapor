"""Module that contain tools for sampling from distributions."""

from typing import Tuple

import numpy as np
import torch
import torch.distributions as D
import torch_bingham


class GMM:
    """Gaussian Mixture Model."""

    def __init__(
        self, locations: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Construct the mixture model with N components in M dimensions.

        Args:
            locations: locations of mixture components, shape (N, M)
            covariances: covariance matrices of mixture components, shape (N, M, M)
            weights: mixture weights, shape (N,)
        """
        mix = D.Categorical(weights)
        comp = D.MultivariateNormal(locations, covariances)
        self._gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, sample_shape: Tuple = torch.Size) -> torch.Tensor:
        """Draw samples from the mixture model.

        Args:
            sample_shape: shape of samples to be drawn

        Returns:
            drawn samples
        """
        return self._gmm.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value: value(s) at which the density is evaluated

        Returns:
            log probability at the queried values
        """
        return self._gmm.log_prob(value)


class BMM:
    """Bingham Mixture Model."""

    def __init__(
        self, locations: torch.Tensor, lambdas: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Construct the mixture model with N components in 4 dimensions.

        Args:
            locations:
                locations of mixture components in quaternion parameterization
                [w, x, y, z], shape (N, 4)
            lambdas: concentration matrix values of components, shape (N, 3)
            weights: mixture weights, shape (N,)
        """
        self._locations = locations
        self._lambdas = lambdas
        self._weights = weights / torch.norm(weights)
        self._device = locations.device
        self._density_upper_bound = torch.sum(
            self._weights * torch.exp(self.log_prob(self._locations))
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value:
                value(s) at which the density is evaluated in quaternion
                parameterization [w, x, y, z], shape (M, 4)

        Returns:
            log probability at the queried values, shape (M,)
        """
        n = len(self._locations)
        if len(value.shape) == 1:
            value = value.reshape(1, 4)
        m = len(value)
        log_probs = torch_bingham.bingham_prob(
            self._locations[:, None, :].repeat(1, m, 1).reshape(-1, 4),
            self._lambdas[:, None, :].repeat(1, m, 1).reshape(-1, 3),
            value[None, :, :].repeat(n, 1, 1).reshape(-1, 4),
        ).reshape(n, m)
        return torch.log(
            torch.sum(self._weights[:, None] * torch.exp(log_probs), dim=0)
        )

    def _generate_uniform_quaternion(self, num_samples: int) -> torch.Tensor:
        """Generate a normalized uniform quaternion.

        Following the method from K. Shoemake, Uniform Random Rotations, 1992.

        See: http://planning.cs.uiuc.edu/node198.html

        Args:
            num_samples: number of samples

        Returns:
            Uniformly distributed unit quaternion [w, x, y, z], shape (num_samples, 4)
        """
        u1 = np.random.random(num_samples)
        u2 = np.random.random(num_samples)
        u3 = np.random.random(num_samples)

        return torch.tensor(
            np.vstack(
                [
                    np.sqrt(u1) * np.cos(2 * np.pi * u3),
                    np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                    np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                    np.sqrt(u1) * np.sin(2 * np.pi * u3),
                ]
            ).T,
            device=self._device,
        )

    def sample(self, num_samples: int, num_candidates: int = 1000) -> torch.Tensor:
        """Draw samples from the mixture model.

        Args:
            num_samples: number of samples to be drawn
            num_candidates: number of candidates considered at every step

        Returns:
            drawn quaternion samples [w, x, y, z], shape (num_samples, 4)
        """
        samples = torch.empty((0, 4), device=self._device)
        while len(samples) < num_samples:
            u = torch.rand(num_candidates, device=self._device)
            candidate_q = self._generate_uniform_quaternion(num_candidates)
            candidate_density = torch.exp(self.log_prob(candidate_q))
            f_over_g = candidate_density / self._density_upper_bound
            samples = torch.cat((samples, candidate_q[u < f_over_g]))
        return samples[:num_samples]
