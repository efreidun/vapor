"""Module that contain tools for sampling from distributions."""

from typing import Tuple, Union
from collections import Counter

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
        self._comps = [
            Bingham(location, lmbda) for location, lmbda in zip(locations, lambdas)
        ]
        self._weights = weights
        self._device = locations.device

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value:
                value(s) at which the density is evaluated in quaternion
                parameterization [w, x, y, z], shape (M, 4)

        Returns:
            log probability at the queried values, shape (M,)
        """
        log_probs = torch.vstack([comp.log_prob(value) for comp in self._comps])
        return torch.log(
            torch.sum(self._weights[:, None] * torch.exp(log_probs), dim=0)
        )

    def sample(
        self, num_samples: Union[int, np.ndarray], num_candidates: int = 1000
    ) -> torch.Tensor:
        """Draw samples from the mixture model.

        Args:
            num_samples:
                number of samples to be drawn. if int, samples are randomly drawn from
                the components according to their weights. if array, specifies the
                number of samples drawn from each component, must be of shape (N,)
            num_candidates: number of candidates considered at every step

        Returns:
            drawn quaternion samples [w, x, y, z], shape (num_samples, 4)
        """
        if type(num_samples) is int:
            comp_counts = Counter(
                np.random.choice(
                    len(self._comps), num_samples, replace=True, p=self._weights
                )
            )
            num_samples = np.zeros(len(self._comps), dtype=int)
            num_samples[list(comp_counts.keys())] = list(comp_counts.values())
        else:
            assert len(num_samples) == len(
                self._comps
            ), "must specify number of samples for each component"

        return torch.cat(
            [
                comp.sample(num_comp_samples, num_candidates)
                for comp, num_comp_samples in zip(self._comps, num_samples)
            ]
        )


class Bingham:
    """Bingham distribution."""

    def __init__(self, location: torch.Tensor, lambdas: torch.Tensor) -> None:
        """Construct a Bingham distribution.

        Args:
            locations:
                location in quaternion parameterization [w, x, y, z], shape (4,)
            lambdas: concentration matrix values, shape (3,)
        """
        self._location = location
        self._lambdas = lambdas
        self._device = self._location.device
        self._density_upper_bound = torch.exp(self.log_prob(self._location))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value:
                value(s) at which the density is evaluated in quaternion
                parameterization [w, x, y, z], shape (M, 4)

        Returns:
            log probability at the queried values, shape (M,)
        """
        if len(value.shape) == 1:
            value = value.reshape(1, 4)
        m = len(value)
        return torch_bingham.bingham_prob(
            self._location[None, :].repeat(m, 1),
            self._lambdas[None, :].repeat(m, 1),
            value,
        ).squeeze()

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
        """Draw samples from the distribution via rejection sampling.

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
