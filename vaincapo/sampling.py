"""Module that contain tools for sampling from distributions."""

from typing import Tuple

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

    def sample(self, sample_shape: Tuple = torch.Size()) -> torch.Tensor:
        """Draw samples from the modelled distribution.

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
