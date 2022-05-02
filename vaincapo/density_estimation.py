"""Module that contains tools for density estimation of distributions from samples."""

from typing import Tuple

import torch
import torch.distributions as D
import torch_bingham


class R3Gaussian:
    """Approximation of a trivariate distribution with a Gaussian kernel.

    KDE is applied on n i.i.d. samples from an unknown distribution to be modelled,
    using a zero-mean isotropic Gaussian kernel with tunable smooting parameter.
    """

    def __init__(self, samples: torch.Tensor, sigma: float) -> None:
        """Construct the approximation of the distribution.

        Args:
            samples: samples drawn from the dsitribution to be modelled, shape (N, 3)
            sigma: standard deviation of Gaussian kernel along all directions
        """
        n = len(samples)
        mix = D.Categorical(torch.ones(n))
        comp = D.MultivariateNormal(
            samples, (sigma ** 2 * torch.eye(3)).unsqueeze(0).repeat(n, 1, 1)
        )
        self._gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, sample_shape: Tuple = torch.Size()) -> torch.Tensor():
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


class SO3Bingham:
    """Approximation of a distribution on SO(3) with a Bingham kernel.

    KDE is applied on n i.i.d. samples from an unknown distribution to be modelled,
    using a Bingham kernel with a triple of tunable smooting parameters.
    """

    def __init__(self, samples: torch.Tensor, lambdas: torch.Tensor) -> None:
        """Construct the approximation of the distribution.

        Args:
            samples:
                samples drawn from the dsitribution to be modelled in quaternion
                parameterization [w, x, y, z], shape (N, 4)
            lambdas:
                smoothing parameters, which must be non-positive and in descending
                order. they are sorted in descnding order before use, shape (3,)
        """
        assert torch.all(lambdas <= 0), "lambda values must be non-positive"
        self._samples = samples
        self._lambdas = torch.sort(lambdas, descending=True)[0][None, :]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value:
                value(s) at which the density is evaluated in quaternion
                parameterization [w, x, y, z], shape (M, 4)

        Returns:
            log probability at the queried values
        """
        n = len(self._samples)
        if len(value.shape) == 1:
            value = value.reshape(1, 4)
        m = len(value)
        log_probs = torch_bingham.bingham_prob(
            self._samples[:, None, :].repeat(1, m, 1).reshape(-1, 4),
            self._lambdas,
            value[None, :, :].repeat(n, 1, 1).reshape(-1, 4),
        ).reshape(n, m)
        return torch.logsumexp(log_probs - torch.log(torch.tensor(n)), dim=0)
