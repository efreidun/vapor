"""Module defining inference tools."""

from typing import Iterable, Tuple

import torch

from vaincapo.models import Encoder, PoseMap
from vaincapo.utils import scale_trans, cont_to_rotmat


def infer(
    encoder: Encoder,
    posemap: PoseMap,
    images: torch.Tensor,
    num_samples: int,
    mins: Iterable[float],
    maxs: Iterable[float],
    margins: Iterable[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer the posterior distribution for query images.

    Args:
        encoder: Encoder model
        posemap: PoseMap model
        image: query images, shape (N, 3, H, W)
        num_samples: number of samples to be drawn from the posterior
        mins: minimum valid value for translations, shape (3,)
        maxs: maximum valid value for translations, shape (3,)
        margins: margins for translations, shape (3,)

    Returns:
        translation samples from the inferred posterior, shape (N, num_samples, 3)
        rotation samples from the inferred posterior, shape (N, num_samples, 3, 3)
    """
    num_images = images.shape[0]
    latent_mu, latent_logvar = encoder(images)
    latent_std = torch.exp(0.5 * latent_logvar)
    eps = torch.randn(
        (num_images, num_samples, encoder.get_latent_dim()), device=images.device
    )
    latent_sample = eps * latent_std.unsqueeze(1) + latent_mu.unsqueeze(1)
    tvec, rvec = posemap(latent_sample.flatten(end_dim=1))

    tra_hat = scale_trans(tvec, mins, maxs, margins).reshape(num_images, num_samples, 3)
    rot_hat = cont_to_rotmat(rvec).reshape(num_images, num_samples, 3, 3)

    return tra_hat, rot_hat
