"""Module defining inference tools."""

from typing import Tuple

import torch

from vaincapo.models import Encoder, PoseMap
from vaincapo.utils import scale_trans, cont_to_rotmat
from vaincapo.losses import (
    kl_divergence,
    winners_take_all,
    euclidean_dist,
    chordal_dist,
)


def forward_pass(
    encoder: Encoder,
    posemap: PoseMap,
    batch: Tuple[torch.Tensor, torch.Tensor],
    num_samples: int,
    num_winners: int,
    tra_weight: float,
    rot_weight: float,
    scene_dims: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one forward pass and compute the losses.

    Args:
        encoder: Encoder model
        posemap: PoseMap model
        batch:
            batch of data, tuple of image, shape (N, 3, H, W), and pose, shape (N, 12)
        num_samples: number of samples to be drawn from the posterior
        num_winners: number of winners whose loss is computed
        tra_weight: weight used for mixing translation and rotation
        rot_weight: weight used for mixing translation and rotation
        scene_dims: tensor containing scene minima, maxima and margins, shape (3, 3)
        device: device on which forward pass is carried out

    Returns:
        groundtruth translations, shape (N, 3)
        groundtruth rotation matrices, shape (N, 3, 3)
        sample translations, shape (N, num_samples, 3)
        sample rotation matrices, shape (N, num_samples, 3, 3)
        mean winners-take-all loss
        mean KL divergence
        mean translation Euclidean distance
        mean rotation chordal distance
    """
    image = batch[0].to(device)
    pose = batch[1].to(device)
    tra = pose[:, :3]
    rot = pose[:, 3:].reshape(-1, 3, 3)

    tra_hat, rot_hat, lat_mu, lat_logvar = infer(
        encoder, posemap, image, num_samples, scene_dims
    )

    kld_loss = kl_divergence(lat_mu, lat_logvar)
    tra_error = euclidean_dist(tra_hat, tra)
    rot_error = chordal_dist(rot_hat, rot)
    wta_loss, tra_loss, rot_loss = winners_take_all(
        tra_error, rot_error, tra_weight, rot_weight, num_winners
    )

    return tra, rot, tra_hat, rot_hat, wta_loss, kld_loss, tra_loss, rot_loss


def infer(
    encoder: Encoder,
    posemap: PoseMap,
    image: torch.Tensor,
    num_samples: int,
    scene_dims: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer the posterior distribution for query images.

    Args:
        encoder: Encoder model
        posemap: PoseMap model
        image: query images, shape (N, 3, H, W)
        num_samples: number of samples to be drawn from the posterior
        scene_dims: tensor containing scene minima, maxima and margins, shape (3, 3)

    Returns:
        translation samples from the inferred posterior, shape (N, num_samples, 3)
        rotation samples from the inferred posterior, shape (N, num_samples, 3, 3)
        latent Gaussian means, shape (N, L)
        latent Gaussian log of variances, shape (N, L)
    """
    num_images = image.shape[0]
    lat_mu, lat_logvar = encoder(image)
    lat_std = torch.exp(0.5 * lat_logvar)
    eps = torch.randn(
        (num_images, num_samples, encoder.get_latent_dim()), device=image.device
    )
    lat_sample = eps * lat_std.unsqueeze(1) + lat_mu.unsqueeze(1)
    tvec, rvec = posemap(lat_sample.flatten(end_dim=1))

    tra_hat = scale_trans(tvec, scene_dims).reshape(num_images, num_samples, 3)
    rot_hat = cont_to_rotmat(rvec).reshape(num_images, num_samples, 3, 3)

    return tra_hat, rot_hat, lat_mu, lat_logvar
