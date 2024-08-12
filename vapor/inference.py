"""Module defining inference tools."""

from typing import Optional, Tuple

import torch
from torch.distributions.normal import Normal

from vapor.losses import chordal_dist, euclidean_dist, kl_divergence, winners_take_all
from vapor.models import (
    CVAEPoRBackbone,
    CVAEPoRDecoder,
    CVAEPoREncoder,
    CVAEPoRFlows,
    Encoder,
    PoseMap,
)
from vapor.utils import cont_to_rotmat, rotmat_to_cont, scale_trans


def cvaepor_train_forward_pass(
    encoder: CVAEPoREncoder,
    decoder: CVAEPoRDecoder,
    backbone: CVAEPoRBackbone,
    flows: Optional[CVAEPoRFlows],
    batch: Tuple[torch.Tensor, torch.Tensor],
    num_samples: int,
    tra_weight: float,
    rot_weight: float,
    scene_dims: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one forward pass and compute the losses.

    Args:
        encoder: Encoder model
        decoder: Decoder model
        backbone: Backbone model
        flows: Normalizing flows model
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
        mean reconstruction error
        mean KL divergence
        mean translation Euclidean distance
        mean rotation chordal distance
        median mu,
        median logvar,
        samples from the Gaussian-modelled latent posterior
        samples from the possibly modified latent posterior with normalizing flows
        latent posterior mean,
        latent posterior log-variance,
    """
    image = batch[0].to(device)
    pose = batch[1].to(device)
    tra = pose[:, :3]  # (N, 3)
    rot = pose[:, 3:].reshape(-1, 3, 3)  # (N, 3, 3)
    rvec = rotmat_to_cont(rot)  # (N, 6)

    tra_hat, rot_hat, lat_mu, lat_logvar, kld_loss, z0, z = cvaepor_train_reconstruct(
        encoder, decoder, backbone, flows, image, tra, rvec, num_samples, scene_dims
    )

    tra_error = euclidean_dist(tra_hat, tra)
    rot_error = chordal_dist(rot_hat, rot)
    rec_error = tra_weight * tra_error + rot_weight * rot_error

    return (
        tra,
        rot,
        tra_hat,
        rot_hat,
        torch.mean(rec_error),
        kld_loss,
        torch.mean(tra_error),
        torch.mean(rot_error),
        torch.median(lat_mu),
        torch.median(lat_logvar),
        z0,
        z,
        lat_mu,
        lat_logvar,
    )


def cvaepor_test_forward_pass(
    decoder: CVAEPoRDecoder,
    backbone: CVAEPoRBackbone,
    batch: Tuple[torch.Tensor, torch.Tensor],
    num_samples: int,
    tra_weight: float,
    rot_weight: float,
    scene_dims: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one forward pass and compute the losses.

    Args:
        decoder: Decoder model
        backbone: Backbone model
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
        mean reconstruction error
        mean translation Euclidean distance
        mean rotation chordal distance
        samples from the latent prior
    """
    image = batch[0].to(device)
    pose = batch[1].to(device)
    tra = pose[:, :3]  # (N, 3)
    rot = pose[:, 3:].reshape(-1, 3, 3)  # (N, 3, 3)

    tra_hat, rot_hat, z = cvaepor_test_generate(
        decoder, backbone, image, num_samples, scene_dims
    )

    tra_error = euclidean_dist(tra_hat, tra)
    rot_error = chordal_dist(rot_hat, rot)
    rec_error = tra_weight * tra_error + rot_weight * rot_error

    return (
        tra,
        rot,
        tra_hat,
        rot_hat,
        torch.mean(rec_error),
        torch.mean(tra_error),
        torch.mean(rot_error),
        z,
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


def cvaepor_train_reconstruct(
    encoder: CVAEPoREncoder,
    decoder: CVAEPoRDecoder,
    backbone: CVAEPoRBackbone,
    flows: Optional[CVAEPoRFlows],
    image: torch.Tensor,
    tvec: torch.Tensor,
    rvec: torch.Tensor,
    num_samples: int,
    scene_dims: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Reconstruct the pose for an image.

    Args:
        encoder: Encoder model
        decoder: Decoder model
        backbone: Backbone model
        flows: Normalizing flows model
        image: query images, shape (N, 3, H, W)
        tvec: translation component of pose, shape (N, 3)
        rvec: rotation component of pose in 6D parameterization, shape (N, 6)
        num_samples: number of samples to be drawn from the posterior
        scene_dims: tensor containing scene minima, maxima and margins, shape (3, 3)

    Returns:
        translation samples from the inferred posterior, shape (N, num_samples, 3)
        rotation samples from the inferred posterior, shape (N, num_samples, 3, 3)
        latent Gaussian means, shape (N, L)
        latent Gaussian log of variances, shape (N, L)
        KLD term
        samples from the Gaussian-modelled latent posterior
        samples from the possibly modified latent posterior with normalizing flows
    """
    num_images = image.shape[0]
    lat_dim = encoder.get_latent_dim()
    pose = torch.cat((tvec, rvec), dim=1)  # (N, 9)
    img_feat = backbone(image)  # (N, 512)
    lat_mu, lat_logvar = encoder(pose, img_feat)  # (N, L), (N, L)

    if torch.any(torch.isnan(lat_mu)) or torch.any(torch.isnan(lat_logvar)):
        breakpoint()

    if num_samples == 0:
        raise NotImplementedError("Single sample reconstruction not implemented")
    else:
        lat_std = torch.exp(0.5 * lat_logvar)  # (N, L)
        eps = torch.randn(
            (num_images, num_samples, lat_dim), device=image.device
        )  # (N, M, L)
        z0 = eps * lat_std.unsqueeze(1) + lat_mu.unsqueeze(1)  # (N, M, L)

        if flows is None:
            z = z0.view(num_images * num_samples, lat_dim)
            kld_loss = kl_divergence(lat_mu, lat_logvar)
        else:
            z, log_det = flows(
                z0.view(num_images * num_samples, lat_dim)
            )  # (N*M, L), (N*M,)

            if torch.any(torch.isnan(z)) or torch.any(torch.isnan(log_det)):
                breakpoint()

            q0 = Normal(
                lat_mu.view(num_images, 1, lat_dim),
                torch.exp((0.5 * lat_logvar.view(num_images, 1, lat_dim))),
            )
            prior = Normal(0.0, 1.0)

            # KLD including logdet term
            kld_loss = torch.mean(
                (
                    torch.sum(
                        q0.log_prob(z0).view(num_images * num_samples, lat_dim),
                        -1,
                    )
                    - torch.sum(prior.log_prob(z), -1)
                    - log_det
                    # - torch.clamp(log_det, min=-1000.0, max=-1e-3)
                )
            )
            # print(log_det.min().item(), log_det.max().item(), torch.isnan(log_det).any().item(), torch.isinf(log_det).any().item())

            if torch.any(torch.isnan(kld_loss)):
                breakpoint()

        tvec, rvec = decoder(
            z.view(num_images, num_samples, lat_dim), img_feat
        )  # (N*M, 3), (N*M, 6)

    tra_hat = scale_trans(tvec, scene_dims).reshape(num_images, num_samples, 3)
    rot_hat = cont_to_rotmat(rvec).reshape(num_images, num_samples, 3, 3)

    return (
        tra_hat,
        rot_hat,
        lat_mu,
        lat_logvar,
        kld_loss,
        z0,
        z.view(num_images, num_samples, lat_dim),
    )


def cvaepor_test_generate(
    decoder: CVAEPoRDecoder,
    backbone: CVAEPoRBackbone,
    image: torch.Tensor,
    num_samples: int,
    scene_dims: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct the pose for an image.

    Args:
        decoder: Decoder model
        backbone: Backbone model
        image: query images, shape (N, 3, H, W)
        num_samples: number of samples to be drawn from the posterior
        scene_dims: tensor containing scene minima, maxima and margins, shape (3, 3)

    Returns:
        translation samples from the inferred posterior, shape (N, num_samples, 3)
        rotation samples from the inferred posterior, shape (N, num_samples, 3, 3)
        samples from the latent prior
    """
    num_images = image.shape[0]
    img_feat = backbone(image)  # (N, 512)
    if num_samples == 0:
        raise NotImplementedError("Single sample reconstruction not implemented")
    else:
        lat_sample = torch.randn(
            (num_images, num_samples, decoder.get_latent_dim()), device=image.device
        )  # (N, M, L)
        tvec, rvec = decoder(lat_sample, img_feat)  # (N*M, 3), (N*M, 6)

    tra_hat = scale_trans(tvec, scene_dims).reshape(num_images, num_samples, 3)
    rot_hat = cont_to_rotmat(rvec).reshape(num_images, num_samples, 3, 3)

    return tra_hat, rot_hat, lat_sample


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
    if num_samples == 0:
        num_samples = 1
        tvec, rvec = posemap(lat_mu)
    else:
        lat_std = torch.exp(0.5 * lat_logvar)
        eps = torch.randn(
            (num_images, num_samples, encoder.get_latent_dim()), device=image.device
        )
        lat_sample = eps * lat_std.unsqueeze(1) + lat_mu.unsqueeze(1)
        tvec, rvec = posemap(lat_sample.flatten(end_dim=1))

    tra_hat = scale_trans(tvec, scene_dims).reshape(num_images, num_samples, 3)
    rot_hat = cont_to_rotmat(rvec).reshape(num_images, num_samples, 3, 3)

    return tra_hat, rot_hat, lat_mu, lat_logvar
