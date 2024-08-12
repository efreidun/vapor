"""Module that contains utils functions."""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.manifold import TSNE


def visualize_latent(
    images: torch.Tensor,
    z0: torch.Tensor,
    z: torch.Tensor,
    tra_hat: torch.Tensor,
    rot_hat: torch.Tensor,
    tra: torch.Tensor,
    rot: torch.Tensor,
    num_vis: int = 4,
    perplexity: float = 50.0,
) -> torch.Tensor:
    """Visualize latent samples and the prior.

    Args:
        images: input images, shape (N, 3, H, W)
        z0: Samples from Gaussian-modelled latent posterior, shape (N, M, L)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (N, M, L)
        tra_hat: Predicted translation vectors, shape (N, M, 3)
        rot_hat: Predicted rotation matrices, shape (N, M, 3, 3)
        tra: Ground-truth translation vectors, shape (N, 3)
        rot: Ground-truth rotation matrices, shape (N, 3, 3)
        num_vis: How many images visualize latent samples for, must be <=N
        perplexity: Parameter for t-SNE (recommended between 5 to 50, must be <= M)

    Returns:
        The torch tensor corresponding to the k
    """
    num_images, num_samples, lat_dim = z0.shape

    images = images.numpy(force=True)

    rvec_hat = rotmat_to_cont(rot_hat.view(-1, 3, 3)).view(
        rot_hat.shape[:2] + (6,)
    )  # (N, M, 6)
    rvec = rotmat_to_cont(rot)  # (N, 6)

    z0 = z0.numpy(force=True)
    z = z.numpy(force=True)
    zp = np.random.multivariate_normal(
        np.zeros(lat_dim), np.eye(lat_dim), (num_images, num_samples)
    )  # (N, M, L)

    tra_hat = tra_hat.numpy(force=True)
    rvec_hat = rvec_hat.numpy(force=True)
    tra = tra.numpy(force=True)
    rvec = rvec.numpy(force=True)
    pose_hat = np.concatenate((tra_hat, rvec_hat), axis=2)  # (N, M, 9)
    pose = np.concatenate((tra, rvec), axis=1)  # (N, 9)
    poses = np.concatenate((pose_hat, pose[:, None, :]), axis=1)  # (N, M+1, 9)
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perplexity,
    )
    poses_emb = np.stack(
        [tsne.fit_transform(poses[i]) for i in range(num_vis)]
    )  # (num_vis, M+1, 2)
    pose_hat_emb = poses_emb[:, :-1, :]  # (num_vis, M, 2)
    pose_emb = poses_emb[:, -1, :]  # (num_vis, 2)

    if lat_dim == 2:
        z0_emb = z0[:num_vis]  # (N, M, 2)
        z_emb = z[:num_vis]  # (N, M, 2)
        zp_emb = z[:num_vis]  # (N, M, 2)
    else:
        zs = np.concatenate((z0, z, zp), axis=1)  # (N, 3*M, L)
        zs_emb = np.stack(
            [tsne.fit_transform(zs[i]) for i in range(num_vis)]
        )  # (num_vis, 3*M, 2)
        z0_emb = zs_emb[:, :num_samples]  # (num_vis, M, 2)
        z_emb = zs_emb[:, num_samples : 2 * num_samples]  # (num_vis, M, 2)
        zp_emb = zs_emb[:, 2 * num_samples : 3 * num_samples]  # (num_vis, M, 2)

    fig, axs = plt.subplots(num_vis, 4, figsize=(10, 7))
    for i in range(num_vis):
        axs[i, 0].set_ylabel(f"image {i}")
        axs[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].scatter(*zp_emb[i].T, c="blue", alpha=0.5, s=2)
        axs[i, 1].scatter(*z0_emb[i].T, c="red", alpha=0.5, s=2)
        axs[i, 2].scatter(*zp_emb[i].T, c="blue", alpha=0.5, s=2)
        axs[i, 2].scatter(*z_emb[i].T, c="lime", alpha=0.5, s=2)
        axs[i, 3].scatter(*pose_emb[i], c="magenta", s=10, marker="*")
        axs[i, 3].scatter(*pose_hat_emb[i].T, c="lime", alpha=0.5, s=2)
    axs[0, 0].set_title("query img")
    axs[0, 1].set_title("Gaussian posterior")
    axs[0, 2].set_title("After norm. flows")
    axs[0, 3].set_title("Pose samples")
    fig.tight_layout()

    return fig


def schedule_warmup(
    epoch: int, max_value: float, start: int = 0, period: int = 0
) -> float:
    """Schedule a weight linear warmup from 0 to max_value.

    If start and period are 0 (default) warmup is disabled and max_value is used.

    Args:
        epoch: current epoch
        max_value: final value of the weight
        start: epoch when warmup starts
        period: warmup period until max_value is reached

    Returns:
        weight value for current epoch
    """
    if epoch < start:
        return 0
    elif epoch < start + period:
        m = max_value / period
        c = -m * start
        return m * epoch + c
    else:
        return max_value


def quat_to_hopf(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to hopf coordinates.

    Args:
        quaternion: unit quaternions in [w, x, y, z], shape (N, 4)

    Returns:
        hopf coordinates phi, theta, psi in [-pi, pi), [-pi/2, pi/2], [0, 2pi),
        shape (N, 3)
    """
    w, x, y, z = quat.T
    psi = 2 * np.arctan2(x, w)
    theta = 2 * np.arctan2(np.sqrt(z**2 + y**2), np.sqrt(w**2 + x**2))
    phi = np.arctan2(z * w - x * y, y * w + x * z)

    # Note for the following correction use while instead of if, to support
    # float32, because atan2 range for float32 ([-np.float32(np.pi),
    # np.float32(np.pi)]) is larger than for float64 ([-np.pi,np.pi]).

    # Phi must be [-pi, pi) and wraps around at pi, so this correction just makes
    # sure the angle is in the expected range
    while np.any(phi < -np.pi):
        phi[phi < np.pi] += 2 * np.pi
    while np.any(phi >= np.pi):
        phi[phi >= np.pi] -= 2 * np.pi

    # Theta must be [-pi/2, pi]
    # theta -= np.pi / 2  # TODO: why did we have this

    # Psi must be [0, 2pi) and wraps around at 4*pi, so this correction changes the
    # the half-sphere
    while np.any(psi < 0):
        psi[psi < 0] += 2 * np.pi
    while np.any(psi >= 2 * np.pi):
        psi[psi >= 2 * np.pi] -= 2 * np.pi

    return np.vstack((phi, theta, psi)).T


def scale_trans(
    tra: torch.Tensor,
    scene_dims: np.ndarray,
) -> torch.Tensor:
    """scale canonical translations to metric space.

    Args:
        tra: translations confined in [0, 1], shape (N, 3)
        scene_dims: tensor containing scene minima, maxima and margins, shape (3, 3)

    Returns:
        metric translation vectors, shape (N, 3)
    """
    scene_dims = scene_dims.to(tra.device)
    a = (scene_dims[1] - scene_dims[0] + 2 * scene_dims[2])[None, :]
    b = (scene_dims[0] - scene_dims[2])[None, :]
    return a * tra + b


def cont_to_rotmat(rot: torch.Tensor) -> torch.Tensor:
    """Convert 6D continuous rotation parameterization to roation matrix.

    Args:
        rot: continuous 6D rotation parameterization, shape (N, 6)

    Returns:
        rotation matrices, shape (N, 3, 3)
    """
    a1 = rot[:, :3]
    a2 = rot[:, 3:]
    b1 = a1 / torch.norm(a1, dim=1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
    b3 = torch.cross(b1, b2)
    return torch.cat((b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)), dim=2)


def rotmat_to_cont(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D continuous rotation parameterization.

    Args:
        rotmat: rotation matrix, shape (N, 3, 3)

    Returns:
        continuous 6D rotation parameterization, shape (N, 6)
    """
    return torch.permute(rotmat[:, :, :2], (0, 2, 1)).flatten(start_dim=1)  # (N, 6)


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions.

    Args:
        rot: rotation matrices, shape (N, 3, 3)

    Returns:
        quaternions in [w, x, y, z] parameterization, shape (N, 4)
    """
    rotation = Rotation.from_matrix(rot)
    quat = rotation.as_quat()
    return np.roll(quat, 1, axis=1)


def quat_to_rotmat(
    quat: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Conver quaternions to rotation matrices.

    Args:
        quat: quaternions in [w, x, y, z] parameterization, shape (N, 4)

    Returns:
        rotation matrices, shape (N, 3, 3)
    """
    qw, qx, qy, qz = quat.T
    qw2, qx2, qy2, qz2 = qw**2, qx**2, qy**2, qz**2
    qwx, qwy, qwz = qw * qx, qw * qy, qw * qz
    qxy, qxz, qyz = qx * qy, qx * qz, qy * qz
    vstack = torch.vstack if type(quat) is torch.Tensor else np.vstack
    return vstack(
        [
            qw2 + qx2 - qy2 - qz2,
            2 * qxy - 2 * qwz,
            2 * qwy + 2 * qxz,
            2 * qwz + 2 * qxy,
            qw2 - qx2 + qy2 - qz2,
            2 * qyz - 2 * qwx,
            2 * qxz - 2 * qwy,
            2 * qwx + 2 * qyz,
            qw2 - qx2 - qy2 + qz2,
        ]
    ).T.reshape(-1, 3, 3)


def create_tfmat(
    tvec: Union[torch.Tensor, np.ndarray], rotmat: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Create tfmat from translation vector and rotation matrix.

    Args:
        tra: translation vector, shape (N, 3)
        rotmat: rotation matrix, shape (N, 3, 3)

    Returns:
        transformation matrix, shape (N, 4, 4)
    """
    if type(tvec) is torch.Tensor:
        return torch.cat(
            (
                torch.cat((rotmat, tvec[:, :, None]), dim=2),
                torch.tensor([[[0, 0, 0, 1]]], device=tvec.device).repeat(
                    len(tvec), 1, 1
                ),
            ),
            dim=1,
        )
    else:
        return np.concatenate(
            (
                np.concatenate((rotmat, tvec[:, :, None]), axis=2),
                np.tile(np.array([[[0, 0, 0, 1]]]), (len(tvec), 1, 1)),
            ),
            axis=1,
        )


def get_ingp_transform(tvec: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """Get the transform for instant neural graphics primitives pipeline.

    The input translation and rotation must define a camera-to-world transformation.

    Args:
        tvec: translation vector, shape (N, 3)
        rotmat: rotation matrix, shape (N, 3, 3)

    Returns:
        transformation matrix as required by ingp, shape (N, 4, 4)
    """
    transform = create_tfmat(tvec, rotmat)

    # manipulate as done in ingp
    transform[:, 0:3, 2] *= -1  # flip the y and z axis
    transform[:, 0:3, 1] *= -1
    transform = transform[:, [1, 0, 2, 3], :]  # swap y and z
    transform[:, 2, :] *= -1  # flip whole world upside down

    return transform


def average_pose(tra_hat: torch.Tensor, rot_hat: torch.Tensor) -> torch.Tensor:
    """Compute the average pose of a number of samples.

    L2 mean is computed for translations and chordal L2 mean for rotations.
    Ref "Rotation averaging" by Hartley et al (2013).

    Args:
        tra_hat: translation samples, shape (N, M, 3)
        rot_hat: rotation matrix samples, shape (N, M, 3, 3)

    Returns:
        average translation, shape (N, 3),
        average rotation, shape (N, 3, 3)
    """
    tra = torch.mean(tra_hat, dim=1)
    rot = torch.cat([chordal_l2_mean(R)[None, :, :] for R in rot_hat])

    return tra, rot


def chordal_l2_mean(rot_samples: torch.Tensor) -> torch.Tensor:
    """Compute the a single rotation average of many samples.

    Chordal L2 mean for rotations as done in "Rotation averaging" Hartley et al (2013).

    Args:
        rot_samples: rotation matrix samples, shape (N, 3, 3)

    Returns:
        average rotation, shape (3, 3)
    """
    C = torch.sum(rot_samples, dim=0)
    U, _, Vt = torch.linalg.svd(C)

    S = U @ Vt
    if torch.linalg.det(S) < 0:
        S = (
            U
            @ torch.diag(
                torch.tensor(
                    [1.0, 1.0, -1.0], device=rot_samples.device, dtype=rot_samples.dtype
                )
            )
            @ Vt
        )

    return S
