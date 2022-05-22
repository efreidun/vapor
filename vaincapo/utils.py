"""Module that contains utils functions."""

from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation
import torch


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
    theta = 2 * np.arctan2(np.sqrt(z ** 2 + y ** 2), np.sqrt(w ** 2 + x ** 2))
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
    theta -= np.pi / 2

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
    qw2, qx2, qy2, qz2 = qw ** 2, qx ** 2, qy ** 2, qz ** 2
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
