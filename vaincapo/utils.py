"""Module that contains utils functions."""

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def quat_to_hopf(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to hopf coordinates.

    Args:
        quaternion: unit quaternions in [w, x, y, z], shape (N, 4)

    Returns:
        hopf coordinates psi, theta, phi in [0, 2pi), [0, pi], [0, 2pi), shape (N, 3)
    """
    w, x, y, z = quat.T
    psi = 2 * np.arctan2(x, w)
    theta = 2 * np.arctan2(np.sqrt(z ** 2 + y ** 2), np.sqrt(w ** 2 + x ** 2))
    phi = np.arctan2(z * w - x * y, y * w + x * z)

    # Note for the following correction use while instead of if, to support
    # float32, because atan2 range for float32 ([-np.float32(np.pi),
    # np.float32(np.pi)]) is larger than for float64 ([-np.pi,np.pi]).

    # Psi must be [0, 2pi) and wraps around at 4*pi, so this correction changes the
    # the half-sphere
    while np.any(psi < 0):
        psi[psi < 0] += 2 * np.pi
    while np.any(psi >= 2 * np.pi):
        psi[psi >= 2 * np.pi] -= 2 * np.pi

    # Phi must be [0, 2pi) and wraps around at 2*pi, so this correction just makes
    # sure the angle is in the expected range
    while np.any(phi < 0):
        phi[phi < 0] += 2 * np.pi
    while np.any(phi >= 2 * np.pi):
        phi[phi >= 2 * np.pi] -= 2 * np.pi
    return np.vstack((psi, theta, phi)).T


def chordal_to_geodesic(dist: torch.Tensor, deg: bool = False) -> torch.Tensor:
    """Convert rotation chordal distance to geodesic distance.

    Args:
        dist: chordal distance, shape (N,)
        deg: if True, geodesic distance is returned in degrees, otherwise radians

    Returns:
        geodesic distance, shape (N,)
    """
    geo_dist = 2 * torch.asin(dist / (8 ** 0.5))
    if deg:
        geo_dist = geo_dist / np.pi * 180
    return geo_dist


def scale_trans(
    tra: torch.Tensor, mins: Iterable, maxs: Iterable, margins: Iterable
) -> torch.Tensor:
    """scale canonical translations to metric space.

    Args:
        tra: translations confined in [0, 1], shape (N, 3)
        mins: minimum valid value for translations, shape (3,)
        maxs: maximum valid value for translations, shape (3,)
        margins: margins for translations, shape (3,)

    Returns:
        metric translation vectors, shape (N, 3)
    """
    return torch.hstack(
        tuple(
            tra[:, i : i + 1] * (maxs[i] - mins[i] + 2 * margins[i])
            + (mins[i] - margins[i])
            for i in range(3)
        )
    )


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
