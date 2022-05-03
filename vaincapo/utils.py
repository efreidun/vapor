"""Module that contains utils functions."""

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
import torch


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
