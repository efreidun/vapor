"""Module that contains utils functions."""

from typing import Tuple
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def read_poses(poses_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read poses of a sequence from text file.

    Every row of the text file is one pose containing
    sequence id, frame id, qw, qx, qy, qz, tx, ty, tz.

    Args:
        poses_path: path to the sequence poses text file

    Returns:
        image IDs, shape (N,)
        image poses [qw, qx, qy, qz, tx, ty, tz], shape (N, 7)
    """
    with open(poses_path) as f:
        content = f.readlines()
    parsed_poses = np.array(
        [[float(entry) for entry in line.strip().split(", ")] for line in content],
        dtype=np.float32,
    )
    return parsed_poses[:, 1].astype(int), parsed_poses[:, 2:]


def compute_scene_dims(scene_path: Path, margin_ratio: float) -> np.ndarray:
    """Compute scene dimensions and write them onto text file.

    Args:
        scene_path: path to the sequence that contains scene.txt file
        margin_ratio: ratio of dim width that margin is set to

    Returns:
        2D array with rows containing minimum, maximum and margin values repectively,
        and columns the x, y, z axes, shape (3, 3)
    """
    _, train_poses = read_poses(scene_path / "train/seq00/poses_seq00.txt")
    _, test_poses = read_poses(scene_path / "test/seq01/poses_seq01.txt")
    positions = np.concatenate((train_poses, test_poses))[:, 4:]
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    margins = margin_ratio * (maxs - mins)
    with open(scene_path / "scene.txt", "w") as f:
        f.write(scene_path.stem + "\n")
        f.write("quantity x y z\n")
        f.write(f"mins {mins[0]} {mins[1]} {mins[2]}\n")
        f.write(f"maxs {maxs[0]} {maxs[1]} {maxs[2]}\n")
        f.write(f"margins {margins[0]} {margins[1]} {margins[2]}\n")
    return np.vstack((mins, maxs, margins))


def read_scene_dims(scene_path: Path) -> np.ndarray:
    """Read scene dimensions from text file.

    Args:
        scene_path: path to the sequence that contains scene.txt file

    Returns:
        2D array with rows containing minimum, maximum and margin values repectively,
        and columns the x, y, z axes, shape (3, 3)
    """
    with open(scene_path / "scene.txt") as f:
        content = f.readlines()
    return torch.tensor(
        [float(entry) for line in content[2:] for entry in line.strip().split()[1:]]
    ).reshape(3, 3)


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
