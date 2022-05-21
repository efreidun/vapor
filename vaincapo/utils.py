"""Module that contains utils functions."""

from typing import Tuple, Union, List, Iterable
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def write_metrics(
    recalls: List[List[List[float]]],
    tra_log_likelihoods: List[List[torch.Tensor]],
    rot_log_likelihoods: List[List[torch.Tensor]],
    recall_thresholds: Iterable[Iterable[float]],
    recall_min_samples: Iterable[int],
    kde_gaussian_sigmas: Iterable[float],
    kde_bingham_lambdas: Iterable[float],
    split_names: Iterable[str],
    header: str,
    run_path: Path,
) -> None:
    """Write evaluation results in metrics.txt.

    Args:
        recalls: recorded recalls,
            if in array, has shape (num_splits, num_min_samples, num_thresholds)
        tra_log_likelihoods: recorded translation log-likelihoods,
            if in array, has shape (num_splits, num_kde_sigmas)
        rot_log_likelihoods: recorded rotation log-likelihoods,
            if in array, has shape (num_splits, num_kde_sigmas)
        recall_thresholds: evaluated recall thresholds,
            a sequence of (tra (m), rot (deg)) pairs
        recall_min_samples: minimum number of samples that define a true-positive
        kde_gaussian_sigmas: evaluated Gaussian kernel sigmas for KDE
        kde_bingham_lambdas: evaluated Bingham kernel lambdas for KDe
        split_names: sequence of split names
        header: text written on the first line of the saved file (without newline char)
        run_path: path to the run where the results will be saved
    """
    recalls = np.transpose(np.array(recalls), (2, 0, 1))
    recall_matrix = [["rec_th", "samps"] + [str(val) for val in recall_min_samples]]
    for (tra_thr, rot_thr), thr_recalls in zip(recall_thresholds, recalls):
        for split_name, vals in zip(split_names, thr_recalls):
            recall_matrix.append(
                [f"{tra_thr}/{int(rot_thr)}", split_name]
                + [f"{val:.2f}" for val in vals]
            )
    recall_matrix = np.array(recall_matrix).T
    m, n = recall_matrix.shape

    loglik_matrix = [
        ["t_logl", "sigma"] + [f"{val:.2f}" for val in kde_gaussian_sigmas]
    ]
    for i, split_name in enumerate(split_names):
        loglik_matrix.append(
            ["t_logl", split_name] + [f"{val:.2f}" for val in tra_log_likelihoods[i]]
        )
    loglik_matrix.append(
        ["r_logl", "lambda"]
        + [f"{val:.2f}" if val < 1000 else f"{val:.1f}" for val in kde_bingham_lambdas]
    )
    for i, split_name in enumerate(split_names):
        loglik_matrix.append(
            ["r_logl", split_name] + [f"{val:.2f}" for val in rot_log_likelihoods[i]]
        )
    loglik_matrix = np.array(loglik_matrix).T
    p, q = loglik_matrix.shape

    if m > p:
        filler = np.empty((m - p, q), dtype=str)
        filler[:] = "-"
        loglik_matrix = np.concatenate((loglik_matrix, filler))
    elif m < p:
        filler = np.empty((p - m, n), dtype=str)
        filler[:] = "-"
        recall_matrix = np.concatenate((recall_matrix, filler))
    results_matrix = np.concatenate((loglik_matrix, recall_matrix), axis=1)

    with open(run_path / "metrics.txt", "w") as f:
        f.write(header + "\n")
        for row in results_matrix:
            f.write("\t".join([val.ljust(6) for val in row]) + "\n")


def write_sample_transforms(
    tra_samples: np.ndarray,
    rot_samples: np.ndarray,
    names: List[str],
    num_renders: int,
    run_path: Path,
    scene_path: Path,
    split_name: str,
) -> None:
    """Write transforms.json file from posterior sample set for rendering with iNGP.

    Args:
        tra_samples: translation samples, shape (N, M, 3)
        rot_samples: rotation matrix samples, shape (N, M, 3, 3)
        names: names of original query images, length (N,)
        run_renders: how many transforms to write for rendering
        run_path: path where the transforms.json file will be saved
        scene_path: path where scene's original transforms.json exists
        split_name: "train" or "valid"
    """
    assert num_renders <= tra_samples.shape[1], "Cannot render more images than samples"
    id_digits = len(str(len(names) * num_renders))
    render_digits = len(str(num_renders))
    with open(scene_path / "transforms.json") as f:
        scene_transforms = json.load(f)
    ingp_params = {
        "scene": scene_path.stem,
        "num_renders": num_renders,
        "split": split_name,
        "camera_angle_x": scene_transforms["camera_angle_x"],
        "frames": [
            {
                "query_image": name,
                "file_path": f"{'/'.join(name.split('/')[:-1])}/"
                + f"{str(num_renders * j + i).zfill(id_digits)}"
                + f"_{name[:-4].split('/')[-1]}_{str(i).zfill(render_digits)}.png",
                "transform_matrix": transform.tolist(),
            }
            for j, (name, transforms) in enumerate(
                zip(
                    names,
                    get_ingp_transform(
                        tra_samples[:, :num_renders].reshape(-1, 3),
                        rot_samples[:, :num_renders].reshape(-1, 3, 3),
                    ).reshape(-1, num_renders, 4, 4),
                )
            )
            for i, transform in enumerate(transforms)
        ],
    }
    with open(run_path / "transforms.json", "w") as f:
        json.dump(ingp_params, f, indent=4)


def read_poses(
    poses_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read poses of a sequence from text file.

    Every row of the text file is one pose containing
    sequence id, frame id, qw, qx, qy, qz, tx, ty, tz.

    Args:
        poses_path: path to the sequence poses text file

    Returns:
        image sequence identifier, shape (N,)
        image IDs, shape (N,)
        image translation vectors, shape (N, 3)
        image rotation matrices, shape (N, 3, 3)
    """
    with open(poses_path) as f:
        content = f.readlines()
    parsed_poses = np.array(
        [[float(entry) for entry in line.strip().split(", ")] for line in content],
        dtype=np.float32,
    )
    seq_ids = parsed_poses[:, 0].astype(int)
    img_ids = parsed_poses[:, 1].astype(int)
    tvecs = parsed_poses[:, 6:]
    rotmats = quat_to_rotmat(parsed_poses[:, 2:6])
    return seq_ids, img_ids, tvecs, rotmats


def compute_scene_dims(scene_path: Path, margin_ratio: float = 0.2) -> np.ndarray:
    """Compute scene dimensions and write them onto text file.

    Args:
        scene_path: path to the sequence that contains scene.txt file
        margin_ratio: ratio of dim width that margin is set to

    Returns:
        2D array with rows containing minimum, maximum and margin values repectively,
        and columns the x, y, z axes, shape (3, 3)
    """
    poses_paths = scene_path.glob("**/poses_seq*.txt")
    positions = np.concatenate(
        [read_poses(poses_path)[2] for poses_path in poses_paths]
    )
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    margins = margin_ratio * (maxs - mins)
    with open(scene_path / "scene.txt", "w") as f:
        f.write(scene_path.stem + "\n")
        f.write("quantity x y z\n")
        f.write(f"mins {mins[0]} {mins[1]} {mins[2]}\n")
        f.write(f"maxs {maxs[0]} {maxs[1]} {maxs[2]}\n")
        f.write(f"margins {margins[0]} {margins[1]} {margins[2]}\n")
    return torch.tensor(np.vstack((mins, maxs, margins)))


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
