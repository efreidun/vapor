"""Module that contains read and write helper functions."""

from typing import Tuple, List, Iterable, Optional
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from vaincapo.utils import quat_to_rotmat, get_ingp_transform


def read_rendered_samples(
    transforms_path: Path,
    renders_path: Path,
    query_images_path: Path,
    query_renders_path: Path,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read saved renders of samples and their corresponding query images.

    Args:
        transforms_path: path to the transforms.json file used to render the samples
        renders_path: directory where the renders are saved
        query_images_path: directory of the original query images
        query_renders_path: directory of the rendered query images
        resize:
            (width, height), to which the query images/renders are resized,
            if not set, they are resized to sample render size

    Returns:
        rendered samples, shape (N, M, H, W, 3) for N query images and M samples,
        query images, shape (N, H, W, 3)
    """
    with open(transforms_path) as f:
        transforms = json.load(f)
    sample_image_paths = sorted(renders_path.glob("**/*.png"))
    frame_ids = [
        int(str((sample_image_path.stem)).split("_")[0])
        for sample_image_path in sample_image_paths
    ]
    num_renders = transforms["num_renders"]
    sample_renders = np.concatenate(
        [
            np.array(Image.open(sample_image_path))[None, ...]
            for sample_image_path in sample_image_paths
        ]
    )[:, :, :, :3]
    im_h, im_w = sample_renders.shape[1:3]
    if resize is None:
        resize = (im_w, im_h)
    sample_renders = sample_renders.reshape(-1, num_renders, im_h, im_w, 3)
    query_images, query_renders = (
        np.concatenate(
            [
                np.array(
                    Image.open(
                        query_path / transforms["frames"][frame_id]["query_image"]
                    ).resize(resize)
                )[None, ...]
                for frame_id in frame_ids[::num_renders]
            ]
        )[:, :, :, :3]
        for query_path in (query_images_path, query_renders_path)
    )
    assert len(query_images) == len(
        sample_renders
    ), "number of query images and produced sample sets must be equal"
    assert len(query_renders) == len(
        sample_renders
    ), "number of query renders and produced sample sets must be equal"

    return sample_renders, query_images, query_renders


def write_metrics(
    median_errors: List[List[float]],
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
        median_errors: recorded median errors,
            if in array, has shape (num_splits, 2)
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
        f.write(
            ", ".join(
                [
                    f"{split_name} median {quant} error: {err:.2f}"
                    for split_name, errs in zip(split_names, median_errors)
                    for quant, err in zip(("tra", "rot"), errs)
                ]
            )
            + "\n"
        )
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
    dataset: str = "AmbiguousReloc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read poses of a sequence from text file.

    Every row of the text file is one pose containing
    sequence id, frame id, qw, qx, qy, qz, tx, ty, tz for "AmbiguousReloc" dataset,
    and row 3 onward of the text file is one pose containing
    sequence/image_file_path tx ty tz qw qx qy qz for "CambridgeLandmarks" dataset.

    Args:
        poses_path: path to the sequence poses text file
        dataset: dataset that the scene belongs to

    Returns:
        image sequence identifier, shape (N,)
        image IDs, shape (N,)
        image translation vectors, shape (N, 3)
        image rotation matrices, shape (N, 3, 3)
    """
    with open(poses_path) as f:
        content = f.readlines()
    if dataset == "AmbiguousReloc":
        parsed_poses = np.array(
            [[float(entry) for entry in line.strip().split(", ")] for line in content],
            dtype=np.float32,
        )
        seq_ids = parsed_poses[:, 0].astype(int)
        img_ids = parsed_poses[:, 1].astype(int)
        tvecs = parsed_poses[:, 6:]
        rotmats = quat_to_rotmat(parsed_poses[:, 2:6])
        return seq_ids, img_ids, tvecs, rotmats
    elif dataset == "CambridgeLandmarks":
        parsed_poses = np.array(
            [[entry for entry in line.strip().split()] for line in content[3:]],
            dtype=str,
        )
        file_paths = parsed_poses[:, 0]
        seq_ids, img_ids = np.array(
            [file_path.split("/") for file_path in file_paths]
        ).T
        try:
            seq_ids = np.array([seq_id[3:] for seq_id in seq_ids], dtype=int)
            img_ids = np.array([img_id.split(".")[0][5:] for img_id in img_ids], dtype=int)
        except ValueError:
            seq_ids = None
            img_ids = None
        tvecs = parsed_poses[:, 1:4].astype(float)
        rotmats = quat_to_rotmat(parsed_poses[:, 4:].astype(float))
        return seq_ids, img_ids, tvecs, rotmats, file_paths
    elif dataset == "Rig":
        parsed_poses = np.array(
            [[entry for entry in line.strip().split()] for line in content],
            dtype=np.float32,
        )
        seq_ids = parsed_poses[:, 0].astype(int)
        img_ids = parsed_poses[:, 1].astype(int)
        tvecs = parsed_poses[:, 2:5]
        rotmats = quat_to_rotmat(parsed_poses[:, 5:])
        return seq_ids, img_ids, tvecs, rotmats

    else:
        raise ValueError("Invalid dataset name.")


def read_tfmat(tfmat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 4x4 transformation matrix from text file.

    Args:
        tfmat_path: path to the transformation matrix text file

    Returns:
        translation vector, shape (3,)
        rotation matrix, shape (3, 3)
    """
    with open(tfmat_path) as f:
        content = f.readlines()
    tfmat = np.array(
        [[float(entry) for entry in line.strip().split()] for line in content],
        dtype=np.float32,
    )
    return tfmat[:3, 3], tfmat[:3, :3]


def compute_scene_dims(
    scene_path: Path, dataset: str = "AmbiguousReloc", margin_ratio: float = 0.2
) -> torch.Tensor:
    """Compute scene dimensions and write them onto text file.

    Args:
        scene_path: path to the sequence that contains scene.txt file
        dataset: dataset that the scene belongs to
        margin_ratio: ratio of dim width that margin is set to

    Returns:
        2D array with rows containing minimum, maximum and margin values repectively,
        and columns the x, y, z axes, shape (3, 3)
    """
    if dataset == "AmbiguousReloc":
        poses_paths = scene_path.glob("**/poses_seq*.txt")
        positions = np.concatenate(
            [read_poses(poses_path, dataset)[2] for poses_path in poses_paths]
        )
    elif dataset == "SevenScenes":
        pose_paths = scene_path.glob("**/*.pose.txt")
        positions = np.vstack([read_tfmat(pose_path)[0] for pose_path in pose_paths])
    elif dataset == "CambridgeLandmarks":
        positions = np.concatenate(
            [
                read_poses(scene_path / split_file_path, dataset)[2]
                for split_file_path in ("dataset_train.txt", "dataset_test.txt")
            ]
        )
    elif dataset == "Rig":
        poses_paths = scene_path.glob("**/poses.txt")
        positions = np.concatenate(
            [read_poses(poses_path, dataset)[2] for poses_path in poses_paths]
        )
    elif dataset == "SketchUpCircular":
        sin = 1 / np.sqrt(3)
        cos = np.sqrt(2) * sin
        distance = 3
        z = distance * sin
        r = distance * cos
        positions = np.array([[-r, -r, z - 0.1], [r, r, z + 0.1]])
    else:
        raise ValueError("Invalid dataset name.")

    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    margins = margin_ratio * (maxs - mins)
    with open(scene_path / "scene.txt", "w") as f:
        f.write(scene_path.stem + "\n")
        f.write("quantity x y z\n")
        f.write(f"mins {mins[0]} {mins[1]} {mins[2]}\n")
        f.write(f"maxs {maxs[0]} {maxs[1]} {maxs[2]}\n")
        f.write(f"margins {margins[0]} {margins[1]} {margins[2]}\n")
    return torch.tensor(np.vstack((mins, maxs, margins))).float()


def read_scene_dims(scene_path: Path) -> torch.Tensor:
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
