"""Script for preparing JSON file for a scene."""

from types import SimpleNamespace
import os
from pathlib import Path
import json
import yaml

import argparse
import cv2
import numpy as np
from tqdm import tqdm

from vaincapo.utils import get_ingp_transform
from vaincapo.read_write import read_poses, read_tfmat


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Write JSON file for a scene.")
    parser.add_argument("scene", type=str)
    parser.add_argument("--dataset", type=str, default="AmbiguousReloc")
    parser.add_argument("--split", type=str, nargs="+", default=["train", "test"])
    args = parser.parse_args()

    return vars(args)


def get_frame(
    img_path: Path, position: np.ndarray, rotmat: np.ndarray, seq_path: Path
) -> dict:
    """Create a frame dictionary.

    Args:
        img_path: path to the image
        position: image position, shape (3,)
        rotmat: image c2w rotation matrix, shape (3, 3)
        seq_path: path to the sequence that contains the image

    Returns:
        dictionary with keys "file_path", "sharpness", "transform_matrix"
    """
    file_path = str(img_path.relative_to(seq_path.parents[1]))
    image = cv2.imread(str(img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    transform = get_ingp_transform(position[None, :], rotmat[None, :, :])[0].tolist()
    return {
        "file_path": file_path,
        "sharpness": sharpness,
        "transform_matrix": transform,
    }


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    dataset_path = Path.home() / "data" / cfg.dataset
    if cfg.dataset == "AmbiguousReloc":
        with open(dataset_path / "camera.yaml") as f:
            camera_params = yaml.load(f, Loader=yaml.FullLoader)
        params = {
            "fl_x": camera_params["camera_matrix"]["data"][0],
            "fl_y": camera_params["camera_matrix"]["data"][4],
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "cx": camera_params["camera_matrix"]["data"][2],
            "cy": camera_params["camera_matrix"]["data"][5],
            "w": camera_params["image_width"],
            "h": camera_params["image_height"],
            "aabb_scale": 4,
        }
    elif cfg.dataset == "SevenScenes":
        params = {
            "fl_x": 585,
            "fl_y": 585,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "cx": 320,
            "cy": 240,
            "w": 640,
            "h": 480,
            "aabb_scale": 1,
        }
    params["camera_angle_x"] = np.arctan(params["w"] / (params["fl_x"] * 2)) * 2
    params["camera_angle_y"] = np.arctan(params["h"] / (params["fl_y"] * 2)) * 2

    scene_path = dataset_path / cfg.scene
    split_names = cfg.split
    split_paths = [scene_path / split for split in split_names]
    seq_paths = sorted(
        [
            split_path / seq
            for split_path in split_paths
            for seq in next(os.walk(split_path))[1]
        ]
    )
    frames = []
    for seq_path in tqdm(seq_paths):
        if cfg.dataset == "AmbiguousReloc":
            poses_path = seq_path / f"poses_{seq_path.stem}.txt"
            _, img_ids, positions, rotmats = read_poses(poses_path, cfg.dataset)
            img_paths = [
                seq_path / f"rgb_matched/frame-color-{str(img_id).zfill(4)}.png"
                for img_id in img_ids
            ]
        elif cfg.dataset == "SevenScenes":
            img_paths = sorted(seq_path.glob("*.color.png"))
            pose_paths = [
                Path(str(img_path).replace("color.png", "pose.txt"))
                for img_path in img_paths
            ]
            positions = []
            rotmats = []
            for pose_path in pose_paths:
                position, rotmat = read_tfmat(pose_path)
                positions.append(position)
                rotmats.append(rotmat)
        else:
            raise ValueError("Invalid dataset name.")

        frames.extend(
            [
                get_frame(img_path, position, rotmat, seq_path)
                for img_path, position, rotmat in zip(img_paths, positions, rotmats)
            ]
        )
    params["frames"] = frames

    with open(
        scene_path
        / f"transforms{f'_{split_names[0]}' if len(split_names) == 1 else ''}.json",
        "w",
    ) as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
