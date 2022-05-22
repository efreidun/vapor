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

from vaincapo.utils import read_poses, get_ingp_transform


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Write JSON file for a scene.")
    parser.add_argument("scene", type=str)
    parser.add_argument("--split", type=str, nargs="+")
    args = parser.parse_args()

    return vars(args)


def get_frame(
    seq_path: Path, img_id: int, position: np.ndarray, rotmat: np.ndarray
) -> dict:
    """Create a frame dictionary.

    Args:
        seq_path: path to the sequence that contains the image
        img_id: image ID read from poses file
        position: image position, shape (3,)
        rotmat: image c2w rotation matrix, shape (3, 3)

    Returns:
        dictionary with keys "file_path", "sharpness", "transform_matrix"
    """
    image_rel_path = f"rgb_matched/frame-color-{str(img_id).zfill(4)}.png"
    file_path = str(seq_path.relative_to(seq_path.parents[1]) / image_rel_path)

    image_path = seq_path / image_rel_path
    image = cv2.imread(str(image_path))
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
    dataset_path = Path.home() / "data/Ambiguous_ReLoc_Dataset"
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
    params["camera_angle_x"] = np.arctan(params["w"] / (params["fl_x"] * 2)) * 2
    params["camera_angle_y"] = np.arctan(params["h"] / (params["fl_y"] * 2)) * 2

    scene_path = dataset_path / cfg.scene
    split_names = cfg.split or ["train", "test"]
    split_paths = [scene_path / split for split in split_names]
    seq_paths = [
        split_path / seq
        for split_path in split_paths
        for seq in next(os.walk(split_path))[1]
    ]
    poses_paths = [seq_path / f"poses_{seq_path.stem}.txt" for seq_path in seq_paths]
    frames = []
    for seq_path, poses_path in tqdm(zip(seq_paths, poses_paths), total=len(seq_paths)):
        _, img_ids, positions, rotmats = read_poses(poses_path)
        frames.extend(
            [
                get_frame(seq_path, img_id, position, rotmat)
                for img_id, position, rotmat in zip(img_ids, positions, rotmats)
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
