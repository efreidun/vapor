"""Prepare scene for caffe posenet implementation."""

import os
from pathlib import Path

import numpy as np
import argparse


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Prepare scene for Caffe PoseNet.")
    parser.add_argument("scene", type=str)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    scene_path = Path.home() / "data" / config["dataset"] / config["scene"]
    for split in ("train", "test"):
        split_path = scene_path / split
        seq_paths = [split_path / seq for seq in next(os.walk(split_path))[1]]
        poses = []
        for seq_path in seq_paths:
            with open(
                seq_path / f"poses_{seq_path.stem}.txt"
                if config["dataset"] == "AmbiguousReloc" else "poses.txt"
            ) as f:
                content = f.readlines()
            poses.append(
                np.array(
                    [
                        [
                            float(entry) for entry in line.strip().split(
                                ", " if config["dataset"] == "AmbiguousReloc" else " "
                            )
                        ]
                        for line in content
                    ],
                    dtype=np.float32,
                )
            )
        poses = np.concatenate(poses)
        with open(scene_path / f"dataset_{split}.txt", "w") as f:
            f.write(f"{config['dataset']} dataset {config['scene']}\n")
            f.write(f"ImageFile, Camera Position [X Y Z W P Q R]\n\n")
            if config["dataset"] == "AmbiguousReloc":
                for seq, frame, qw, qx, qy, qz, tx, ty, tz in poses:
                    f.write(
                        f"{split}/seq{str(int(seq)).zfill(2)}/rgb_matched"
                        + f"/frame-color-{str(int(frame)).zfill(4)}.png"
                        + f" {tx} {ty} {tz} {qw} {qx} {qy} {qz}\n"
                    )
            else:
                for seq, frame, tx, ty, tz, qw, qx, qy, qz in poses:
                    f.write(
                        f"{split}/{int(seq)}/images"
                        + f"/{int(frame):06}.jpg"
                        + f" {tx} {ty} {tz} {qw} {qx} {qy} {qz}\n"
                    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
