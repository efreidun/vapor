"""Script that evaluates saved samples from PoseNet."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import argparse
import torch
from tqdm import tqdm

from vaincapo.data import AmbiguousReloc
from vaincapo.utils import quat_to_rotmat
from vaincapo.evaluation import evaluate_recall
from vaincapo.read_write import write_sample_transforms


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate camera pose posterior samples."
    )
    parser.add_argument("path", type=str)
    parser.add_argument("scene", type=str)
    parser.add_argument("--num_renders", type=int, default=10)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0], [1.0, 60.0]]
    recall_min_samples = [100]

    cfg = SimpleNamespace(**config)

    path = Path(cfg.path)
    if path.is_file():
        mixmod = dict(np.load(cfg.path))
        tra = torch.from_numpy(mixmod["tra_gt"])
        rot_quat = torch.from_numpy(mixmod["rot_gt"])
        tra_hat = torch.from_numpy(mixmod["tra_samples"])
        rot_quat_hat = torch.from_numpy(mixmod["rot_samples"])
        run_path = path.parent / "posenet"
    else:
        paths = path.glob("bayesian_posenet_*.npz")
        tra = []
        tra_hat = []
        rot_quat = []
        rot_quat_hat = []
        for pth in paths:
            mixmod = dict(np.load(pth))
            tra.append(torch.from_numpy(mixmod["tra_gt"]))
            rot_quat.append(torch.from_numpy(mixmod["rot_gt"]))
            tra_hat.append(torch.from_numpy(mixmod["tra_samples"]))
            rot_quat_hat.append(torch.from_numpy(mixmod["rot_samples"]))
        tra = tra[0]
        rot_quat = rot_quat[0]
        tra_hat = torch.cat(tra_hat, dim=1)
        rot_quat_hat = torch.cat(rot_quat_hat, dim=1)
        run_path = path / "bayesian_posenet"
    run_path.mkdir(exist_ok=True)

    num_queries, num_samples, _ = tra_hat.shape
    if num_samples == 1:
        cfg.num_renders = 1
        recall_min_samples = [1]
        print("Running evaluation in single-hypothesis mode.")

    rot = quat_to_rotmat(rot_quat)
    rot_hat = quat_to_rotmat(rot_quat_hat.reshape(-1, 4)).reshape(
        num_queries, num_samples, 3, 3
    )

    recall = [
        evaluate_recall(
            tra,
            tra_hat,
            rot,
            rot_hat,
            recall_thresholds,
            recall_min_sample,
        )
        for recall_min_sample in tqdm(recall_min_samples)
    ]

    print(recall)

    scene_path = Path.home() / "data/AmbiguousReloc" / cfg.scene
    valid_set = AmbiguousReloc(scene_path / "test", 64)

    names = []
    idcs = []
    for i in range(len(valid_set)):
        _, pose, name = valid_set[i]
        data_tra = pose[:3]
        data_rot = pose[3:].reshape(3, 3)
        tra_dist = torch.linalg.norm(tra - data_tra[None, :], dim=1)
        rot_dist = torch.linalg.norm(rot - data_rot[None, :, :], dim=(1, 2))
        dist = tra_dist + rot_dist
        idcs.append(torch.argmin(dist).item())
        names.append(name)

    tra = tra[idcs]
    tra_hat = tra_hat[idcs]
    rot = rot[idcs]
    rot_hat = rot_hat[idcs]

    write_sample_transforms(
        tra_hat.numpy(),
        rot_hat.numpy(),
        names,
        cfg.num_renders,
        run_path,
        scene_path,
        "valid",
    )

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
