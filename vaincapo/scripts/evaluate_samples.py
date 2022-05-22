"""Script that evaluates saved samples."""

from pathlib import Path
from types import SimpleNamespace

import argparse
import numpy as np
import torch
from tqdm import tqdm

from vaincapo.read_write import write_sample_transforms, write_metrics
from vaincapo.utils import quat_to_rotmat
from vaincapo.evaluation import (
    evaluate_tras_likelihood,
    evaluate_rots_likelihood,
    evaluate_recall,
)


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate camera pose posterior samples."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--num_renders", type=int, default=10)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0], [1.0, 60.0]]
    kde_gaussian_sigmas = np.linspace(0.01, 0.50, num=100, endpoint=True)
    kde_bingham_lambdas = np.linspace(100.0, 400.0, num=100, endpoint=True)
    recall_min_samples = [1, 5, 10, 15, 20, 25, 50, 75, 100, 200]

    cfg = SimpleNamespace(**config)
    run_path = Path.home() / "code/vaincapo/bingham_runs" / cfg.run
    scene = "_".join(cfg.run.split("_")[:-1])
    scene_path = Path.home() / "data/Ambiguous_ReLoc_Dataset" / scene

    recalls = []
    tra_log_likelihoods = []
    rot_log_likelihoods = []

    split_names = ("train", "valid")
    for split_name in split_names:
        split_path = run_path / f"{split_name}.npz"
        mixmod = dict(np.load(split_path))
        tra = torch.from_numpy(mixmod["tra_gt"])
        rot_quat = torch.from_numpy(mixmod["rot_gt"])
        tra_hat = torch.from_numpy(mixmod["tra_samples"])
        rot_quat_hat = torch.from_numpy(mixmod["rot_samples"])
        names = mixmod["names"]

        num_queries, num_samples, _ = tra_hat.shape
        rot = quat_to_rotmat(rot_quat)
        rot_hat = quat_to_rotmat(rot_quat_hat.reshape(-1, 4)).reshape(
            num_queries, num_samples, 3, 3
        )

        recalls.append(
            [
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
        )
        tra_log_likelihoods.append(
            [
                evaluate_tras_likelihood(tra, tra_hat, sigma).item()
                for sigma in tqdm(kde_gaussian_sigmas)
            ]
        )
        rot_log_likelihoods.append(
            [
                evaluate_rots_likelihood(rot_quat, rot_quat_hat, sigma).item()
                for sigma in tqdm(kde_bingham_lambdas)
            ]
        )

        if split_name == "valid":
            write_sample_transforms(
                tra_hat.cpu().numpy(),
                rot_hat.cpu().numpy(),
                names,
                cfg.num_renders,
                run_path,
                scene_path,
                split_name,
            )

    write_metrics(
        recalls,
        tra_log_likelihoods,
        rot_log_likelihoods,
        recall_thresholds,
        recall_min_samples,
        kde_gaussian_sigmas,
        kde_bingham_lambdas,
        split_names,
        f"{cfg.run} with {num_samples} samples",
        run_path,
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
