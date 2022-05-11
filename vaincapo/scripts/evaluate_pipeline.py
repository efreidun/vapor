"""Script for evaluting the pipeline."""

from pathlib import Path
from types import SimpleNamespace
import yaml

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from vaincapo.data import AmbiguousImages
from vaincapo.models import Encoder, PoseMap
from vaincapo.inference import forward_pass
from vaincapo.utils import read_scene_dims, rotmat_to_quat
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
        description="Train camera pose posterior inference pipeline."
    )
    parser.add_argument("--run", type=str)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0], [1.0, 60.0]]
    kde_gaussian_sigmas = np.linspace(0.01, 0.50, num=100, endpoint=True)
    kde_bingham_lambdas = np.linspace(200.0, 300.0, num=100, endpoint=True)
    recall_min_samples = [1, 25, 50, 100, 150, 200, 300, 400, 500]

    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code" / "vaincapo"
    run_path = base_path / "runs" / cfg.run
    with open(run_path / "config.yaml") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg = SimpleNamespace(**train_config)

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / "Ambiguous_ReLoc_Dataset" / train_cfg.sequence
    scene_dims = read_scene_dims(scene_path)
    train_set = AmbiguousImages(
        scene_path / "train/seq00", train_cfg.image_size, train_cfg.augment
    )
    valid_set = AmbiguousImages(
        scene_path / "test/seq01", train_cfg.image_size, train_cfg.augment
    )

    if cfg.epoch is None:
        encoder_path = sorted(run_path.glob("encoder_*.pth"))[-1]
        posemap_path = sorted(run_path.glob("posemap_*.pth"))[-1]
    else:
        encoder_path = run_path / f"encoder_{cfg.epoch}.pth"
        posemap_path = run_path / f"posemap_{cfg.epoch}.pth"

    encoder = Encoder(train_cfg.latent_dim)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    posemap = PoseMap(train_cfg.latent_dim)
    posemap.load_state_dict(torch.load(posemap_path, map_location=device))
    encoder.to(device)
    posemap.to(device)
    encoder.eval()
    posemap.eval()

    recalls = []
    tra_log_likelihoods = []
    rot_log_likelihoods = []
    datasets = {"train": train_set, "valid": valid_set}
    for dataset in datasets.values():
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size or len(dataset),
            num_workers=cfg.num_workers,
        )
        tras = []
        rots = []
        tra_hats = []
        rot_hats = []
        for batch in dataloader:
            tra, rot, tra_hat, rot_hat, _, _, _, _, = forward_pass(
                encoder,
                posemap,
                batch,
                cfg.num_samples,
                train_cfg.num_winners,
                train_cfg.tra_weight,
                train_cfg.rot_weight,
                scene_dims,
                device,
            )
            tras.append(tra)
            rots.append(rot)
            tra_hats.append(tra_hat)
            rot_hats.append(rot_hat)
        tra = torch.cat(tras)
        tra_hat = torch.cat(tra_hats)
        rot = torch.cat(rots)
        rot_hat = torch.cat(rot_hats)
        rot_quat = torch.tensor(rotmat_to_quat(rot.cpu().numpy()))
        rot_quat_hat = torch.tensor(
            rotmat_to_quat(rot_hat.reshape(-1, 3, 3).cpu().numpy())
        ).reshape(*rot_hat.shape[:2], 4)
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

    recalls = np.transpose(np.array(recalls), (2, 0, 1))
    recall_matrix = [["rec_th", "samps"] + [str(val) for val in recall_min_samples]]
    for (tra_thr, rot_thr), thr_recalls in zip(recall_thresholds, recalls):
        for split, vals in zip(datasets.keys(), thr_recalls):
            recall_matrix.append(
                [f"{tra_thr}/{int(rot_thr)}", split] + [f"{val:.2f}" for val in vals]
            )
    recall_matrix = np.array(recall_matrix).T
    m, n = recall_matrix.shape

    loglik_matrix = [
        ["t_logl", "sigma"] + [f"{val:.2f}" for val in kde_gaussian_sigmas]
    ]
    for i, split in enumerate(datasets.keys()):
        loglik_matrix.append(
            ["t_logl", split] + [f"{val:.2f}" for val in tra_log_likelihoods[i]]
        )
    loglik_matrix.append(
        ["r_logl", "lambda"]
        + [f"{val:.2f}" if val < 1000 else f"{val:.1f}" for val in kde_bingham_lambdas]
    )
    for i, split in enumerate(datasets.keys()):
        loglik_matrix.append(
            ["r_logl", split] + [f"{val:.2f}" for val in rot_log_likelihoods[i]]
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
        f.write(
            f"{cfg.run} epoch {encoder_path.stem.split('_')[1]}"
            + f" with {cfg.num_samples} samples\n"
        )
        for row in results_matrix:
            f.write("\t".join([val.ljust(6) for val in row]) + "\n")


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
