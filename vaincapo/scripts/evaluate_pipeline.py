"""Script for evaluting the pipeline."""

from pathlib import Path
from types import SimpleNamespace
import yaml

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from vaincapo.data import (
    AmbiguousReloc,
    SevenScenes,
    CambridgeLandmarks,
    SketchUpCircular,
    Rig,
)
from vaincapo.models import Encoder, PoseMap
from vaincapo.inference import forward_pass
from vaincapo.utils import average_pose, rotmat_to_quat
from vaincapo.losses import euclidean_dist, geodesic_dist
from vaincapo.read_write import (
    write_metrics,
    write_sample_transforms,
    read_scene_dims,
    compute_scene_dims,
)
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
        description="Evaluate camera pose posterior inference pipeline."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_renders", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0], [1.0, 60.0]]
    # kde_gaussian_sigmas = np.linspace(0.01, 0.50, num=2, endpoint=True)
    # kde_bingham_lambdas = np.linspace(100.0, 400.0, num=2, endpoint=True)
    kde_gaussian_sigmas = [0.1]
    kde_bingham_lambdas = [40.0]
    cfg = SimpleNamespace(**config)
    recall_min_samples = [cfg.num_samples // 20]

    run_path = Path.home() / "code/vaincapo/runs" / cfg.run
    with open(run_path / "config.yaml") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    clean_train_config = {
        key: val["value"] for key, val in train_config.items() if type(val) == dict
    }
    for key, val in train_config.items():
        if type(val) != dict:
            clean_train_config[key] = val
    train_cfg = SimpleNamespace(**clean_train_config)
    if train_cfg.num_samples == 0:
        cfg.num_samples = 0
        cfg.num_renders = 1
        recall_min_samples = [1]
        print("Running evaluation in single-hypothesis mode.")

    torch.set_grad_enabled(False)
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / train_cfg.dataset / train_cfg.sequence
    try:
        scene_dims = read_scene_dims(scene_path)
    except FileNotFoundError:
        scene_dims = compute_scene_dims(scene_path, train_cfg.dataset)
    # dataset_cfg = {
    #     "image_size": train_cfg.image_size,
    #     "mode": train_cfg.image_mode,
    #     "crop": train_cfg.image_crop,
    #     "gauss_kernel": train_cfg.gauss_kernel,
    #     "gauss_sigma": train_cfg.gauss_sigma,
    #     "jitter_brightness": train_cfg.jitter_brightness,
    #     "jitter_contrast": train_cfg.jitter_contrast,
    #     "jitter_saturation": train_cfg.jitter_saturation,
    #     "jitter_hue": train_cfg.jitter_hue,
    # }
    dataset_cfg = {
        "image_size": train_cfg.image_size,
        "mode": "ceiling_test",
        "crop": train_cfg.image_crop,
        "gauss_kernel": None,
        "gauss_sigma": None,
        "jitter_brightness": None,
        "jitter_contrast": None,
        "jitter_saturation": None,
        "jitter_hue": None,
    }
    if train_cfg.dataset in ("AmbiguousReloc", "SketchUpCircular"):
        dataset_cfg["half_image"] = train_cfg.half_image
    if train_cfg.dataset == "AmbiguousReloc":
        train_set = AmbiguousReloc(scene_path / "train", **dataset_cfg)
        valid_set = AmbiguousReloc(scene_path / "test", **dataset_cfg)
    elif train_cfg.dataset == "SevenScenes":
        train_set = SevenScenes(scene_path / "train", **dataset_cfg)
        valid_set = SevenScenes(scene_path / "test", **dataset_cfg)
    elif train_cfg.dataset == "CambridgeLandmarks":
        train_set = CambridgeLandmarks(scene_path / "dataset_train.txt", **dataset_cfg)
        valid_set = CambridgeLandmarks(scene_path / "dataset_test.txt", **dataset_cfg)
    elif train_cfg.dataset == "SketchUpCircular":
        train_set = SketchUpCircular(scene_path, "train", **dataset_cfg)
        valid_set = SketchUpCircular(scene_path, "valid", **dataset_cfg)
    elif train_cfg.dataset == "Rig":
        train_set = Rig(scene_path / "train", **dataset_cfg)
        valid_set = Rig(scene_path / "test", **dataset_cfg)
    else:
        raise ValueError("Invalid dataset.")

    if cfg.epoch is None:
        encoder_path = sorted(run_path.glob("encoder_*.pth"))[-1]
        posemap_path = sorted(run_path.glob("posemap_*.pth"))[-1]
    else:
        encoder_path = run_path / f"encoder_{str(cfg.epoch).zfill(4)}.pth"
        posemap_path = run_path / f"posemap_{str(cfg.epoch).zfill(4)}.pth"

    encoder = Encoder(train_cfg.latent_dim, train_cfg.backbone)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    posemap = PoseMap(
        train_cfg.latent_dim,
        train_cfg.map_depth,
        train_cfg.map_breadth,
        train_cfg.map_sin_mu,
        train_cfg.map_sin_sigma,
    )
    posemap.load_state_dict(torch.load(posemap_path, map_location=device))
    encoder.to(device)
    posemap.to(device)
    encoder.eval()
    posemap.eval()

    recalls = []
    tra_log_likelihoods = []
    rot_log_likelihoods = []
    median_errors = []

    split_sets = {"train": train_set, "valid": valid_set}
    for split_name, split_set in split_sets.items():
        dataloader = DataLoader(
            split_set,
            batch_size=cfg.batch_size or len(split_set),
            num_workers=cfg.num_workers,
        )
        tras = []
        rots = []
        tra_hats = []
        rot_hats = []
        names = []
        for batch in dataloader:
            tra, rot, tra_hat, rot_hat, _, _, _, _, = forward_pass(
                encoder,
                posemap,
                batch,
                cfg.num_samples,
                int(train_cfg.top_percent * train_cfg.num_samples),
                train_cfg.tra_weight,
                train_cfg.rot_weight,
                scene_dims,
                device,
            )
            tras.append(tra)
            rots.append(rot)
            tra_hats.append(tra_hat)
            rot_hats.append(rot_hat)
            names.extend(batch[2])
        tra = torch.cat(tras)
        tra_hat = torch.cat(tra_hats)
        rot = torch.cat(rots)
        rot_hat = torch.cat(rot_hats)

        rot_quat = rotmat_to_quat(rot.cpu().numpy())
        rot_quat_hat = rotmat_to_quat(rot_hat.reshape(-1, 3, 3).cpu().numpy()).reshape(
            *rot_hat.shape[:2], 4
        )
        np.savez(
            run_path / f"{split_name}.npz",
            tra_gt=tra.cpu().numpy(),
            rot_gt=rot_quat,
            tra_samples=tra_hat.cpu().numpy(),
            rot_samples=rot_quat_hat,
            names=names,
        )
        rot_quat = torch.from_numpy(rot_quat)
        rot_quat_hat = torch.from_numpy(rot_quat_hat)

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
                evaluate_tras_likelihood(tra.float(), tra_hat, sigma).item()
                for sigma in tqdm(kde_gaussian_sigmas)
            ]
        )
        rot_log_likelihoods.append(
            [
                evaluate_rots_likelihood(rot_quat, rot_quat_hat, sigma).item()
                for sigma in tqdm(kde_bingham_lambdas)
            ]
        )

        tra_hat_point, rot_hat_point = average_pose(tra_hat, rot_hat)
        median_errors.append(
            [
                torch.median(euclidean_dist(tra_hat_point[:, None, :], tra)).item(),
                torch.median(
                    geodesic_dist(rot_hat_point[:, None, :], rot, deg=True)
                ).item(),
            ]
        )

        if train_cfg.dataset == "AmbiguousReloc" and split_name == "valid":
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
        median_errors,
        recalls,
        tra_log_likelihoods,
        rot_log_likelihoods,
        recall_thresholds,
        recall_min_samples,
        kde_gaussian_sigmas,
        kde_bingham_lambdas,
        split_sets.keys(),
        f"{cfg.run} epoch {int(encoder_path.stem.split('_')[1])}"
        + f" with {cfg.num_samples} samples",
        run_path,
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
