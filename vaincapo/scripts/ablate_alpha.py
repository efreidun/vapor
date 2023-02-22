"""Script for evaluting the pipeline."""

from pathlib import Path
from types import SimpleNamespace
import yaml

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from vaincapo.data import SketchUpCircular
from vaincapo.models import Encoder, PoseMap
from vaincapo.inference import forward_pass
from vaincapo.read_write import (
    read_scene_dims,
    compute_scene_dims,
)
from vaincapo.evaluation import evaluate_recall


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate camera pose posterior inference pipeline."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("alpha", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0]]
    cfg = SimpleNamespace(**config)
    recall_min_samples = cfg.num_samples // 10

    run_path = Path.home() / "code/vaincapo/runs" / cfg.run
    with open(run_path / "config.yaml") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg = SimpleNamespace(**train_config)

    torch.set_grad_enabled(False)
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / train_cfg.dataset / train_cfg.sequence
    try:
        scene_dims = read_scene_dims(scene_path)
    except FileNotFoundError:
        scene_dims = compute_scene_dims(scene_path, train_cfg.dataset)
    dataset_cfg = {
        "image_size": train_cfg.image_size,
        "mode": train_cfg.image_mode,
        "crop": train_cfg.image_crop,
        "gauss_kernel": train_cfg.gauss_kernel,
        "gauss_sigma": train_cfg.gauss_sigma,
        "jitter_brightness": train_cfg.jitter_brightness,
        "jitter_contrast": train_cfg.jitter_contrast,
        "jitter_saturation": train_cfg.jitter_saturation,
        "jitter_hue": train_cfg.jitter_hue,
    }
    valid_set = SketchUpCircular(scene_path, "valid", **dataset_cfg)

    if cfg.epoch is None:
        encoder_path = sorted(run_path.glob("encoder_*.pth"))[-1]
        posemap_path = sorted(run_path.glob("posemap_*.pth"))[-1]
    else:
        encoder_path = run_path / f"encoder_{str(cfg.epoch).zfill(3)}.pth"
        posemap_path = run_path / f"posemap_{str(cfg.epoch).zfill(3)}.pth"

    encoder = Encoder(train_cfg.latent_dim, train_cfg.backbone)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    posemap = PoseMap(train_cfg.latent_dim, train_cfg.map_depth, train_cfg.map_breadth)
    posemap.load_state_dict(torch.load(posemap_path, map_location=device))
    encoder.to(device)
    posemap.to(device)
    encoder.eval()
    posemap.eval()

    dataloader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size or len(valid_set),
        num_workers=cfg.num_workers,
    )
    recalls = []
    for _ in tqdm(range(cfg.num_trials)):
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
            names.extend(batch[2])
        tra = torch.cat(tras)
        tra_hat = torch.cat(tra_hats)
        rot = torch.cat(rots)
        rot_hat = torch.cat(rot_hats)

        recalls.append(
            evaluate_recall(
                tra,
                tra_hat,
                rot,
                rot_hat,
                recall_thresholds,
                recall_min_samples,
            )
        )

    recalls = np.array(recalls)
    mean = np.mean(recalls, axis=0)
    std = np.std(recalls, axis=0)
    with open(cfg.output, "a") as f:
        f.write(cfg.alpha + " ")
        f.write(" ".join([f"{rec}" for rec in mean]) + " ")
        f.write(" ".join([f"{rec}" for rec in std]) + "\n")


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
