"""Script for visualing the latent space."""

from pathlib import Path
from types import SimpleNamespace
import yaml

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from vaincapo.data import (
    AmbiguousReloc,
    SevenScenes,
    CambridgeLandmarks,
    SketchUpCircular,
)
from vaincapo.models import Encoder
from vaincapo.plotting import plot_latent
from vaincapo.utils import rotmat_to_quat


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize latent encodings of train and validation sets."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    run_path = Path.home() / "code/vaincapo/runs" / cfg.run
    with open(run_path / "config.yaml") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg = SimpleNamespace(**train_config)

    torch.set_grad_enabled(False)
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / train_cfg.dataset / train_cfg.sequence
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
    if train_cfg.dataset == "AmbiguousReloc":
        DataSet = AmbiguousReloc
    elif train_cfg.dataset == "SevenScenes":
        DataSet = SevenScenes
    elif train_cfg.dataset == "CambridgeLandmarks":
        DataSet = CambridgeLandmarks
    elif train_cfg.dataset == "SketchUpCircular":
        pass
    else:
        raise ValueError("Invalid dataset.")

    if train_cfg.dataset == "SketchUpCircular":
        train_set = SketchUpCircular(scene_path, "train", **dataset_cfg)
        valid_set = SketchUpCircular(scene_path, "valid", **dataset_cfg)
    else:
        train_set = DataSet(scene_path / "train", **dataset_cfg)
        valid_set = DataSet(scene_path / "test", **dataset_cfg)

    if cfg.epoch is None:
        encoder_path = sorted(run_path.glob("encoder_*.pth"))[-1]
    else:
        encoder_path = run_path / f"encoder_{str(cfg.epoch).zfill(3)}.pth"

    encoder = Encoder(train_cfg.latent_dim, train_cfg.backbone)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    split_sets = {"train": train_set, "valid": valid_set}
    tras_splits = []
    quats_splits = []
    codes_splits = []
    for split_set in split_sets.values():
        dataloader = DataLoader(
            split_set,
            batch_size=cfg.batch_size or len(split_set),
            num_workers=cfg.num_workers,
        )
        tras = []
        rots = []
        lat_mus = []
        for batch in dataloader:
            image = batch[0].to(device)
            pose = batch[1].to(device)
            tra = pose[:, :3]
            rot = pose[:, 3:].reshape(-1, 3, 3)
            lat_mu, _ = encoder(image)

            tras.append(tra)
            rots.append(rot)
            lat_mus.append(lat_mu)
        tras = torch.cat(tras).cpu().numpy()
        quats = rotmat_to_quat(torch.cat(rots).cpu().numpy())
        lat_mus = torch.cat(lat_mus).cpu().numpy()

        codes = TSNE(n_components=2, init="random").fit_transform(lat_mus)

        tras_splits.append(tras)
        quats_splits.append(quats)
        codes_splits.append(codes)

    plot_latent(
        *tras_splits,
        *quats_splits,
        *codes_splits,
        scene_path,
        cfg.run,
        run_path / "latents.png",
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
