"""Script for visalizing the latent space of CVAEPoR."""

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from vapor.data import (
    AmbiguousReloc,
    CambridgeLandmarks,
    Rig,
    SevenScenes,
    SketchUpCircular,
    Synthetic,
)
from vapor.inference import cvaepor_test_forward_pass, cvaepor_train_forward_pass
from vapor.models import CVAEPoRBackbone, CVAEPoRDecoder, CVAEPoREncoder, CVAEPoRFlows
from vapor.plotting import plot_bluechairs_generation, plot_bluechairs_latent_3d
from vapor.read_write import read_scene_dims


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Visualize latent space of CVAEPoR.")
    parser.add_argument("run", type=str)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)

    run_path = Path.home() / "code/vapor/runs" / cfg.run
    with open(run_path / "config.yaml") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg = SimpleNamespace(**train_config)
    plots_path = run_path / "plots"
    scene = train_cfg.sequence
    dataset_path = Path.home() / "data" / train_cfg.dataset

    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / train_cfg.dataset / train_cfg.sequence
    scene_dims = read_scene_dims(scene_path)
    dataset_cfg = {
        "image_size": train_cfg.image_size,
        "mode": train_cfg.image_mode,
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
    elif cfg.dataset == "Synthetic":
        train_set = Synthetic(scene_path, "train")
        valid_set = Synthetic(scene_path, "valid")
    else:
        raise ValueError("Invalid dataset.")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    if cfg.epoch is None:
        encoder_path = sorted(run_path.glob("encoder_*.pth"))[-1]
        decoder_path = sorted(run_path.glob("decoder_*.pth"))[-1]
    else:
        print(cfg.epoch)
        encoder_path = run_path / f"encoder_{str(cfg.epoch).zfill(6)}.pth"
        decoder_path = run_path / f"decoder_{str(cfg.epoch).zfill(6)}.pth"
    encoder = CVAEPoREncoder(
        train_cfg.latent_dim,
        train_cfg.encoder_depth,
        train_cfg.encoder_breadth,
        train_cfg.conditional_encoder
        if "conditional_encoder" in train_cfg.__dict__
        else False,
    )
    decoder = CVAEPoRDecoder(
        train_cfg.latent_dim,
        train_cfg.decoder_depth,
        train_cfg.decoder_breadth,
        train_cfg.bottleneck_dim if "bottleneck_dim" in train_cfg.__dict__ else None,
    )
    backbone = CVAEPoRBackbone(train_cfg.backbone)
    encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location=device,
        )
    )
    decoder.load_state_dict(
        torch.load(
            decoder_path,
            map_location=device,
        )
    )
    if "freeze_backbone" not in train_cfg.__dict__ or not train_cfg.freeze_backbone:
        if cfg.epoch is None:
            backbone_path = sorted(run_path.glob("backbone_*.pth"))[-1]
        else:
            backbone_path = run_path / f"backbone_{str(cfg.epoch).zfill(6)}.pth"
        backbone.load_state_dict(
            torch.load(
                backbone_path,
                map_location=device,
            )
        )
    encoder.to(device)
    decoder.to(device)
    backbone.to(device)
    encoder.eval()
    decoder.eval()
    backbone.eval()
    if train_cfg.num_flows == 0:
        flows = None
    else:
        if cfg.epoch is None:
            flows_path = sorted(run_path.glob("flows_*.pth"))[-1]
        else:
            flows_path = run_path / f"flows_{str(cfg.epoch).zfill(6)}.pth"
        flows = CVAEPoRFlows(
            train_cfg.latent_dim, train_cfg.num_flows, train_cfg.flow_type
        )
        flows.load_state_dict(
            torch.load(
                flows_path,
                map_location=device,
            )
        )
        flows.to(device)
        flows.eval()

    with torch.no_grad():
        test_tras = []
        test_rots = []
        test_tra_hats = []
        test_rot_hats = []
        zps = []
        for batch in valid_loader:
            (
                tra,
                rot,
                tra_hat,
                rot_hat,
                _,
                _,
                _,
                zp,
            ) = cvaepor_test_forward_pass(
                decoder,
                backbone,
                batch,
                cfg.num_samples,
                train_cfg.tra_weight,
                train_cfg.rot_weight,
                scene_dims,
                device,
            )
            test_tras.append(tra)
            test_rots.append(rot)
            test_tra_hats.append(tra_hat)
            test_rot_hats.append(rot_hat)
            zps.append(zp)
        test_tra = torch.cat(test_tras).numpy(force=True)  # (N, 3)
        test_tra_hat = torch.cat(test_tra_hats).numpy(force=True)  # (N, M, 3)
        test_rot = torch.cat(test_rots).numpy(force=True)  # (N, 3, 3)
        test_rot_hat = torch.cat(test_rot_hats).numpy(force=True)  # (N, M, 3, 3)
        zp = torch.cat(zps).numpy(force=True)  # (N, M, L)

        tras = []
        rots = []
        tra_hats = []
        rot_hats = []
        z0s = []
        zs = []
        names = []
        for batch in valid_loader:
            (
                tra,
                rot,
                tra_hat,
                rot_hat,
                _,
                _,
                _,
                _,
                _,
                _,
                z0,
                z,
                _,
                _,
            ) = cvaepor_train_forward_pass(
                encoder,
                decoder,
                backbone,
                flows,
                batch,
                cfg.num_samples,
                train_cfg.tra_weight,
                train_cfg.rot_weight,
                scene_dims,
                device,
            )
            tras.append(tra)
            rots.append(rot)
            tra_hats.append(tra_hat)
            rot_hats.append(rot_hat)
            z0s.append(z0)
            zs.append(z)
            names.extend(batch[2])
        tra = torch.cat(tras).numpy(force=True)  # (N, 3)
        tra_hat = torch.cat(tra_hats).numpy(force=True)  # (N, M, 3)
        rot = torch.cat(rots).numpy(force=True)  # (N, 3, 3)
        rot_hat = torch.cat(rot_hats).numpy(force=True)  # (N, M, 3, 3)
        z0 = torch.cat(z0s).numpy(force=True)  # (N, M, L)
        z = torch.cat(zs).numpy(force=True)  # (N, M, L)

    valid_set_mode_idcs = [111, 13, 36, 46, 57, 81, 96]  # O

    names_with_split = ["test/" + names[i] for i in valid_set_mode_idcs]  # O
    plot_path = plots_path / "modes.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    query_file_names = [
        scene + "/" + name_with_split for name_with_split in names_with_split
    ]  # O
    q = 0  # which chair to predict posterior for
    plot_bluechairs_generation(
        [
            np.array(Image.open(dataset_path / query_file_name))
            for query_file_name in query_file_names
        ],  # O
        test_tra_hat[valid_set_mode_idcs[q]],  # (M, 3)
        test_rot_hat[valid_set_mode_idcs[q]],  # (M, 3, 3)
        tra[valid_set_mode_idcs],  # (O, 3)
        zp[valid_set_mode_idcs[q]],  # (M, L)
        scene_path,
        1,
        plot_path,
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
