"""Script for training the pipeline."""

from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import yaml

import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

from vaincapo.data import AmbiguousReloc, SevenScenes, CambridgeLandmarks
from vaincapo.models import Encoder, PoseMap
from vaincapo.read_write import read_scene_dims, compute_scene_dims
from vaincapo.utils import schedule_warmup, rotmat_to_quat
from vaincapo.inference import forward_pass
from vaincapo.losses import chordal_to_geodesic, euclidean_dist, geodesic_dist
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
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--load_encoder", type=str, default=None)
    parser.add_argument("--load_posemap", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_mode", type=str, default="resize")
    parser.add_argument("--image_crop", type=float, default=0.9)
    parser.add_argument("--gauss_kernel", type=int, default=3)
    parser.add_argument("--gauss_sigma", type=float, default=(0.1, 1.0))
    parser.add_argument("--jitter_brightness", type=float, default=0.05)
    parser.add_argument("--jitter_contrast", type=float, default=0.05)
    parser.add_argument("--jitter_saturation", type=float, default=0.05)
    parser.add_argument("--jitter_hue", type=float, default=0.05)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--top_percent", type=float, default=0.2)
    parser.add_argument("--tra_weight", type=float, default=10)
    parser.add_argument("--rot_weight", type=float, default=1)
    parser.add_argument("--wta_weight", type=float, default=1)
    parser.add_argument("--kld_warmup_start", type=int, default=50)
    parser.add_argument("--kld_warmup_period", type=int, default=50)
    parser.add_argument("--kld_max_weight", type=float, default=0.1)
    parser.add_argument("--kde_gaussian_sigma", type=float, default=0.1)
    parser.add_argument("--kde_bingham_lambda", type=float, default=40.0)
    parser.add_argument("--recall_min_samples", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0], [1.0, 60.0]]
    base_path = Path.home() / "code/vaincapo"
    wandb.init(
        project="vaincapo_pipeline",
        entity="efreidun",
        config=config,
        dir=str(base_path),
    )
    config["datetime"] = str(datetime.now())
    config["run_name"] = wandb.run.name
    config["num_winners"] = int(config["top_percent"] * config["num_samples"])
    cfg = SimpleNamespace(**config)
    runs_path = base_path / "runs"
    run_path = runs_path / cfg.run_name
    run_path.mkdir()
    with open(run_path / "config.yaml", "w") as f:
        yaml.dump(config, f)
    wandb.save(str(run_path / "config.yaml"))

    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / cfg.dataset / cfg.sequence
    try:
        scene_dims = read_scene_dims(scene_path)
    except FileNotFoundError:
        scene_dims = compute_scene_dims(scene_path, cfg.dataset)
    dataset_cfg = {
        "image_size": cfg.image_size,
        "mode": cfg.image_mode,
        "crop": cfg.image_crop,
        "gauss_kernel": cfg.gauss_kernel,
        "gauss_sigma": cfg.gauss_sigma,
        "jitter_brightness": cfg.jitter_brightness,
        "jitter_contrast": cfg.jitter_contrast,
        "jitter_saturation": cfg.jitter_saturation,
        "jitter_hue": cfg.jitter_hue,
    }
    if cfg.dataset == "AmbiguousReloc":
        DataSet = AmbiguousReloc
    elif cfg.dataset == "SevenScenes":
        DataSet = SevenScenes
    elif cfg.dataset == "CambridgeLandmarks":
        DataSet = CambridgeLandmarks
    else:
        raise ValueError("Invalid dataset.")
    train_set = DataSet(scene_path / "train", **dataset_cfg)
    valid_set = DataSet(scene_path / "test", **dataset_cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    encoder = Encoder(cfg.latent_dim, cfg.image_size)
    if cfg.load_encoder is not None:
        encoder.load_state_dict(
            torch.load(
                runs_path / cfg.load_encoder / "encoder.pth",
                map_location=device,
            )
        )
    posemap = PoseMap(cfg.latent_dim)
    if cfg.load_posemap is not None:
        posemap.load_state_dict(
            torch.load(
                runs_path / cfg.load_map / "posemap.pth",
                map_location=device,
            )
        )
    encoder.to(device)
    posemap.to(device)
    wandb.watch(encoder)
    wandb.watch(posemap)

    optimizer = Adam(
        list(encoder.parameters()) + list(posemap.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    epochs_digits = len(str(cfg.epochs))
    for epoch in tqdm(range(cfg.epochs)):
        kld_weight = schedule_warmup(
            epoch, cfg.kld_max_weight, cfg.kld_warmup_start, cfg.kld_warmup_period
        )
        encoder.train()
        posemap.train()

        for i, batch in enumerate(train_loader):
            (
                tra,
                rot,
                tra_hat,
                rot_hat,
                wta_loss,
                kld_loss,
                tra_loss,
                rot_loss,
            ) = forward_pass(
                encoder,
                posemap,
                batch,
                cfg.num_samples,
                cfg.num_winners,
                cfg.tra_weight,
                cfg.rot_weight,
                scene_dims,
                device,
            )
            loss = cfg.wta_weight * wta_loss + kld_weight * kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rot_quat = torch.tensor(rotmat_to_quat(rot.cpu().numpy()))
                rot_quat_hat = torch.tensor(
                    rotmat_to_quat(rot_hat.reshape(-1, 3, 3).cpu().numpy())
                ).reshape(*rot_hat.shape[:2], 4)
                tra_log_likelihood = evaluate_tras_likelihood(
                    tra, tra_hat, cfg.kde_gaussian_sigma
                )
                rot_log_likelihood = evaluate_rots_likelihood(
                    rot_quat, rot_quat_hat, cfg.kde_bingham_lambda
                )
                recalls = evaluate_recall(
                    tra,
                    tra_hat,
                    rot,
                    rot_hat,
                    recall_thresholds,
                    cfg.recall_min_samples,
                )

            wandb_log = {
                "epoch.step": epoch + i / len(train_loader),
                "train_loss": loss.item(),
                "train_wta_loss": wta_loss.item(),
                "train_kld_loss": kld_loss.item(),
                "train_tra_loss": tra_loss.item(),
                "train_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                "train_tra_log_likelihood": tra_log_likelihood.item(),
                "train_rot_log_likelihood": rot_log_likelihood.item(),
                "train_med_tra_loss": torch.median(
                    torch.median(euclidean_dist(tra_hat, tra), dim=1)[0]
                ).item(),
                "train_med_rot_loss": torch.median(
                    torch.median(geodesic_dist(rot_hat, rot, deg=True), dim=1)[0]
                ).item(),
            }
            for j, (tra_thr, rot_thr) in enumerate(recall_thresholds):
                wandb_log[f"train_recall_{tra_thr}m_{rot_thr}deg"] = recalls[j]
            wandb.log(wandb_log)

        encoder.eval()
        posemap.eval()
        with torch.no_grad():
            tras = []
            rots = []
            tra_hats = []
            rot_hats = []
            losses = []
            for batch in valid_loader:
                (
                    tra,
                    rot,
                    tra_hat,
                    rot_hat,
                    wta_loss,
                    kld_loss,
                    tra_loss,
                    rot_loss,
                ) = forward_pass(
                    encoder,
                    posemap,
                    batch,
                    cfg.num_samples,
                    cfg.num_winners,
                    cfg.tra_weight,
                    cfg.rot_weight,
                    scene_dims,
                    device,
                )
                tras.append(tra)
                rots.append(rot)
                tra_hats.append(tra_hat)
                rot_hats.append(rot_hat)
                losses.append(torch.stack([wta_loss, kld_loss, tra_loss, rot_loss]))

            wta_loss, kld_loss, tra_loss, rot_loss = torch.mean(
                torch.vstack(losses), dim=0
            )
            loss = cfg.wta_weight * wta_loss + kld_weight * kld_loss

            tra = torch.cat(tras)
            tra_hat = torch.cat(tra_hats)
            rot = torch.cat(rots)
            rot_hat = torch.cat(rot_hats)
            rot_quat = torch.tensor(rotmat_to_quat(rot.cpu().numpy()))
            rot_quat_hat = torch.tensor(
                rotmat_to_quat(rot_hat.reshape(-1, 3, 3).cpu().numpy())
            ).reshape(*rot_hat.shape[:2], 4)
            tra_log_likelihood = evaluate_tras_likelihood(
                tra, tra_hat, cfg.kde_gaussian_sigma
            )
            rot_log_likelihood = evaluate_rots_likelihood(
                rot_quat, rot_quat_hat, cfg.kde_bingham_lambda
            )
            recalls = evaluate_recall(
                tra,
                tra_hat,
                rot,
                rot_hat,
                recall_thresholds,
                cfg.recall_min_samples,
            )
            wandb_log = {
                "epoch.step": epoch + 1,
                "valid_loss": loss.item(),
                "valid_wta_loss": wta_loss.item(),
                "valid_kld_loss": kld_loss.item(),
                "valid_tra_loss": tra_loss.item(),
                "valid_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                "valid_tra_log_likelihood": tra_log_likelihood.item(),
                "valid_rot_log_likelihood": rot_log_likelihood.item(),
                "valid_med_tra_loss": torch.median(
                    torch.median(euclidean_dist(tra_hat, tra), dim=1)[0]
                ).item(),
                "valid_med_rot_loss": torch.median(
                    torch.median(geodesic_dist(rot_hat, rot, deg=True), dim=1)[0]
                ).item(),
            }
            for j, (tra_thr, rot_thr) in enumerate(recall_thresholds):
                wandb_log[f"valid_recall_{tra_thr}m_{rot_thr}deg"] = recalls[j]
            wandb.log(wandb_log)

        if (epoch + 1) % 50 == 0:
            torch.save(
                encoder.state_dict(),
                run_path / f"encoder_{str(epoch + 1).zfill(epochs_digits)}.pth",
            )
            torch.save(
                posemap.state_dict(),
                run_path / f"posemap_{str(epoch + 1).zfill(epochs_digits)}.pth",
            )

    wandb.save(str(run_path / f"encoder_{cfg.epochs}.pth"))
    wandb.save(str(run_path / f"posemap_{cfg.epochs}.pth"))


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
