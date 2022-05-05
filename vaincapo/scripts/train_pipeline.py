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

from vaincapo.data import AmbiguousImages
from vaincapo.models import Encoder, PoseMap
from vaincapo.utils import read_scene_dims, schedule_warmup
from vaincapo.inference import forward_pass
from vaincapo.losses import chordal_to_geodesic


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Train camera pose posterior inference pipeline."
    )
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--load_encoder", type=str, default=None)
    parser.add_argument("--load_posemap", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--top_percent", type=float, default=0.2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--tra_weight", type=float, default=1)
    parser.add_argument("--rot_weight", type=float, default=1)
    parser.add_argument("--wta_weight", type=float, default=1)
    parser.add_argument("--kld_warmup_start", type=int, default=0)
    parser.add_argument("--kld_warmup_period", type=int, default=0)
    parser.add_argument("--kld_max_weight", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    wandb.init(
        project="vaincapo_pipeline",
        entity="efreidun",
        config=config,
    )
    config["datetime"] = str(datetime.now())
    config["run_name"] = wandb.run.name
    config["num_winners"] = int(config["top_percent"] * config["num_samples"])
    cfg = SimpleNamespace(**config)
    runs_path = Path.home() / "code" / "vaincapo" / "runs"
    run_path = runs_path / cfg.run_name
    run_path.mkdir()
    with open(run_path / "config.yml", "w") as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_path = Path.home() / "data" / "Ambiguous_ReLoc_Dataset" / cfg.sequence
    scene_dims = read_scene_dims(scene_path)
    train_set = AmbiguousImages(scene_path / "train/seq00", cfg.image_size, cfg.augment)
    valid_set = AmbiguousImages(scene_path / "test/seq01", cfg.image_size, cfg.augment)

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

    encoder = Encoder(cfg.latent_dim)
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

    optimizer = Adam(posemap.parameters(), lr=cfg.lr)

    for epoch in tqdm(range(cfg.epochs)):
        kld_weight = schedule_warmup(
            epoch, cfg.kld_max_weight, cfg.kld_warmup_start, cfg.kld_warmup_period
        )
        encoder.train()
        posemap.train()
        for i, batch in enumerate(train_loader):
            wta_loss, kld_loss, tra_loss, rot_loss = forward_pass(
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

            wandb.log(
                {
                    "epoch.step": epoch + i / len(train_loader),
                    "train_loss": loss.item(),
                    "train_wta_loss": wta_loss.item(),
                    "train_kld_loss": kld_loss.item(),
                    "train_tra_loss": tra_loss.item(),
                    "train_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                }
            )

        encoder.eval()
        posemap.eval()
        with torch.no_grad():
            losses = torch.vstack(
                [
                    torch.stack(
                        forward_pass(
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
                    )
                    for batch in valid_loader
                ]
            )
            wta_loss, kld_loss, tra_loss, rot_loss = torch.mean(losses, dim=0)
            loss = cfg.wta_weight * wta_loss + kld_weight * kld_loss

            wandb.log(
                {
                    "epoch.step": epoch + 1,
                    "valid_loss": loss.item(),
                    "valid_wta_loss": wta_loss.item(),
                    "valid_kld_loss": kld_loss.item(),
                    "valid_tra_loss": tra_loss.item(),
                    "valid_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                }
            )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
