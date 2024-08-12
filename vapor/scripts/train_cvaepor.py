"""Script for training the pipeline."""

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
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
from vapor.evaluation import (
    evaluate_recall,
    evaluate_rots_likelihood,
    evaluate_tras_likelihood,
)
from vapor.inference import cvaepor_test_forward_pass, cvaepor_train_forward_pass
from vapor.losses import chordal_to_geodesic, euclidean_dist, geodesic_dist
from vapor.models import CVAEPoRBackbone, CVAEPoRDecoder, CVAEPoREncoder, CVAEPoRFlows
from vapor.read_write import compute_scene_dims, read_scene_dims
from vapor.utils import average_pose, rotmat_to_quat, schedule_warmup, visualize_latent


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
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)
    parser.add_argument("--scheduler_step", type=int, default=50)
    parser.add_argument("--scheduler_end", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--half_image", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_mode", type=str, default="ceiling_train")
    parser.add_argument("--image_crop", type=float, default=None)
    parser.add_argument("--gauss_kernel", type=int, default=3)
    parser.add_argument("--gauss_sigma", type=float, nargs="*", default=(0.05, 5.0))
    parser.add_argument("--jitter_brightness", type=float, default=0.2)
    parser.add_argument("--jitter_contrast", type=float, default=0.2)
    parser.add_argument("--jitter_saturation", type=float, default=0.2)
    parser.add_argument("--jitter_hue", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--tra_gauss_noise", type=float, default=None)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--encoder_depth", type=int, default=5)
    parser.add_argument("--encoder_breadth", type=int, default=128)
    parser.add_argument("--decoder_depth", type=int, default=5)
    parser.add_argument("--decoder_breadth", type=int, default=128)
    parser.add_argument("--bottleneck_dim", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--tra_weight", type=float, default=5)
    parser.add_argument("--rot_weight", type=float, default=2)
    parser.add_argument("--rec_weight", type=float, default=1)
    parser.add_argument("--kld_warmup_start", type=int, default=0)
    parser.add_argument("--kld_warmup_period", type=int, default=0)
    parser.add_argument("--kld_max_weight", type=float, default=0.3)
    parser.add_argument("--kde_gaussian_sigma", type=float, default=0.1)
    parser.add_argument("--kde_bingham_lambda", type=float, default=40.0)
    parser.add_argument("--recall_min_samples", type=int, default=100)
    parser.add_argument("--num_flows", type=int, default=10)
    parser.add_argument("--flow_type", type=str, default="planar")
    parser.add_argument("--conditional_encoder", action="store_true")
    parser.add_argument("--lens", action="store_true")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    recall_thresholds = [
        [0.1, 10.0],
        [0.2, 15.0],
        [0.3, 20.0],
        [1.0, 60.0],
        [0.1, 180.0],
        [20.0, 10.0],
    ]
    base_path = Path.home() / "code/vapor"
    config["recall_min_samples"] = 50 if config["dataset"] == "Rig" else 100
    wandb.init(
        project="vapor",
        config=config,
        dir=str(base_path),
        # mode="disabled",
    )
    config["datetime"] = str(datetime.now())
    config["run_name"] = wandb.run.name
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
        "half_image": cfg.half_image,
        "crop": cfg.image_crop,
        "gauss_kernel": cfg.gauss_kernel,
        "gauss_sigma": cfg.gauss_sigma,
        "jitter_brightness": cfg.jitter_brightness,
        "jitter_contrast": cfg.jitter_contrast,
        "jitter_saturation": cfg.jitter_saturation,
        "jitter_hue": cfg.jitter_hue,
    }
    valid_dataset_cfg = {
        "image_size": cfg.image_size,
        "mode": "posenet_plusplus" if cfg.image_mode == "posenet" else cfg.image_mode,
        "crop": cfg.image_crop,
        "gauss_kernel": None,
        "gauss_sigma": None,
        "jitter_brightness": None,
        "jitter_contrast": None,
        "jitter_saturation": None,
        "jitter_hue": None,
    }
    if cfg.dataset in ("AmbiguousReloc", "SketchUpCircular"):
        dataset_cfg["half_image"] = cfg.half_image
    if cfg.dataset in ("AmbiguousReloc", "Rig"):
        dataset_cfg["tra_gauss_noise"] = cfg.tra_gauss_noise
        dataset_cfg["scene_dims"] = scene_dims
        valid_dataset_cfg["tra_gauss_noise"] = None
        valid_dataset_cfg["scene_dims"] = None
    if cfg.dataset == "AmbiguousReloc":
        dataset_cfg["lens"] = cfg.lens
        valid_dataset_cfg["lens"] = False
        train_set = AmbiguousReloc(scene_path / "train", **dataset_cfg)
        valid_set = AmbiguousReloc(scene_path / "test", **valid_dataset_cfg)
    elif cfg.dataset == "SevenScenes":
        train_set = SevenScenes(scene_path / "train", **dataset_cfg)
        valid_set = SevenScenes(scene_path / "test", **valid_dataset_cfg)
    elif cfg.dataset == "CambridgeLandmarks":
        train_set = CambridgeLandmarks(scene_path / "dataset_train.txt", **dataset_cfg)
        valid_set = CambridgeLandmarks(
            scene_path / "dataset_test.txt", **valid_dataset_cfg
        )
    elif cfg.dataset == "SketchUpCircular":
        train_set = SketchUpCircular(scene_path, "train", **dataset_cfg)
        valid_set = SketchUpCircular(scene_path, "valid", **valid_dataset_cfg)
    elif cfg.dataset == "Rig":
        train_set = Rig(scene_path / "train", **dataset_cfg)
        valid_set = Rig(scene_path / "test", **valid_dataset_cfg)
    elif cfg.dataset == "Synthetic":
        train_set = Synthetic(scene_path, "train")
        valid_set = Synthetic(scene_path, "valid")
    else:
        raise ValueError("Invalid dataset.")

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
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    encoder = CVAEPoREncoder(
        cfg.latent_dim, cfg.encoder_depth, cfg.encoder_breadth, cfg.conditional_encoder
    )
    decoder = CVAEPoRDecoder(
        cfg.latent_dim, cfg.decoder_depth, cfg.decoder_breadth, cfg.bottleneck_dim
    )
    backbone = CVAEPoRBackbone(cfg.backbone)
    if cfg.load_checkpoint is not None:
        encoder.load_state_dict(
            torch.load(
                sorted((runs_path / cfg.load_checkpoint).glob("encoder_*.pth"))[-1],
                map_location=device,
            )
        )
        decoder.load_state_dict(
            torch.load(
                sorted((runs_path / cfg.load_checkpoint).glob("decoder_*.pth"))[-1],
                map_location=device,
            )
        )
        if not cfg.freeze_backbone:
            backbone.load_state_dict(
                torch.load(
                    sorted((runs_path / cfg.load_checkpoint).glob("backbone_*.pth"))[-1],
                    map_location=device,
                )
            )
    encoder.to(device)
    decoder.to(device)
    backbone.to(device)
    backbone.eval()
    if cfg.num_flows == 0:
        flows = None
    else:
        flows = CVAEPoRFlows(cfg.latent_dim, cfg.num_flows, cfg.flow_type)
        if cfg.load_checkpoint is not None:
            flows.load_state_dict(
                torch.load(
                    sorted((runs_path / cfg.load_checkpoint).glob("flows_*.pth"))[-1],
                    map_location=device,
                )
            )
        flows.to(device)

    params_to_optimize = [
        {"params": encoder.parameters(), "weight_decay": cfg.weight_decay},
        {"params": decoder.parameters(), "weight_decay": cfg.weight_decay},
    ]
    if not cfg.freeze_backbone:
        params_to_optimize.append(
            {"params": backbone.parameters(), "weight_decay": cfg.weight_decay}
        )
    if flows is not None:
        params_to_optimize.append(
            {"params": flows.parameters(), "weight_decay": cfg.weight_decay}
        )
    optimizer = AdamW(
        params_to_optimize,
        lr=cfg.lr,
    )
    # scheduler = ExponentialLR(optimizer, cfg.scheduler_gamma)

    # epochs_digits = len(str(cfg.epochs))
    iters_digits = len(str(cfg.iters))
    # for epoch in tqdm(range(cfg.epochs)):
    iter_no = 0
    last_val_iter = -100
    last_train_log_iter = -100
    last_checkpoint_iter = 0
    last_train_fig_iter = -2000
    last_valid_fig_iter = -2000
    pbar = tqdm(total=cfg.iters)
    while iter_no < cfg.iters:
        kld_weight = schedule_warmup(
            iter_no, cfg.kld_max_weight, cfg.kld_warmup_start, cfg.kld_warmup_period
        )
        encoder.train()
        decoder.train()
        if flows is not None:
            flows.train()

        for i, batch in enumerate(train_loader):
            pbar.update(1)
            iter_no += 1
            (
                tra,
                rot,
                tra_hat,
                rot_hat,
                rec_loss,
                kld_loss,
                tra_loss,
                rot_loss,
                mu,
                logvar,
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
                cfg.tra_weight,
                cfg.rot_weight,
                scene_dims,
                device,
            )
            loss = cfg.rec_weight * rec_loss + kld_weight * kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iter_no - last_train_fig_iter > 2000:
            #     last_train_fig_iter = iter_no
            #     fig = visualize_latent(batch[0], z0, z, tra_hat, rot_hat, tra, rot, 4)
            #     wandb_log = {
            #         "iter": iter_no,
            #         "train_latent": fig,
            #     }
            #     wandb.log(wandb_log)

            if iter_no - last_train_log_iter > 100:
                last_train_log_iter = iter_no
                with torch.no_grad():
                    rot_quat = torch.tensor(rotmat_to_quat(rot.cpu().numpy()))
                    rot_quat_hat = torch.tensor(
                        rotmat_to_quat(rot_hat.reshape(-1, 3, 3).cpu().numpy())
                    ).reshape(*rot_hat.shape[:2], 4)
                    tra_log_likelihood = evaluate_tras_likelihood(
                        tra.float(), tra_hat.float(), cfg.kde_gaussian_sigma
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
                tra_hat_point, rot_hat_point = average_pose(tra_hat, rot_hat)
                wandb_log = {
                    # "epoch.step": epoch + i / len(train_loader),
                    "iter": iter_no,
                    "train_loss": loss.item(),
                    "train_rec_loss": rec_loss.item(),
                    "train_kld_loss": kld_loss.item(),
                    "train_tra_loss": tra_loss.item(),
                    "train_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                    "train_tra_log_likelihood": tra_log_likelihood.item(),
                    "train_rot_log_likelihood": rot_log_likelihood.item(),
                    "train_med_tra_loss": torch.median(
                        euclidean_dist(tra_hat_point[:, None, :], tra)
                    ).item(),
                    "train_med_rot_loss": torch.median(
                        geodesic_dist(rot_hat_point[:, None, :], rot, deg=True)
                    ).item(),
                    "train_mu": mu.item(),
                    "train_std": (torch.exp(logvar) ** 0.5).item(),
                }
                for j, (tra_thr, rot_thr) in enumerate(recall_thresholds):
                    wandb_log[f"train_recall_{tra_thr}m_{rot_thr}deg"] = recalls[j]
                wandb.log(wandb_log)

        # if (
        #     epoch != 0
        #     and epoch % cfg.scheduler_gamma == 0
        #     and epoch <= cfg.scheduler_end
        # ):
        #     scheduler.step()

        if iter_no - last_val_iter > 100:
            last_val_iter = iter_no
            encoder.eval()
            decoder.eval()
            if flows is not None:
                flows.eval()
            with torch.no_grad():
                test_tras = []
                test_rots = []
                test_tra_hats = []
                test_rot_hats = []
                test_losses = []
                for batch in valid_loader:
                    (
                        tra,
                        rot,
                        tra_hat,
                        rot_hat,
                        rec_loss,
                        tra_loss,
                        rot_loss,
                        _,
                    ) = cvaepor_test_forward_pass(
                        decoder,
                        backbone,
                        batch,
                        cfg.num_samples,
                        cfg.tra_weight,
                        cfg.rot_weight,
                        scene_dims,
                        device,
                    )
                    test_tras.append(tra)
                    test_rots.append(rot)
                    test_tra_hats.append(tra_hat)
                    test_rot_hats.append(rot_hat)
                    test_losses.append(torch.stack([rec_loss, tra_loss, rot_loss]))

                test_rec_loss, test_tra_loss, test_rot_loss = torch.mean(
                    torch.vstack(test_losses), dim=0
                )

                test_tra = torch.cat(test_tras)
                test_tra_hat = torch.cat(test_tra_hats)
                test_rot = torch.cat(test_rots)
                test_rot_hat = torch.cat(test_rot_hats)
                test_recalls = evaluate_recall(
                    test_tra,
                    test_tra_hat,
                    test_rot,
                    test_rot_hat,
                    recall_thresholds,
                    cfg.recall_min_samples,
                )
                test_tra_hat_point, test_rot_hat_point = average_pose(
                    test_tra_hat, test_rot_hat
                )

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
                        rec_loss,
                        kld_loss,
                        tra_loss,
                        rot_loss,
                        mu,
                        logvar,
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
                        cfg.tra_weight,
                        cfg.rot_weight,
                        scene_dims,
                        device,
                    )
                    tras.append(tra)
                    rots.append(rot)
                    tra_hats.append(tra_hat)
                    rot_hats.append(rot_hat)
                    losses.append(torch.stack([rec_loss, kld_loss, tra_loss, rot_loss]))

                # if iter_no - last_valid_fig_iter > 2000:
                #     last_valid_fig_iter = iter_no
                #     fig = visualize_latent(
                #         batch[0], z0, z, tra_hat, rot_hat, tra, rot, 4
                #     )
                #     wandb_log = {
                #         "iter": iter_no,
                #         "valid_latent": fig,
                #     }
                #     wandb.log(wandb_log)

                rec_loss, kld_loss, tra_loss, rot_loss = torch.mean(
                    torch.vstack(losses), dim=0
                )
                loss = cfg.rec_weight * rec_loss + kld_weight * kld_loss

                tra = torch.cat(tras)
                tra_hat = torch.cat(tra_hats)
                rot = torch.cat(rots)
                rot_hat = torch.cat(rot_hats)
                recalls = evaluate_recall(
                    tra,
                    tra_hat,
                    rot,
                    rot_hat,
                    recall_thresholds,
                    cfg.recall_min_samples,
                )
                tra_hat_point, rot_hat_point = average_pose(tra_hat, rot_hat)

                wandb_log = {
                    # "epoch.step": epoch + 1,
                    "iter": iter_no,
                    "valid_loss": loss.item(),
                    "valid_rec_loss": rec_loss.item(),
                    "valid_kld_loss": kld_loss.item(),
                    "valid_tra_loss": tra_loss.item(),
                    "valid_rot_loss": chordal_to_geodesic(rot_loss, deg=True).item(),
                    "valid_med_tra_loss": torch.median(
                        euclidean_dist(tra_hat_point[:, None, :], tra)
                    ).item(),
                    "valid_med_rot_loss": torch.median(
                        geodesic_dist(rot_hat_point[:, None, :], rot, deg=True)
                    ).item(),
                    "valid_mu": mu.item(),
                    "valid_std": (torch.exp(logvar) ** 0.5).item(),
                    "test_rec_loss": test_rec_loss.item(),
                    "test_tra_loss": test_tra_loss.item(),
                    "test_rot_loss": chordal_to_geodesic(
                        test_rot_loss, deg=True
                    ).item(),
                    "test_med_tra_loss": torch.median(
                        euclidean_dist(test_tra_hat_point[:, None, :], tra)
                    ).item(),
                    "test_med_rot_loss": torch.median(
                        geodesic_dist(test_rot_hat_point[:, None, :], rot, deg=True)
                    ).item(),
                }
                for j, (tra_thr, rot_thr) in enumerate(recall_thresholds):
                    wandb_log[f"valid_recall_{tra_thr}m_{rot_thr}deg"] = recalls[j]
                    wandb_log[f"test_recall_{tra_thr}m_{rot_thr}deg"] = test_recalls[j]
                wandb.log(wandb_log)

        # if (epoch + 1) == 500:
        # if (iter_no + 1) % (cfg.iters / 5) == 0:
        if iter_no - last_checkpoint_iter > 10000:
            last_checkpoint_iter = iter_no
            torch.save(
                encoder.state_dict(),
                run_path / f"encoder_{str(iter_no + 1).zfill(iters_digits)}.pth",
            )
            torch.save(
                decoder.state_dict(),
                run_path / f"decoder_{str(iter_no + 1).zfill(iters_digits)}.pth",
            )
            if not cfg.freeze_backbone:
                torch.save(
                    backbone.state_dict(),
                    run_path / f"backbone_{str(iter_no + 1).zfill(iters_digits)}.pth",
                )
            if flows is not None:
                torch.save(
                    flows.state_dict(),
                    run_path / f"flows_{str(iter_no + 1).zfill(iters_digits)}.pth",
                )

    torch.save(encoder.state_dict(), run_path / f"encoder_{cfg.iters}.pth")
    torch.save(decoder.state_dict(), run_path / f"decoder_{cfg.iters}.pth")
    if not cfg.freeze_backbone:
        torch.save(backbone.state_dict(), run_path / f"backbone_{cfg.iters}.pth")
    if flows is not None:
        torch.save(flows.state_dict(), run_path / f"flows_{cfg.iters}.pth")
    # wandb.save(str(run_path / f"encoder_{cfg.epochs}.pth"))
    # wandb.save(str(run_path / f"decoder_{cfg.epochs}.pth"))


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
