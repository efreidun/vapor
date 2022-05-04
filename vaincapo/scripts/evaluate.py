"""Script to evaluate a trained model."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vaincapo.utils import scale_trans, cont_to_rotmat, rotmat_to_quat
from vaincapo.evaluation import (
    evaluate_tras_likelihood,
    evaluate_rots_likelihood,
    evaluate_tras_recall,
    evaluate_rots_recall,
    evaluate_recall,
)
from vaincapo.models import Encoder, PoseMap
from vaincapo.data import AmbiguousImages


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 16
    image_size = 64
    num_latent_samples = 1000
    sequence = "blue_chairs"
    scene_path = Path.home() / "data" / "Ambiguous_ReLoc_Dataset" / sequence
    saved_weights_path = Path.home() / "code/vaincapo/runs" / "old"
    encoder = Encoder(latent_dim)
    encoder.load_state_dict(
        torch.load(saved_weights_path / "encoder.pth", map_location=device)
    )
    posemap = PoseMap(latent_dim)
    posemap.load_state_dict(
        torch.load(saved_weights_path / f"posemap.pth", map_location=device)
    )
    encoder.to(device)
    posemap.to(device)
    encoder.eval()
    posemap.eval()

    augment = False
    dataset_mean = [0.5417, 0.5459, 0.4945]
    dataset_std = [0.1745, 0.1580, 0.1627]
    train_set = AmbiguousImages(
        scene_path / "train" / "seq00", image_size, augment, dataset_mean, dataset_std
    )
    test_set = AmbiguousImages(
        scene_path / "test" / "seq01", image_size, augment, dataset_mean, dataset_std
    )
    mins = [-1.135581, -0.763927, -0.058301]
    maxs = [0.195657, 0.464706, 0.058201]
    margins = [2, 2, 0.5]

    for dataset in (train_set, test_set):
        num_val_images = len(dataset)
        val_data = DataLoader(
            dataset,
            batch_size=num_val_images,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch in val_data:
                image = batch[0].to(device)
                pose = batch[1].to(device)

                latent_mu, latent_logvar = encoder(image)
                latent_std = torch.exp(0.5 * latent_logvar)
                eps = torch.randn(
                    (num_val_images, num_latent_samples, latent_dim), device=device
                )
                latent_sample = eps * latent_std.unsqueeze(1) + latent_mu.unsqueeze(1)
                tvec, rvec = posemap(latent_sample.flatten(end_dim=1))

                tra_hat = (
                    scale_trans(tvec, mins, maxs, margins)
                    .reshape(num_val_images, num_latent_samples, 3)
                    .cpu()
                )
                rot_hat = (
                    cont_to_rotmat(rvec)
                    .reshape(num_val_images, num_latent_samples, 3, 3)
                    .cpu()
                )
                break

        tra = pose[:, :3].cpu()
        rot = pose[:, 3:].reshape(-1, 3, 3).cpu()

        rot_quat = torch.tensor(rotmat_to_quat(rot.numpy()))
        rot_hat_quat = torch.tensor(
            rotmat_to_quat(rot_hat.reshape(-1, 3, 3).numpy())
        ).reshape(*rot_hat.shape[:2], 4)

        print("tra loglik", evaluate_tras_likelihood(tra, tra_hat, 0.1).item())
        print("rot loglik", evaluate_rots_likelihood(rot_quat, rot_hat_quat, 40).item())
        min_samples = 10
        print(
            "tra recall",
            evaluate_tras_recall(tra, tra_hat, [0.1, 0.2, 0.3], min_samples),
        )
        print(
            "rot recall",
            evaluate_rots_recall(rot, rot_hat, [10.0, 15.0, 20.0], min_samples),
        )
        print(
            "joint recall",
            evaluate_recall(
                tra,
                tra_hat,
                rot,
                rot_hat,
                [[0.1, 10.0], [0.2, 15.0], [0.3, 20.0]],
                min_samples,
            ),
        )


if __name__ == "__main__":
    main()
