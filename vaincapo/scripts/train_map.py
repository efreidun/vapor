from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose

from nerfcity.data import AmbiguousImages
from nerfcity.models import Encoder, PoseMap
from nerfcity.losses import kl_divergence


def canonical_to_pose(
    canonical_pose: torch.Tensor,
    mins,
    maxs,
    margins,
) -> torch.Tensor:
    """Convert posemap output which is confined in [0, 1] to metric space.

    Args:
        canonical_pose: pose confined in [0, 1], shape (N, 9),
        mins: minimum valid value for translations, shape (3,),
        maxs: maximum valid value for translations, shape (3,),
        margins: margins for translations, shape (3,),

    Returns:
        translation vectors, shape (N, 3),
        rotation matrices, shape (N, 3, 3)
    """
    assert len(canonical_pose.shape) == 2, "Wrong shape for canonical pose."
    canonical_trans = canonical_pose[:, :3]
    metric_trans = torch.hstack(
        tuple(
            canonical_trans[:, i : i + 1] * (maxs[i] - mins[i] + 2 * margins[i])
            + (mins[i] - margins[i])
            for i in range(3)
        )
    )
    canonical_rot = canonical_pose[:, 3:]
    a1 = canonical_rot[:, :3]
    a2 = canonical_rot[:, 3:]
    b1 = a1 / torch.norm(a1, dim=1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
    b3 = torch.cross(b1, b2)
    rotmat = torch.cat((b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)), dim=2)

    return metric_trans, rotmat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 30000
    latent_dim = 16
    image_size = 64
    batch_size = 16
    lr = 1e-4
    t_weight = 1
    r_weight = 1
    geo_weight = 1
    kld_warmup_start = 0
    kld_warmup_period = 0
    kld_max_weight = 0
    top_percent = 0.05
    num_latent_samples = 200
    top_samples = int(top_percent * num_latent_samples)

    wandb.init(
        project="map_training",
        entity="efreidun",
        config={
            "num_epochs": num_epochs,
            "latent_dim": latent_dim,
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "t_weight": t_weight,
            "r_weight": r_weight,
            "geo_weight": geo_weight,
            "kld_warmup_start": kld_warmup_start,
            "kld_warmup_period": kld_warmup_period,
            "kld_max_weight": kld_max_weight,
            "num_latent_samples": num_latent_samples,
            "top_percent": top_percent,
        },
    )

    # meeting_room
    # dataset_mean = [0.3396, 0.3422, 0.3201]
    # dataset_std = [0.3855, 0.3720, 0.3587]
    # blue_chairs
    dataset_mean = [0.5417, 0.5459, 0.4945]
    dataset_std = [0.1745, 0.1580, 0.1627]
    # dataset_mean = None
    # dataset_std = None
    sequence = "blue_chairs"
    scene_path = Path.home() / "data" / "Ambiguous_ReLoc_Dataset" / sequence
    augment = True
    train_set = AmbiguousImages(
        scene_path / "train" / "seq00", image_size, augment, dataset_mean, dataset_std
    )
    test_set = AmbiguousImages(
        scene_path / "test" / "seq01", image_size, augment, dataset_mean, dataset_std
    )
    dataset_mean = torch.tensor(dataset_mean, device=device)[:, None, None]
    dataset_std = torch.tensor(dataset_std, device=device)[:, None, None]
    # scene_path = Path.home() / "data" / "mnist"
    # transforms = Compose([Resize((image_size, image_size)), ToTensor()])
    # train_set = MNIST(root=scene_path, train=True, transform=transforms, download=True)
    # test_set = MNIST(root=scene_path, train=False, transform=transforms, download=False)

    # all_pixels = torch.cat(
    #     [
    #         dataset[i][0].unsqueeze(1)
    #         for dataset in (train_set, test_set)
    #         for i in tqdm(range(len(dataset)))
    #     ],
    #     dim=1,
    # )
    # dataset_mean = torch.mean(all_pixels, dim=(1, 2, 3))
    # dataset_std = torch.std(all_pixels, dim=(1, 2, 3))
    # print("mean", dataset_mean)
    # print("std", dataset_std)
    # assert 2 + 2 == 5

    train_data = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_data = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    margins = [2, 2, 0.5]
    mins = [-1.135581, -0.763927, -0.058301]
    maxs = [0.195657, 0.464706, 0.058201]
    # mins = [9999, 9999, 9999]
    # maxs = [-9999, -9999, -9999]
    # for dataset in (train_set, test_set):
    #     for i in tqdm(range(len(dataset))):
    #         position = train_set[i][1]
    #         for j in range(3):
    #             if position[j] > maxs[j]:
    #                 maxs[j] = position[j]
    #             if position[j] < mins[j]:
    #                 mins[j] = position[j]
    # print("mins", mins)
    # print("maxs", maxs)

    saved_weights_path = Path.home() / "code" / "nerfcity" / "saved_weights"
    encoder = Encoder(latent_dim)
    encoder.load_state_dict(
        torch.load(
            saved_weights_path / f"encoder_blue_{sequence}_{latent_dim}.pth",
            map_location=device,
        )
    )
    posemap = PoseMap(latent_dim)
    posemap.load_state_dict(
        torch.load(
        saved_weights_path / f"new_posemap_blue_{sequence}_{latent_dim}.pth",
        map_location=device,
    ))
    # if torch.cuda.device_count() > 1:
    #     encoder = torch.nn.DataParallel(encoder)
    #     posemap = torch.nn.DataParallel(posemap)
    encoder.to(device)
    posemap.to(device)
    wandb.watch(encoder)
    wandb.watch(posemap)

    # optimizer = Adam(list(encoder.parameters()) + list(posemap.parameters()), lr=lr)
    optimizer = Adam(posemap.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        if epoch < kld_warmup_start:
            kld_weight = 0
        elif epoch < kld_warmup_start + kld_warmup_period:
            m = kld_max_weight / kld_warmup_period
            c = -m * kld_warmup_start
            kld_weight = m * epoch + c
        else:
            kld_weight = kld_max_weight

        encoder.train()
        posemap.train()
        for i, batch in enumerate(train_data):
            x = batch[0].to(device)  # (N, 3, 32, 32)
            h = batch[1].to(device)  # (N, 12)
            t = h[:, :3]  # (N, 3)
            r = h[:, 3:].reshape(-1, 3, 3)  # (N, 3, 3)

            z_mu, z_logvar = encoder(x)  # (N, L), (N, L)
            kld_loss = kl_divergence(z_mu, z_logvar)

            z_std = torch.exp(0.5 * z_logvar)  # (N, L)
            eps = torch.randn(
                (len(x), num_latent_samples, latent_dim), device=device
            )  # (N, S, L)
            z_tilde = eps * z_std.unsqueeze(1) + z_mu.unsqueeze(1)  # (N, L)
            t_hat, r_hat = canonical_to_pose(
                posemap(z_tilde.flatten(end_dim=1)),
                mins,
                maxs,
                margins,
            )  # (N, S, 3), (N, S, 3, 3)
            t_error = torch.norm(t.unsqueeze(1) - t_hat, dim=2)  # (N, S)
            r_error = torch.norm(r.unsqueeze(1) - r_hat, dim=(2, 3))  # (N, S)
            g_error = t_weight * t_error + r_weight * r_error  # (N, S)
            sort_g_error, sort_idcs = torch.sort(g_error, dim=1)  # (N, S), (N, S)
            geo_loss = torch.mean(sort_g_error[:, :top_samples])
            loss = geo_weight * geo_loss + kld_weight * kld_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_error = t_error.detach().cpu().numpy()
            r_error = r_error.detach().cpu().numpy()
            sort_idcs = sort_idcs.detach().cpu().numpy()
            t_loss = np.mean(
                [error[idcs[:top_samples]] for error, idcs in zip(t_error, sort_idcs)]
            )
            r_loss = 2 * np.arcsin(
                np.mean(
                    [
                        error[idcs[:top_samples]]
                        for error, idcs in zip(r_error, sort_idcs)
                    ]
                )
                / np.sqrt(8)
            )

            wandb.log(
                {
                    "epoch.step": epoch + i / len(train_data),
                    "loss": loss.item(),
                    "geo_loss": geo_loss.item(),
                    "kld_loss": kld_loss.item(),
                    "t_loss": t_loss,
                    "r_loss": r_loss,
                }
            )

        encoder.eval()
        posemap.eval()
        with torch.no_grad():
            kld_losses = []
            geo_losses = []
            t_losses = []
            r_losses = []
            for i, batch in enumerate(test_data):
                x = batch[0].to(device)  # (N, 3, 32, 32)
                h = batch[1].to(device)  # (N, 12)
                t = h[:, :3]  # (N, 3)
                r = h[:, 3:].reshape(-1, 3, 3)  # (N, 3, 3)

                z_mu, z_logvar = encoder(x)  # (N, L), (N, L)
                kld_losses.append(kl_divergence(z_mu, z_logvar).item())

                z_std = torch.exp(0.5 * z_logvar)  # (N, L)
                eps = torch.randn(
                    (len(x), num_latent_samples, latent_dim), device=device
                )  # (N, S, L)
                z_tilde = eps * z_std.unsqueeze(1) + z_mu.unsqueeze(1)  # (N, L)
                t_hat, r_hat = canonical_to_pose(
                    posemap(z_tilde.flatten(end_dim=1)),
                    mins,
                    maxs,
                    margins,
                )  # (N, S, 3), (N, S, 3, 3)
                t_error = torch.norm(t.unsqueeze(1) - t_hat, dim=2)  # (N, S)
                r_error = torch.norm(r.unsqueeze(1) - r_hat, dim=(2, 3))  # (N, S)
                g_error = t_weight * t_error + r_weight * r_error  # (N, S)
                sort_g_error, sort_idcs = torch.sort(g_error, dim=1)  # (N, S), (N, S)
                geo_losses.append(torch.mean(sort_g_error[:, :top_samples]).item())

                t_error = t_error.detach().cpu().numpy()
                r_error = r_error.detach().cpu().numpy()
                sort_idcs = sort_idcs.detach().cpu().numpy()
                t_losses.append(
                    np.mean(
                        [
                            error[idcs[:top_samples]]
                            for error, idcs in zip(t_error, sort_idcs)
                        ]
                    )
                )
                r_losses.append(
                    np.mean(
                        [
                            error[idcs[:top_samples]]
                            for error, idcs in zip(r_error, sort_idcs)
                        ]
                    )
                )

        kld_loss = np.mean(kld_losses)
        geo_loss = np.mean(geo_losses)
        t_loss = np.mean(t_losses)
        r_loss = 2 * np.arcsin(np.mean(r_losses) / np.sqrt(8))
        loss = geo_weight * geo_loss + kld_weight * kld_loss

        wandb.log(
            {
                "epoch.step": epoch,
                "test_loss": loss,
                "test_geo_loss": geo_loss,
                "test_kld_loss": kld_loss,
                "test_t_loss": t_loss,
                "test_r_loss": r_loss,
            }
        )

        # torch.save(
        #     encoder.state_dict(),
        #     saved_weights_path / f"encoder_blue_{sequence}_{latent_dim}.pth",
        # )
        torch.save(
            posemap.state_dict(),
            saved_weights_path / f"posemap_blue_{sequence}_{latent_dim}.pth",
        )


if __name__ == "__main__":
    main()
