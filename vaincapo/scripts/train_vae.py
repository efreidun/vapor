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
from nerfcity.models import Encoder, Decoder
from nerfcity.losses import kl_divergence


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 2000
    latent_dim = 16
    image_size = 64
    batch_size = 16
    lr = 1e-4
    rec_weight = 1
    kld_warmup_start = 0
    kld_warmup_period = 0
    kld_max_weight = 0.001
    num_recs = 4

    wandb.init(
        project="vae_training",
        entity="efreidun",
        config={
            "num_epochs": num_epochs,
            "latent_dim": latent_dim,
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "rec_weight": rec_weight,
            "kld_warmup_start": kld_warmup_start,
            "kld_warmup_period": kld_warmup_period,
            "kld_max_weight": kld_max_weight,
            "num_recs": num_recs,
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
    train_rec_idcs = [i for i in range(0, len(train_data), len(train_data) // num_recs)]
    test_rec_idcs = [i for i in range(0, len(test_data), len(test_data) // num_recs)]

    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    # if torch.cuda.device_count() > 1:
    #     encoder = torch.nn.DataParallel(encoder)
    #     decoder = torch.nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)
    wandb.watch(encoder)
    wandb.watch(decoder)

    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

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
        decoder.train()
        xs = []
        x_hats = []
        for i, batch in enumerate(train_data):
            x = batch[0].to(device)  # (N, 3, 32, 32)
            z_mu, z_logvar = encoder(x)  # (N, L), (N, L)
            kld_loss = kl_divergence(z_mu, z_logvar)

            z_std = torch.exp(0.5 * z_logvar)  # (N, L)
            eps = torch.randn((len(x), latent_dim), device=device)  # (N, L)
            z_tilde = eps * z_std + z_mu  # (N, L)
            x_hat = decoder(z_tilde)  # (N, 3, 32, 32)
            rec_loss = torch.mean((x - x_hat) ** 2)
            loss = rec_weight * rec_loss + kld_weight * kld_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "epoch.step": epoch + i / len(train_data),
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "kld_loss": kld_loss.item(),
                }
            )

            if i in train_rec_idcs:
                xs.append(x[0])
                x_hats.append(x_hat[0])

        xs.extend(x_hats)
        train_recs = (
            make_grid(xs, nrow=len(train_rec_idcs)) * dataset_std + dataset_mean
        )

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            kld_losses = []
            rec_losses = []
            xs = []
            x_hats = []
            for i, batch in enumerate(test_data):
                x = batch[0].to(device)  # (N, 3, 32, 32)
                z_mu, z_logvar = encoder(x)  # (N, L), (N, L)
                kld_losses.append(kl_divergence(z_mu, z_logvar).item())

                z_std = torch.exp(0.5 * z_logvar)  # (N, L)
                eps = torch.randn((len(x), latent_dim), device=device)  # (N, L)
                z_tilde = eps * z_std + z_mu  # (N, L)
                x_hat = decoder(z_tilde)  # (N, 3, 32, 32)
                rec_losses.append(torch.mean((x - x_hat) ** 2).item())

                if i in test_rec_idcs:
                    xs.append(x[0])
                    x_hats.append(x_hat[0])

            xs.extend(x_hats)
            test_recs = (
                make_grid(xs, nrow=len(test_rec_idcs)) * dataset_std + dataset_mean
            )

            z_tilde = torch.randn((10, latent_dim), device=device)  # (N, L)
            x_hat = decoder(z_tilde)  # (N, 3, 32, 32)
            sample_recs = make_grid(x_hat, nrow=5) * dataset_std + dataset_mean

        kld_loss = np.mean(kld_losses)
        rec_loss = np.mean(rec_losses)
        loss = rec_weight * rec_loss + kld_weight * kld_loss

        wandb.log(
            {
                "epoch.step": epoch,
                "test_loss": loss,
                "test_rec_loss": rec_loss,
                "test_kld_loss": kld_loss,
                "train_recs": wandb.Image(train_recs, caption="top: x, bottom: x_hat"),
                "test_recs": wandb.Image(test_recs, caption="top: x, bottom: x_hat"),
                "sample_recs": wandb.Image(
                    sample_recs, caption="unconstrained samples"
                ),
            }
        )

    saved_weights_path = Path.home() / "code" / "nerfcity" / "saved_weights"
    torch.save(
        encoder.state_dict(),
        saved_weights_path / f"encoder_blue_{sequence}_{latent_dim}.pth",
    )
    torch.save(
        decoder.state_dict(),
        saved_weights_path / f"decoder_blue_{sequence}_{latent_dim}.pth",
    )


if __name__ == "__main__":
    main()
