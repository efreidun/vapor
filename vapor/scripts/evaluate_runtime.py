from pathlib import Path
import time

import numpy as np
import torch

from vapor.models import Encoder, PoseMap
from vapor.data import AmbiguousReloc

dataset = AmbiguousReloc(
    Path.home() / "data/AmbiguousReloc/blue_chairs/test",
    224,
    "posenet",
    False,
    None,
    3,
    (0.001, 1.0),
    0.05,
    0.05,
    0.05,
    0.05,
)
device = torch.device("cpu")
image = dataset[0][0][None, ...].to(device)
num_images = image.shape[0]

encoder = Encoder(16, "resnet18")
posemap = PoseMap(16, 3, 128)
encoder.to(device)
posemap.to(device)
with torch.no_grad():
    for num_samples in 1, 10, 100, 1000, 10000, 100000, 1000000:
        times = []
        for i in range(105):
            torch.cuda.synchronize()
            t0 = time.time()
            lat_mu, lat_logvar = encoder(image)
            lat_std = torch.exp(0.5 * lat_logvar)
            eps = torch.randn(
                (num_images, num_samples, encoder.get_latent_dim()), device=image.device
            )
            lat_sample = eps * lat_std.unsqueeze(1) + lat_mu.unsqueeze(1)
            tvec, rvec = posemap(lat_sample.flatten(end_dim=1))
            torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)
        print(
            f"{num_samples} samples: {np.mean(times[5:])*1000:.2f}, "
            + f"({np.std(times[5:])*1000:.2f})"
        )
