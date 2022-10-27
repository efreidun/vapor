from pathlib import Path
from types import SimpleNamespace

import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from vaincapo.data import Rig


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize augmentations."
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--half_image", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_mode", type=str, default="center_crop")
    parser.add_argument("--image_crop", type=float, default=None)
    parser.add_argument("--gauss_kernel", type=int, default=3)
    parser.add_argument(
        "--gauss_sigma", type=float, nargs="*", default=None) # (0.05, 5.0))
    parser.add_argument("--jitter_brightness", type=float, default=0.7)
    parser.add_argument("--jitter_contrast", type=float, default=0.7)
    parser.add_argument("--jitter_saturation", type=float, default=0.7)
    parser.add_argument("--jitter_hue", type=float, default=0.5)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    scene_path = Path.home() / "data" / cfg.dataset / cfg.sequence
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
    dataset = Rig(scene_path / "test", **dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
    )

    for batch in dataloader:
        images = make_grid(batch[0])
        images_np = images.transpose(0, 2).transpose(0, 1).numpy()

        plt.figure()
        plt.imshow(images_np)
        plt.show()

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
