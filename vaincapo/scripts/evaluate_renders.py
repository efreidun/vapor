"""Script for evaluating renders."""

from pathlib import Path
from types import SimpleNamespace
import yaml
import json

from PIL import Image
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate renders from camera pose posterior samples."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--source", type=str, default="pipeline")
    parser.add_argument("--reference", type=str, default="render")
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code" / "vaincapo"

    assert cfg.source in (
        "pipeline",
        "bingham",
    ), "Source has to be either 'pipeline' or 'bingham'"
    if cfg.source == "pipeline":
        run_path = base_path / "runs" / cfg.run
        with open(run_path / "config.yaml") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        scene = train_config["sequence"]
    else:
        run_path = base_path / "bingham_runs" / cfg.run
        scene = "_".join(cfg.run.split("_")[:-1])

    assert cfg.reference in (
        "image",
        "render",
    ), "Reference has to be either 'image' or 'render'"
    scene_path = Path.home() / "data/Ambiguous_ReLoc_Dataset" / scene
    reference_path = (
        (scene_path / "test") if cfg.reference == "image" else scene_path / "render"
    )

    with open(run_path / "transforms.json") as f:
        transforms = json.load(f)
    renders_path = run_path / "renders"
    sample_image_paths = sorted(renders_path.glob("*.png"))
    frame_ids = [
        int(str((sample_image_path.stem)).split("_")[0])
        for sample_image_path in sample_image_paths
    ]
    query_images = np.concatenate(
        [
            np.array(
                Image.open(
                    reference_path / transforms["frames"][frame_id]["query_image"]
                ).resize((cfg.width, cfg.height)),
                dtype=float,
            )[None, ...]
            for frame_id in frame_ids
        ]
    )
    sample_images = np.concatenate(
        [
            np.array(Image.open(sample_image_path), dtype=float)[None, ...]
            for sample_image_path in sample_image_paths
        ]
    )[:, :, :, :3]
    q = len(query_images)
    n, im_h, im_w = sample_images.shape[:3]
    m = transforms["num_renders"]
    assert q == n // m, "number of query images and produced sample sets must be equal"
    sample_images = sample_images.reshape(q, m, im_h, im_w, 3)

    ssims = []
    mses = []
    for query_image, sample_image in tqdm(zip(query_images, sample_images), total=q):
        for samp_image in sample_image:
            mses.append(mse(query_image, samp_image))
            ssims.append(
                ssim(query_image, samp_image, data_range=255, multichannel=True)
            )

    with open(run_path / "photometrics.txt", "w") as f:
        f.write(
            f"{cfg.run} {len(query_images)} test {scene} query"
            + f" {cfg.reference}s with {m} {cfg.width}x{cfg.height} renders each\n"
        )
        f.write(f"MSE: {np.mean(mses)}\n")
        f.write(f"SSIM: {np.mean(ssims)}\n")


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
