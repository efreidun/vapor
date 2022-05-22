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

from vaincapo.utils import read_rendered_samples


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
    base_path = Path.home() / "code/vaincapo"

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

    sample_renders, query_images, query_renders = read_rendered_samples(
        run_path / "transforms.json",
        run_path / "renders",
        scene_path / "test",
        scene_path / "renders/test",
        (cfg.width, cfg.height),
    )

    query_sets = {"images": query_images, "renders": query_renders}
    mse_avgs = []
    ssim_avgs = []
    for queries in query_sets.values():
        mses = []
        ssims = []
        for query, sample_render in tqdm(
            zip(queries, sample_renders), total=len(queries)
        ):
            for samp_render in sample_render:
                mses.append(mse(query, samp_render))
                ssims.append(
                    ssim(query, samp_render, data_range=255, multichannel=True)
                )
        mse_avgs.append(np.mean(mses))
        ssim_avgs.append(np.mean(ssims))

    with open(run_path / "photometrics.txt", "w") as f:
        f.write(
            f"{cfg.run} {len(query_images)} test {scene} queries with"
            + f" {sample_renders.shape[1]} {cfg.width}x{cfg.height} rendered samples"
            + f" each\n"
        )
        for queries, avg_ssim, avg_mse in zip(query_sets.keys(), ssim_avgs, mse_avgs):
            f.write(f"sample renders agains query {queries}:\n")
            f.write(f"MSE: {avg_mse}\n")
            f.write(f"SSIM: {np.mean(avg_ssim)}\n")


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
