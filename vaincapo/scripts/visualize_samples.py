"""Script that visualizes saved samples."""

from pathlib import Path
from types import SimpleNamespace
import yaml
import json

from PIL import Image
import numpy as np
import argparse

from vaincapo.plot_utils import plot_posterior


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize camera pose posterior samples."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--source", type=str, default="pipeline")
    parser.add_argument("--split", type=str, nargs="+")
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code/vaincapo"
    dataset_path = Path.home() / "data/Ambiguous_ReLoc_Dataset"
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
    scene_path = dataset_path / scene
    reference_path = scene_path / "renders/test"
    with open(run_path / "transforms.json") as f:
        transforms = json.load(f)
    renders_path = run_path / "renders"
    sample_image_paths = sorted(renders_path.glob("**/*.png"))
    frame_ids = [
        int(str((sample_image_path.stem)).split("_")[0])
        for sample_image_path in sample_image_paths
    ]
    m = transforms["num_renders"]
    query_renders = np.concatenate(
        [
            np.array(
                Image.open(
                    reference_path / transforms["frames"][frame_id]["query_image"]
                ),
            )[None, ...]
            for frame_id in frame_ids[::m]
        ]
    )[:, :, :, :3]
    sample_renders = np.concatenate(
        [
            np.array(Image.open(sample_image_path))[None, ...]
            for sample_image_path in sample_image_paths
        ]
    )[:, :, :, :3]
    im_h, im_w = sample_renders.shape[1:3]
    sample_renders = sample_renders.reshape(-1, m, im_h, im_w, 3)

    data = np.load(run_path / "valid.npz")

    q = 0
    query_image_path = scene_path / "test" / data["names"][q]
    image = np.array(Image.open(scene_path / "test" / data["names"][q]))
    query_name = str(query_image_path.relative_to(dataset_path).with_suffix(""))
    plot_posterior(
        image,
        data["tra_samples"][q],
        data["rot_samples"][q],
        cfg.num_samples,
        scene_path,
        query_name,
        query_renders[q],
        sample_renders[q],
        data["tra_gt"][q],
        data["rot_gt"][q],
        Path.home() / "Desktop/figoor.png",
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
