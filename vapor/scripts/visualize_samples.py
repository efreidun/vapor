"""Script that visualizes saved samples."""

from pathlib import Path
from types import SimpleNamespace
import yaml

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

from vapor.read_write import read_rendered_samples
from vapor.plotting import plot_posterior


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
    parser.add_argument("--dataset", type=str, default="AmbiguousReloc")
    parser.add_argument("--source", type=str, default="pipeline")
    parser.add_argument("--query", type=int, nargs="+")
    parser.add_argument("--split", type=str, nargs="+", default=["valid"])
    parser.add_argument("--norender", action="store_true")
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code/vapor"
    dataset_path = Path.home() / "data" / cfg.dataset
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
    plots_path = run_path / "plots"

    for split in cfg.split:
        data = np.load(run_path / f"{split}.npz")
        plot_renders = split == "valid" and not cfg.norender
        if plot_renders:
            sample_renders, query_images, query_renders = read_rendered_samples(
                run_path / "transforms.json",
                run_path / "renders",
                scene_path / "test",
                scene_path / "renders/test",
            )

        queries = cfg.query or range(len(data["names"]))
        for i in tqdm(queries):
            name_with_split = (
                f"{'test' if split == 'valid' else 'train'}/" + data["names"][i]
            )
            query_file_name = (
                scene
                + "/"
                + (
                    data["names"][i]
                    if cfg.dataset == "SketchUpCircular"
                    else name_with_split
                )
            )
            plot_path = plots_path / name_with_split
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_posterior(
                query_images[i]
                if plot_renders
                else np.array(Image.open(dataset_path / query_file_name)),
                data["tra_samples"][i],
                data["rot_samples"][i],
                cfg.num_samples,
                scene_path,
                cfg.run + " : " + query_file_name[:-4],
                query_renders[i] if plot_renders else None,
                sample_renders[i, : cfg.num_samples] if plot_renders else None,
                data["tra_gt"][[i]],
                data["rot_gt"][[i]],
                plot_path,
            )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
