"""Script that visualizes saved mixture models."""

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm

from vaincapo.plotting import plot_mixture_model


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize camera pose posterior samples."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--query", type=int, nargs="+")
    parser.add_argument("--split", type=str, nargs="+", default=["valid"])
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code/vaincapo"
    dataset_path = Path.home() / "data/AmbiguousReloc"
    run_path = base_path / "bingham_runs" / cfg.run
    scene = "_".join(cfg.run.split("_")[:-1])
    scene_path = dataset_path / scene
    plots_path = run_path / "plots"

    for split in cfg.split:
        data = np.load(run_path / f"{split}.npz")

        queries = cfg.query or range(len(data["names"]))
        for i in tqdm(queries):
            name_with_split = (
                f"{'test' if split == 'valid' else 'train'}/" + data["names"][i]
            )
            query_file_name = scene + "/" + name_with_split
            plot_path = plots_path / name_with_split
            plot_path = plot_path.parent / "mixture" / plot_path.name
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_mixture_model(
                np.array(Image.open(dataset_path / query_file_name)),
                data["tra_locs"][i],
                data["tra_stds"][i],
                data["rot_locs"][i],
                data["rot_lams"][i],
                data["coeffs"][i],
                scene_path,
                cfg.run + " : " + query_file_name[:-4],
                data["tra_gt"][i],
                data["rot_gt"][i],
                plot_path,
            )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
