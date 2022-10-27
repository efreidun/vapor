"""Script that visualizes runs on ceiling sequence."""

from pathlib import Path
from types import SimpleNamespace
import yaml

from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

from vaincapo.plotting import plot_ceiling


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize camera pose posterior samples."
    )
    parser.add_argument("run", type=str)
    parser.add_argument("--mapnet", type=str)
    parser.add_argument("--bingham", type=str)
    parser.add_argument(
        "--source", type=str, default="pipeline", choices=("pipeline", "bingham")
    )
    parser.add_argument("--query", type=int, nargs="+")
    parser.add_argument("--split", type=str, nargs="+", default=["valid"])
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    base_path = Path.home() / "code/vaincapo"
    dataset_path = Path.home() / "data/Rig"
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

    Hs = np.loadtxt(Path.home() / "data/Rig/Ceiling/homographies.txt")

    for split in cfg.split:
        data = np.load(run_path / f"{split}.npz")
        plot_mapnet = split == "valid" and cfg.mapnet is not None
        if plot_mapnet:
            mapnet_data = np.load(cfg.mapnet)
        plot_bingham = split == "valid" and cfg.bingham is not None
        if plot_bingham:
            bingham_data = np.load(cfg.bingham)

        queries = cfg.query or range(len(data["names"]))
        for i in tqdm(queries):
            name_with_split = (
                f"{'test' if split == 'valid' else 'train'}/" + data["names"][i]
            )
            query_file_name = scene + "/" + name_with_split
            plot_path = plots_path / name_with_split
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_ceiling(
                np.array(Image.open(dataset_path / query_file_name)),
                data["tra_samples"][i],
                data["rot_samples"][i],
                scene_path,
                cfg.run + " : " + query_file_name[:-4],
                data["tra_gt"][i],
                data["rot_gt"][i],
                mapnet_data["tra_samples"][i].squeeze() if plot_mapnet else None,
                mapnet_data["rot_samples"][i].squeeze() if plot_mapnet else None,
                bingham_data["tra_samples"][i].squeeze() if plot_bingham else None,
                bingham_data["rot_samples"][i].squeeze() if plot_bingham else None,
                Hs[i, 1:].reshape(3, 3) if split == "valid" else None,
                save=plot_path.with_suffix(".png"),
            )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
