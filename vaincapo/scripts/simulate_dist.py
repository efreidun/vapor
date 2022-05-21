"""Script that simulates mixture models and saves the samples."""

from types import SimpleNamespace
from pathlib import Path

import argparse
import numpy as np
from tqdm import tqdm
import torch

from vaincapo.sampling import GMM, BMM


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Simulate mixture models.")
    parser.add_argument("run", type=str)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_candidates", type=int, default=1000000)
    parser.add_argument("--split", type=str, default=["train", "valid"])

    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    runs_path = Path.home() / "code/vaincapo/bingham_runs"
    run_path = runs_path / cfg.run

    for split in cfg.split:
        split_path = run_path / f"{split}.npz"
        mixmod = dict(np.load(split_path))
        tra_locs = torch.from_numpy(mixmod["tra_locs"])
        tra_vars = torch.from_numpy(mixmod["tra_vars"])
        rot_locs = torch.from_numpy(mixmod["rot_locs"])
        rot_lams = torch.from_numpy(mixmod["rot_lams"])
        coeffs = torch.from_numpy(mixmod["coeffs"])

        tra_samples = []
        rot_samples = []
        for i in tqdm(range(len(tra_locs))):
            covariances = torch.cat(
                [torch.diag(variances)[None, :, :] for variances in tra_vars[i]]
            )
            tra_gmm = GMM(tra_locs[i], covariances, coeffs[i])
            rot_bmm = BMM(rot_locs[i], rot_lams[i], coeffs[i])
            tra_samples.append(tra_gmm.sample((cfg.num_samples,))[None, :, :])
            rot_samples.append(
                rot_bmm.sample(cfg.num_samples, cfg.num_candidates)[None, :, :]
            )
        tra_samples = torch.cat(tra_samples).numpy()
        rot_samples = torch.cat(rot_samples).numpy()

        mixmod["tra_samples"] = tra_samples
        mixmod["rot_samples"] = rot_samples
        np.savez(split_path, **mixmod)


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
