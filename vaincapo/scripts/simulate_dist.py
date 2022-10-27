"""Script that simulates mixture models and saves the samples."""

from types import SimpleNamespace
from pathlib import Path
from collections import Counter

import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.distributions as D

from vaincapo.sampling import GMM, BMM


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Simulate mixture models.")
    parser.add_argument("run", type=str)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_candidates", type=int, default=10000)
    parser.add_argument("--split", type=str, nargs="+", default=["valid"])
    parser.add_argument("--device", type=str)

    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_path = Path.home() / "code/vaincapo/bingham_runs" / cfg.run

    for split_name in cfg.split:
        split_path = run_path / f"{split_name}.npz"
        mixmod = dict(np.load(split_path))
        tra_locs = torch.from_numpy(mixmod["tra_locs"]).to(device)
        tra_stds = torch.from_numpy(mixmod["tra_stds"]).to(device)
        rot_locs = torch.from_numpy(mixmod["rot_locs"]).to(device)
        rot_lams = torch.from_numpy(mixmod["rot_lams"]).to(device)
        coeffs = mixmod["coeffs"]
        num_queries, num_comps = coeffs.shape

        tra_samples = []
        rot_samples = []
        for i in tqdm(range(num_queries)):
            if np.isnan(coeffs[i]).any():
                coeffs[i] = np.ones(num_comps) / num_comps
            comp_counts = Counter(
                np.random.choice(num_comps, cfg.num_samples, replace=True, p=coeffs[i])
            )
            num_samples = np.zeros(num_comps, dtype=int)
            num_samples[list(comp_counts.keys())] = list(comp_counts.values())
            rot_bmm = BMM(rot_locs[i], rot_lams[i], coeffs[i])
            rot_samples.append(
                rot_bmm.sample(num_samples, cfg.num_candidates)[None, :, :]
            )
            tra_samples.append(
                torch.cat(
                    [
                        D.MultivariateNormal(
                            tra_locs[i, j], torch.diag(tra_stds[i, j] ** 2)
                        ).sample((count,))
                        for j, count in enumerate(num_samples)
                        if count != 0
                    ]
                )[None, :, :]
            )
        tra_samples = torch.cat(tra_samples).cpu().numpy()
        rot_samples = torch.cat(rot_samples).cpu().numpy()

        mixmod["tra_samples"] = tra_samples
        mixmod["rot_samples"] = rot_samples
        np.savez(split_path, **mixmod)


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
