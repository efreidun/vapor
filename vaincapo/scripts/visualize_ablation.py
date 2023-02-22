"""Script for visualizing ablation results."""

from types import SimpleNamespace

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(description="Visualize ablation results.")
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)
    with open(cfg.output) as f:
        content = f.readlines()
    results = np.array(
        [float(entry) for line in content for entry in line.strip().split()]
    ).reshape(-1, 7)
    idcs = np.argsort(results[:, 0])
    results = results[idcs].T

    alphas = results[0]
    means = results[1:4]
    stds = results[4:7]

    unique_alphas = np.unique(alphas)
    unique_means = []
    unique_stds = []
    unique_mins = []
    unique_maxs = []
    data = []
    for alpha in unique_alphas:
        unique_means.append(np.median(means[:, alphas == alpha], axis=1))
        unique_stds.append(np.std(means[:, alphas == alpha], axis=1))
        unique_mins.append(np.min(means[:, alphas == alpha], axis=1))
        unique_maxs.append(np.max(means[:, alphas == alpha], axis=1))
        data.append(means[:, alphas == alpha])
    unique_means = np.array(unique_means)
    unique_stds = np.array(unique_stds)
    unique_mins = np.array(unique_mins)
    unique_maxs = np.array(unique_maxs)
    alphas = unique_alphas
    data = np.array(data)
    means = unique_means.T
    stds = unique_stds.T
    maxs = unique_maxs.T
    mins = unique_mins.T

    _, ax = plt.subplots()
    ax.boxplot(data[:, 0, :].T, showfliers=True, whis=(0, 100))
    tikzplotlib.save("ablation.tex")
    plt.show()
    

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
