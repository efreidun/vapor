"""Script that visualizes saved prediction."""

from pathlib import Path
from types import SimpleNamespace

import argparse
import numpy as np

def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize camera pose predictions."
    )
    parser.add_argument("file", type=str)
    parser.add_argument("--dataset", type=str, default="AmbiguousReloc")
    parser.add_argument("--query", type=int, nargs="+")
    args = parser.parse_args()

    return vars(args)

def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
