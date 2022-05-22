"""Script for visualizing mesh of a scene."""

from types import SimpleNamespace
from pathlib import Path

import argparse
import open3d as o3d


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        Passed arguments as dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Visualize mesh of a scene."
    )
    parser.add_argument("source", type=str)
    args = parser.parse_args()

    return vars(args)


def main(config: dict) -> None:
    cfg = SimpleNamespace(**config)

    source_path = Path(cfg.source)
    if source_path.suffix == ".ply":
        mesh_path = source_path
    else:
        scene_path = Path.home() / "data/Ambiguous_ReLoc_Dataset" / source_path
        mesh_path = scene_path / "mesh.ply"
        if not mesh_path.is_file():
            raise ValueError(f"Cannot find 'mesh.ply' in scene {source_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            ),
            mesh,
        ]
    )


if __name__ == "__main__":
    config = parse_arguments()
    main(config)
