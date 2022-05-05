"""Module containing plotting tools."""

from typing import Optional, Iterable, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
from matplotlib.patches import Circle
import open3d as o3d

from vaincapo.utils import quat_to_hopf


def render_3d(
    scene_path: Path,
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a projective image of the scene with samples.

    Args:
        scene_path: path to the scene that contains camera.json and mesh.ply
        tra_samples: translation samples , shape (N, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
    """
    vis_camera = o3d.io.read_pinhole_camera_parameters(str(scene_path / "camera.json"))
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=vis_camera.intrinsic.width,
        height=vis_camera.intrinsic.height,
        visible=False,
    )
    mesh = o3d.io.read_triangle_mesh(str(scene_path / "mesh.ply"))
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    for tra, quat in zip(tra_samples, quat_samples):
        sample_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0.0, 0.0, 0.0]
        )
        sample_frame.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat))
        sample_frame.translate(tra)
        vis.add_geometry(sample_frame)

    if tra_gt is not None and quat_gt is not None:
        gt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0.0, 0.0, 0.0]
        )
        gt_frame.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat_gt))
        gt_frame.translate(tra_gt)
        vis.add_geometry(gt_frame)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(vis_camera)

    vis.poll_events()
    vis.update_renderer()

    return np.array(vis.capture_screen_float_buffer(do_render=True))


def plot_posterior(
    query_id: int,
    image: np.ndarray,
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    tra_mins: Iterable[float],
    tra_maxs: Iterable[float],
    num_samples: int,
    scene_path: Path,
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_id: id of the query image
        image: query image, shape (H, W, 3)
        tra_samples: translation samples , shape (N, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        tra_mins: minimum boundaries of translations, shape (3,)
        tra_maxs: maximum boundaries of translations, shape (3,)
        num_samples: number of samples to draw explicitly, maximum 13
        scene_path: path to the scene that contains camera.json and mesh.ply
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
        save: save path including file extension

    Returns:
        figure instance
    """
    s1_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=2 * np.pi), cmap=plt.get_cmap("hsv")
    )
    z_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=tra_mins[2], vmax=tra_maxs[2]),
        cmap=plt.get_cmap("plasma"),
    )

    fig = plt.figure(figsize=(20, 10))

    image_ax = fig.add_subplot(231)
    image_ax.imshow(image)
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    image_ax.set_title(f"query image {query_id}")

    render_ax = fig.add_subplot(234)
    render = render_3d(
        scene_path,
        tra_samples[:num_samples],
        quat_samples[:num_samples],
        tra_gt,
        quat_gt,
    )
    render_ax.imshow(render)
    render_ax.set_xticks([])
    render_ax.set_yticks([])
    render_ax.set_title("samples from posterior")

    tra_ax = fig.add_subplot(232)
    tra_ax.set_xlim([tra_mins[0], tra_maxs[0]])
    tra_ax.set_ylim([tra_mins[1], tra_maxs[1]])
    tra_ax.set_aspect("equal")
    tra_ax.set_title("translation posterior")
    if tra_gt is not None:
        tra_ax.add_patch(
            Circle(
                xy=tra_gt[:2],
                radius=0.2,
                linewidth=2,
                facecolor="none",
                edgecolor=z_cm.to_rgba(tra_gt[2]),
            )
        )
    tra_ax.scatter(*tra_samples.T[:2], s=2, color=z_cm.to_rgba(tra_samples.T[2]))

    rot_ax = fig.add_subplot(233, projection="mollweide")
    rot_ax.grid(True)
    rot_ax.set_title("rotation posterior")
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt[None, :])[0]
        rot_ax.add_patch(
            Circle(
                xy=hopf_gt[:2],
                radius=0.2,
                linewidth=2,
                facecolor="none",
                edgecolor=s1_cm.to_rgba(hopf_gt[2]),
            )
        )
    hopf_samples = quat_to_hopf(quat_samples)
    rot_ax.scatter(*hopf_samples.T[:2], s=2, color=s1_cm.to_rgba(hopf_samples.T[2]))

    markers = ["o", "+", "x", "*", ".", "X", "p", "h", "D", "d", "^", "v", "s"]

    tra_samp_ax = fig.add_subplot(235)
    tra_samp_ax.set_xlim([tra_mins[0], tra_maxs[0]])
    tra_samp_ax.set_ylim([tra_mins[1], tra_maxs[1]])
    tra_samp_ax.set_aspect("equal")
    tra_samp_ax.set_title("translation samples")
    if tra_gt is not None:
        tra_samp_ax.add_patch(
            Circle(
                xy=tra_gt[:2],
                radius=0.2,
                linewidth=2,
                facecolor="none",
                edgecolor=z_cm.to_rgba(tra_gt[2]),
            )
        )
    for (x, y, z), marker in zip(
        tra_samples[:num_samples],
        markers[:num_samples],
    ):
        tra_samp_ax.scatter(x, y, s=50, color=z_cm.to_rgba(z), marker=marker)

    rot_samp_ax = fig.add_subplot(236, projection="mollweide")
    rot_samp_ax.grid(True)
    rot_samp_ax.set_title("rotation samples")
    if quat_gt is not None:
        rot_samp_ax.add_patch(
            Circle(
                xy=hopf_gt[:2],
                radius=0.2,
                linewidth=2,
                facecolor="none",
                edgecolor=s1_cm.to_rgba(hopf_gt[2]),
            )
        )
    for (phi, theta, psi), marker in zip(
        hopf_samples[:num_samples],
        markers[:num_samples],
    ):
        rot_samp_ax.scatter(phi, theta, s=50, color=s1_cm.to_rgba(psi), marker=marker)

    if save is not None:
        fig.savefig(save)

    plt.show()
    return fig
