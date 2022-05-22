"""Module containing plotting tools."""

from typing import Optional, Iterable, Union
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec, SubplotSpec
import open3d as o3d

from vaincapo.utils import quat_to_hopf, read_scene_dims


def render_3d(
    scene_path: Path,
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a projective image of the scene with samples.

    Args:
        scene_path: path to the scene that contains vis_camera.json and mesh.ply
        tra_samples: translation samples , shape (N, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
    """
    vis_camera = o3d.io.read_pinhole_camera_parameters(
        str(scene_path / "vis_camera.json")
    )
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
        num_samples: number of samples to draw explicitly, maximum 13
        scene_path: path to the scene that contains camera.json and mesh.ply
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
        save: save path including file extension

    Returns:
        figure instance
    """
    scene_dims = read_scene_dims(scene_path)
    tra_mins = scene_dims[0] - scene_dims[2]
    tra_maxs = scene_dims[1] + scene_dims[2]

    s1_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=2 * np.pi), cmap=plt.get_cmap("hsv")
    )
    z_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=tra_mins[2], vmax=tra_maxs[2]),
        cmap=plt.get_cmap("plasma"),
    )

    fig = plt.figure(figsize=(20, 10))
    grid_spec = GridSpec(100, 100)

    show_image(fig, grid_spec[0:25, 0:25], f"query image {query_id}", image)

    render = render_3d(
        scene_path,
        tra_samples[:num_samples],
        quat_samples[:num_samples],
        tra_gt,
        quat_gt,
    )
    show_image(fig, grid_spec[30:55, 0:25], "samples from posterior", render)

    plot_tras_on_plane(
        fig,
        grid_spec[0:25, 30:55],
        "translations posterior",
        scene_dims,
        tra_samples,
        tra_gt[None, :],
        None,
        grid_spec[0:25, 55:56],
    )
    # plot_rots_on_plane(fig, 233, "rotation posterior", quat_samples, quat_gt[None, :])

    markers = ["o", "+", "x", "*", ".", "X", "p", "h", "D", "d", "^", "v", "s"]

    plot_tras_on_plane(
        fig,
        grid_spec[30:55, 30:55],
        "translations samples",
        scene_dims,
        tra_samples[:num_samples],
        tra_gt[None, :],
        markers,
        grid_spec[30:55, 55:56],
    )

    # plot_rots_on_plane(
    #     fig,
    #     236,
    #     "rotation samples",
    #     quat_samples[:num_samples],
    #     quat_gt[None, :],
    #     markers,
    # )

    if save is not None:
        fig.savefig(save)

    return fig


def show_image(
    figure: plt.Figure, position: SubplotSpec, title: str, image: np.ndarray
) -> None:
    """Plot an image.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        image: image array, shape (H, W, 3)
    """
    ax = figure.add_subplot(position)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_tras_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    scene_dims: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    markers: Optional[Iterable[str]] = None,
    cm_position: Optional[SubplotSpec] = None,
) -> None:
    """Plot rotation samples on a 2D plane.

    Translation samples are plotted on a plane, with height of shown by a color map.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        scene_dims:
            2D array with rows containing minimum, maximum and margin values repectively
            and columns the x, y, z axes, shape (3, 3)
        tra_samples: translation samples to be plotted, shape (N, 3)
        tra_gt: groundtruth translations plotted as circles, shape (M, 3)
        markers: marks to be used for individual samples
        cm_position: colorbar position on the figure
    """
    tra_mins = scene_dims[0] - scene_dims[2]
    tra_maxs = scene_dims[1] + scene_dims[2]

    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=tra_mins[2], vmax=tra_maxs[2])
    z_cm = cmx.ScalarMappable(norm=norm, cmap=cmap)

    ax = figure.add_subplot(position)
    ax.set_xlim([tra_mins[0], tra_maxs[0]])
    ax.set_ylim([tra_mins[1], tra_maxs[1]])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if tra_gt is not None:
        for tra in tra_gt:
            ax.add_patch(
                Circle(
                    xy=tra[:2],
                    radius=0.2,
                    linewidth=2,
                    facecolor="none",
                    edgecolor=z_cm.to_rgba(tra[2]),
                )
            )
    if markers is None:
        ax.scatter(*tra_samples.T[:2], s=2, color=z_cm.to_rgba(tra_samples.T[2]))
    else:
        assert len(markers) >= len(tra_samples), "Must have as many markers as samples"
        for (x, y, z), marker in zip(tra_samples, markers[: len(tra_samples)]):
            ax.scatter(x, y, s=50, color=z_cm.to_rgba(z), marker=marker)

    if cm_position is not None:
        cm_ax = figure.add_subplot(cm_position)
        mpl.colorbar.ColorbarBase(cm_ax, cmap=cmap, norm=norm)
        cm_ax.set_ylabel("z")


def plot_rots_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_samples: np.ndarray,
    quat_gt: Optional[np.ndarray] = None,
    markers: Optional[Iterable[str]] = None,
) -> None:
    """Plot rotation samples on a 2D plane.

    Rotation samples are converted to Hopf coordinates, S^2 element of which is
    projected onto a 2D plane using Mollweide projection, and the S^1 element marcated
    by coloring the samples according to a color wheel.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        quat_samples: samples to be plotted in quaternions [w, x, y, z], shape (N, 4)
        quat_gt: groundtruth quaternions [w, x, y, z] plotted as circles, shape (M, 4)
        markers: marks to be used for individual samples
    """
    s1_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=2 * np.pi), cmap=plt.get_cmap("hsv")
    )

    ax = figure.add_subplot(position, projection="mollweide")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt)
        for hopf in hopf_gt:
            ax.add_patch(
                Circle(
                    xy=hopf[:2],
                    radius=0.2,
                    linewidth=2,
                    facecolor="none",
                    edgecolor=s1_cm.to_rgba(hopf[2]),
                )
            )
    hopf_samples = quat_to_hopf(quat_samples)
    if markers is None:
        ax.scatter(*hopf_samples.T[:2], s=2, color=s1_cm.to_rgba(hopf_samples.T[2]))
    else:
        assert len(markers) >= len(hopf_samples), "Must have as many markers as samples"
        for (phi, theta, psi), marker in zip(
            hopf_samples, markers[: len(hopf_samples)]
        ):
            ax.scatter(phi, theta, s=50, color=s1_cm.to_rgba(psi), marker=marker)
