"""Module containing plotting tools."""

from typing import Optional, Iterable, Union
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec, SubplotSpec
import open3d as o3d

from vaincapo.utils import quat_to_hopf
from vaincapo.read_write import read_scene_dims


def plot_latent(
    tras_train: np.ndarray,
    tras_valid: np.ndarray,
    quats_train: np.ndarray,
    quats_valid: np.ndarray,
    codes_train: np.ndarray,
    codes_valid: np.ndarray,
    scene_path: Path,
    title: Optional[str],
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        tras_train: training set translations, (N, 3)
        tras_valid: validation set translations, (M, 3)
        quats_train: training set rotation matrices, (N, 3, 3)
        quats_valid: validation set rotation matrices, (M, 3, 3)
        codes_train: training set latent codes, (N, 2)
        codes_valid: validation set latent codes, (M, 2)
        scene_path: path to the scene that contains scene.txt
        title: title of figure
        save: save path including file extension

    Returns:
        figure instance
    """
    scene_dims = read_scene_dims(scene_path)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title)
    r = [0, 50]
    c = [0, 25, 50, 75]
    w = 25
    h = 40
    grid_spec = GridSpec(100, 100)

    # first row of subplots, training samples
    try:
        render = render_3d(
            scene_path,
            tras_train,
            quats_train,
        )
        show_image(
            fig,
            grid_spec[r[0] : r[0] + h, c[0] : c[0] + w],
            "training samples",
            render,
        )
    except AttributeError:
        pass
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=len(tras_train))
    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plot_tra_colored_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[1] : c[1] + w],
        "translation components",
        scene_dims,
        tras_train,
        colormap,
    )
    plot_rot_colored_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[2] : c[2] + w],
        "rotation components",
        quats_train,
        colormap,
    )
    plot_code_colored_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[3] : c[3] + w],
        "latent codes",
        codes_train,
        colormap,
    )

    # second row of subplots, validation samples
    try:
        render = render_3d(
            scene_path,
            tras_valid,
            quats_valid,
        )
        show_image(
            fig,
            grid_spec[r[1] : r[1] + h, c[0] : c[0] + w],
            "validation samples",
            render,
        )
    except AttributeError:
        pass
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=len(tras_valid))
    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plot_tra_colored_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[1] : c[1] + w],
        "translation components",
        scene_dims,
        tras_valid,
        colormap,
    )
    plot_rot_colored_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[2] : c[2] + w],
        "rotation components",
        quats_valid,
        colormap,
    )
    plot_code_colored_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[3] : c[3] + w],
        "latent codes",
        codes_valid,
        colormap,
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)

    return fig


def plot_mixture_model(
    query_image: np.ndarray,
    tra_locs: np.ndarray,
    tra_stds: np.ndarray,
    quat_locs: np.ndarray,
    quat_lams: np.ndarray,
    coeffs: np.ndarray,
    scene_path: Path,
    title: Optional[str],
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot components of a mixture model SE(3).

    Args:
        query_image: query image, shape (H, W, 3)
        tra_locs: translation component locations, shape (N, 3)
        tra_stds: translation component standard deviations, shape (N, 3)
        quat_locs: rotation locations in quaternion [w, x, y, z], shape (N, 4)
        quat_lams: rotation bingham lambdas in quaternion [w, x, y, z], shape (N, 3)
        coeffs: component coefficient weights, shape (N,)
        scene_path: path to the scene that contains camera.json and mesh.ply
        title: title of figure
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
        save: save path including file extension

    Returns:
        figure instance
    """
    scene_dims = read_scene_dims(scene_path)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title)
    r = [0, 50]
    c = [0, 40]
    w = 40
    h = 45
    cb_o = 1
    cb_w = 1
    cw_o = 1
    cw_w = 4
    grid_spec = GridSpec(100, 100)

    # first row of subplots, query image and posterior samples
    show_image(
        fig,
        grid_spec[r[0] : r[0] + h, c[0] : c[0] + w],
        "query image",
        query_image,
    )
    plot_rot_dists_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[1] : c[1] + w],
        "rotation components",
        quat_locs,
        quat_lams,
        coeffs,
        quat_gt[None, :],
        grid_spec[r[0] : r[0] + h, c[1] + w + cw_o : c[1] + w + cw_o + cw_w],
    )

    # second row of subplots, projecive view and individual samples
    try:
        render = render_3d(
            scene_path,
            tra_locs,
            quat_locs,
            tra_gt,
            quat_gt,
        )
        show_image(
            fig,
            grid_spec[r[1] : r[1] + h, c[0] : c[0] + w],
            "mixture components",
            render,
        )
    except AttributeError:
        pass
    plot_tra_dists_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[1] : c[1] + w],
        "translations components",
        scene_dims,
        tra_locs,
        tra_stds,
        coeffs,
        tra_gt[None, :],
        grid_spec[r[1] : r[1] + h, c[1] + w + cb_o : c[1] + w + cb_o + cb_w],
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)

    return fig


def plot_posterior(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    num_samples: int,
    scene_path: Path,
    title: Optional[str],
    query_render: Optional[np.ndarray] = None,
    sample_renders: Optional[np.ndarray] = None,
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (H, W, 3)
        tra_samples: translation samples , shape (N, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        num_samples: number of samples to show individually, maximum 13
        scene_path: path to the scene that contains camera.json and mesh.ply
        title: title of figure
        query_render: NeRF render of the query image, shape (H, W, 3)
        sample_renders: NeRF rendesr of the samples, shape (M, H, W, 3)
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)
        save: save path including file extension

    Returns:
        figure instance
    """
    scene_dims = read_scene_dims(scene_path)
    markers = ["o", "+", "x", "*", ".", "X", "p", "h", "D", "d", "^", "v", "s"]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title)
    r = [0, 40, 80]
    c = [0, 33, 66]
    w = 25
    h = 35
    cb_o = 1
    cb_w = 1
    cw_o = 1
    cw_w = 4
    grid_spec = GridSpec(
        r[1] + h if (query_render is None and sample_renders is None) else 100, 100
    )

    # first row of subplots, query image and posterior samples
    show_image(
        fig,
        grid_spec[r[0] : r[0] + h, c[0] : c[0] + w],
        "query image",
        query_image,
    )
    plot_tra_hist_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[1] : c[1] + w],
        "translations posterior",
        scene_dims,
        tra_samples,
        tra_gt[None, :],
    )
    plot_rot_hist_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[2] : c[2] + w],
        "rotation posterior",
        quat_samples,
        quat_gt[None, :],
    )

    # second row of subplots, projecive view and individual samples
    try:
        render = render_3d(
            scene_path,
            tra_samples[:num_samples],
            quat_samples[:num_samples],
            tra_gt,
            quat_gt,
        )
        show_image(
            fig,
            grid_spec[r[1] : r[1] + h, c[0] : c[0] + w],
            "samples from posterior",
            render,
        )
    except AttributeError:
        pass
    plot_tra_samples_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[1] : c[1] + w],
        "translations samples",
        scene_dims,
        tra_samples[:num_samples],
        tra_gt[None, :],
        markers,
        grid_spec[r[1] : r[1] + h, c[1] + w + cb_o : c[1] + w + cb_o + cb_w],
    )
    plot_rot_samples_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[2] : c[2] + w],
        "rotation samples",
        quat_samples[:num_samples],
        quat_gt[None, :],
        markers,
        grid_spec[r[1] : r[1] + h, c[2] + w + cw_o : c[2] + w + cw_o + cw_w],
    )

    # third row of subplots, renders of query image and samples
    if query_render is not None and sample_renders is not None:
        assert len(markers) >= len(
            sample_renders
        ), "Must have as many markers as renders"
        renders = np.concatenate((query_render[None, ...], sample_renders))
        ren_w = 100 // len(renders)
        for i, render in enumerate(renders):
            show_image(
                fig,
                grid_spec[r[2] :, c[0] + i * ren_w : c[0] + (i + 1) * ren_w],
                "query render" if i == 0 else f"render {markers[i-1]}",
                render,
            )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)

    return fig


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


def plot_tra_samples_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    scene_dims: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    markers: Optional[Iterable[str]] = None,
    cm_position: Optional[SubplotSpec] = None,
) -> None:
    """Plot translation samples on a 2D plane.

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
    z_cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax = figure.add_subplot(position)
    ax.set_xlim([tra_mins[0], tra_maxs[0]])
    ax.set_ylim([tra_mins[1], tra_maxs[1]])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if tra_gt is not None:
        for tra in tra_gt:
            ax.add_patch(
                Circle(
                    xy=tra[:2],
                    radius=0.1,
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
        cm_ax.set_axis_off()


def plot_rot_samples_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_samples: np.ndarray,
    quat_gt: Optional[np.ndarray] = None,
    markers: Optional[Iterable[str]] = None,
    cm_position: Optional[SubplotSpec] = None,
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
        cm_position: colorbar position on the figure
    """
    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
    s1_cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

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

    if cm_position is not None:
        cm_ax = figure.add_subplot(cm_position, projection="polar")
        cb = mpl.colorbar.ColorbarBase(
            cm_ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        cb.outline.set_visible(False)
        cm_ax.set_axis_off()
        cm_ax.set_rlim([-1, 1])


def plot_tra_dists_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    scene_dims: np.ndarray,
    tra_locs: np.ndarray,
    tra_stds: np.ndarray,
    coeffs: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    cm_position: Optional[SubplotSpec] = None,
) -> None:
    """Plot components of Gaussian mixture model translation distribution on a 2D plane.

    Translations are plotted on a plane, with height of shown by a color map.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        scene_dims:
            2D array with rows containing minimum, maximum and margin values repectively
            and columns the x, y, z axes, shape (3, 3)
        tra_locs: location of components, shape (N, 3)
        tra_stds: standard deviations of components, shape (N, 3)
        coeffs: weight coeffients of mixutre components, shape (N,)
        markers: marks to be used for individual samples
        cm_position: colorbar position on the figure
    """
    tra_mins = scene_dims[0] - scene_dims[2]
    tra_maxs = scene_dims[1] + scene_dims[2]

    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=tra_mins[2], vmax=tra_maxs[2])
    z_cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax = figure.add_subplot(position)
    ax.set_xlim([tra_mins[0], tra_maxs[0]])
    ax.set_ylim([tra_mins[1], tra_maxs[1]])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if tra_gt is not None:
        ax.scatter(*tra_gt.T[:2], s=100, marker="*", color=z_cm.to_rgba(tra_gt.T[2]))

    radii = np.mean(tra_stds, axis=1)
    coeffs = coeffs * 0.8 + 0.2
    for tra, radius, coeff in zip(tra_locs, radii, coeffs):
        ax.add_patch(
            Circle(
                xy=tra[:2],
                radius=radius,
                linewidth=2,
                facecolor=z_cm.to_rgba(tra[2]),
                edgecolor=z_cm.to_rgba(tra[2]),
                alpha=coeff,
            )
        )

    if cm_position is not None:
        cm_ax = figure.add_subplot(cm_position)
        mpl.colorbar.ColorbarBase(cm_ax, cmap=cmap, norm=norm)
        cm_ax.set_axis_off()


def plot_rot_dists_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_locs: np.ndarray,
    quat_lams: np.ndarray,
    coeffs: np.ndarray,
    quat_gt: Optional[np.ndarray] = None,
    cm_position: Optional[SubplotSpec] = None,
) -> None:
    """Plot components of Bingham mixture model rotation distribution on a 2D plane.

    Rotations are converted to Hopf coordinates, S^2 element of which is
    projected onto a 2D plane using Mollweide projection, and the S^1 element marcated
    by coloring the samples according to a color wheel.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        quat_locs: location of components in quaternions [w, x, y, z], shape (N, 4)
        quat_lams: concenrtration parameters of components, shape (N, 3)
        coeffs: weight coeffients of mixutre components, shape (N,)
        quat_gt: groundtruth quaternions [w, x, y, z] plotted as circles, shape (M, 4)
        cm_position: colorbar position on the figure
    """
    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
    s1_cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax = figure.add_subplot(position, projection="mollweide")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt)
        ax.scatter(*hopf_gt.T[:2], s=100, marker="*", color=s1_cm.to_rgba(hopf_gt.T[2]))

    hopf_locs = quat_to_hopf(quat_locs)
    radii = -5 * (np.mean(quat_lams, axis=1) ** -1)
    coeffs = coeffs * 0.8 + 0.2
    for hopf, radius, coeff in zip(hopf_locs, radii, coeffs):
        ax.add_patch(
            Circle(
                xy=hopf[:2],
                radius=radius,
                linewidth=2,
                facecolor=s1_cm.to_rgba(hopf[2]),
                edgecolor=s1_cm.to_rgba(hopf[2]),
                alpha=coeff,
            )
        )

    if cm_position is not None:
        cm_ax = figure.add_subplot(cm_position, projection="polar")
        cb = mpl.colorbar.ColorbarBase(
            cm_ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        cb.outline.set_visible(False)
        cm_ax.set_axis_off()
        cm_ax.set_rlim([-1, 1])


def plot_tra_hist_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    scene_dims: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
) -> None:
    """Plot translation samples in a histogram on a 2D plane.

    Z-axis is marginalized.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        scene_dims:
            2D array with rows containing minimum, maximum and margin values repectively
            and columns the x, y, z axes, shape (3, 3)
        tra_samples: translation samples to be plotted, shape (N, 3)
        tra_gt: groundtruth translations plotted as circles, shape (M, 3)
    """
    tra_mins = scene_dims[0] - scene_dims[2]
    tra_maxs = scene_dims[1] + scene_dims[2]
    xlim = [tra_mins[0], tra_maxs[0]]
    ylim = [tra_mins[1], tra_maxs[1]]

    ax = figure.add_subplot(position)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.hist2d(*tra_samples.T[:2], bins=50, range=[xlim, ylim])

    if tra_gt is not None:
        ax.scatter(*tra_gt.T[:2], s=20, color="red")


def plot_rot_hist_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_samples: np.ndarray,
    quat_gt: Optional[np.ndarray] = None,
) -> None:
    """Plot rotation samples in a histogram on a 2D plane.

    Rotation samples are converted to Hopf coordinates, S^2 element of which is
    projected onto a 2D plane using Mollweide projection, and the S^1 element
    marginalized.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        quat_samples: samples to be plotted in quaternions [w, x, y, z], shape (N, 4)
        quat_gt: groundtruth quaternions [w, x, y, z] plotted as circles, shape (M, 4)
    """
    ax = figure.add_subplot(position, projection="mollweide")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    hopf_samples = quat_to_hopf(quat_samples)
    hist, phi_edges, theta_edges = np.histogram2d(
        *hopf_samples.T[:2],
        bins=50,
        range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
    )
    ax.pcolor(
        phi_edges[:-1],
        theta_edges[:-1],
        hist.T,  # transpose from (row, column) to (x, y)
        cmap=mpl.cm.viridis,
        shading="auto",
        vmin=0,
        vmax=np.max(hist),
    )
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt)
        ax.scatter(*hopf_gt.T[:2], s=20, color="red")


def plot_tra_colored_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    scene_dims: np.ndarray,
    tra_samples: np.ndarray,
    colormap: mpl.cm.ScalarMappable,
) -> None:
    """Plot color-coded translation samples on a 2D plane.

    Z-axis is marginalized.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        scene_dims:
            2D array with rows containing minimum, maximum and margin values repectively
            and columns the x, y, z axes, shape (3, 3)
        tra_samples: translation samples to be plotted, shape (N, 3)
        colormap: colormap used for coloring samples
    """
    tra_mins = scene_dims[0] - scene_dims[2]
    tra_maxs = scene_dims[1] + scene_dims[2]

    ax = figure.add_subplot(position)
    ax.set_xlim([tra_mins[0], tra_maxs[0]])
    ax.set_ylim([tra_mins[1], tra_maxs[1]])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(
        *tra_samples.T[:2], s=10, color=colormap.to_rgba(np.arange(len(tra_samples)))
    )


def plot_rot_colored_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_samples: np.ndarray,
    colormap: mpl.cm.ScalarMappable,
) -> None:
    """Plot color-coded rotation samples on a 2D plane.

    Rotation samples are converted to Hopf coordinates, S^2 element of which is
    projected onto a 2D plane using Mollweide projection, and the S^1 element
    marginalized.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        quat_samples: samples to be plotted in quaternions [w, x, y, z], shape (N, 4)
        colormap: colormap used for coloring samples
    """
    ax = figure.add_subplot(position, projection="mollweide")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    hopf_samples = quat_to_hopf(quat_samples)
    ax.scatter(
        *hopf_samples.T[:2], s=10, color=colormap.to_rgba(np.arange(len(quat_samples)))
    )


def plot_code_colored_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    codes: np.ndarray,
    colormap: mpl.cm.ScalarMappable,
) -> None:
    """Plot color-coded latent encodings on a 2D plane.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        codes: latent encodings to be plotted, shape (N, 2)
        colormap: colormap used for coloring samples
    """
    ax = figure.add_subplot(position)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(*codes.T, s=10, color=colormap.to_rgba(np.arange(len(codes))))
