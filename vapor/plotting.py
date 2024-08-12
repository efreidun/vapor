"""Module containing plotting tools."""

from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tikzplotlib
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import Circle, Ellipse, Rectangle
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

from vapor.read_write import read_scene_dims
from vapor.utils import quat_to_hopf, quat_to_rotmat, rotmat_to_quat


def plot_ceiling_generation(
    query_image: Iterable[np.ndarray],
    test_tra_samples: np.ndarray,
    tra_gt: np.ndarray,
    ceil_img_corners: np.ndarray,
    vapor_tra_samples: np.ndarray,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, len N, shape (H, W, 3)
        test_tra_samples: translation samples, shape (N, M, 3)
        tra_gt: groundtruth translation, shape (N, 3)
        ceiling_img_corners: corners of images projection onto the ceiling, shape (N, 5, 2)
        vapor_tra_hat: vapor translation samples, shape (N, M, 3)
        save: save path including file extension

    Returns:
        figure instance
    """
    cmap = plt.cm.get_cmap("Dark2")

    fig = plt.figure(figsize=(15, 10))
    grid_spec = GridSpec(60, 100)

    tra_gt_vis = (tra_gt - np.array([[13.214, 0.232, 0]])) * np.array(
        [[1, -1, 1]]
    ) * 1000 + np.array(
        [[778, 443, 0]]
    )  # (N, 3)
    # test_tra_samples_vis = (
    #     test_tra_samples - np.array([[[13.214, 0.232, 0]]])
    # ) * np.array([[[1, -1, 1]]]) * 1000 + np.array(
    #     [[[778, 443, 0]]]
    # )  # (N, M, 3)
    # vapor_tra_samples_vis = (
    #     vapor_tra_samples - np.array([[[13.214, 0.232, 0]]])
    # ) * np.array([[[1, -1, 1]]]) * 1000 + np.array(
    #     [[[778, 443, 0]]]
    # )  # (N, M, 3)

    # KDE
    factor = 0.02
    test_kernels = [
        gaussian_kde(samples[:, 0], bw_method=factor) for samples in test_tra_samples
    ]
    vapor_kernels = [
        gaussian_kde(samples[:, 0], bw_method=factor) for samples in vapor_tra_samples
    ]
    pixel_positions = np.arange(0, 16500, 30)  # (K,)
    metric_positions = (pixel_positions - 778) / 1000 + 13.214  # (K,)
    test_densities = [kernel(metric_positions) for kernel in test_kernels]  # P, (K,)
    vapor_densities = [kernel(metric_positions) for kernel in vapor_kernels]  # P, (K,)

    # cropping the images
    size = int(480 / 256 * 224)  # 420
    w_margin = (640 - size) // 2
    h_margin = (480 - size) // 2
    num_examples = len(query_image)
    # for each example
    for i in range(num_examples):
        # show the query image
        show_image(
            fig,
            grid_spec[i * 15 + 3 : (i + 1) * 15, 0:15],
            "",
            np.transpose(
                query_image[i][
                    h_margin + size : h_margin : -1, w_margin + size : w_margin : -1, :
                ],
                (1, 0, 2),
            ),
        )

        # show the distribution
        ax = fig.add_subplot(grid_spec[i * 15 + 3 : (i + 1) * 15, 15:100])
        ax.fill_between(
            pixel_positions,
            vapor_densities[i],
            color=cmap.colors[1],
            alpha=0.5,
            zorder=0,
        )
        ax.plot(pixel_positions, vapor_densities[i], color=cmap.colors[1], zorder=1)
        ax.fill_between(
            pixel_positions,
            test_densities[i],
            color=cmap.colors[0],
            alpha=0.6,
            zorder=2,
        )
        ax.plot(
            pixel_positions,
            test_densities[i],
            color=cmap.colors[0],
            alpha=0.9,
            zorder=3,
        )
        # ax.hist(
        #     vapor_tra_samples_vis[i, :, 0],
        #     bins=16500 // 100,
        #     range=[0, 16500],
        #     density=True,
        #     alpha=0.8,
        #     color=cmap.colors[num_examples + 2],
        #     label="CVAEPoR",
        # )
        # ax.hist(
        #     test_tra_samples_vis[i, :, 0],
        #     bins=16500 // 100,
        #     range=[0, 16500],
        #     density=True,
        #     alpha=0.8,
        #     color=cmap.colors[num_examples + 1],
        #     label="CVAEPoR",
        # )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([0, 16500])
        ax.set_ylim(
            [
                0.0,
                1.05 * np.max(np.concatenate((vapor_densities[i], test_densities[i]))),
            ]
        )
        ax.axvline(
            tra_gt_vis[i, 0],
            linewidth=2,
            color=cmap.colors[i + 2],
            label="Ground truth",
        )
        # ax.legend(loc="upper right")

    # show stitched ceiling
    ax = fig.add_subplot(grid_spec[50:60, 15:100])
    ax.set_aspect("equal")
    stitched_img = np.array(Image.open(Path.home() / "data/Rig/Ceiling/stitched.jpg"))
    ax.imshow(stitched_img)
    xticks = np.linspace(0, 16000, 9)
    xticklabels = xticks / 1000
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks([])
    ax.set_xlabel("x (m)")
    for i in range(num_examples):
        ax.plot(
            *ceil_img_corners[i].T,
            linewidth=2,
            color=cmap.colors[i + 2],
        )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        tikzplotlib.save(save.with_suffix(".tex"))


def plot_colors_generation(
    query_image: np.ndarray,
    test_tra_samples: np.ndarray,
    tra_gt: Iterable[np.ndarray],
    zp: np.ndarray,
    vapor_tra_samples: np.ndarray,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (N, H, W, 3)
        test_tra_samples: translation samples, shape (N, M, 3)
        tra_gt: groundtruth translation, len N, shape (O, 3)
        zp: sampels from the latent prior, shape (N, M, L)
        vapor_tra_samples: vapor translation samples, shape (N, M, 3)
        save: save path including file extension

    Returns:
        figure instance
    """
    cmap = plt.cm.get_cmap("Dark2")

    # training fiture
    fig = plt.figure(figsize=(15, 10))
    grid_spec = GridSpec(100, 100)

    # test_tra_samples_vis = test_tra_samples * np.array([[[100, 1, 1]]]) + np.array(
    #     [[[112, 0, 0]]]
    # )  # (N, M, 3)
    # vapor_tra_samples_vis = vapor_tra_samples * np.array([[[100, 1, 1]]]) + np.array(
    #     [[[112, 0, 0]]]
    # )  # (N, M, 3)

    assignments = np.array(
        [
            np.argmin(np.abs(test[None, ..., 0] - gt[:, None, 0]), axis=0)
            for gt, test in zip(tra_gt, test_tra_samples)  # (O, 3), (M, 3)
        ]
    )  # (N, M)

    # KDE
    factor = 0.01
    single_kernels = [
        gaussian_kde(samples[:, 0], bw_method=factor) for samples in test_tra_samples
    ]  # N
    vapor_kernels = [
        gaussian_kde(samples[:, 0], bw_method=factor) for samples in vapor_tra_samples
    ]  # N
    pixel_positions = np.arange(0, 5120, 10)  # (K,)
    metric_positions = (pixel_positions - 112) / 100  # (K,)
    test_densities = [kernel(metric_positions) for kernel in single_kernels]  # P, (K,)
    vapor_densities = [kernel(metric_positions) for kernel in vapor_kernels]  # P, (K,)

    density_assignments = np.array(
        [
            np.argmin(np.abs(metric_positions[None, :] - gt[:, None, 0]), axis=0)
            for gt in tra_gt  # (O, 3)
        ]
    )  # (N, K)

    # for each example
    for i in range(len(tra_gt)):  # N
        # show query image
        show_image(
            fig,
            grid_spec[5 + i * 20 : (i + 1) * 20, 0:15],
            "",
            query_image[i],
        )

        # show latent space
        ax = fig.add_subplot(grid_spec[5 + i * 20 : (i + 1) * 20, 15:30])
        for radius, color in zip(
            [1.0, 2.0, 3.0],
            np.array([[178] * 3, [204] * 3, [229] * 3]) / 255,
        ):
            ax.add_patch(
                Circle(
                    xy=(0, 0),
                    radius=radius,
                    linewidth=3,
                    facecolor="none",
                    edgecolor=color,
                    zorder=0,
                )
            )
        for j in range(len(tra_gt[i])):  # O
            mask = assignments[i] == j  # (M,)
            ax.scatter(*zp[i][mask].T, color=cmap.colors[j], alpha=0.5, s=1)
        ax.set_aspect("equal")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

        # show pose histogram
        ax = fig.add_subplot(grid_spec[5 + i * 20 : (i + 1) * 20, 30:100])
        ax.fill_between(
            pixel_positions,
            vapor_densities[i],
            color=cmap.colors[-1],
            alpha=0.5,
            zorder=0,
        )
        ax.plot(pixel_positions, vapor_densities[i], color=cmap.colors[-1], zorder=1)
        # ax.hist(
        #     vapor_tra_samples_vis[i][:, 0],
        #     bins=100,
        #     range=[0, 5120],
        #     density=False,
        #     zorder=0,
        #     alpha=1.0,
        #     label="VaPoR",
        #     color=cmap.colors[-1],
        # )
        for j in range(len(tra_gt[i])):  # O
            mask = density_assignments[i] == j
            ax.fill_between(
                pixel_positions[mask],
                test_densities[i][mask],
                color=cmap.colors[j],
                alpha=0.6,
                zorder=2,
            )
            ax.plot(
                pixel_positions[mask],
                test_densities[i][mask],
                color=cmap.colors[j],
                alpha=0.9,
                zorder=3,
            )
            # mask = assignments[i] == j  # (M,)
            # ax.hist(
            #     test_tra_samples_vis[i][mask][:, 0],
            #     bins=100,
            #     range=[0, 5120],
            #     density=False,
            #     zorder=0,
            #     alpha=0.8,
            #     label="Reconstruction",
            #     color=cmap.colors[j],
            # )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([0, 5120])
        ax.set_ylim(
            [0, 1.05 * np.max(np.concatenate((test_densities[i], vapor_densities[i])))]
        )

    # show colors sequence
    ax = fig.add_subplot(grid_spec[50:90, 30:100])
    stitched_img = np.array(
        Image.open(Path.home() / "data/Synthetic/colors/colors.png")
    )
    ax.imshow(stitched_img)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        tikzplotlib.save(save.with_suffix(".tex"))


def plot_colors_paper(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    test_tra_samples: np.ndarray,
    tra_gt: np.ndarray,
    z: np.ndarray,
    zp: np.ndarray,
    save: Optional[Union[Path, str]] = None,
    perplexity: float = 50.0,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (N, H, W, 3)
        tra_samples: translation samples, shape (N, O, M, 3)
        test_tra_samples: translation samples, shape (N, M, 3)
        tra_gt: groundtruth translation, shape (N, O, 3)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (N, O, M, L)
        zp:
            Sampels from the latent prior, shape (N, M, L)
        save: save path including file extension

    Returns:
        figure instance
    """
    z_emb = z  # (N, O, M, 2)

    colors = (
        np.array(
            [
                [31, 119, 180],  # tab:blue
                [255, 127, 14],  # tab:orange
                [44, 160, 44],  # tab:green
                [214, 39, 40],  # tab:red
                [148, 103, 189],  # tab:purple
                [188, 189, 34],  # tab:olive
                [227, 119, 194],  # tab:pink
            ]
        )
        / 255
    )
    cmap = plt.cm.get_cmap("Dark2")

    # training fiture
    fig = plt.figure(figsize=(15, 10))
    grid_spec = GridSpec(100, 100)

    tra_samples_vis = [
        samps * np.array([[[100, 1, 1]]]) + np.array([[[112, 0, 0]]])
        for samps in tra_samples
    ]
    tra_gt_vis = [
        gt * np.array([[100, 1, 1]]) + np.array([[112, 0, 0]]) for gt in tra_gt
    ]
    test_tra_samples_vis = test_tra_samples * np.array([[[100, 1, 1]]]) + np.array(
        [[[112, 0, 0]]]
    )

    assignments = [
        np.argmin(np.abs(test[None, ..., 0] - gt[:, None, 0]), axis=0)
        for gt, test in zip(tra_gt, test_tra_samples)
    ]  # (N, M)

    for i in range(3):  # N
        show_image(
            fig,
            grid_spec[5 + i * 20 : (i + 1) * 20, 0:15],
            "",
            query_image[i],
        )

        ax = fig.add_subplot(grid_spec[5 + i * 20 : (i + 1) * 20, 15:30])
        for j in range(len(tra_gt[i])):  # O
            mask = assignments[i] == j
            ax.scatter(*zp[i][mask].T, color=cmap.colors[j], alpha=0.5, s=1)
        # ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
        ax.set_aspect("equal")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=1.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([178, 178, 178]) / 255,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=2.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([204, 204, 204]) / 255,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=3.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([229, 229, 229]) / 255,
                zorder=0,
            )
        )
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

        ax = fig.add_subplot(grid_spec[5 + i * 20 : (i + 1) * 20, 30:100])
        print("HELOOOO")
        for j in range(len(tra_gt[i])):  # O
            mask = assignments[i] == j
            print(np.sum(mask))
            ax.hist(
                test_tra_samples_vis[i][mask][:, 0],
                bins=100,
                range=[0, 5120],
                density=False,
                zorder=0,
                alpha=1.0,
                label="Reconstruction",
                color=cmap.colors[j],
            )
        # ax.hist(
        #     tra_samples_vis[i, :, 0],
        #     bins=16500 // 100,
        #     range=[0, 16500],
        #     density=True,
        #     zorder=0,
        #     alpha=0.7,
        #     label="Reconstruction",
        # )
        ax.set_yticks([])
        ax.set_xticks([])
        # xticks = np.linspace(0, 16000, 9)
        # xticklabels = xticks / 1000
        # yticks = np.linspace(0, 1000, 3)
        # yticklabels = yticks / 1000
        # ax.set_xticklabels(xticklabels)
        # ax.set_xlabel("x (m)")
        # ax.set_ylabel("density")
        ax.set_xlim([0, 5120])
        # ax.set_ylim([0, 0.002])
        # ax.axvline(
        #     tra_gt_vis[i][0],
        #     linewidth=2,
        #     color=colors[i],
        #     zorder=2,
        #     label="Ground truth",
        # )
        # ax.legend(loc="upper right")

    ax = fig.add_subplot(grid_spec[50:90, 30:100])
    ax.set_aspect("equal")
    stitched_img = np.array(
        Image.open(Path.home() / "data/Synthetic/colors/colors.png")
    )
    ax.imshow(stitched_img)
    # for i in range(3):
    #     ax.add_patch(
    #         Rectangle(
    #             xy=[tra_gt_vis[i, 0] - 112, 0],
    #             width=224,
    #             height=224,
    #             linewidth=2,
    #             facecolor="none",
    #             edgecolor=colors[i],
    #         )
    #     )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))


def get_ceiling_projections(idcs: Iterable) -> np.ndarray:
    with open(Path.home() / "data/Rig/Ceiling/homographies.txt") as f:
        content = f.readlines()
    homographies = np.array(
        [[float(entry) for entry in line.strip().split(" ")] for line in content],
        dtype=np.float32,
    )[:, 1:].reshape(
        -1, 3, 3
    )  # (N, 3, 3)
    # cropped image
    size = int(480 / 256 * 224)  # 420
    w_margin = (640 - size) // 2
    h_margin = (480 - size) // 2
    img_corners = np.array(
        [
            [w_margin, h_margin],
            [w_margin + size, h_margin],
            [w_margin + size, h_margin + size],
            [w_margin, h_margin + size],
        ]
    )  # (4, 2)
    # full image
    # img_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]])  # (4, 2)
    img_corners = np.concatenate((img_corners, np.ones((4, 1))), axis=1)  # (4, 3)
    ceil_img_corners = np.stack(
        [(homographies[idx] @ img_corners.T).T for idx in idcs]
    )  # (N, 4, 3)
    ceil_img_corners = ceil_img_corners[:, :, :2] / ceil_img_corners[:, :, 2:]
    # ceil_img_corners[:, :, 1] += 100
    return np.concatenate(
        (ceil_img_corners, ceil_img_corners[:, :1]), axis=1
    )  # (N, 5, 2)


def plot_ceiling_paper(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    test_tra_samples: np.ndarray,
    tra_gt: np.ndarray,
    z: np.ndarray,
    ceil_img_corners: np.ndarray,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (N, H, W, 3)
        tra_samples: translation samples, shape (N, M, 3)
        test_tra_samples: translation samples, shape (N, M, 3)
        tra_gt: groundtruth translation, shape (N, 3)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (N, M, L)
        save: save path including file extension

    Returns:
        figure instance
    """
    num_modes, num_samples, lat_dim = z.shape

    colors = (
        np.array(
            [
                [31, 119, 180],  # tab:blue
                [255, 127, 14],  # tab:orange
                [44, 160, 44],  # tab:green
                [214, 39, 40],  # tab:red
                [148, 103, 189],  # tab:purple
                [188, 189, 34],  # tab:olive
                [227, 119, 194],  # tab:pink
            ]
        )
        / 255
    )

    # training fiture
    fig = plt.figure(figsize=(15, 10))
    grid_spec = GridSpec(100, 100)

    tra_samples_vis = (tra_samples - np.array([[[13.214, 0.232, 0]]])) * np.array(
        [[[1, -1, 1]]]
    ) * 1000 + np.array([[[778, 443, 0]]])
    tra_gt_vis = (tra_gt - np.array([[13.214, 0.232, 0]])) * np.array(
        [[1, -1, 1]]
    ) * 1000 + np.array([[778, 443, 0]])
    test_tra_samples_vis = (
        test_tra_samples - np.array([[[13.214, 0.232, 0]]])
    ) * np.array([[[1, -1, 1]]]) * 1000 + np.array([[[778, 443, 0]]])

    for i in range(3):
        show_image(
            fig,
            grid_spec[i * 25 : (i + 1) * 25, 0:15],
            "",
            query_image[i][::-1, :, :],
        )

        ax = fig.add_subplot(grid_spec[i * 25 : (i + 1) * 25, 15:100])
        ax.hist(
            test_tra_samples_vis[i, :, 0],
            bins=16500 // 100,
            range=[0, 16500],
            density=True,
            zorder=0,
            alpha=1.0,
            label="Generation",
        )
        # ax.hist(
        #     tra_samples_vis[i, :, 0],
        #     bins=16500 // 100,
        #     range=[0, 16500],
        #     density=True,
        #     zorder=0,
        #     alpha=0.7,
        #     label="Reconstruction",
        # )
        ax.set_yticks([])
        ax.set_xticks([])
        # xticks = np.linspace(0, 16000, 9)
        # xticklabels = xticks / 1000
        # yticks = np.linspace(0, 1000, 3)
        # yticklabels = yticks / 1000
        # ax.set_xticklabels(xticklabels)
        # ax.set_xlabel("x (m)")
        # ax.set_ylabel("density")
        ax.set_xlim([0, 16500])
        # ax.set_ylim([0, 0.002])
        ax.axvline(
            tra_gt_vis[i, 0],
            linewidth=2,
            color=colors[i],
            zorder=2,
            label="Ground truth",
        )
        ax.legend(loc="upper right")

    ax = fig.add_subplot(grid_spec[75:100, 15:100])
    ax.set_aspect("equal")
    stitched_img = np.array(Image.open(Path.home() / "data/Rig/Ceiling/stitched.jpg"))
    ax.imshow(stitched_img)
    # xticks = np.linspace(0, 16000, 9)
    # xticklabels = xticks / 1000
    # yticks = np.linspace(0, 1000, 3)
    # yticklabels = yticks / 1000
    # ax.set_xticklabels(xticklabels)
    # ax.set_yticklabels(yticklabels)
    # ax.set_xlabel("x (m)")
    # ax.set_ylabel("y (m)")
    for i in range(3):
        # ax.add_patch(
        #     Circle(
        #         xy=tra_gt_vis[i, :2],
        #         radius=150,
        #         linewidth=2,
        #         facecolor="none",
        #         edgecolor=colors[i],
        #     )
        # )
        # breakpoint()
        ax.plot(
            *ceil_img_corners[i].T,
            linewidth=2,
            color=colors[i],
        )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))


def plot_staircase_latent_3d(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    rot_samples: np.ndarray,
    test_tra_samples: np.ndarray,
    test_rot_samples: np.ndarray,
    tra_gt: np.ndarray,
    rot_gt: np.ndarray,
    z: np.ndarray,
    scene_path: Path,
    zoom: int,
    save: Optional[Union[Path, str]] = None,
    perplexity: float = 50.0,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (N, H, W, 3)
        tra_samples: translation samples, shape (N, M, 3)
        rot_samples: rotation samples, shape (N, M, 3, 3)
        test_tra_samples: translation samples, shape (N, M, 3)
        test_rot_samples: rotation samples, shape (N, M, 3, 3)
        tra_gt: groundtruth translation, shape (N, 3)
        rot_gt: groundtruth orientation, shape (N, 3, 3)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (N, M, L)
        save: save path including file extension

    Returns:
        figure instance
    """
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perplexity,
    )
    num_modes, num_samples, lat_dim = z.shape
    zp = np.random.multivariate_normal(
        np.zeros(lat_dim), np.eye(lat_dim), num_samples
    )  # (M, L)
    if lat_dim == 2:
        z_emb = z  # (M, 2)
        zp_emb = zp  # (M, 2)
    else:
        zs = np.concatenate((z, zp[None, ...]), axis=0)  # (N+1, M, L)
        zs_emb = tsne.fit_transform(zs.reshape(-1, lat_dim)).reshape(
            num_modes + 1, num_samples, 2
        )  # (N+1, M, 2)
        z_emb = zs_emb[:-1]  # (N, M, 2)
        zp_emb = zs_emb[-1]  # (M, 2)

    colors = (
        np.array(
            [
                [31, 119, 180],  # tab:blue
                [255, 127, 14],  # tab:orange
                [44, 160, 44],  # tab:green
                [214, 39, 40],  # tab:red
                [148, 103, 189],  # tab:purple
                [188, 189, 34],  # tab:olive
                [227, 119, 194],  # tab:pink
            ]
        )
        / 255
    )

    # training fiture
    fig = plt.figure(figsize=(15, 10))
    grid_spec = GridSpec(100, 100)

    quat_samples = rotmat_to_quat(rot_samples.reshape(-1, 3, 3)).reshape(
        num_modes, num_samples, 4
    )
    quat_gt = rotmat_to_quat(rot_gt)
    for i in range(num_modes):
        show_image(
            fig,
            grid_spec[0:20, i * 14 : (i + 1) * 14],
            "",
            query_image[i],
        )

    ax = fig.add_subplot(grid_spec[25:100, 0:50])
    for i in range(num_modes):
        ax.scatter(*z_emb[i].T, color=colors[i], alpha=1.0, s=2)
    # ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=1.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([178, 178, 178]) / 255,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=2.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([204, 204, 204]) / 255,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=3.0,
                linewidth=3,
                facecolor="none",
                edgecolor=np.array([229, 229, 229]) / 255,
                zorder=0,
            )
        )
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
    print("Plotting")

    render = render_3d_modes(
        scene_path,
        tra_samples,
        quat_samples,
        colors,
        tra_gt,
        quat_gt,
    )
    ren_h, ren_w, _ = render.shape
    offset = (1 - 1 / zoom) / 2
    show_image(
        fig,
        grid_spec[25:100, 50:100],
        "",
        # render,
        render[
            int(offset * ren_h) : int((1 - offset) * ren_h),
            int(offset * ren_w) : int((1 - offset) * ren_w),
        ],
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))

    # test fiture
    for i, (test_tra_samps, test_rot_samps) in enumerate(
        zip(test_tra_samples, test_rot_samples)
    ):
        fig = plt.figure(figsize=(15, 10))
        grid_spec = GridSpec(100, 100)

        test_quat_samps = rotmat_to_quat(test_rot_samps)
        show_image(
            fig,
            grid_spec[0:20, i * 14 : (i + 1) * 14],
            "",
            query_image[i],
        )

        ax = fig.add_subplot(grid_spec[25:100, 0:50])
        ax.scatter(*zp_emb.T, color=colors[3], alpha=1.0, s=2)
        # ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
        ax.set_aspect("equal")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        if lat_dim == 2:
            ax.add_patch(
                Circle(
                    xy=(0, 0),
                    radius=1.0,
                    linewidth=3,
                    facecolor="none",
                    edgecolor=np.array([178, 178, 178]) / 255,
                    zorder=0,
                )
            )
            ax.add_patch(
                Circle(
                    xy=(0, 0),
                    radius=2.0,
                    linewidth=3,
                    facecolor="none",
                    edgecolor=np.array([204, 204, 204]) / 255,
                    zorder=0,
                )
            )
            ax.add_patch(
                Circle(
                    xy=(0, 0),
                    radius=3.0,
                    linewidth=3,
                    facecolor="none",
                    edgecolor=np.array([229, 229, 229]) / 255,
                    zorder=0,
                )
            )
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
        print("Plotting")

        render = render_3d_modes(
            scene_path,
            test_tra_samps[None, ...],
            test_quat_samps[None, ...],
            colors[3:4],
            tra_gt[[i]],
            quat_gt[[i]],
        )
        ren_h, ren_w, _ = render.shape
        show_image(
            fig,
            grid_spec[25:100, 50:100],
            "",
            # render,
            render[
                int(offset * ren_h) : int((1 - offset) * ren_h),
                int(offset * ren_w) : int((1 - offset) * ren_w),
            ],
        )

        if save is not None:
            save = save.parent / f"test_{i}.png"
            print(f"Saving plot {save}")
            fig.savefig(save)
            # tikzplotlib.save(save.with_suffix(".tex"))


def plot_staircase_generation(
    query_image: Iterable[np.ndarray],
    tra_samples: np.ndarray,
    rot_samples: np.ndarray,
    tra_gt: np.ndarray,
    rot_gt: np.ndarray,
    zp: np.ndarray,
    scene_path: Path,
    mu: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    zoom: int = 1,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Assumes samples are all from the first ground-truth mode.

    Args:
        query_image: query images, len N, shape (H, W, 3)
        tra_samples: translation samples, shape (M, 3)
        rot_samples: rotation samples, shape (M, 3, 3)
        tra_gt: ground-truth positions, shape (N, 3)
        rot_gt: ground-truth rotations, shape (N, 3, 3)
        zp: samples from the latent prior, shape (M, 2)
        mu: latent posterior mean, shape (2,)
        std: latent posterior standard deviation, shape (2,)
        zoom: How much to zoom in render, e.g. 2 = 2x
        save: save path including file extension

    Returns:
        figure instance
    """
    num_modes = len(query_image)
    cmap = plt.cm.get_cmap("Dark2")
    cmap_gt = plt.cm.get_cmap("Set2")
    assignments = np.zeros(len(tra_samples))  # (M,)

    fig = plt.figure(figsize=(10, 6))
    grid_spec = GridSpec(100, 100)

    # show images
    for i in range(num_modes):
        img = query_image[i]
        strip = (
            np.tile(np.array(cmap.colors[i])[None, None, :], (30, img.shape[1], 1))
            * 255
        ).astype(int)
        img = np.concatenate((img, strip), axis=0)
        show_image(
            fig,
            grid_spec[0:20, i * 14 : (i + 1) * 14],
            "",
            img,
        )

    # show latent samples
    ax = fig.add_subplot(grid_spec[25:100, 0:50])
    for radius, color in zip(
        [1.0, 2.0, 3.0],
        np.array([[178] * 3, [204] * 3, [229] * 3]) / 255,
    ):
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=radius,
                linewidth=3,
                facecolor="none",
                edgecolor=color,
                zorder=0,
            )
        )

    for i in range(num_modes):
        mask = assignments == i
        if np.sum(mask) > 0 and mu is not None and std is not None:
            for j, alpha in enumerate([0.8, 0.6, 0.4]):
                ax.add_patch(
                    Ellipse(
                        xy=mu,
                        width=2 * (j + 1) * std[0],
                        height=2 * (j + 1) * std[1],
                        linewidth=3,
                        facecolor="none",
                        edgecolor=cmap.colors[i],
                        alpha=np.mean(mask) * alpha,
                        zorder=1,
                    )
                )
        ax.scatter(*zp[mask].T, color=cmap.colors[i], alpha=0.8, s=1)
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    # show render of camera frusta
    quat_samples = rotmat_to_quat(rot_samples)  # (M, 4)
    quat_gt = rotmat_to_quat(rot_gt)  # (N, 4)
    print("rendering in open3d")
    render = render_3d_modes(
        scene_path,
        tra_samples[:100],  # (M, 3)
        quat_samples[:100],  # (M, 4)
        cmap.colors,  # >= N
        assignments[:100],  # (M,) contains an integer index in [0, N)
        tra_gt,  # (N, 3)
        quat_gt,  # (N, 4)
        np.roll(np.array(cmap.colors), -1, axis=0),  # cmap_gt.colors,  # N
        num_sample_noise=10,
    )
    offset = (1 - 1 / zoom) / 2
    ren_h, ren_w, _ = render.shape
    show_image(
        fig,
        grid_spec[25:100, 50:100],
        "",
        render[
            int(offset * ren_h) : int((1 - offset) * ren_h),
            int(offset * ren_w) : int((1 - offset) * ren_w),
        ],
    )

    # save figure
    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_bluechairs_generation(
    query_image: Iterable[np.ndarray],
    tra_samples: np.ndarray,
    rot_samples: np.ndarray,
    tra_gt: np.ndarray,
    zp: np.ndarray,
    scene_path: Path,
    zoom: int = 1,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query images, len N, shape (H, W, 3)
        tra_samples: translation samples, shape (M, 3)
        rot_samples: rotation samples, shape (M, 3, 3)
        tra_gt: ground-truth positions, shape (N, 3)
        zp: samples from the latent prior, shape (M, 2)
        zoom: How much to zoom in render, e.g. 2 = 2x
        save: save path including file extension

    Returns:
        figure instance
    """
    num_modes = len(query_image)
    cmap = plt.cm.get_cmap("Dark2")
    assignments = np.argmin(
        np.linalg.norm(tra_samples[None, ...] - tra_gt[:, None, :], axis=2), axis=0
    )  # (M,) contains an integer index in [0, N)

    fig = plt.figure(figsize=(10, 6))
    grid_spec = GridSpec(100, 100)

    # show images
    for i in range(num_modes):
        img = query_image[i]
        strip = (
            np.tile(np.array(cmap.colors[i])[None, None, :], (30, img.shape[1], 1))
            * 255
        ).astype(int)
        img = np.concatenate((img, strip), axis=0)
        show_image(
            fig,
            grid_spec[0:20, i * 14 : (i + 1) * 14],
            "",
            img,
        )

    # show latent samples
    ax = fig.add_subplot(grid_spec[25:100, 0:50])
    for radius, color in zip(
        [1.0, 2.0, 3.0],
        np.array([[178] * 3, [204] * 3, [229] * 3]) / 255,
    ):
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=radius,
                linewidth=3,
                facecolor="none",
                edgecolor=color,
                zorder=0,
            )
        )
    for i in range(num_modes):
        mask = assignments == i
        ax.scatter(*zp[mask].T, color=cmap.colors[i], alpha=0.8, s=1)
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    # show render of camera frasta
    quat_samples = rotmat_to_quat(rot_samples)  # (M, 4)
    print("rendering in open3d")
    render = render_3d_modes(
        scene_path,
        tra_samples[:100],  # (M, 3)
        quat_samples[:100],  # (M, 4)
        cmap.colors,  # >= N
        assignments[:100],  # (M,) contains an integer index in [0, N)
        num_sample_noise=10,
    )
    offset = (1 - 1 / zoom) / 2
    ren_h, ren_w, _ = render.shape
    show_image(
        fig,
        grid_spec[25:100, 50:100],
        "",
        render[
            int(offset * ren_h) : int((1 - offset) * ren_h),
            int(offset * ren_w) : int((1 - offset) * ren_w),
        ],
    )

    # save figure
    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_bluechairs_latent_3d(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    rot_samples: np.ndarray,
    z: np.ndarray,
    scene_path: Path,
    save: Optional[Union[Path, str]] = None,
    perplexity: float = 50.0,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (N, H, W, 3)
        tra_samples: translation samples, shape (N, M, 3)
        rot_samples: rotation samples, shape (N, M, 3, 3)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (N, M, L)
        save: save path including file extension

    Returns:
        figure instance
    """
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perplexity,
    )
    num_modes, num_samples, lat_dim = z.shape
    zp = np.random.multivariate_normal(
        np.zeros(lat_dim), np.eye(lat_dim), num_samples
    )  # (M, L)
    if lat_dim == 2:
        z_emb = z  # (M, 2)
        zp_emb = zp  # (M, 2)
    else:
        zs = np.concatenate((z, zp[None, ...]), axis=0)  # (N+1, M, L)
        zs_emb = tsne.fit_transform(zs.reshape(-1, lat_dim)).reshape(
            num_modes + 1, num_samples, 2
        )  # (N+1, M, 2)
        z_emb = zs_emb[:-1]  # (N, M, 2)
        zp_emb = zs_emb[-1]  # (M, 2)

    colors = (
        np.array(
            [
                [31, 119, 180],  # tab:blue
                [255, 127, 14],  # tab:orange
                [44, 160, 44],  # tab:green
                [214, 39, 40],  # tab:red
                [148, 103, 189],  # tab:purple
                [188, 189, 34],  # tab:olive
                [227, 119, 194],  # tab:pink
            ]
        )
        / 255
    )

    fig = plt.figure(figsize=(10, 6))
    # fig.suptitle(title)
    grid_spec = GridSpec(100, 100)

    quat_samples = rotmat_to_quat(rot_samples.reshape(-1, 3, 3)).reshape(
        num_modes, num_samples, 4
    )
    for i in range(num_modes):
        show_image(
            fig,
            grid_spec[0:20, i * 14 : (i + 1) * 14],
            "",
            query_image[i],
        )

    ax = fig.add_subplot(grid_spec[25:100, 0:50])
    for i in range(num_modes):
        ax.scatter(*z_emb[i].T, color=colors[i], alpha=1.0, s=2)
    # ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
    print("YOO")

    render = render_3d_modes(
        scene_path,
        tra_samples,
        quat_samples,
        colors,
    )
    ren_h, ren_w, _ = render.shape
    show_image(
        fig,
        grid_spec[25:100, 50:100],
        "",
        render[
            int(0.3 * ren_h) : int(0.7 * ren_h), int(0.3 * ren_w) : int(0.65 * ren_w)
        ],
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_colors_latent(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: np.ndarray,
    z0: np.ndarray,
    z: np.ndarray,
    test_tra_samples: np.ndarray,
    save: Optional[Union[Path, str]] = None,
    perplexity: float = 50.0,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (H, W, 3)
        tra_samples: translation samples, shape (N, 3)
        tra_gt: groundtruth translation, shape (3,)
        z0: Samples from Gaussian-modelled latent posterior, shape (M, L)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (M, L)
        test_tra_samples: test translation samples, shape (N, 3)
        save: save path including file extension

    Returns:
        figure instance
    """
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perplexity,
    )
    num_samples, lat_dim = z0.shape
    zp = np.random.multivariate_normal(
        np.zeros(lat_dim), np.eye(lat_dim), num_samples
    )  # (M, L)
    if lat_dim == 2:
        z0_emb = z0  # (M, 2)
        z_emb = z  # (M, 2)
        zp_emb = zp  # (M, 2)
    else:
        zs = np.concatenate((z0, z, zp), axis=0)  # (3*M, L)
        zs_emb = tsne.fit_transform(zs)  # (3*M, 2)
        z0_emb = zs_emb[:num_samples]  # (M, 2)
        z_emb = zs_emb[num_samples : 2 * num_samples]  # (M, 2)
        zp_emb = zs_emb[2 * num_samples : 3 * num_samples]  # (M, 2)

    fig = plt.figure(figsize=(25, 6))
    # fig.suptitle(title)
    grid_spec = GridSpec(58, 100)

    img_corner = (np.array(list(query_image.shape[:2])) - 224) // 2

    # latent space
    ax = fig.add_subplot(grid_spec[0:25, 0:20])
    ax.scatter(*zp_emb.T, c="blue", alpha=0.3, s=2)
    ax.scatter(*z0_emb.T, c="red", alpha=0.3, s=2)
    ax.set_title("Gaussian posterior")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

    ax = fig.add_subplot(grid_spec[40:57, 0:20])
    ax.scatter(*zp_emb.T, c="blue", alpha=0.3, s=2)
    ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
    ax.set_title("normalizing flows")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

    # first row
    show_image(
        fig,
        grid_spec[00:25, 90:100],  # grid_spec[0:30, 20:40],
        "query image",
        np.transpose(query_image.numpy(force=True), (1, 2, 0)),
    )

    tra_samples_vis = tra_samples * np.array([[100, 1, 1]]) + np.array([[112, 0, 0]])
    tra_gt_vis = tra_gt * np.array([100, 1, 1]) + np.array([112, 0, 0])
    test_tra_samples_vis = test_tra_samples * np.array([[100, 1, 1]]) + np.array(
        [[112, 0, 0]]
    )
    ax = fig.add_subplot(grid_spec[0:25, 25:85])
    ax.set_title("marginal histogram on x-axis")
    ax.hist(
        tra_samples_vis[:, 0],
        bins=100,
        range=[0, 5120],
        density=True,
        zorder=0,
        alpha=1.0,
        label="reconstruction",
    )
    ax.hist(
        test_tra_samples_vis[:, 0],
        bins=100,
        range=[0, 5120],
        density=True,
        zorder=0,
        alpha=0.7,
        label="generation",
    )
    # ax.set_yticks([])
    # xticks = np.linspace(0, 16000, 9)
    # xticklabels = xticks / 1000
    # yticks = np.linspace(0, 1000, 3)
    # yticklabels = yticks / 1000
    # ax.set_xticklabels(xticklabels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("density")
    ax.set_xlim([0, 5120])
    ax.set_ylim([0, 0.005])
    ax.vlines(
        tra_gt_vis[0],
        0,
        0.002,
        linewidth=2,
        edgecolor="limegreen",
        zorder=2,
        label="Ground truth",
    )
    ax.legend(loc="upper right")

    # second row
    ax = fig.add_subplot(grid_spec[40:57, 25:85])
    ax.set_aspect("equal")
    ax.set_title("scene")
    stitched_img = np.array(
        Image.open(Path.home() / "data/Synthetic/colors/colors.png")
    )
    ax.imshow(stitched_img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(
        Rectangle(
            xy=[tra_gt_vis[0] - 112, 0],
            width=224,
            height=224,
            linewidth=2,
            facecolor="none",
            edgecolor="black",
        )
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_ceiling_latent(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: np.ndarray,
    z0: np.ndarray,
    z: np.ndarray,
    test_tra_samples: np.ndarray,
    save: Optional[Union[Path, str]] = None,
    perplexity: float = 50.0,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (H, W, 3)
        tra_samples: translation samples, shape (N, 3)
        tra_gt: groundtruth translation, shape (3,)
        z0: Samples from Gaussian-modelled latent posterior, shape (M, L)
        z:
            Sampels from possibly-modified latent posterior with normalizing flows,
            shape (M, L)
        test_tra_samples: test translation samples, shape (N, 3)
        save: save path including file extension

    Returns:
        figure instance
    """
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perplexity,
    )
    num_samples, lat_dim = z0.shape
    zp = np.random.multivariate_normal(
        np.zeros(lat_dim), np.eye(lat_dim), num_samples
    )  # (M, L)
    if lat_dim == 2:
        z0_emb = z0  # (M, 2)
        z_emb = z  # (M, 2)
        zp_emb = zp  # (M, 2)
    else:
        zs = np.concatenate((z0, z, zp), axis=0)  # (3*M, L)
        zs_emb = tsne.fit_transform(zs)  # (3*M, 2)
        z0_emb = zs_emb[:num_samples]  # (M, 2)
        z_emb = zs_emb[num_samples : 2 * num_samples]  # (M, 2)
        zp_emb = zs_emb[2 * num_samples : 3 * num_samples]  # (M, 2)

    fig = plt.figure(figsize=(25, 6))
    # fig.suptitle(title)
    grid_spec = GridSpec(58, 100)

    img_corner = (np.array(list(query_image.shape[:2])) - 224) // 2

    # latent space
    ax = fig.add_subplot(grid_spec[0:25, 0:20])
    ax.scatter(*zp_emb.T, c="blue", alpha=0.3, s=2)
    ax.scatter(*z0_emb.T, c="red", alpha=0.3, s=2)
    ax.set_title("Gaussian posterior")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

    ax = fig.add_subplot(grid_spec[40:57, 0:20])
    ax.scatter(*zp_emb.T, c="blue", alpha=0.3, s=2)
    ax.scatter(*z_emb.T, c="lime", alpha=0.3, s=2)
    ax.set_title("normalizing flows")
    ax.set_xticks([])
    ax.set_yticks([])
    if lat_dim == 2:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

    # first row
    show_image(
        fig,
        grid_spec[00:25, 90:100],  # grid_spec[0:30, 20:40],
        "query image",
        query_image[
            img_corner[0] : img_corner[0] + 224, img_corner[1] : img_corner[1] + 224
        ],
    )

    tra_samples_vis = (tra_samples - np.array([[13.214, 0.232, 0]])) * np.array(
        [[1, -1, 1]]
    ) * 1000 + np.array([[778, 443, 0]])
    tra_gt_vis = (tra_gt - np.array([13.214, 0.232, 0])) * np.array(
        [1, -1, 1]
    ) * 1000 + np.array([778, 443, 0])
    test_tra_samples_vis = (
        test_tra_samples - np.array([[13.214, 0.232, 0]])
    ) * np.array([[1, -1, 1]]) * 1000 + np.array([[778, 443, 0]])
    ax = fig.add_subplot(grid_spec[0:25, 25:85])
    ax.set_title("marginal histogram on x-axis")
    ax.hist(
        tra_samples_vis[:, 0],
        bins=16500 // 100,
        range=[0, 16500],
        density=True,
        zorder=0,
        alpha=1.0,
        label="reconstruction",
    )
    ax.hist(
        test_tra_samples_vis[:, 0],
        bins=16500 // 100,
        range=[0, 16500],
        density=True,
        zorder=0,
        alpha=0.7,
        label="generation",
    )
    ax.set_yticks([])
    xticks = np.linspace(0, 16000, 9)
    xticklabels = xticks / 1000
    yticks = np.linspace(0, 1000, 3)
    yticklabels = yticks / 1000
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("density")
    ax.set_xlim([0, 16500])
    ax.set_ylim([0, 0.002])
    ax.vlines(
        tra_gt_vis[0],
        0,
        0.002,
        linewidth=2,
        edgecolor="limegreen",
        zorder=2,
        label="Ground truth",
    )
    ax.legend(loc="upper right")

    # second row
    ax = fig.add_subplot(grid_spec[40:57, 25:85])
    ax.set_aspect("equal")
    ax.set_title("posterior on xy-plane")
    stitched_img = np.array(Image.open(Path.home() / "data/Rig/Ceiling/stitched.jpg"))
    ax.imshow(stitched_img)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.hist2d(
        *tra_samples_vis.T[[0, 1]],
        bins=[16500 // 100, 1250 // 100],
        alpha=0.5,
        range=[[0, 16500], [0, 1250]],
        cmap="plasma",
    )
    ax.add_patch(
        Circle(
            xy=tra_gt_vis[:2],
            radius=150,
            linewidth=2,
            facecolor="none",
            edgecolor="limegreen",
        )
    )
    ax = fig.add_subplot(grid_spec[40:57, 90:100])
    ax.set_title("marginal histogram on y-axis")
    ax.hist(
        tra_samples_vis[:, 1],
        bins=1250 // 100,
        range=[0, 1250],
        density=True,
        orientation="horizontal",
        zorder=0,
        label="Ours",
        alpha=1.0,
    )
    ax.set_xticks([])
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("density")
    ax.set_ylabel("y (m)")
    ax.set_xlim([0, 0.01])
    ax.set_ylim([0, 1250])
    ax.hlines(
        tra_gt_vis[1],
        0,
        0.01,
        linewidth=2,
        edgecolor="limegreen",
        zorder=2,
        label="Groundtruth",
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_ceiling(
    query_image: np.ndarray,
    tra_samples: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    tra_pred: Optional[np.ndarray] = None,
    tra_bing: Optional[np.ndarray] = None,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        query_image: query image, shape (H, W, 3)
        tra_samples: translation samples, shape (N, 3)
        tra_gt: groundtruth translation, shape (3,)
        tra_pred: single translation prediction, shape (3,)
        tra_bing: translation samples from Bingham method, shape (N, 3)
        save: save path including file extension

    Returns:
        figure instance
    """
    fig = plt.figure(figsize=(25, 6))
    # fig.suptitle(title)
    grid_spec = GridSpec(58, 100)

    img_corner = (np.array(list(query_image.shape[:2])) - 224) // 2

    # first row
    show_image(
        fig,
        grid_spec[00:25, 90:100],  # grid_spec[0:30, 20:40],
        "query image",
        query_image[
            img_corner[0] : img_corner[0] + 224, img_corner[1] : img_corner[1] + 224
        ],
    )

    # second row
    tra_samples_vis = (tra_samples - np.array([[13.214, 0.232, 0]])) * np.array(
        [[1, -1, 1]]
    ) * 1000 + np.array([[778, 443, 0]])
    tra_gt_vis = (tra_gt - np.array([13.214, 0.232, 0])) * np.array(
        [1, -1, 1]
    ) * 1000 + np.array([778, 443, 0])
    # tra_pred_vis = (tra_pred - np.array([13.214, 0.232, 0])) * np.array(
    #     [1, -1, 1]
    # ) * 1000 + np.array([778, 443, 0])
    # tra_bing_vis = (tra_bing - np.array([[13.214, 0.232, 0]])) * np.array(
    #     [[1, -1, 1]]
    # ) * 1000 + np.array([[778, 443, 0]])
    ax = fig.add_subplot(grid_spec[0:25, 0:85])
    ax.set_title("marginal histogram on x-axis")
    ax.hist(
        tra_samples_vis[:, 0],
        bins=16500 // 100,
        range=[0, 16500],
        density=True,
        zorder=0,
        alpha=1.0,
        label="Ours",
    )
    # ax.hist(
    #     tra_bing_vis[:, 0],
    #     bins=16500 // 100,
    #     range=[0, 16500],
    #     density=True,
    #     zorder=1,
    #     alpha = 0.7,
    #     label = "Bingham MDN"
    # )
    ax.set_yticks([])
    xticks = np.linspace(0, 16000, 9)
    xticklabels = xticks / 1000
    yticks = np.linspace(0, 1000, 3)
    yticklabels = yticks / 1000
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("density")
    ax.set_xlim([0, 16500])
    ax.set_ylim([0, 0.002])
    # ax.vlines(
    #     tra_pred_vis[0],
    #     0,
    #     0.002,
    #     linewidth=2,
    #     edgecolor="magenta",
    #     zorder=1,
    #     label="MapNet"
    # )
    ax.vlines(
        tra_gt_vis[0],
        0,
        0.002,
        linewidth=2,
        edgecolor="limegreen",
        zorder=2,
        label="Ground truth",
    )
    ax.legend(loc="upper right")

    ax = fig.add_subplot(grid_spec[40:57, 0:85])
    ax.set_aspect("equal")
    ax.set_title("posterior on xy-plane")
    stitched_img = np.array(Image.open(Path.home() / "data/Rig/Ceiling/stitched.jpg"))
    ax.imshow(stitched_img)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.hist2d(
        *tra_samples_vis.T[[0, 1]],
        bins=[16500 // 100, 1250 // 100],
        alpha=0.5,
        range=[[0, 16500], [0, 1250]],
        cmap="plasma",
    )
    # ax.add_patch(
    #     Circle(
    #         xy=tra_pred_vis[:2],
    #         radius=150,
    #         linewidth=2,
    #         facecolor="none",
    #         edgecolor="magenta",
    #     )
    # )
    ax.add_patch(
        Circle(
            xy=tra_gt_vis[:2],
            radius=150,
            linewidth=2,
            facecolor="none",
            edgecolor="limegreen",
        )
    )
    ax = fig.add_subplot(grid_spec[40:57, 90:100])
    ax.set_title("marginal histogram on y-axis")
    ax.hist(
        tra_samples_vis[:, 1],
        bins=1250 // 100,
        range=[0, 1250],
        density=True,
        orientation="horizontal",
        zorder=0,
        label="Ours",
        alpha=1.0,
    )
    # ax.hist(
    #     tra_bing_vis[:, 1],
    #     bins=1250 // 100,
    #     range=[0, 1250],
    #     density=True,
    #     orientation="horizontal",
    #     zorder=1,
    #     label="Bui et al.",
    #     alpha=0.7
    # )
    ax.set_xticks([])
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("density")
    ax.set_ylabel("y (m)")
    ax.set_xlim([0, 0.01])
    ax.set_ylim([0, 1250])
    # ax.hlines(
    #     tra_pred_vis[1],
    #     0,
    #     0.01,
    #     linewidth=2,
    #     edgecolor="magenta",
    #     zorder=1,
    #     label="MapNet"
    # )
    ax.hlines(
        tra_gt_vis[1],
        0,
        0.01,
        linewidth=2,
        edgecolor="limegreen",
        zorder=2,
        label="Groundtruth",
    )

    if save is not None:
        print(f"Saving plot {save}")
        fig.savefig(save)
        # tikzplotlib.save(save.with_suffix(".tex"))

    return fig


def plot_hist_on_axis(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    flipped: bool,
    num_bins: Optional[int],
    lims: Iterable[float],
    max_density: float,
    samples: np.ndarray,
    gt: Optional[np.ndarray] = None,
    pred: Optional[float] = None,
) -> None:
    """Plot 1D histogram on axis for marginalized quantities.

    Args:
        figure: figure instance on which plot is made in-place
        position: subplot position on the figure
        title: title of the subplot
        flipped: if True, histogram is plotted vertically
        num_bins: number of histogram bins
        lims: [min, max] range
        max_density: maximum density to fix the histogram scale
        samples: samples to be plotted, shape (N, 3)
        gt: groundtruth plotted as lines, shape (M, 3)
        pred: single prediction
    """
    ax = figure.add_subplot(position)
    ax.set_title(title)
    if flipped:
        ax.set_xlim([0, max_density])
    else:
        ax.set_ylim([0, max_density])

    ax.hist(
        samples,
        bins=num_bins,
        range=lims,
        orientation="horizontal" if flipped else "vertical",
        density=True,
        zorder=0,
    )

    if gt is not None:
        for g in gt:
            if flipped:
                ax.hlines(
                    g,
                    0,
                    max_density,
                    linewidth=2,
                    edgecolor="red",
                    zorder=2,
                )
            else:
                ax.vlines(
                    g,
                    0,
                    max_density,
                    linewidth=2,
                    edgecolor="red",
                    zorder=2,
                )

    if pred is not None:
        if flipped:
            ax.hlines(
                pred,
                0,
                max_density,
                linewidth=4,
                edgecolor="cyan",
                zorder=1,
            )
        else:
            ax.vlines(
                pred,
                0,
                max_density,
                linewidth=4,
                edgecolor="cyan",
                zorder=1,
            )


def plot_latent(
    tras_train: np.ndarray,
    tras_valid: np.ndarray,
    quats_train: np.ndarray,
    quats_valid: np.ndarray,
    codes_train: np.ndarray,
    codes_valid: np.ndarray,
    scene_path: Path,
    title: Optional[str],
    color_latent: bool = False,
    save: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        tras_train: training set translations, (N, 3)
        tras_valid: validation set translations, (M, 3)
        quats_train: training set rotation matrices, (N, 3, 3)
        quats_valid: validation set rotation matrices, (M, 3, 3)
        codes_train: training set latent codes, (N, 2) or (N, K, 2)
        codes_valid: validation set latent codes, (M, 2) or (M, L, 2)
        scene_path: path to the scene that contains scene.txt
        title: title of figure
        color_latent: if True, samples are colored based on t-SNE encodings
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
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=len(tras_train))
    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if color_latent:
        if len(codes_train.shape) == 2:
            mu_train = codes_train
        elif len(codes_train.shape) == 3:
            mu_train = np.mean(codes_train, axis=1)
        else:
            raise ValueError("Invalid shape of latent codes.")
        train_ref = mu_train[
            np.argmax(
                np.linalg.norm(
                    mu_train - np.mean(mu_train, axis=0, keepdims=True), axis=1
                )
            )
        ]
        train_sort = np.argsort(np.linalg.norm(mu_train - train_ref[None, :], axis=1))
        tras_train = tras_train[train_sort]
        quats_train = quats_train[train_sort]
        codes_train = codes_train[train_sort]
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
    cmap = mpl.cm.gist_rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=len(tras_valid))
    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if color_latent:
        if len(codes_valid.shape) == 2:
            mu_valid = codes_valid
        elif len(codes_valid.shape) == 3:
            mu_valid = np.mean(codes_valid, axis=1)
        else:
            raise ValueError("Invalid shape of latent codes.")
        valid_ref = mu_valid[
            np.argmax(
                np.linalg.norm(
                    mu_valid - np.mean(mu_valid, axis=0, keepdims=True), axis=1
                )
            )
        ]
        valid_sort = np.argsort(np.linalg.norm(mu_valid - valid_ref[None, :], axis=1))
        tras_valid = tras_valid[valid_sort]
        quats_valid = quats_valid[valid_sort]
        codes_valid = codes_valid[valid_sort]
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
        tra_gt: groundtruth translation, shape (M, 3)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (M, 4)
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
    grid_spec = GridSpec(100, 100)

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
        tra_gt,
    )
    plot_tra_hist_on_plane(
        fig,
        grid_spec[r[2] : r[2] + h, c[0] : c[2] + w],
        "translations posterior",
        scene_dims,
        tra_samples,
        tra_gt,
    )
    plot_rot_hist_on_plane(
        fig,
        grid_spec[r[0] : r[0] + h, c[2] : c[2] + w],
        "rotation posterior",
        quat_samples,
        quat_gt,
    )

    # second row of subplots, projecive view and individual samples
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
    plot_tra_samples_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[1] : c[1] + w],
        "translations samples",
        scene_dims,
        tra_samples[:num_samples],
        tra_gt,
        markers,
        grid_spec[r[1] : r[1] + h, c[1] + w + cb_o : c[1] + w + cb_o + cb_w],
    )
    plot_rot_samples_on_plane(
        fig,
        grid_spec[r[1] : r[1] + h, c[2] : c[2] + w],
        "rotation samples",
        quat_samples[:num_samples],
        quat_gt,
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
        tra_gt: groundtruth translation, shape (M, 3)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (M, 4)
    """
    mesh_path = scene_path / "mesh.ply"
    if mesh_path.is_file():
        vis_camera = o3d.io.read_pinhole_camera_parameters(
            str(scene_path / "vis_camera.json")
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=vis_camera.intrinsic.width,
            height=vis_camera.intrinsic.height,
            visible=False,
        )
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)
    else:
        # default vis camera
        vis_camera = o3d.io.read_pinhole_camera_parameters(
            str(Path.home() / "code/vapor/default_vis.json")
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=vis_camera.intrinsic.width,
            height=vis_camera.intrinsic.height,
            visible=False,
        )
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0.0, 0.0, 0.0]
        )
        vis.add_geometry(origin_frame)

    intrinsic = np.array(
        [
            [7.5879234898141704e02, 0.0, 2.7354155451863954e02],
            [0.0, 7.6005168616062099e02, 4.8594391325399761e02],
            [0.0, 0.0, 1.0],
        ]
    )
    noises = np.random.uniform(-1, 1, size=(100, 3)) * 0.002
    for tra, quat in zip(tra_samples, quat_samples):
        for noise in noises:
            sample_frame = o3d.geometry.LineSet.create_camera_visualization(
                540, 960, intrinsic, np.eye(4), scale=0.2
            )
            sample_frame.paint_uniform_color(np.array([190, 0, 255]) / 255)
            sample_frame.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat))
            sample_frame.translate(tra + noise)
            vis.add_geometry(sample_frame)

    if tra_gt is not None and quat_gt is not None:
        noises = np.random.uniform(-1, 1, size=(100, 3)) * 0.008
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0.4, 1]])
        for tra, quat, color in zip(tra_gt, quat_gt, cycle(colors)):
            for noise in noises:
                gt_frame = o3d.geometry.LineSet.create_camera_visualization(
                    540, 960, intrinsic, np.eye(4), scale=0.4
                )
                gt_frame.paint_uniform_color(color)
                gt_frame.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat))
                gt_frame.translate(tra + noise)
                vis.add_geometry(gt_frame)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(vis_camera)

    vis.poll_events()
    vis.update_renderer()

    return np.array(vis.capture_screen_float_buffer(do_render=True))


def render_3d_modes(
    scene_path: Path,
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    colors: Iterable[Iterable[float]],
    assignments: np.ndarray,
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
    colors_gt: Optional[Iterable[Iterable[float]]] = None,
    num_sample_noise: int = 1,
    num_gt_noise: int = 100,
) -> np.ndarray:
    """Render a projective image of the scene with samples.

    Args:
        scene_path: path to the scene that contains vis_camera.json and mesh.ply
        tra_samples: translation samples , shape (M, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (M, 4)
        colors: colors corresponding to modes, len >= N
        assignments:
            assigment of samples to modes, contains an integer index in [0, N),
            shape (M,)
        tra_gt: groundtruth translation, shape (N, 3)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (N, 4)
        num_sample_noise: How much noise jitter apply to translation for thicker lines
        num_gt_noise: How much noise jitter apply to translation for thicker lines
    """
    mesh_path = scene_path / "mesh.ply"
    if mesh_path.is_file():
        vis_camera = o3d.io.read_pinhole_camera_parameters(
            str(scene_path / "vis_camera.json")
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=vis_camera.intrinsic.width,
            height=vis_camera.intrinsic.height,
            visible=False,
        )
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)
    else:
        # default vis camera
        vis_camera = o3d.io.read_pinhole_camera_parameters(
            str(Path.home() / "code/vapor/default_vis.json")
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=vis_camera.intrinsic.width,
            height=vis_camera.intrinsic.height,
            visible=False,
        )
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0.0, 0.0, 0.0]
        )
        vis.add_geometry(origin_frame)

    intrinsic = np.array(
        [
            [7.5879234898141704e02, 0.0, 2.7354155451863954e02],
            [0.0, 7.6005168616062099e02, 4.8594391325399761e02],
            [0.0, 0.0, 1.0],
        ]
    )
    noises = np.random.uniform(-1, 1, size=(num_sample_noise, 3)) * 0.002
    num_modes = tra_gt.shape[0] if tra_gt is not None else np.max(assignments) + 1
    for i in range(num_modes):
        mask = assignments == i
        tras = tra_samples[mask]  # (K, 3)
        quats = quat_samples[mask]  # (K, 4)
        color = colors[i]
        for tra, quat in zip(tras, quats):  # (3,), (4,)
            for noise in noises:
                sample_frame = o3d.geometry.LineSet.create_camera_visualization(
                    540, 960, intrinsic, np.eye(4), scale=0.2
                )
                sample_frame.paint_uniform_color(color)
                sample_frame.rotate(
                    o3d.geometry.get_rotation_matrix_from_quaternion(quat)
                )
                sample_frame.translate(tra + noise)
                vis.add_geometry(sample_frame)

    if tra_gt is not None and quat_gt is not None:
        noises = np.random.uniform(-1, 1, size=(num_gt_noise, 3)) * 0.008
        for tra, quat, color in zip(tra_gt, quat_gt, colors_gt):
            for noise in noises:
                gt_frame = o3d.geometry.LineSet.create_camera_visualization(
                    540, 960, intrinsic, np.eye(4), scale=0.4
                )
                gt_frame.paint_uniform_color(color)
                gt_frame.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat))
                gt_frame.translate(tra + noise)
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
                    radius=0.5,
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

    ax.hist2d(*tra_samples.T[:2], bins=50, range=[xlim, ylim], cmap="plasma")

    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0.4, 1]])
    if tra_gt is not None:
        for tra, color in zip(tra_gt, cycle(colors)):
            ax.add_patch(
                Circle(
                    xy=tra[[0, 1]],
                    radius=0.7,
                    linewidth=7,
                    facecolor="none",
                    edgecolor=color,
                )
            )


def plot_rot_hist_on_plane(
    figure: plt.Figure,
    position: SubplotSpec,
    title: str,
    quat_samples: np.ndarray,
    quat_gt: Optional[np.ndarray] = None,
    quat_pred: Optional[np.ndarray] = None,
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
        quat_pred: single prediction quaternion [w, x, y, z], shape (4,)
    """
    ax = figure.add_subplot(position, projection="mollweide")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    rot_samples = quat_to_rotmat(quat_samples)
    rot_gt = quat_to_rotmat(quat_gt)
    angle = 2.1
    rotmat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rot_samples = rotmat @ rot_samples
    rot_gt = rotmat @ rot_gt
    quat_samples = rotmat_to_quat(rot_samples)
    quat_gt = rotmat_to_quat(rot_gt)

    hopf_samples = quat_to_hopf(quat_samples)
    hist, phi_edges, theta_edges = np.histogram2d(
        *hopf_samples.T[:2],
        bins=50,
        range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
    )
    hist = np.concatenate((hist, np.zeros((1, hist.shape[1]))), axis=0)
    hist = np.concatenate((hist, np.zeros((hist.shape[0], 1))), axis=1)
    ax.pcolor(
        phi_edges,
        theta_edges,
        hist.T,  # transpose from (row, column) to (x, y)
        cmap=mpl.cm.plasma,
        shading="auto",
        vmin=0,
        vmax=np.max(hist),
    )
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0.4, 1]])
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt)
        for hopf, color in zip(hopf_gt, cycle(colors)):
            ax.add_patch(
                Circle(
                    xy=hopf[:2],
                    radius=0.4,
                    linewidth=5,
                    facecolor="none",
                    edgecolor=color,
                )
            )
    if quat_pred is not None:
        hopf_pred = quat_to_hopf(quat_pred[None, :])[0]
        ax.add_patch(
            Circle(
                xy=hopf_pred[:2],
                radius=0.3,
                linewidth=2,
                facecolor="none",
                edgecolor="magenta",
            )
        )


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
        *tra_samples.T[:2], s=50, color=colormap.to_rgba(np.arange(len(tra_samples)))
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
        codes: latent encodings to be plotted, shape (N, 2) or (N, M, 2)
        colormap: colormap used for coloring samples
    """
    ax = figure.add_subplot(position)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(codes.shape) == 2:
        ax.scatter(*codes.T, s=10, color=colormap.to_rgba(np.arange(len(codes))))
    else:
        for i, code in enumerate(codes):
            ax.scatter(*code.T, s=50, color=colormap.to_rgba(i))
