from IPython.core.display import display, HTML
from typing import Tuple
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from torch.utils.data import DataLoader
import torch.distributions as D
from torch.optim import Adam
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.colors as plt_colors
import matplotlib.cm as cmx
from PIL import Image
from scipy.spatial.transform import Rotation
from pathlib import Path
import open3d as o3d

from nerfcity.data import AmbiguousImages, AmbiguousToy
from nerfcity.models import Encoder, PoseMap
from nerfcity.losses import kl_divergence
from nerfcity.train_map import canonical_to_pose
from nerfcity.r3grid import R3Grid
from nerfcity.so3grid import SO3Grid


def render_3d(scene_path, tras, rots, gt_tra, gt_rot, visualize=False):
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

    for tra, rot in zip(tras, rots):
        sample_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0.0, 0.0, 0.0]
        )
        sample_frame.rotate(rot)
        sample_frame.translate(tra)
        vis.add_geometry(sample_frame)

    gt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )
    gt_frame.rotate(gt_rot)
    gt_frame.translate(gt_tra)
    vis.add_geometry(gt_frame)

    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(vis_camera)

    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(str(scene_path / "leo.png"), do_render=True)
    im = np.array(vis.capture_screen_float_buffer(do_render=True))

    return im


if __name__ == "__main__":
    tqdm_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 16
    image_size = 64
    batch_size = 16
    sequence = "blue_chairs"
    scene_path = Path.home() / "data" / "Ambiguous_ReLoc_Dataset" / sequence
    saved_weights_path = Path.home() / "code" / "nerfcity" / "saved_weights"
    encoder = Encoder(latent_dim)
    encoder.load_state_dict(
        torch.load(
            saved_weights_path / f"encoder_blue_{sequence}_{latent_dim}.pth",
            map_location=device,
        )
    )
    posemap = PoseMap(latent_dim)
    posemap.load_state_dict(
        torch.load(
            Path.home() / "code/nerfcity/nerfcity/blue_chairs_50000ep_top15" /  f"posemap_blue_{sequence}_{latent_dim}.pth",
            map_location=device,
        )
    )
    # if torch.cuda.device_count() > 1:
    #     encoder = torch.nn.DataParallel(encoder)
    #     posemap = torch.nn.DataParallel(posemap)
    encoder.to(device)
    posemap.to(device)
    encoder.eval()
    posemap.eval()

    augment = False
    dataset_mean = [0.5417, 0.5459, 0.4945]
    dataset_std = [0.1745, 0.1580, 0.1627]
    train_set = AmbiguousImages(
        scene_path / "train" / "seq00", image_size, augment, dataset_mean, dataset_std
    )
    test_set = AmbiguousImages(
        scene_path / "test" / "seq01", image_size, augment, dataset_mean, dataset_std
    )
    margins = [2, 2, 0.5]
    mins = [-1.135581, -0.763927, -0.058301]
    maxs = [0.195657, 0.464706, 0.058201]
    num_latent_samples = 1000
    r3_probses = []
    so3_probses = []
    r3_pmfses = []
    so3_pmfses = []
    r3_idcses = []
    so3_idcses = []
    for dataset in (train_set, test_set):
        num_val_images = len(dataset)
        val_data = DataLoader(
            dataset,
            batch_size=num_val_images,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        encoder.eval()
        posemap.eval()
        with torch.no_grad():
            for batch in val_data:
                image = batch[0].to(device)
                pose = batch[1].to(device)
                latent_mu, latent_logvar = encoder(image)
                latent_std = torch.exp(0.5 * latent_logvar)
                eps = torch.randn((num_val_images, num_latent_samples, latent_dim), device=device)
                latent_sample = eps * latent_std.unsqueeze(1) + latent_mu.unsqueeze(1)
                tra_hat, rot_hat = canonical_to_pose(
                    posemap(latent_sample.flatten(end_dim=1)),
                    mins,
                    maxs,
                    margins,
                )
                tra_hat = tra_hat.reshape(num_val_images, num_latent_samples, 3)
                rot_hat = rot_hat.reshape(num_val_images, num_latent_samples, 3, 3)

                break

        inferno = plt.get_cmap("inferno")
        rainbow = plt.get_cmap("gist_rainbow")
        test_set_color_map = cmx.ScalarMappable(
            norm=plt_colors.Normalize(vmin=0, vmax=num_val_images), cmap=rainbow
        )

        tras = pose[:, :3].cpu().numpy()
        rots = pose[:, 3:].reshape(-1, 3, 3).cpu().numpy()
        tras_hat = tra_hat.cpu().numpy()
        rots_hat = rot_hat.cpu().numpy()
        latents_mu = latent_mu.cpu().numpy()
        latents_std = latent_std.cpu().numpy()

        rotations = Rotation.from_matrix(rots)
        ypr = rotations.as_euler("zyx")
        quat = rotations.as_quat()
        rotations_hat = Rotation.from_matrix(rots_hat.reshape(-1, 3, 3))
        ypr_hat = rotations_hat.as_euler("zyx").reshape(*rots_hat.shape[:2], 3)
        quat_hat = rotations_hat.as_quat().reshape(*rots_hat.shape[:2], 4)
        embeddings = TSNE(n_components=2, init="random").fit_transform(latents_mu)

        r3_grid = R3Grid(0.5, np.vstack((mins, maxs)).T)
        so3_grid = SO3Grid(0)
        r3_idcs = r3_grid.r3_to_index(tras)
        so3_idcs = np.array([so3_grid.quat_to_index(q) for q in quat])
        r3_idcses.append(r3_idcs)
        so3_idcses.append(so3_idcs)
        r3_idcs_hat = r3_grid.r3_to_index(tras_hat.reshape(-1, 3)).reshape(
            *tras_hat.shape[:2]
        )
        so3_idcs_hat = np.array(
            [so3_grid.quat_to_index(q) for q in quat_hat.reshape(-1, 4)]
        ).reshape(*tras_hat.shape[:2])

        r3_pmfs = []
        so3_pmfs = []
        r3_probs = []
        so3_probs = []
        for query in tqdm(range(num_val_images), bar_format=tqdm_bar):
            r3_pmf = np.zeros(r3_grid.num_cells())
            so3_pmf = np.zeros(so3_grid.num_cells())
            for r3_idx in r3_idcs_hat[query]:
                r3_pmf[r3_idx] += 1
            r3_pmf /= np.sum(r3_pmf)
            for so3_idx in so3_idcs_hat[query]:
                so3_pmf[so3_idx] += 1
            so3_pmf /= np.sum(so3_pmf)

            r3_pmfs.append(r3_pmf)
            so3_pmfs.append(so3_pmf)
            r3_probs.append(r3_pmf[r3_idcs[query]])
            so3_probs.append(so3_pmf[so3_idcs[query]])
        r3_pmfses.append(np.array(r3_pmfs))
        so3_pmfses.append(np.array(so3_pmfs))
        r3_probses.append(r3_probs)
        so3_probses.append(so3_probs)

    for query in tqdm(range(num_val_images), bar_format=tqdm_bar):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{query} query, {r3_grid.num_cells()} R3 cells, {so3_grid.num_cells()} SO(3) cells")
        for i, dataset in enumerate(("training", "test")):
            axs[i, 0].bar(np.arange(r3_grid.num_cells()), r3_pmfses[i][query])
            axs[i, 0].vlines(r3_idcses[i][query], 0, 1, color="red")
            axs[i, 0].set_title(f"{dataset} translation porbability masses")
            axs[i, 0].set_xlabel("pmf")
            axs[i, 0].set_ylabel(f"{dataset} set samples")
            axs[i, 1].bar(np.arange(so3_grid.num_cells()), so3_pmfses[i][query])
            axs[i, 1].vlines(so3_idcses[i][query], 0, 1, color="red")
            axs[i, 1].set_title(f"{dataset} rotation porbability masses")
            axs[i, 1].set_xlabel("pmf")
            axs[i, 1].set_ylabel(f"{dataset} set samples")
        fig.savefig(f"sanity/{query}.png")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{r3_grid.num_cells()} R3 cells, {so3_grid.num_cells()} SO(3) cells")
    for i, dataset in enumerate(("training", "test")):
        axs[i, 0].imshow(r3_pmfses[i])
        axs[i, 0].set_title(f"{dataset} translation porbability masses")
        axs[i, 0].set_xlabel("pmf")
        axs[i, 0].set_ylabel(f"{dataset} set samples")
        axs[i, 1].imshow(so3_pmfses[i])
        axs[i, 1].set_title(f"{dataset} rotation porbability masses")
        axs[i, 1].set_xlabel("pmf")
        axs[i, 1].set_ylabel(f"{dataset} set samples")
    fig.savefig(f"pmfs.pdf")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{r3_grid.num_cells()} R3 cells, {so3_grid.num_cells()} SO(3) cells")
    for i, dataset in enumerate(("training", "test")):
        axs[i, 0].hist(r3_probses[i], bins=100)
        axs[i, 0].set_title(f"{dataset} translation likelihood histogram")
        axs[i, 0].set_xlabel("p(t | x)")
        axs[i, 0].set_ylabel(f"count across {dataset} set")
        axs[i, 1].hist(so3_probses[i], bins=100)
        axs[i, 1].set_title(f"{dataset} rotation likelihood histogram")
        axs[i, 1].set_xlabel("p(R | x)")
        axs[i, 1].set_ylabel(f"count across {dataset} set")
    fig.savefig(f"likelihoods.pdf")

    # prox_ori_radius = 35
    # samp_ori_radius = 20
    # num_vis_samples = 10
    # samples_color_map = cmx.ScalarMappable(
    #     norm=plt_colors.Normalize(vmin=0, vmax=num_vis_samples), cmap=rainbow
    # )
    # tsne_threshold = 2.0
    # tsne_xlim = [
    #     np.min(embeddings, axis=0)[0] - tsne_threshold,
    #     np.max(embeddings, axis=0)[0] + tsne_threshold,
    # ]
    # tsne_ylim = [
    #     np.min(embeddings, axis=0)[1] - tsne_threshold,
    #     np.max(embeddings, axis=0)[1] + tsne_threshold,
    # ]
    # xlim = [mins[0] - 1, maxs[0] + 1]
    # ylim = [mins[1] - 1, maxs[1] + 1]
    # zlim = [mins[2] - 0.01, maxs[2] + 0.01]

    # # query = 2
    # for query in tqdm(range(num_val_images), bar_format=tqdm_bar):
    #     tsne_dists = np.linalg.norm(embeddings[query][None, :] - embeddings, axis=1)
    #     close_tsne = np.arange(num_val_images)[tsne_dists < tsne_threshold]

    #     yaw_hist_vals, ori_hist_bin_edges = np.histogram(
    #         ypr_hat[query, :, 0], bins=40, range=[-np.pi, np.pi], normed=True
    #     )
    #     ori_hist_bins = (ori_hist_bin_edges[:-1] + ori_hist_bin_edges[1:]) / 2
    #     ori_hist_comp_bins = np.vstack((np.cos(ori_hist_bins), np.sin(ori_hist_bins))).T
    #     pitch_hist_vals, _ = np.histogram(
    #         ypr_hat[query, :, 1], bins=40, range=[-np.pi, np.pi], normed=True
    #     )
    #     roll_hist_vals, _ = np.histogram(
    #         ypr_hat[query, :, 2], bins=40, range=[-np.pi, np.pi], normed=True
    #     )

    #     fig = plt.figure(figsize=(23, 14), constrained_layout=True)
    #     axd = fig.subplot_mosaic(
    #         """
    #         QQQAAABCCCDDDOOO
    #         PPPEEEFGGGHHHIII
    #         RRRJJJKLLLMMMNNN
    #         """
    #     )
    #     for i, (
    #         emb_test,
    #         (x_test, y_test, z_test),
    #         (yaw_test, pitch_test, roll_test),
    #     ) in enumerate(zip(embeddings, tras, ypr)):
    #         test_color = test_set_color_map.to_rgba(i)
    #         test_alpha = 1.0 if i == query or i in close_tsne else 0.1
    #         test_size = 40 if i == query or i in close_tsne else 20
    #         test_marker = "*" if i == query else "."
    #         test_linewidth = 3 if i == query else 1
    #         test_linestyle = "--" if i == query else "-"
    #         axd["Q"].scatter(
    #             *emb_test,
    #             color=test_color,
    #             s=test_size,
    #             zorder=3,
    #             alpha=test_alpha,
    #             marker=test_marker,
    #         )
    #         axd["A"].scatter(
    #             x_test,
    #             y_test,
    #             color=test_color,
    #             s=test_size,
    #             zorder=3,
    #             alpha=test_alpha,
    #             marker=test_marker,
    #         )
    #         axd["B"].hlines(
    #             z_test,
    #             0,
    #             1,
    #             color=test_color,
    #             linewidth=test_linewidth,
    #             linestyle=test_linestyle,
    #             zorder=3,
    #             alpha=test_alpha,
    #         )
    #         axd["C"].scatter(
    #             np.cos(yaw_test),
    #             np.sin(yaw_test),
    #             color=test_color,
    #             s=test_size,
    #             zorder=3,
    #             alpha=test_alpha,
    #             marker=test_marker,
    #         )
    #         axd["C"].plot(
    #             [0, np.cos(yaw_test)],
    #             [0, np.sin(yaw_test)],
    #             color=test_color,
    #             linewidth=test_linewidth,
    #             linestyle=test_linestyle,
    #             alpha=test_alpha,
    #         )
    #         axd["D"].scatter(
    #             np.cos(pitch_test),
    #             np.sin(pitch_test),
    #             color=test_color,
    #             s=test_size,
    #             zorder=3,
    #             alpha=test_alpha,
    #             marker=test_marker,
    #         )
    #         axd["D"].plot(
    #             [0, np.cos(pitch_test)],
    #             [0, np.sin(pitch_test)],
    #             color=test_color,
    #             linewidth=test_linewidth,
    #             linestyle=test_linestyle,
    #             alpha=test_alpha,
    #         )
    #         axd["O"].scatter(
    #             np.cos(roll_test),
    #             np.sin(roll_test),
    #             color=test_color,
    #             s=test_size,
    #             zorder=3,
    #             alpha=test_alpha,
    #             marker=test_marker,
    #         )
    #         axd["O"].plot(
    #             [0, np.cos(roll_test)],
    #             [0, np.sin(roll_test)],
    #             color=test_color,
    #             linewidth=test_linewidth,
    #             linestyle=test_linestyle,
    #             alpha=test_alpha,
    #         )

    #     tsne_circle = Circle(
    #         (embeddings[query, 0], embeddings[query, 1]),
    #         tsne_threshold,
    #         edgecolor=test_set_color_map.to_rgba(query),
    #         facecolor="none",
    #     )
    #     axd["Q"].add_artist(tsne_circle)
    #     axd["Q"].set_title("test set latent t-SNE")
    #     axd["Q"].set_xlim(tsne_xlim)
    #     axd["Q"].set_ylim(tsne_ylim)
    #     axd["Q"].set_aspect("equal")
    #     axd["A"].set_title("test set (x, y)")
    #     axd["A"].set_aspect("equal")
    #     axd["A"].set_xlabel("x")
    #     axd["A"].set_ylabel("y")
    #     axd["A"].set_xlim(xlim)
    #     axd["A"].set_ylim(ylim)
    #     axd["B"].set_title("test set z")
    #     axd["B"].set_xticks([])
    #     axd["B"].set_ylabel("z")
    #     axd["B"].set_ylim(zlim)
    #     axd["C"].set_title("test set yaw")
    #     axd["D"].set_title("test set pitch")
    #     axd["O"].set_title("test set roll")

    #     axd["P"].imshow(image[query].transpose(0, 1).transpose(1, 2).cpu().numpy())
    #     axd["P"].set_xticks([])
    #     axd["P"].set_yticks([])
    #     axd["P"].set_title(f"query image {query}")

    #     axd["E"].set_title(f"(x,y) posterior")
    #     axd["E"].hist2d(*tras_hat[query, :, :2].T, bins=50, range=[xlim, ylim])
    #     axd["E"].scatter(*tras[query, :2], color="red", s=20)
    #     axd["E"].set_aspect("equal")
    #     axd["E"].set_xlim(xlim)
    #     axd["E"].set_ylim(ylim)
    #     axd["E"].set_xlabel("x")
    #     axd["E"].set_ylabel("y")

    #     axd["F"].set_title(f"z posterior")
    #     vals, _, _ = axd["F"].hist(
    #         tras_hat[query, :, 2],
    #         bins=50,
    #         range=zlim,
    #         orientation="horizontal",
    #         density=True,
    #     )
    #     axd["F"].hlines(tras[query, 2], 0, np.max(vals), color="red", zorder=5)
    #     axd["F"].set_xlabel("density")
    #     axd["F"].set_ylabel("z")
    #     axd["F"].set_ylim(zlim)

    #     for axis in ("C", "D", "O", "G", "H", "I", "L", "M", "N"):
    #         axd[axis].vlines(0, -1.5, 1.5, color="gray", alpha=0.5, linewidth=1)
    #         axd[axis].hlines(0, -1.5, 1.5, color="gray", alpha=0.5, linewidth=1)
    #         unit_circle = Circle(
    #             (0, 0), 1, edgecolor="gray", alpha=0.5, facecolor="none"
    #         )
    #         axd[axis].add_artist(unit_circle)
    #         axd[axis].set_aspect("equal")
    #         axd[axis].set_xlim([-1.5, 1.5])
    #         axd[axis].set_ylim([-1.5, 1.5])
    #         axd[axis].set_xlabel("real")
    #         axd[axis].set_ylabel("imaginary")
    #     axd["G"].set_title("yaw posterior")
    #     axd["H"].set_title("pitch posterior")
    #     axd["I"].set_title("roll posterior")
    #     for val, complex_bin in zip(yaw_hist_vals, ori_hist_comp_bins):
    #         axd["G"].scatter(*complex_bin, s=40 * val, zorder=2, color="blue")
    #     for val, complex_bin in zip(pitch_hist_vals, ori_hist_comp_bins):
    #         axd["H"].scatter(*complex_bin, s=40 * val, zorder=2, color="blue")
    #     for val, complex_bin in zip(roll_hist_vals, ori_hist_comp_bins):
    #         axd["I"].scatter(*complex_bin, s=40 * val, zorder=2, color="blue")
    #     axd["G"].scatter(
    #         np.cos(ypr[query, 0]), np.sin(ypr[query, 0]), color="red", s=30, zorder=3
    #     )
    #     axd["H"].scatter(
    #         np.cos(ypr[query, 1]), np.sin(ypr[query, 1]), color="red", s=30, zorder=3
    #     )
    #     axd["I"].scatter(
    #         np.cos(ypr[query, 2]), np.sin(ypr[query, 2]), color="red", s=30, zorder=3
    #     )
    #     axd["G"].plot(
    #         [0, np.cos(ypr[query, 0])],
    #         [0, np.sin(ypr[query, 0])],
    #         color="red",
    #         linewidth=2,
    #     )
    #     axd["H"].plot(
    #         [0, np.cos(ypr[query, 1])],
    #         [0, np.sin(ypr[query, 1])],
    #         color="red",
    #         linewidth=2,
    #     )
    #     axd["I"].plot(
    #         [0, np.cos(ypr[query, 2])],
    #         [0, np.sin(ypr[query, 2])],
    #         color="red",
    #         linewidth=2,
    #     )

    #     for i, (
    #         (x_samp, y_samp, z_samp),
    #         (yaw_samp, pitch_samp, roll_samp),
    #     ) in enumerate(
    #         zip(tras_hat[query, :num_vis_samples], ypr_hat[query, :num_vis_samples])
    #     ):
    #         sample_color = samples_color_map.to_rgba(i)
    #         axd["J"].scatter(x_samp, y_samp, color=sample_color, s=20, zorder=3)
    #         axd["K"].hlines(z_samp, 0, 1, color=sample_color, linewidth=1, zorder=3)
    #         axd["L"].scatter(
    #             np.cos(yaw_samp), np.sin(yaw_samp), color=sample_color, s=15, zorder=3
    #         )
    #         axd["L"].plot(
    #             [0, np.cos(yaw_samp)],
    #             [0, np.sin(yaw_samp)],
    #             color=sample_color,
    #             linewidth=1,
    #         )
    #         axd["M"].scatter(
    #             np.cos(pitch_samp),
    #             np.sin(pitch_samp),
    #             color=sample_color,
    #             s=15,
    #             zorder=3,
    #         )
    #         axd["M"].plot(
    #             [0, np.cos(pitch_samp)],
    #             [0, np.sin(pitch_samp)],
    #             color=sample_color,
    #             linewidth=1,
    #         )
    #         axd["N"].scatter(
    #             np.cos(roll_samp), np.sin(roll_samp), color=sample_color, s=15, zorder=3
    #         )
    #         axd["N"].plot(
    #             [0, np.cos(roll_samp)],
    #             [0, np.sin(roll_samp)],
    #             color=sample_color,
    #             linewidth=1,
    #         )
    #     axd["J"].scatter(
    #         tras[query, 0], tras[query, 1], color="red", s=100, marker="*", zorder=3
    #     )
    #     axd["K"].hlines(tras[query, 2], 0, 1, color="red", linewidth=3, zorder=3)
    #     axd["L"].scatter(
    #         np.cos(ypr[query, 0]),
    #         np.sin(ypr[query, 0]),
    #         color="red",
    #         s=100,
    #         zorder=3,
    #         marker="*",
    #     )
    #     axd["L"].plot(
    #         [0, np.cos(ypr[query, 0])],
    #         [0, np.sin(ypr[query, 0])],
    #         color="red",
    #         linewidth=2,
    #     )
    #     axd["M"].scatter(
    #         np.cos(ypr[query, 1]),
    #         np.sin(ypr[query, 1]),
    #         color="red",
    #         s=100,
    #         zorder=3,
    #         marker="*",
    #     )
    #     axd["M"].plot(
    #         [0, np.cos(ypr[query, 1])],
    #         [0, np.sin(ypr[query, 1])],
    #         color="red",
    #         linewidth=2,
    #     )
    #     axd["N"].scatter(
    #         np.cos(ypr[query, 2]),
    #         np.sin(ypr[query, 2]),
    #         color="red",
    #         s=100,
    #         zorder=3,
    #         marker="*",
    #     )
    #     axd["N"].plot(
    #         [0, np.cos(ypr[query, 2])],
    #         [0, np.sin(ypr[query, 2])],
    #         color="red",
    #         linewidth=2,
    #     )

    #     axd["J"].set_title("(x, y) samples from posterior")
    #     axd["J"].set_aspect("equal")
    #     axd["J"].set_xlim(xlim)
    #     axd["J"].set_ylim(ylim)
    #     axd["J"].set_xlabel("x")
    #     axd["J"].set_ylabel("y")
    #     axd["K"].set_title("z samples from posterior")
    #     axd["L"].set_title("yaw samples from posterior")
    #     axd["M"].set_title("pitch samples from posterior")
    #     axd["N"].set_title("roll samples from posterior")
    #     axd["K"].set_xticks([])
    #     axd["K"].set_ylabel("z")
    #     axd["K"].set_ylim(zlim)

    #     render = render_3d(
    #         scene_path,
    #         tras_hat[query, :num_vis_samples],
    #         rots_hat[query, :num_vis_samples],
    #         tras[query],
    #         rots[query],
    #         visualize=True,
    #     )
    #     axd["R"].imshow(render)
    #     axd["R"].set_xticks([])
    #     axd["R"].set_yticks([])
    #     axd["R"].set_title("samples from posterior")

    #     # plt.show()

    #     fig.savefig(f"queries/{query}.png")
    #     plt.close()
