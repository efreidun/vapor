from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from vaincapo.read_write import read_poses


def get_panel_loc(num: int, staircase: bool, flipped: bool, quarter: bool):
    """Get location of panel in the stitched image."""
    cm_to_px = 10
    btwn_panels_cm = 4 * 60 + 5
    panel_x_cm = 55
    panel_y_cm = 107.5
    panel_stair_y_cm = 71.5
    panel_half_y_cm = 28.5
    offset_x_cm = 50
    offset_y_cm = 10

    x0 = offset_x_cm + num * (panel_x_cm + btwn_panels_cm)
    y0 = offset_y_cm
    y0 += panel_y_cm - panel_half_y_cm if flipped else 0
    x1 = x0
    y1 = y0 + (panel_stair_y_cm if staircase else panel_half_y_cm)
    x2 = x0 + (panel_x_cm / 2 if quarter else panel_x_cm)
    y2 = y0
    x3 = x2
    y3 = y1

    return cm_to_px * np.array(
        [[x2, y2], [x3, y3], [x0, y0], [x1, y1]], dtype=np.float32
    )


def idx_to_id(x):
    return x * 10 + 5200


def id_to_idx(x):
    return (x - 5200) // 10


def warp_image(img, img_c2w, ref_c2w, K, pi, size, H0=None):
    if H0 is None:
        H0 = np.eye(3)
    # T = np.linalg.inv(ref_c2w) @ img_c2w
    T = np.linalg.inv(img_c2w) @ ref_c2w
    H = K @ (T[:3, :3] - (T[:3, 3:] @ pi[None, :3]) / pi[3]) @ np.linalg.inv(K)
    H = H0 @ np.linalg.inv(H)
    return cv2.warpPerspective(img, H, size), H


if __name__ == "__main__":
    seq_path = Path.home() / "data/Rig/Ceiling/test/0"
    poses_path = seq_path / "poses.txt"
    K = np.array(
        [
            [386.88853898703957, 0, 329.8143910097873],
            [0, 387.2276336575777, 245.30885289613698],
            [0, 0, 1],
        ]
    )
    normal = np.array([-0.0117612, 0.0514473, 0.998606])
    center = np.array([15.015343666077, 0.018681228161, 0.608011424541])
    v = normal / (-normal @ center)
    pi = np.concatenate((v, [1]))

    _, img_ids, tvecs_c2w, rotmats_c2w = read_poses(seq_path / "poses.txt", "Rig")
    c2ws = np.concatenate((rotmats_c2w, tvecs_c2w[..., None]), axis=2)
    c2ws = np.concatenate(
        (c2ws, np.tile(np.array([[[0, 0, 0, 1]]]), (len(c2ws), 1, 1))), axis=1
    )
    img_paths = [seq_path / f"images/{img_id:06}.jpg" for img_id in img_ids]
    imgs = [np.array(Image.open(img_path)) for img_path in img_paths]

    ctrl_ids = [5340, 5600, 5890, 6150, 6410, 6660, 6920, 7170, 7430, 7660, 7900]
    ctrl_nums = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ctrl_staircase = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    ctrl_flipped = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    ctrl_quarter = [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    ctrl_idcs = [id_to_idx(id) for id in ctrl_ids]
    ctrl_imgs = [imgs[idx] for idx in ctrl_idcs]
    ctrl_c2ws = [c2ws[idx] for idx in ctrl_idcs]
    ctrl_pis = [c2w.T @ pi for c2w in ctrl_c2ws]

    ctrl_srcs = np.array(
        [
            [[430, 126], [46, 107], [426, 414], [34, 413]],
            [[427, 119], [274, 111], [419, 415], [265, 414]],
            [[470, 138], [303, 133], [457, 456], [292, 452]],
            [[427, 149], [273, 148], [429, 446], [274, 448]],
            [[522, 156], [356, 150], [518, 476], [351, 477]],
            [[35, 271], [199, 283], [48, 109], [212, 119]],
            [[207, 52], [44, 38], [182, 369], [16, 357]],
            [[567, 97], [397, 92], [562, 426], [391, 424]],
            [[582, 145], [427, 134], [567, 446], [410, 437]],
            [[580, 119], [425, 107], [565, 422], [407, 414]],
            [[515, 134], [112, 116], [505, 448], [95, 429]],
        ],
        dtype=np.float32,
    )

    ctrl_dsts = [
        get_panel_loc(ctrl_nums[i], ctrl_staircase[i], ctrl_flipped[i], ctrl_quarter[i])
        for i in range(len(ctrl_ids))
    ]
    ctrl_H0s = [
        cv2.getPerspectiveTransform(src, dst) for src, dst in zip(ctrl_srcs, ctrl_dsts)
    ]

    all_idcs = np.arange(len(img_ids))
    ctrl_map = np.argmin(
        np.abs(all_idcs[:, None] - np.array(ctrl_idcs)[None, :]), axis=1
    )
    # ctrl_map = [
    #     ctrl_idcs[i]
    #     for i in np.argmin(
    #         np.abs(all_idcs[:, None] - np.array(ctrl_idcs)[None, :]), axis=1
    #     )
    # ]

    size = (16500, 1250)
    start = 0
    end = len(img_ids)  # id_to_idx(6700)
    step = 1
    stitched_img = np.zeros(size[::-1] + (3,))
    stitch_count = np.zeros(size[::-1])
    Hs = []
    for img, c2w, i in tqdm(
        zip(
            imgs[start:end:step],
            c2ws[start:end:step],
            ctrl_map[start:end:step],
        ),
        total=(end - start) // step,
    ):
        wrpd_img, H = warp_image(img, c2w, ctrl_c2ws[i], K, ctrl_pis[i], size, ctrl_H0s[i])
        Hs.append(H.reshape(1, -1))
        mask = np.sum(wrpd_img > 0, axis=2).astype(bool).astype(float)
        stitched_img += wrpd_img
        stitch_count += mask
    stitched_img = np.round(
        np.where(stitched_img > 0, stitched_img / stitch_count[..., None], stitched_img)
    ).astype(int)
    Hs = np.concatenate(Hs)
    Hs = np.concatenate((img_ids[start:end:step, None], Hs), axis=1)
    np.savetxt("homographies.txt", Hs)

    # wrpd_imgs = [
    #     warp_image(img, c2w, c2ws[ctrl_idcs[i]], K, v, size, ctrl_H0s[i])[None, ...]
    #     for img, c2w, i in zip(
    #         imgs[start:end:step], c2ws[start:end:step], ctrl_map[start:end:step]
    #     )
    # ]
    # wrpd_imgs = np.concatenate(wrpd_imgs)
    # stitched_img = np.max(wrpd_imgs, axis=0)

    # wrpd_ctrl_imgs = [
    #     cv2.warpPerspective(img, H0, size)[None, ...]
    #     for img, H0 in zip(ctrl_imgs, ctrl_H0s)
    # ]
    # wrpd_ctrl_imgs = np.concatenate(wrpd_ctrl_imgs)
    # stitched_ctrl_img = np.max(wrpd_ctrl_imgs, axis=0)

    # np.savez(
    #     "stitching.npz", stitched_img=stitched_img, stitched_ctrl_img=stitched_ctrl_img
    # )

    # stitching = np.load("stitching.npz")
    # stitched_img = stitching["stitched_img"]
    # stitched_ctrl_img = stitching["stitched_ctrl_img"]

    # stitched_img = np.where(stitched_ctrl_img > 0, stitched_ctrl_img, stitched_img)
    stitched_img = np.where(stitched_img > 0, stitched_img, 100 * np.ones_like(stitched_img))

    img = Image.fromarray(np.uint8(stitched_img))
    img.save("stitched.jpg")

    fig, axs = plt.subplots(1)
    axs.imshow(stitched_img)
    plt.show()
