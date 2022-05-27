"""Module that defines the dataloader tools."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    GaussianBlur,
    CenterCrop,
    ColorJitter,
    RandomCrop,
)
from tqdm import tqdm

from vaincapo.read_write import read_poses, read_tfmat


class CambridgeLandmarks(Dataset):
    """Dataset class for the Cambridge Landmarks dataset."""

    def __init__(
        self,
        split_file_path: Path,
        image_size: int,
        mode: str = "resize",
        crop: Optional[float] = None,
        gauss_kernel: Optional[int] = None,
        gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
        jitter_brightness: Optional[float] = None,
        jitter_contrast: Optional[float] = None,
        jitter_saturation: Optional[float] = None,
        jitter_hue: Optional[float] = None,
    ) -> None:
        """Construct the cambridge landmarks dataset class.

        Args:
            split_file_path: path of the text file listing samples in the data split
            image_size: image size (images are made square)
            mode: how image is made smaller,
                options: "resize", "random_crop", "vertical_crop", "center_crop"
            crop: what ratio of original height and width is cropped
            gauss_kernel: size of Gaussian blur kernel
            gauss_sigma: [min and max of] std dev for creating Gaussian blur kernel
            jitter_brightness: brightness jitter
            jitter_contrast: contrast jitter
            jitter_saturation: saturation jitter
            jitter_hue: hue jitter
        """
        print("Preparing dataset...")
        seq_ids, img_ids, tvecs, rotmats = read_poses(
            split_file_path, "CambridgeLandmarks"
        )

        self._img_paths = [
            split_file_path.parent
            / f"seq{str(seq_id)}"
            / f"frame{str(img_id).zfill(5)}.png"
            for seq_id, img_id in zip(seq_ids, img_ids)
        ]

        self._poses = torch.from_numpy(
            np.concatenate((tvecs, rotmats.reshape(-1, 9)), axis=1)
        )

        self._names = [
            str(img_path.relative_to(split_file_path.parent))
            for img_path in self._img_paths
        ]

        self._transform = create_transforms(
            *ToTensor()(Image.open(self._img_paths[0])).shape[1:],
            True,
            image_size,
            mode,
            crop,
            gauss_kernel,
            gauss_sigma,
            jitter_brightness,
            jitter_contrast,
            jitter_saturation,
            jitter_hue,
        )

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get sample by index.

        Args:
            sample index

        Returns:
            image tensor, shape (3, H, W),
            image pose, shape (12,)
                formatted as (tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33),
            image name
        """
        return (
            self._transform(Image.open(self._img_paths[index])),
            self._poses[index],
            self._names[index],
        )


class SevenScenes(Dataset):
    """Dataset class for the 7-Scenes dataset."""

    def __init__(
        self,
        root_path: Path,
        image_size: int,
        mode: str = "resize",
        crop: Optional[float] = None,
        gauss_kernel: Optional[int] = None,
        gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
        jitter_brightness: Optional[float] = None,
        jitter_contrast: Optional[float] = None,
        jitter_saturation: Optional[float] = None,
        jitter_hue: Optional[float] = None,
    ) -> None:
        """Construct the 7-scenes dataset class.

        Args:
            root_path: path of the root directory of the data split
            image_size: image size (images are made square)
            mode: how image is made smaller,
                options: "resize", "random_crop", "vertical_crop", "center_crop"
            crop: what ratio of original height and width is cropped
            gauss_kernel: size of Gaussian blur kernel
            gauss_sigma: [min and max of] std dev for creating Gaussian blur kernel
            jitter_brightness: brightness jitter
            jitter_contrast: contrast jitter
            jitter_saturation: saturation jitter
            jitter_hue: hue jitter
        """
        print("Preparing dataset...")
        seq_paths = [root_path / seq for seq in sorted(next(os.walk(root_path))[1])]

        self._img_paths = [
            img_path
            for seq_path in seq_paths
            for img_path in seq_path.glob("*.color.png")
        ]
        self._names = [
            str(img_path.relative_to(root_path)) for img_path in self._img_paths
        ]
        pose_paths = [
            Path(str(img_path).replace("color.png", "pose.txt"))
            for img_path in self._img_paths
        ]

        tvecs = []
        rotmats = []
        for pose_path in tqdm(pose_paths):
            tvec, rotmat = read_tfmat(pose_path)
            tvecs.append(tvec[None, :])
            rotmats.append(rotmat[None, :, :])
        tvecs = np.concatenate(tvecs)
        rotmats = np.concatenate(rotmats)
        self._poses = torch.from_numpy(
            np.concatenate((tvecs, rotmats.reshape(-1, 9)), axis=1)
        )

        self._transform = create_transforms(
            *ToTensor()(Image.open(self._img_paths[0])).shape[1:],
            True,
            image_size,
            mode,
            crop,
            gauss_kernel,
            gauss_sigma,
            jitter_brightness,
            jitter_contrast,
            jitter_saturation,
            jitter_hue,
        )

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get sample by index.

        Args:
            sample index

        Returns:
            image tensor, shape (3, H, W),
            image pose, shape (12,)
                formatted as (tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33),
            image name
        """
        return (
            self._transform(Image.open(self._img_paths[index])),
            self._poses[index],
            self._names[index],
        )


class AmbiguousReloc(Dataset):
    """Dataset class for ambiguous relocalisation dataset."""

    def __init__(
        self,
        root_path: Path,
        image_size: int,
        mode: str = "resize",
        crop: Optional[float] = None,
        gauss_kernel: Optional[int] = None,
        gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
        jitter_brightness: Optional[float] = None,
        jitter_contrast: Optional[float] = None,
        jitter_saturation: Optional[float] = None,
        jitter_hue: Optional[float] = None,
    ) -> None:
        """Construct the ambiguous relocalisation dataset class.

        Args:
            root_path: path of the root directory of the data split
            image_size: image size (images are made square)
            mode: how image is made smaller,
                options: "resize", "random_crop", "vertical_crop", "center_crop"
            crop: what ratio of original height and width is cropped
            gauss_kernel: size of Gaussian blur kernel
            gauss_sigma: [min and max of] std dev for creating Gaussian blur kernel
            jitter_brightness: brightness jitter
            jitter_contrast: contrast jitter
            jitter_saturation: saturation jitter
            jitter_hue: hue jitter
        """
        print("Preparing dataset...")
        seq_paths = [root_path / seq for seq in sorted(next(os.walk(root_path))[1])]
        poses_paths = [
            seq_path / f"poses_{seq_path.stem}.txt" for seq_path in seq_paths
        ]

        seq_ids = []
        img_ids = []
        tvecs = []
        rotmats = []
        for poses_path in poses_paths:
            seq_id, img_id, tvec, rotmat = read_poses(poses_path, "AmbiguousReloc")
            seq_ids.append(seq_id)
            img_ids.append(img_id)
            tvecs.append(tvec)
            rotmats.append(rotmat)
        seq_ids = np.concatenate(seq_ids)
        img_ids = np.concatenate(img_ids)
        tvecs = np.concatenate(tvecs)
        rotmats = np.concatenate(rotmats)
        img_paths = [
            root_path
            / f"seq{str(seq_id).zfill(2)}"
            / "rgb_matched"
            / f"frame-color-{str(img_id).zfill(4)}.png"
            for seq_id, img_id in zip(seq_ids, img_ids)
        ]

        self._poses = torch.from_numpy(
            np.concatenate((tvecs, rotmats.reshape(-1, 9)), axis=1)
        )
        self._images = torch.cat(
            [
                ToTensor()(Image.open(img_path))[None, ...]
                for img_path in tqdm(img_paths)
            ]
        )
        self._names = [str(img_path.relative_to(root_path)) for img_path in img_paths]

        self._transform = create_transforms(
            *self._images.shape[2:],
            False,
            image_size,
            mode,
            crop,
            gauss_kernel,
            gauss_sigma,
            jitter_brightness,
            jitter_contrast,
            jitter_saturation,
            jitter_hue,
        )

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get sample by index.

        Args:
            sample index

        Returns:
            image tensor, shape (3, H, W),
            image pose, shape (12,)
                formatted as (tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33),
            image name
        """
        return (
            self._transform(self._images[index]),
            self._poses[index],
            self._names[index],
        )


def create_transforms(
    height: int,
    width: int,
    to_tensor: bool,
    image_size: int,
    mode: str = "resize",
    crop: Optional[float] = None,
    gauss_kernel: Optional[int] = None,
    gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
    jitter_brightness: Optional[float] = None,
    jitter_contrast: Optional[float] = None,
    jitter_saturation: Optional[float] = None,
    jitter_hue: Optional[float] = None,
) -> Compose:
    """Create transforms.

    Args:
        height: original height of images
        width: original width of images
        to_tensor: if True, ToTensor transform is included
        image_size: image size (images are made square)
        mode: how image is made smaller,
            options: "resize", "random_crop", "vertical_crop", "center_crop"
        crop: what ratio of original height and width is cropped
        gauss_kernel: size of Gaussian blur kernel
        gauss_sigma: [min and max of] std dev for creating Gaussian blur kernel
        jitter_brightness: brightness jitter
        jitter_contrast: contrast jitter
        jitter_saturation: saturation jitter
        jitter_hue: hue jitter
    """
    transforms = [ToTensor()] if to_tensor else []
    if crop is not None:
        print(f"Images are cropped by {crop} of its height and width")
        transforms.append(RandomCrop((int(crop * height), int(crop * width))))
    if mode == "resize":
        print(f"Images are resized to {image_size}x{image_size}")
        transforms.append(Resize((image_size, image_size)))
    elif mode == "center_crop":
        print(f"Images are center cropped to {image_size}x{image_size}")
        transforms.append(CenterCrop(image_size))
    elif mode == "random_crop" or mode == "vertical_crop":
        if mode == "vertical_crop":
            print(f"Images are cropped such that smallest edge is {image_size}")
            transforms.append(Resize(image_size))
        print(f"Images are random cropped to {image_size}x{image_size}")
        transforms.append(RandomCrop((image_size, image_size)))
    else:
        raise ValueError("Invalid image resizing mode.")
    if gauss_kernel is not None and gauss_sigma is not None:
        print(
            f"Gaussian blur of kernel size {gauss_kernel}"
            + f" and std dev {gauss_sigma} is applied"
        )
        transforms.append(GaussianBlur(gauss_kernel, sigma=gauss_sigma))
    if (
        jitter_brightness is not None
        and jitter_contrast is not None
        and jitter_saturation is not None
        and jitter_hue is not None
    ):
        print(
            f"Color jitter of {jitter_brightness} brightness,"
            + f" {jitter_contrast} contrast,"
            + f" {jitter_saturation} saturation, and {jitter_hue} hue is applied"
        )
        transforms.append(
            ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            )
        )
    return Compose(transforms)
