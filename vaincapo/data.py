"""Module that defines the dataloader tools."""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
from glob import glob

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
    Normalize,
)
from tqdm import tqdm

from vaincapo.utils import read_poses


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
        seq_paths = [root_path / seq for seq in sorted(next(os.walk(root_path))[1])]
        poses_paths = [
            seq_path / f"poses_{seq_path.stem}.txt" for seq_path in seq_paths
        ]

        seq_ids = []
        img_ids = []
        tvecs = []
        rotmats = []
        for poses_path in poses_paths:
            seq_id, img_id, tvec, rotmat = read_poses(poses_path)
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

        print("Preparing dataset...")
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

        transforms = []
        if crop is not None:
            print(f"Images are cropped by {crop} of its height and width")
            height, width = self._images.shape[2:]
            transforms.append(RandomCrop((int(crop * height), int(crop * width))))
        if mode == "resize":
            print(f"Images are resized to {image_size}x{image_size}")
            transforms.append(Resize((image_size, image_size)))
        elif mode == "center_crop":
            print(f"Images are center cropped to {image_size}x{image_size}")
            transforms.append(CenterCrop(image_size))
        else:
            if mode == "vertical_crop":
                print(f"Images are cropped such that smallest edge is {image_size}")
                transforms.append(Resize(image_size))
            print(f"Images are random cropped to {image_size}x{image_size}")
            transforms.append(RandomCrop((image_size, image_size)))
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
        self._transform = Compose(transforms)

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
        return self._images[index], self._poses[index], self._names[index]


class AmbiguousImages(Dataset):
    """Dataset class for ambiguous relocalisation dataset."""

    def __init__(
        self,
        root: str,
        image_size: int,
        augment: bool = False,
        mean: Optional[List] = None,
        std: Optional[List] = None,
    ) -> None:
        """construct the ambiguous scene images dataset class.

        Args:
            root: path of the root directory of the scene image collection
            image_size: size of the smaller edge of the image
            augment: if True, dataset is augmented
            mean: dataset mean for normalization
            std: dataset std for normalization
        """
        self._augment = augment
        self._root_dir = Path(root)
        self._images_dir = self._root_dir / "rgb_matched"
        _, self._im_ids, self._trans, self._rotmats = read_poses(
            self._root_dir / f"poses_{self._root_dir.name}.txt"
        )
        image_paths = sorted(glob(str(self._images_dir / "*.png")))
        assert len(self._im_ids) == len(image_paths)

        height = 960
        width = 540
        transforms = [ToTensor()]
        if self._augment:
            transforms.append(RandomCrop((int(0.9 * height), int(0.9 * width))))
            # transforms.append(Resize(256))
            # transforms.append(RandomCrop((image_size, image_size)))
        transforms.append(Resize((image_size, image_size)))
        if mean is not None and std is not None:
            transforms.append(Normalize(mean, std))
        if self._augment:
            transforms.extend(
                [
                    GaussianBlur(3, sigma=(0.1, 1.0)),
                    ColorJitter(
                        brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
                    ),
                ]
            )
            # transforms.append(RandomCrop((image_size, image_size)))
        self._transform = Compose(transforms)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get sample by index.

        Args:
            sample index

        Returns:
            image tensor,
            image pose,
                formatted as (tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33)
        """
        im_id = self._im_ids[index]
        with Image.open(
            # self._image_paths[index]
            self._images_dir
            / f"frame-color-{str(im_id).zfill(4)}.png"
        ) as image:
            image_tensor = self._transform(image)

        tran = self._trans[index]
        rotmat = self._rotmats[index]

        # if self._augment:
        #     tran_noise = np.random.uniform(low=-0.05, high=0.05, size=3)
        #     tran += tran_noise
        #     euler_noise = np.random.uniform(low=-5.0, high=5.0, size=3)
        #     rot_noise = Rotation.from_euler(
        #         "zyx", euler_noise, degrees=True
        #     ).as_matrix()
        #     rotmat = rotmat @ rot_noise

        return image_tensor, np.concatenate((tran, rotmat.flatten()))

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._trans)
