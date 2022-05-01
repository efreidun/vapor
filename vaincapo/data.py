"""Module that defines the dataloader tools."""

from pathlib import Path
from typing import Optional, List
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
    RandomAffine,
    ColorJitter,
    RandomCrop,
    Normalize,
)


class AmbiguousToy(Dataset):
    def __init__(self):
        self._images = torch.zeros(10, 3, 64, 64)
        self._poses = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]).repeat(10, 1)
        self._images[0:4, 0] = 1
        self._poses[0, 0] = 0
        self._poses[1, 0] = 1
        self._poses[2, 0] = 2
        self._poses[3, 0] = 3
        self._images[4:7, 1] = 1
        self._poses[4, 1] = 1
        self._poses[5, 1] = 2
        self._poses[6, 1] = 3
        self._images[7:10, 2] = 1
        self._poses[7, 2] = 1
        self._poses[8, 2] = 2
        self._poses[9, 2] = 3

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index], self._poses[index]


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
        """Initialize the scene collection dataset class.

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
        with open(self._root_dir / f"poses_{self._root_dir.name}.txt") as f:
            content = f.readlines()
        parsed_poses = [
            [float(entry) for entry in line.strip().split(", ")] for line in content
        ]
        poses = np.array(parsed_poses, dtype=np.float32)[:, 2:]
        self._im_ids = np.array(parsed_poses, dtype=int)[:, 1]
        image_paths = sorted(glob(str(self._images_dir / "*.png")))
        assert len(poses) == len(image_paths)

        qw = poses[:, 0]
        qx = poses[:, 1]
        qy = poses[:, 2]
        qz = poses[:, 3]
        qw2, qx2, qy2, qz2 = qw ** 2, qx ** 2, qy ** 2, qz ** 2
        qwx, qwy, qwz = qw * qx, qw * qy, qw * qz
        qxy, qxz, qyz = qx * qy, qx * qz, qy * qz
        self._rotmats = np.vstack(
            [
                qw2 + qx2 - qy2 - qz2,
                2 * qxy - 2 * qwz,
                2 * qwy + 2 * qxz,
                2 * qwz + 2 * qxy,
                qw2 - qx2 + qy2 - qz2,
                2 * qyz - 2 * qwx,
                2 * qxz - 2 * qwy,
                2 * qwx + 2 * qyz,
                qw2 - qx2 - qy2 + qz2,
            ]
        ).T.reshape(-1, 3, 3)
        self._trans = poses[:, -3:]

        height = 960
        width = 540
        transforms = [ToTensor()]
        if self._augment:
            transforms.append(RandomCrop((int(0.9 * height), int(0.9 * width))))
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
