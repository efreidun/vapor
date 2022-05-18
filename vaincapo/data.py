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

from vaincapo.utils import read_poses


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
