# dataset.py
"""
Dataset classes. 
TODO: add class & methods docstrings
"""

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor
from torchgeo.datasets.landcoverai import LandCoverAI
from torchvision.transforms import RandomResizedCrop

from config import (
    COLOR_JITTER,
    IMAGE_SIZE,
    MEAN,
    P_HORIZONTAL_FLIP,
    RATIO,
    ROTATION_DEGREES,
    SCALE,
    STD,
)


# Helper function to ensure mask has a channel dimension
def ensure_mask_channel_dim(mask):
    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
        return mask.unsqueeze(0)  # From [H, W] to [1, H, W]
    elif hasattr(mask, "mode") and mask.mode != "L":
        return mask.convert("L")  # Ensure it's in mode 'L' for single channel
    return mask


class TrainTransform:
    def __init__(
        self,
        image_size: tuple = IMAGE_SIZE,
        scale: tuple = SCALE,
        ratio: tuple = RATIO,
        rotation_degrees: int = ROTATION_DEGREES,
        p_horizontal_flip: float = P_HORIZONTAL_FLIP,
        color_jitter: float = COLOR_JITTER,
        mean: Tensor = MEAN,
        std: Tensor = STD,
    ):
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.rotation_degrees = rotation_degrees
        self.p_horizontal_flip = p_horizontal_flip
        self.color_jitter = color_jitter
        self.mean = mean
        self.std = std

        # Initialize color jitter (only for images)
        self.color_jitter_transform = T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter,
        )

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        mask = ensure_mask_channel_dim(mask)

        # Scale image to [0, 1]
        image = image / 255.0

        # ----- Spatial Transformations -----

        # Random Horizontal Flip
        if random.random() < self.p_horizontal_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random Resized Crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )
        image = TF.resized_crop(
            image, i, j, h, w, self.image_size, TF.InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, self.image_size, TF.InterpolationMode.NEAREST
        )

        # ----- Color Jitter (only on image) -----
        image = self.color_jitter_transform(image)

        # ----- Normalize Image -----
        image = T.Normalize(mean=self.mean, std=self.std)(image)

        # ----- Convert Mask to Tensor -----
        mask = torch.as_tensor(np.array(mask), dtype=torch.long).squeeze(0)

        sample["image"] = image
        sample["mask"] = mask

        return sample


class ValTestTransform:
    def __init__(
        self, image_size: tuple = IMAGE_SIZE, mean: Tensor = MEAN, std: Tensor = STD
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        mask = ensure_mask_channel_dim(mask)

        # Scale image to [0, 1]
        image = image / 255.0

        # Resize Image and Mask
        image = TF.resize(
            image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST
        )

        # Normalize Image
        image = T.Normalize(mean=self.mean, std=self.std)(image)

        # Convert Mask to Tensor
        mask = torch.as_tensor(np.array(mask), dtype=torch.long).squeeze(0)

        sample["image"] = image
        sample["mask"] = mask

        return sample


def get_datasets(
    data_root: str, train_transform: TrainTransform, val_transform: ValTestTransform
):
    train_dataset = LandCoverAI(
        root=data_root, split="train", transforms=train_transform
    )
    val_dataset = LandCoverAI(root=data_root, split="val", transforms=val_transform)
    test_dataset = LandCoverAI(root=data_root, split="test", transforms=val_transform)

    return train_dataset, val_dataset, test_dataset
