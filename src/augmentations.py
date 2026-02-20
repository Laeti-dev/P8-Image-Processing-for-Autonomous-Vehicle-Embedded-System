"""
Data augmentation pipelines for Cityscapes semantic segmentation.

Uses albumentations to apply the same geometric and photometric transforms
to both image and mask, so that segmentation labels stay aligned.
Compatible with CityscapesDataGenerator (expects callable returning dict with 'image' and 'mask').
"""

import albumentations as A
from typing import Tuple, Optional, Sequence




def get_light_augmentation(
    image_size: Tuple[int, int],
    p: float = 0.5,
) -> A.Compose:
    """
    Light augmentation pipeline for faster training.

    Applies only horizontal flip and mild brightness/contrast changes.
    Use when you want some regularization without slowing down epochs too much.

    Args:
        image_size: (height, width) - used only for consistency; resizing is done in the data generator.
        p: Probability of applying each transform.

    Returns:
        albumentations.Compose callable. Use as: out = transform(image=img, mask=mask).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=p,
            ),
        ],
    )


def get_training_augmentation(
    image_size: Tuple[int, int],
    p: float = 0.5,
    extra_transforms: Optional[Sequence[A.BasicTransform]] = None,
) -> A.Compose:
    transforms = [
        A.HorizontalFlip(p=p),
    ]
    if extra_transforms:
        transforms.extend(extra_transforms)
    return A.Compose(transforms)

