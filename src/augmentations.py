"""
Data augmentation pipelines for Cityscapes semantic segmentation.

Uses albumentations to apply the same geometric and photometric transforms
to both image and mask, so that segmentation labels stay aligned.
Compatible with CityscapesDataGenerator (expects callable returning dict with 'image' and 'mask').
"""

import albumentations as A
from typing import Tuple


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
        additional_targets={},
    )


def get_training_augmentation(
    image_size: Tuple[int, int],
    p: float = 0.5,
) -> A.Compose:
    """
    Full augmentation pipeline for semantic segmentation (driving scenes).

    Includes geometric (flip, shift/scale/rotate) and photometric
    (brightness, contrast, blur, HSV) transforms. All transforms
    are applied to both image and mask so labels remain aligned.

    Args:
        image_size: (height, width) - used only for consistency; resizing is done in the data generator.
        p: Base probability of applying each transform.

    Returns:
        albumentations.Compose callable. Use as: out = transform(image=img, mask=mask).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                value=0,
                mask_value=0,
                p=p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=p,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=p * 0.5),
            A.GaussNoise(var_limit=(5.0, 25.0), p=p * 0.5),
        ],
        additional_targets={},
    )
