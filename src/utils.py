
"""
Utility functions for Cityscapes dataset loading and processing.
"""

import numpy as np
from pathlib import Path
from PIL import Image

# Dataset paths
DATA_ROOT = Path("data/raw")
IMAGES_DIR = DATA_ROOT / "leftImg8bit"
MASKS_DIR = DATA_ROOT / "gtFine"

# Cityscapes class to category mapping
CLASS_TO_CATEGORY = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 1, 9: 1, 10: 1,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
    17: 3, 18: 3, 19: 3, 20: 3,
    21: 4, 22: 4,
    23: 5,
    24: 6, 25: 6,
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,
    -1: 0
}

CATEGORY_NAMES = {
    0: 'void', 1: 'flat', 2: 'construction', 3: 'object',
    4: 'nature', 5: 'sky', 6: 'human', 7: 'vehicle'
}

CATEGORY_COLORS = {
    0: [0, 0, 0],
    1: [128, 64, 128],
    2: [70, 70, 70],
    3: [153, 153, 153],
    4: [107, 142, 35],
    5: [70, 130, 180],
    6: [220, 20, 60],
    7: [0, 0, 142]
}

def get_image_path(city, sequence, frame, split="train"):
    """Get path to an image file."""
    filename = f"{city}_{sequence:06d}_{frame:06d}_leftImg8bit.png"
    return IMAGES_DIR / split / city / filename

def get_mask_path(city, sequence, frame, split="train"):
    """Get path to a labelIds mask file."""
    filename = f"{city}_{sequence:06d}_{frame:06d}_gtFine_labelIds.png"
    return MASKS_DIR / split / city / filename

def load_image(image_path):
    """Load an image as numpy array."""
    img = Image.open(image_path)
    return np.array(img)

def load_mask(mask_path):
    """Load a mask as numpy array."""
    mask = Image.open(mask_path)
    return np.array(mask)

def convert_to_8_categories(mask):
    """Convert Cityscapes 34-class mask to 8-category mask."""
    category_mask = np.zeros_like(mask, dtype=np.uint8)
    for class_id, category_id in CLASS_TO_CATEGORY.items():
        category_mask[mask == class_id] = category_id
    unmapped = ~np.isin(mask, list(CLASS_TO_CATEGORY.keys()))
    if unmapped.any():
        category_mask[unmapped] = 0
    return category_mask

def mask_to_colored(mask, color_map=None):
    """Convert category mask to colored visualization."""
    if color_map is None:
        color_map = CATEGORY_COLORS
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for category_id, color in color_map.items():
        colored[mask == category_id] = color
    return colored
