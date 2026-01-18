
"""
Utility functions for Cityscapes dataset loading and processing.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import os

# Determine project root directory
# This file is in src/, so we go up one level to get project root
# Handle case where __file__ might not be available (e.g., in some notebook environments)

if 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_IP' in os.environ:
    PROJECT_ROOT = Path("/content/dataset")
else:
    try:
        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
        # Try using __file__ first (works when imported as a module)
    except (NameError, AttributeError):
        # Fallback: try to find project root by looking for common markers
        current_dir = Path(os.getcwd())
        # If we're in notebooks/, go up one level first
        if current_dir.name == "notebooks":
            current_dir = current_dir.parent

        # Look for project root markers (README.md, requirements.txt, src/, data/)
        found_root = False
        search_dir = current_dir
        max_depth = 5  # Prevent infinite loops
        depth = 0

    while depth < max_depth and search_dir != search_dir.parent:
        if ((search_dir / "README.md").exists() and
            (search_dir / "requirements.txt").exists() and
            (search_dir / "src").exists() and
            (search_dir / "data").exists()):
            PROJECT_ROOT = search_dir.resolve()
            found_root = True
            break
        search_dir = search_dir.parent
        depth += 1

    if not found_root:
        # Last resort: use current working directory (or parent if in notebooks/)
        PROJECT_ROOT = current_dir.resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent.resolve()

# Dataset paths (relative to project root)
# Ensure all paths are absolute
DATA_ROOT = (PROJECT_ROOT / "data" / "raw").resolve()
# For colab
if not DATA_ROOT.exists() and Path("/content/dataset").exists():
    DATA_ROOT = Path("/content/dataset")

IMAGES_DIR = (DATA_ROOT / "leftImg8bit").resolve()
MASKS_DIR = (DATA_ROOT / "gtFine").resolve()

# Debug: Print paths when module is loaded (can be disabled in production)
# Uncomment the following lines for debugging:
# print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
# print(f"DEBUG: IMAGES_DIR = {IMAGES_DIR}")
# print(f"DEBUG: IMAGES_DIR exists = {IMAGES_DIR.exists()}")

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
    # Lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for class_id, cat_id in CLASS_TO_CATEGORY.items():
        if class_id != -1:
            lut[class_id] = cat_id

    return lut[mask]

def mask_to_colored(mask, color_map=None):
    """Convert category mask to colored visualization."""
    if color_map is None:
        color_map = CATEGORY_COLORS
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for category_id, color in color_map.items():
        colored[mask == category_id] = color
    return colored
