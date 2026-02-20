"""
Keras DataGenerator for Cityscapes dataset.

This module provides a custom DataGenerator class that inherits from
keras.utils.Sequence to efficiently load and preprocess Cityscapes images
and masks for semantic segmentation tasks.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tensorflow import keras
from sklearn.model_selection import train_test_split

from src.utils import (
    IMAGES_DIR,
    load_image,
    load_mask,
    convert_to_8_categories,
    get_image_path,
    get_mask_path,
    mask_to_colored,
)


def create_partition(split: str = "train") -> List[Tuple[str, int, int]]:
    """
    Create a list of (city, sequence, frame) tuples for a given split.

    Note: Only uses the "train" split from the dataset, as validation
    images are not available in leftImg8bit.

    Args:
        split: Dataset split (only 'train' is used, regardless of this parameter)

    Returns:
        List of tuples (city, sequence, frame) representing all samples
    """
    from pathlib import Path
    partition = []
    # Always use "train" split since validation images don't exist
    # Ensure IMAGES_DIR is a Path object
    if not isinstance(IMAGES_DIR, Path):
        images_dir = Path(IMAGES_DIR)
    else:
        images_dir = IMAGES_DIR
    split_dir = images_dir / "train"

    if not split_dir.exists():
        raise ValueError(f"Split directory {split_dir} does not exist")

    # Iterate through all cities
    for city_dir in split_dir.iterdir():
        if city_dir.is_dir():
            city = city_dir.name
            # Get all image files in this city
            image_files = list(city_dir.glob("*_leftImg8bit.png"))

            for img_file in image_files:
                # Parse filename: city_sequence_frame_leftImg8bit.png
                parts = img_file.stem.replace("_leftImg8bit", "").split("_")
                if len(parts) >= 3:
                    sequence = int(parts[1])
                    frame = int(parts[2])
                    partition.append((city, sequence, frame))

    return partition


class CityscapesDataGenerator(keras.utils.Sequence):
    """
    Data generator for Cityscapes semantic segmentation dataset.

    This generator loads images and masks on-the-fly, converts masks to
    8 categories, and optionally applies data augmentation and resizing.

    Attributes:
        list_IDs: List of (city, sequence, frame) tuples
        batch_size: Number of samples per batch
        dim: Target dimensions (height, width) for resizing
        n_channels: Number of image channels (3 for RGB)
        n_classes: Number of output classes (8 for Cityscapes categories)
        shuffle: Whether to shuffle data after each epoch
        augmentation: Optional augmentation function
        normalize: Whether to normalize images to [0, 1]
    """

    def __init__(
        self,
        list_IDs: List[Tuple[str, int, int]],
        split: str = "train",
        batch_size: int = 32,
        dim: Tuple[int, int] = (512, 512),
        n_channels: int = 3,
        n_classes: int = 8,
        shuffle: bool = True,
        augmentation: Optional[callable] = None,
        normalize: bool = True,
    ):
        """
        Initialize the data generator.

        Args:
            list_IDs: List of (city, sequence, frame) tuples
            split: Dataset split ('train', 'val', or 'test')
            batch_size: Number of samples per batch
            dim: Target image dimensions (height, width); images are resized to dim when different.
            n_channels: Number of image channels
            n_classes: Number of segmentation classes
            shuffle: Whether to shuffle data after each epoch
            augmentation: Optional augmentation function (albumentations)
            normalize: Whether to normalize images to [0, 1]
        """
        self.list_IDs = list_IDs
        self.split = split
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.normalize = normalize
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch.

        Returns:
            Number of batches
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Args:
            index: Batch index

        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def visualize_batch(
        self,
        batch_index: int = 0,
        num_samples: Optional[int] = None,
        show_original_then_resized: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize a batch of images and their segmentation masks.

        Args:
            batch_index: Index of the batch to visualize (default: 0).
            num_samples: Number of samples to show (default: min(4, batch_size)).
            show_original_then_resized: If True and dim != original image size, show each sample
                at original size (image | mask) then resized (image | mask). If False or sizes
                match, only the resized batch is shown (image | mask).
            figsize: Figure size (width, height). Auto-sized if None.
            save_path: If provided, save the figure to this path.
        """
        n_show = num_samples if num_samples is not None else min(4, self.batch_size)
        n_show = min(n_show, self.batch_size)

        indexes = self.indexes[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes[:n_show]]

        X, y = self[batch_index]
        if self.normalize:
            X_resized = (X[:n_show] * 255).astype(np.uint8)
        else:
            X_resized = X[:n_show].astype(np.uint8)
        y_resized = y[:n_show]

        need_originals = False
        if show_original_then_resized:
            X_orig_list, y_orig_list = [], []
            for (city, sequence, frame) in list_IDs_batch:
                img_path = get_image_path(city, sequence, frame, split=self.split)
                mask_path = get_mask_path(city, sequence, frame, split=self.split)
                img = load_image(img_path)
                mask = load_mask(mask_path)
                mask = convert_to_8_categories(mask)
                X_orig_list.append(img)
                y_orig_list.append(mask)
            need_originals = any(a.shape[:2] != self.dim for a in X_orig_list)

        if need_originals:
            n_cols = 4
            if figsize is None:
                figsize = (16, 4 * n_show)
            fig, axes = plt.subplots(n_show, n_cols, figsize=figsize, squeeze=False)
            for i in range(n_show):
                axes[i, 0].imshow(X_orig_list[i])
                axes[i, 0].set_title("Image (original)")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(mask_to_colored(y_orig_list[i]))
                axes[i, 1].set_title("Mask (original)")
                axes[i, 1].axis("off")
                axes[i, 2].imshow(X_resized[i])
                axes[i, 2].set_title("Image (resized)")
                axes[i, 2].axis("off")
                axes[i, 3].imshow(mask_to_colored(y_resized[i]))
                axes[i, 3].set_title("Mask (resized)")
                axes[i, 3].axis("off")
        else:
            n_cols = 2
            if figsize is None:
                figsize = (4 * n_show, 8)
            fig, axes = plt.subplots(n_show, n_cols, figsize=figsize, squeeze=False)
            for i in range(n_show):
                axes[i, 0].imshow(X_resized[i])
                axes[i, 0].set_title("Image")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(mask_to_colored(y_resized[i]))
                axes[i, 1].set_title("Mask")
                axes[i, 1].axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

    def __data_generation(
        self, list_IDs_temp: List[Tuple[str, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples.

        Args:
            list_IDs_temp: List of (city, sequence, frame) tuples for this batch

        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        # Initialize arrays
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, (city, sequence, frame) in enumerate(list_IDs_temp):
            try:
                # Get paths
                img_path = get_image_path(city, sequence, frame, split=self.split)
                mask_path = get_mask_path(city, sequence, frame, split=self.split)

                # Load image and mask
                image = load_image(img_path)
                mask = load_mask(mask_path)

                # Convert mask to 8 categories
                mask = convert_to_8_categories(mask)

                # Resize if dimensions differ from target dim
                if image.shape[:2] != self.dim:
                    image = cv2.resize(image, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_NEAREST)

                # Apply augmentation if provided
                if self.augmentation is not None:
                    augmented = self.augmentation(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']

                # Normalize image if needed
                if self.normalize:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)

                # Store sample
                X[i,] = image
                y[i,] = mask

            except Exception as e:
                print(f"Error loading sample ({city}, {sequence}, {frame}): {e}")
                # Fill with zeros if loading fails
                X[i,] = np.zeros((*self.dim, self.n_channels), dtype=np.float32)
                y[i,] = np.zeros(self.dim, dtype=np.uint8)

        # Convert masks to one-hot encoding if needed
        # For segmentation, we typically keep masks as integer labels
        # But if your model requires one-hot, uncomment the following:
        # y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y


def create_data_generators(
    batch_size: int = 32,
    dim: Tuple[int, int] = (512, 512),
    augmentation: Optional[callable] = None,
    normalize: bool = True,
    validation_split: float = 0.2,
    random_state: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[CityscapesDataGenerator, CityscapesDataGenerator]:
    """
    Create training and validation data generators.

    Args:
        batch_size: Batch size
        dim: Target image dimensions (height, width); images are resized to dim when different.
        augmentation: Optional augmentation function (applied only to training)
        normalize: Whether to normalize images
        validation_split: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducible train/val split
        max_samples: Optional. If provided, only this many samples will be randomly
                     selected from the full partition before splitting into train/val.

    Returns:
        Tuple of (training_generator, validation_generator)
    """
    # Create partition from training data
    full_partition = create_partition("train")

    # Optionally select a subset of samples
    if max_samples is not None and max_samples < len(full_partition):
        np.random.seed(random_state) # Ensure reproducibility for sample selection
        # Select random indices and then use them to get the samples
        indices = np.random.choice(len(full_partition), size=max_samples, replace=False)
        full_partition = [full_partition[i] for i in indices]
        print(f"Using a subset of {max_samples} samples from the full partition.")

    # Split into train and validation
    train_partition, val_partition = train_test_split(
        full_partition,
        test_size=validation_split,
        random_state=random_state,
        shuffle=True,
        # Stratify by city to ensure representation across cities
        # However, for stratify to work, each class must have at least 2 samples.
        # If a city only has 1 image, stratify will fail. So we'll disable it for now.
        # stratify=[p[0] for p in full_partition] if full_partition else None
    )

    print(f"Total samples used: {len(full_partition)}")
    print(f"Training samples: {len(train_partition)}")
    print(f"Validation samples: {len(val_partition)}")

    # Create generators
    training_generator = CityscapesDataGenerator(
        list_IDs=train_partition,
        split="train",
        batch_size=batch_size,
        dim=dim,
        shuffle=True,
        augmentation=augmentation,
        normalize=normalize,
    )

    validation_generator = CityscapesDataGenerator(
        list_IDs=val_partition,
        split="train",  # Use "train" split for both, but different data subsets
        batch_size=batch_size,
        dim=dim,
        shuffle=False,  # No need to shuffle validation
        augmentation=None,  # No augmentation for validation
        normalize=normalize,
    )

    return training_generator, validation_generator
