"""
Keras DataGenerator for Cityscapes dataset.

This module provides a custom DataGenerator class that inherits from
keras.utils.Sequence to efficiently load and preprocess Cityscapes images
and masks for semantic segmentation tasks.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from tensorflow import keras
from sklearn.model_selection import train_test_split

from src.utils import (
    IMAGES_DIR,
    load_image,
    load_mask,
    convert_to_8_categories,
    get_image_path,
    get_mask_path
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
    partition = []
    # Always use "train" split since validation images don't exist
    split_dir = IMAGES_DIR / "train"

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
        normalize: bool = True
    ):
        """
        Initialize the data generator.

        Args:
            list_IDs: List of (city, sequence, frame) tuples
            split: Dataset split ('train', 'val', or 'test')
            batch_size: Number of samples per batch
            dim: Target image dimensions (height, width)
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

                # Resize if needed
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
    max_samples: Optional[int] = None
) -> Tuple[CityscapesDataGenerator, CityscapesDataGenerator]:
    """
    Create training and validation data generators.

    Args:
        batch_size: Batch size
        dim: Target image dimensions (height, width)
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
        normalize=normalize
    )

    validation_generator = CityscapesDataGenerator(
        list_IDs=val_partition,
        split="train",  # Use "train" split for both, but different data subsets
        batch_size=batch_size,
        dim=dim,
        shuffle=False,  # No need to shuffle validation
        augmentation=None,  # No augmentation for validation
        normalize=normalize
    )

    return training_generator, validation_generator
