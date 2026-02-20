"""
Semantic segmentation predictor for Cityscapes 8-category masks.

This module provides inference capabilities for segmenting images using
a model loaded from local storage or Azure Blob Storage.
"""

import io
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from tensorflow import keras

from src.utils import PROJECT_ROOT, load_image, mask_to_colored

from src.metrics import (
    DiceCoefficient,
    IoUCoefficient,
    combined_loss,
)

# Default input size (H, W) — must match the trained model's input shape
DEFAULT_INPUT_SIZE = (256, 512)


def _load_image_from_source(
    image: Union[np.ndarray, Path, str, bytes]
) -> np.ndarray:
    """Load image from various sources into a numpy array."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, bytes):
        from PIL import Image
        img = Image.open(io.BytesIO(image))
        return np.array(img)
    if isinstance(image, (Path, str)):
        return load_image(image)
    raise TypeError(
        f"image must be np.ndarray, Path, str, or bytes, got {type(image)}"
    )


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess image for model inference.

    Same logic as the data generator: resize to target size and normalize to [0, 1].

    Args:
        image: Input image as numpy array (H, W, 3), RGB, uint8
        target_size: (height, width) to resize to
        normalize: If True, normalize to [0, 1] (float32)

    Returns:
        Preprocessed image (batch_size=1, H, W, 3), float32
    """
    if image.shape[:2] != target_size:
        image = cv2.resize(
            image,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if normalize:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


class SegmentationPredictor:
    """
    Predictor for semantic segmentation masks.

    Loads model from local path or Azure Blob Storage, with caching
    of Azure-downloaded models to avoid repeated downloads.
    """

    DEFAULT_CACHE_DIR = "models"
    DEFAULT_CACHE_NAME = "cached_model.keras"

    def __init__(
        self,
        azure_blob_name: Optional[str] = None,
        azure_container_name: str = "training-outputs",
        cache_dir: Optional[str] = None,
        input_shape: tuple[int, int, int] = (*DEFAULT_INPUT_SIZE, 3),
    ):
        """
        Initialize the predictor.

        Args:
            azure_blob_name: Name of the blob in Azure (e.g., "model/best_model.keras").
            azure_container_name: Azure container name (default: "training-outputs").
            cache_dir: Directory for caching model downloaded from Azure.
                       Default: PROJECT_ROOT / "models".
            input_shape: Expected input shape (H, W, C) for validation.
        """
        self.azure_blob_name = azure_blob_name
        self.azure_container_name = azure_container_name
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / self.DEFAULT_CACHE_DIR
        self.input_shape = input_shape
        self._model: Optional[keras.Model] = None


    def _load_from_azure(self) -> Path:
        """Download model from Azure and save to cache."""
        from azure.core.exceptions import ResourceNotFoundError

        from src.azure_storage import AzureStorageManager

        blob_name = self.azure_blob_name or "model/best_model.keras"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use blob filename for cache if it has a known extension
        ext = Path(blob_name).suffix or ".keras"
        cache_name = f"cached_model{ext}"
        cache_path = self.cache_dir / cache_name

        try:
            azure_manager = AzureStorageManager(
                container_name=self.azure_container_name,
                create_container_if_not_exists=False,
            )
            azure_manager.download_blob(blob_name, str(cache_path))
        except ResourceNotFoundError:
            raise FileNotFoundError(
                f"Model blob not found in Azure: '{blob_name}' (container: {self.azure_container_name}). "
                "Check AZURE_MODEL_BLOB_NAME and AZURE_CONTAINER_NAME and ensure the file exists."
            ) from None
        return cache_path

    def load_model(self) -> keras.Model:
        """
        Load the segmentation model.

        Uses local path if available, else cached model, else downloads from Azure.
        """
        if self._model is not None:
            return self._model

        has_azure_creds = bool(
            os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            or (
                os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
                and os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            )
        )
        if self.azure_blob_name and has_azure_creds:
            local_path = self._load_from_azure()
        else:
            raise FileNotFoundError(
                "No model found. Provide model_path with an existing file, "
                "or set azure_blob_name and Azure credentials (AZURE_STORAGE_CONNECTION_STRING "
                "or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY) to download from Azure."
            )

        custom_objects = {
            "combined_loss": combined_loss,
            "DiceCoefficient": DiceCoefficient,
            "IoUCoefficient": IoUCoefficient,
        }
        self._model = keras.models.load_model(
            str(local_path),
            custom_objects=custom_objects,
            safe_mode=False,
        )
        return self._model

    def predict_from_image(
        self,
        image: Union[np.ndarray, Path, str, bytes],
        target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> np.ndarray:
        """
        Run inference on an image and return the predicted mask.

        Args:
            image: Input image as numpy array, file path, or bytes.
            target_size: (height, width) for preprocessing (must match model input).

        Returns:
            Predicted mask as numpy array (H, W) with values 0-7 (class indices).
        """
        img = _load_image_from_source(image)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        preprocessed = preprocess_image(img, target_size=target_size)
        model = self.load_model()
        predictions = model.predict(preprocessed, verbose=0)
        mask = np.argmax(predictions[0], axis=-1).astype(np.uint8)
        return mask

    def predict_to_colored_mask(
        self,
        image: Union[np.ndarray, Path, str, bytes],
        target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> np.ndarray:
        """
        Run inference and return the mask as a colored RGB image for visualization.

        Returns:
           Colored mask (H, W, 3), uint8, suitable for display.
        """
        mask = self.predict_from_image(image, target_size=target_size)
        return mask_to_colored(mask)


def predict_from_image(
    image: Union[np.ndarray, Path, str, bytes],
    model_path: Optional[str] = None,
    azure_blob_name: Optional[str] = None,
    target_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """
    Convenience function to predict mask from an image.

    Args:
        image: Input image.
        azure_blob_name: Azure blob name (optional, used if model_path not found).
        target_size: Resize target for preprocessing.

    Returns:
        Predicted mask (H, W) with class indices 0-7.
    """
    predictor = SegmentationPredictor(
        azure_blob_name=azure_blob_name,
    )
    return predictor.predict_from_image(image, target_size=target_size)
