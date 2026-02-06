"""
Custom Keras callbacks for training visualization and monitoring.

This module provides callbacks to:
- Visualize predictions during training
- Visualize encoder feature maps at key training steps (first/last epoch, every N epochs)
- Track training time per epoch and total training time
- Log experiments to MLflow

Example usage:
    from src.callbacks import PredictionVisualizationFromGenerator, TrainingTimeCallback
    from src.mlflow_tracking import MLflowCallback

    # Create callbacks
    viz_callback = PredictionVisualizationFromGenerator(
        validation_generator=val_gen,
        output_dir="outputs/training_visualizations",
        num_samples=4,
        frequency=5  # Visualize every 5 epochs
    )

    time_callback = TrainingTimeCallback()

    mlflow_callback = MLflowCallback(
        tracker=mlflow_tracker,
        log_frequency=1  # Log metrics every epoch
    )

    # Add to callbacks list
    callbacks = [viz_callback, time_callback, mlflow_callback, ...]

    # Train model
    history = model.fit(..., callbacks=callbacks)

    # Access training time metrics
    print(f"Total training time: {time_callback.total_time:.2f} seconds")
    print(f"Average epoch time: {np.mean(time_callback.epoch_times):.2f} seconds")
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from src.feature_maps import get_feature_map_model
from src.utils import CATEGORY_COLORS, CATEGORY_NAMES, mask_to_colored


class PredictionVisualizationCallback(keras.callbacks.Callback):
    """
    Callback to visualize model predictions during training.

    This callback periodically saves visualizations showing:
    - Original images
    - Ground truth masks
    - Predicted masks
    - Overlay of prediction on original image
    """

    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        output_dir: str = "outputs/training_visualizations",
        num_samples: int = 4,
        frequency: int = 5,
        save_format: str = "png"
    ):
        """
        Initialize the visualization callback.

        Args:
            validation_data: Tuple of (images, masks) from validation set
            output_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            frequency: Visualize every N epochs
            save_format: Image format ('png', 'jpg', etc.)
        """
        super().__init__()
        self.validation_images, self.validation_masks = validation_data
        self.output_dir = Path(output_dir)
        self.num_samples = min(num_samples, len(self.validation_images))
        self.frequency = frequency
        self.save_format = save_format

        # Create output directory (clean previous visualizations to avoid stale files)
        if self.output_dir.exists():
            for old_file in self.output_dir.glob(f"epoch_*.{self.save_format}"):
                old_file.unlink()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Select samples to visualize
        indices = np.linspace(0, len(self.validation_images) - 1, self.num_samples, dtype=int)
        self.sample_images = self.validation_images[indices]
        self.sample_masks = self.validation_masks[indices]

        print(f"Visualization callback initialized:")
        print(f"  - Output directory: {self.output_dir} (cleaned)")
        print(f"  - Number of samples: {self.num_samples}")
        print(f"  - Frequency: every {frequency} epochs")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        # Only visualize at specified frequency
        if (epoch + 1) % self.frequency != 0:
            return

        # Make predictions
        predictions = self.model.predict(
            self.sample_images,
            verbose=0,
            batch_size=len(self.sample_images)
        )

        # Convert predictions to class indices
        predicted_masks = np.argmax(predictions, axis=-1)

        # Create visualization
        self._save_visualization(
            epoch=epoch + 1,
            images=self.sample_images,
            true_masks=self.sample_masks,
            pred_masks=predicted_masks
        )

    def _save_visualization(
        self,
        epoch: int,
        images: np.ndarray,
        true_masks: np.ndarray,
        pred_masks: np.ndarray
    ):
        """
        Save visualization of predictions.

        Args:
            epoch: Current epoch number
            images: Original images
            true_masks: Ground truth masks
            pred_masks: Predicted masks
        """
        n_samples = len(images)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

        # Handle case where there's only one sample
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Denormalize image if needed
            img = images[i].copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            true_mask = true_masks[i].astype(np.uint8)
            pred_mask = pred_masks[i].astype(np.uint8)

            # Convert masks to colored
            true_colored = mask_to_colored(true_mask)
            pred_colored = mask_to_colored(pred_mask)

            # Create overlay (prediction on image)
            overlay = img.copy()
            overlay = (overlay * 0.6 + pred_colored * 0.4).astype(np.uint8)

            # Plot original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Sample {i+1}: Original Image")
            axes[i, 0].axis('off')

            # Plot ground truth
            axes[i, 1].imshow(true_colored)
            axes[i, 1].set_title(f"Sample {i+1}: Ground Truth")
            axes[i, 1].axis('off')

            # Plot prediction
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title(f"Sample {i+1}: Prediction")
            axes[i, 2].axis('off')

            # Plot overlay
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f"Sample {i+1}: Overlay")
            axes[i, 3].axis('off')

        plt.suptitle(f"Epoch {epoch} - Predictions", fontsize=16, y=0.995)
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"epoch_{epoch:03d}.{self.save_format}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization: {output_path}")


class FeatureMapVisualizationCallback(keras.callbacks.Callback):
    """
    Callback to visualize feature maps at key steps during training.

    Builds a sub-model from the current training model to extract intermediate
    layer activations (encoder stages) and saves visualizations at configurable
    epochs: first epoch, every N epochs, and optionally at the last epoch.

    Compatible with U-Net MobileNetV2, DeepLabV3 (MobileNetV2/ResNet50), and
    U-Net; uses defaults for MobileNet-based models and falls back to first
    Conv2D/activation layers for plain U-Net.
    """

    def __init__(
        self,
        validation_data: Optional[Union[Tuple[np.ndarray, np.ndarray], object]] = None,
        validation_generator: Optional[object] = None,
        output_dir: str = "outputs/feature_maps",
        frequency: int = 5,
        always_first_epoch: bool = True,
        always_last_epoch: bool = True,
        layer_names: Optional[List[str]] = None,
        sample_idx: int = 0,
        max_channels: int = 16,
        figsize: Tuple[float, float] = (16, 12),
        cmap: str = "viridis",
        save_format: str = "png",
    ):
        """
        Initialize the feature map visualization callback.

        Args:
            validation_data: Optional tuple (images, masks) to use for visualization.
                            Only images are used; one sample is taken (sample_idx).
            validation_generator: Optional data generator; first batch is used to get
                                 a fixed batch of images. Ignored if validation_data
                                 is provided.
            output_dir: Directory to save feature map figures.
            frequency: Visualize every N epochs (in addition to first/last if enabled).
            always_first_epoch: If True, always visualize at epoch 1.
            always_last_epoch: If True, visualize again at the end of training.
            layer_names: Optional list of layer names to extract. If None, uses
                         defaults for MobileNet-based models or first conv layers.
            sample_idx: Index of the sample in the batch to visualize.
            max_channels: Maximum number of channels to plot per layer.
            figsize: Figure size (width, height).
            cmap: Matplotlib colormap for feature maps (e.g. 'viridis', 'gray').
            save_format: Image format ('png', 'jpg', etc.).
        """
        super().__init__()
        if validation_data is not None:
            self._images = validation_data[0]
        elif validation_generator is not None:
            batch = validation_generator[0]
            self._images = batch[0] if isinstance(batch, (list, tuple)) else batch
        else:
            raise ValueError(
                "Provide either validation_data (tuple of arrays) or validation_generator."
            )
        self.output_dir = Path(output_dir)
        self.frequency = frequency
        self.always_first_epoch = always_first_epoch
        self.always_last_epoch = always_last_epoch
        self.layer_names = layer_names
        self.sample_idx = sample_idx
        self.max_channels = max_channels
        self.figsize = figsize
        self.cmap = cmap
        self.save_format = save_format
        self._last_epoch: Optional[int] = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(
            "Feature map visualization callback initialized:\n"
            f"  - Output directory: {self.output_dir}\n"
            f"  - Frequency: every {frequency} epoch(s)\n"
            f"  - First epoch: {always_first_epoch}, Last epoch: {always_last_epoch}"
        )

    def _should_visualize(self, epoch: int) -> bool:
        """Return True if we should visualize at this epoch (1-based)."""
        step = epoch + 1
        if self.always_first_epoch and step == 1:
            return True
        if self.frequency > 0 and step % self.frequency == 0:
            return True
        return False

    def _visualize_feature_maps(self, epoch_label: str) -> None:
        """Build feature model from current training model and save feature map plot."""
        try:
            feature_model, names_used = get_feature_map_model(
                self.model, layer_names=self.layer_names
            )
        except ValueError as e:
            print(f"  Feature map visualization skipped (epoch {epoch_label}): {e}")
            return

        batch = self._images
        if isinstance(batch, tf.Tensor):
            batch = batch.numpy()
        if batch.ndim == 3:
            batch = batch[np.newaxis, ...]
        batch = batch[: self.sample_idx + 1]

        activations = feature_model.predict(batch, verbose=0)
        if not isinstance(activations, list):
            activations = [activations]

        n_layers = len(activations)
        n_cols = self.max_channels
        n_rows = n_layers
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (name, act) in enumerate(zip(names_used, activations)):
            if act.ndim == 3:
                act = act[np.newaxis, ...]
            feat = act[self.sample_idx]
            n_ch = min(feat.shape[-1], n_cols)
            for ch in range(n_cols):
                ax = axes[i, ch]
                if ch < n_ch:
                    ax.imshow(feat[:, :, ch], cmap=self.cmap, aspect="auto")
                    ax.set_title(f"{name[:20]} ch{ch}", fontsize=7)
                ax.axis("off")
        fig.suptitle(
            f"Feature maps (sample {self.sample_idx}) — {epoch_label}",
            fontsize=12,
        )
        plt.tight_layout()
        save_path = self.output_dir / f"feature_maps_{epoch_label}.{self.save_format}"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved feature maps: {save_path}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Visualize feature maps at configured epoch steps."""
        self._last_epoch = epoch + 1
        if not self._should_visualize(epoch):
            return
        epoch_label = f"epoch_{epoch + 1:03d}"
        self._visualize_feature_maps(epoch_label)

    def on_train_end(self, logs: Optional[dict] = None):
        """Optionally visualize feature maps after the last epoch."""
        if not self.always_last_epoch or self._last_epoch is None:
            return
        self._visualize_feature_maps("epoch_last")


class PredictionVisualizationFromGenerator(keras.callbacks.Callback):
    """
    Callback to visualize predictions using a data generator.

    This version works with data generators by extracting a batch
    from the validation generator.
    """

    def __init__(
        self,
        validation_generator,
        output_dir: str = "outputs/training_visualizations",
        num_samples: int = 4,
        frequency: int = 5,
        save_format: str = "png"
    ):
        """
        Initialize the visualization callback.

        Args:
            validation_generator: Validation data generator
            output_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            frequency: Visualize every N epochs
            save_format: Image format ('png', 'jpg', etc.)
        """
        super().__init__()
        self.validation_generator = validation_generator
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.frequency = frequency
        self.save_format = save_format

        # Create output directory (clean previous visualizations to avoid stale files)
        if self.output_dir.exists():
            for old_file in self.output_dir.glob(f"epoch_*.{self.save_format}"):
                old_file.unlink()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get a batch from the generator
        self.sample_images, self.sample_masks = validation_generator[0]
        self.num_samples = min(self.num_samples, len(self.sample_images))

        # Select samples
        indices = np.linspace(0, len(self.sample_images) - 1, self.num_samples, dtype=int)
        self.sample_images = self.sample_images[indices]
        self.sample_masks = self.sample_masks[indices]

        print(f"Visualization callback initialized (from generator):")
        print(f"  - Output directory: {self.output_dir} (cleaned)")
        print(f"  - Number of samples: {self.num_samples}")
        print(f"  - Frequency: every {frequency} epochs")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        # Only visualize at specified frequency
        if (epoch + 1) % self.frequency != 0:
            return

        # Make predictions
        predictions = self.model.predict(
            self.sample_images,
            verbose=0,
            batch_size=len(self.sample_images)
        )

        # Convert predictions to class indices
        predicted_masks = np.argmax(predictions, axis=-1)

        # Create visualization
        self._save_visualization(
            epoch=epoch + 1,
            images=self.sample_images,
            true_masks=self.sample_masks,
            pred_masks=predicted_masks
        )

    def _save_visualization(
        self,
        epoch: int,
        images: np.ndarray,
        true_masks: np.ndarray,
        pred_masks: np.ndarray
    ):
        """
        Save visualization of predictions.

        Args:
            epoch: Current epoch number
            images: Original images
            true_masks: Ground truth masks
            pred_masks: Predicted masks
        """
        n_samples = len(images)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

        # Handle case where there's only one sample
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Denormalize image if needed
            img = images[i].copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            true_mask = true_masks[i].astype(np.uint8)
            pred_mask = pred_masks[i].astype(np.uint8)

            # Convert masks to colored
            true_colored = mask_to_colored(true_mask)
            pred_colored = mask_to_colored(pred_mask)

            # Create overlay (prediction on image)
            overlay = img.copy()
            overlay = (overlay * 0.6 + pred_colored * 0.4).astype(np.uint8)

            # Plot original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Sample {i+1}: Original Image")
            axes[i, 0].axis('off')

            # Plot ground truth
            axes[i, 1].imshow(true_colored)
            axes[i, 1].set_title(f"Sample {i+1}: Ground Truth")
            axes[i, 1].axis('off')

            # Plot prediction
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title(f"Sample {i+1}: Prediction")
            axes[i, 2].axis('off')

            # Plot overlay
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f"Sample {i+1}: Overlay")
            axes[i, 3].axis('off')

        plt.suptitle(f"Epoch {epoch} - Predictions", fontsize=16, y=0.995)
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"epoch_{epoch:03d}.{self.save_format}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization: {output_path}")


class TrainingTimeCallback(keras.callbacks.Callback):
    """
    Callback to track training time per epoch and total training time.

    This callback measures:
    - Time per epoch (in seconds)
    - Total training time (in seconds)
    - Average time per epoch

    The times are added to the training history logs.
    """

    def __init__(self):
        """
        Initialize the training time callback.
        """
        super().__init__()
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.total_time = 0.0

    def on_train_begin(self, logs: Optional[dict] = None):
        """
        Called at the beginning of training.

        Args:
            logs: Dictionary of logs
        """
        self.training_start_time = time.time()
        self.epoch_times = []
        self.total_time = 0.0
        print("Training time tracking started")

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs
        """
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (will be modified to include time metrics)
        """
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            self.total_time += epoch_time

            # Add time metrics to logs
            if logs is not None:
                logs['epoch_time'] = epoch_time
                logs['total_training_time'] = self.total_time
                logs['avg_epoch_time'] = np.mean(self.epoch_times)

    def on_train_end(self, logs: Optional[dict] = None):
        """
        Called at the end of training.

        Args:
            logs: Dictionary of logs
        """
        if self.training_start_time is not None:
            total_training_time = time.time() - self.training_start_time
            avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0.0

            if logs is not None:
                logs['total_training_time'] = total_training_time
                logs['avg_epoch_time'] = avg_epoch_time

            print(f"\nTraining completed!")
            print(f"  Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
            print(f"  Average time per epoch: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)")
            print(f"  Number of epochs: {len(self.epoch_times)}")


class AzureUploadCallback(keras.callbacks.Callback):
    """
    Callback to upload training outputs to Azure Blob Storage.

    This callback uploads:
    - Model checkpoints (when model is saved)
    - Training visualizations
    - TensorBoard logs

    Example usage:
        from src.callbacks import AzureUploadCallback
        from src.azure_storage import AzureStorageManager

        azure_manager = AzureStorageManager(container_name="training-outputs")
        azure_callback = AzureUploadCallback(
            azure_manager=azure_manager,
            model_path="models/unet_cityscapes.h5",
            output_dir="outputs/training_visualizations",
            logs_dir="logs",
            run_name="experiment_001"
        )

        callbacks = [azure_callback, ...]
        model.fit(..., callbacks=callbacks)
    """

    def __init__(
        self,
        azure_manager,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        upload_frequency: int = 1,
        upload_on_checkpoint: bool = True,
        upload_on_train_end: bool = True
    ):
        """
        Initialize Azure upload callback.

        Args:
            azure_manager: AzureStorageManager instance
            model_path: Path to the model file to upload
            output_dir: Directory containing visualizations
            logs_dir: Directory containing TensorBoard logs
            run_name: Name for this training run
            upload_frequency: Upload every N epochs (default: 1 = every epoch)
            upload_on_checkpoint: Upload when model checkpoint is saved
            upload_on_train_end: Upload all outputs at end of training
        """
        super().__init__()
        self.azure_manager = azure_manager
        self.model_path = model_path
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.run_name = run_name
        self.upload_frequency = upload_frequency
        self.upload_on_checkpoint = upload_on_checkpoint
        self.upload_on_train_end = upload_on_train_end
        self.last_uploaded_epoch = -1

        print(f"Azure Upload Callback initialized:")
        print(f"  - Run name: {run_name}")
        print(f"  - Upload frequency: every {upload_frequency} epochs")
        print(f"  - Upload on checkpoint: {upload_on_checkpoint}")
        print(f"  - Upload on train end: {upload_on_train_end}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        # Upload at specified frequency
        if (epoch + 1) % self.upload_frequency == 0:
            try:
                self._upload_outputs(epoch + 1)
                self.last_uploaded_epoch = epoch + 1
            except Exception as e:
                print(f"Warning: Failed to upload to Azure at epoch {epoch + 1}: {e}")

    def on_train_end(self, logs: Optional[dict] = None):
        """
        Called at the end of training.

        Args:
            logs: Dictionary of logs
        """
        if self.upload_on_train_end:
            try:
                print("\nUploading final training outputs to Azure...")
                self._upload_outputs("final")
                print("Upload completed successfully!")
            except Exception as e:
                print(f"Warning: Failed to upload final outputs to Azure: {e}")

    def _upload_outputs(self, epoch_label):
        """
        Upload training outputs to Azure.

        Args:
            epoch_label: Label for this upload (epoch number or "final")
        """
        results = self.azure_manager.upload_training_outputs(
            model_path=self.model_path,
            output_dir=self.output_dir,
            logs_dir=self.logs_dir,
            run_name=self.run_name
        )

        if results["model_url"]:
            print(f"  Model uploaded: {results['model_url']}")
        if results["outputs_urls"]:
            print(f"  Outputs uploaded: {len(results['outputs_urls'])} files")
        if results["logs_urls"]:
            print(f"  Logs uploaded: {len(results['logs_urls'])} files")


class AzureModelCheckpoint(keras.callbacks.Callback):
    """
    ModelCheckpoint callback that saves models directly to Azure Blob Storage
    without saving to local disk.

    This callback monitors a metric and saves the model to Azure when the metric
    improves, without creating a local file.

    Example usage:
        from src.callbacks import AzureModelCheckpoint
        from src.azure_storage import AzureStorageManager

        azure_manager = AzureStorageManager(container_name="training-outputs")
        azure_checkpoint = AzureModelCheckpoint(
            azure_manager=azure_manager,
            blob_name="model/best_model.h5",
            monitor='val_iou_coefficient',
            save_best_only=True,
            mode='max'
        )

        callbacks = [azure_checkpoint, ...]
        model.fit(..., callbacks=callbacks)
    """

    def __init__(
        self,
        azure_manager,
        blob_name: str,
        monitor: str = 'val_loss',
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = 'auto',
        save_freq: str = 'epoch',
        run_name: Optional[str] = None
    ):
        """
        Initialize Azure Model Checkpoint callback.

        Args:
            azure_manager: AzureStorageManager instance
            blob_name: Name of the blob in Azure (e.g., "model/best_model.h5")
            monitor: Metric to monitor (default: 'val_loss')
            verbose: Verbosity mode (0 or 1)
            save_best_only: If True, only save when monitor improves
            mode: One of {'auto', 'min', 'max'}. Determines if improvement is
                  lower (min) or higher (max) monitor value
            save_freq: 'epoch' or integer. When to save
            run_name: Optional run name to prefix blob_name
        """
        super().__init__()
        self.azure_manager = azure_manager
        self.blob_name = blob_name
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.run_name = run_name

        # Determine mode
        if mode == 'auto':
            if 'acc' in self.monitor or 'iou' in self.monitor or 'dice' in self.monitor:
                mode = 'max'
            else:
                mode = 'min'

        self.mode = mode
        self.best = None
        self.best_epoch = None

        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b)
            self.best = np.Inf
        elif self.mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b)
            self.best = -np.Inf

        # Construct full blob name
        if self.run_name:
            self.full_blob_name = f"{self.run_name}/{self.blob_name}"
        else:
            self.full_blob_name = self.blob_name

        print(f"Azure Model Checkpoint initialized:")
        print(f"  - Blob name: {self.full_blob_name}")
        print(f"  - Monitor: {self.monitor}")
        print(f"  - Mode: {self.mode}")
        print(f"  - Save best only: {self.save_best_only}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        logs = logs or {}

        # Check if we should save
        if self.save_freq == 'epoch':
            should_save = True
        else:
            # Integer frequency - not implemented for simplicity
            should_save = True

        if should_save:
            # Get current monitor value
            current = logs.get(self.monitor)
            if current is None:
                if self.verbose > 0:
                    print(f"Warning: {self.monitor} not found in logs. Skipping save.")
                return

            # Check if we should save
            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.best_epoch = epoch
                    self._save_model(epoch, logs)
                else:
                    if self.verbose > 0:
                        print(f"Epoch {epoch + 1}: {self.monitor} did not improve "
                              f"from {self.best:.4f}. Skipping save.")
            else:
                self._save_model(epoch, logs)

    def _save_model(self, epoch: int, logs: Optional[dict] = None):
        """
        Save model directly to Azure.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics
        """
        try:
            import io
            import tempfile

            # Save model to temporary bytes buffer
            buffer = io.BytesIO()

            # Keras requires a file path, so we use a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_path = tmp_file.name

            # Save model to temporary file
            # Note: self.model is set by Keras when callback is attached
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Model not available. Make sure callback is attached to a model.")
            self.model.save(tmp_path)

            # Read the file and upload to Azure
            with open(tmp_path, 'rb') as f:
                model_data = f.read()

            # Upload to Azure
            url = self.azure_manager.upload_from_memory(
                data=model_data,
                blob_name=self.full_blob_name,
                overwrite=True
            )

            # Clean up temporary file
            import os
            os.unlink(tmp_path)

            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Model saved to Azure: {url}")
                if self.save_best_only:
                    print(f"  {self.monitor} improved to {self.best:.4f}")
        except Exception as e:
            print(f"Warning: Failed to save model to Azure: {e}")


# MLflow callbacks
try:
    from src.mlflow_tracking import MLflowTracker, MLflowCallback as BaseMLflowCallback

    class MLflowVisualizationCallback(keras.callbacks.Callback):
        """
        Callback to log training visualizations to MLflow.

        This callback logs prediction visualizations as artifacts to MLflow
        whenever they are created by other visualization callbacks.
        """

        def __init__(self, mlflow_tracker, visualization_dir="outputs/training_visualizations"):
            """
            Initialize MLflow visualization callback.

            Args:
                mlflow_tracker: MLflowTracker instance
                visualization_dir: Directory containing visualizations
            """
            super().__init__()
            self.mlflow_tracker = mlflow_tracker
            self.visualization_dir = Path(visualization_dir)

        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
            """
            Log visualizations at the end of each epoch if they exist.

            Args:
                epoch: Current epoch number
                logs: Dictionary of metrics
            """
            # Check if visualization exists for this epoch
            viz_pattern = f"epoch_{(epoch + 1):03d}.png"
            viz_path = self.visualization_dir / viz_pattern

            if viz_path.exists():
                self.mlflow_tracker.log_visualization(
                    str(viz_path),
                    f"visualizations/{viz_pattern}"
                )

        def on_train_end(self, logs: Optional[dict] = None):
            """Log final visualizations."""
            # Log all visualizations in the directory
            if self.visualization_dir.exists():
                for viz_file in self.visualization_dir.glob("*.png"):
                    artifact_name = f"final_visualizations/{viz_file.name}"
                    self.mlflow_tracker.log_visualization(str(viz_file), artifact_name)


    class MLflowModelCallback(keras.callbacks.Callback):
        """
        Callback to log model artifacts to MLflow.

        This callback logs the trained model and model checkpoints to MLflow.
        """

        def __init__(
            self,
            mlflow_tracker,
            save_model_at_end=True,
            save_model_frequency=None,
            model_name="unet_model"
        ):
            """
            Initialize MLflow model callback.

            Args:
                mlflow_tracker: MLflowTracker instance
                save_model_at_end: Whether to save model at training end
                save_model_frequency: Save model every N epochs (None = never)
                model_name: Name for the model artifact
            """
            super().__init__()
            self.mlflow_tracker = mlflow_tracker
            self.save_model_at_end = save_model_at_end
            self.save_model_frequency = save_model_frequency
            self.model_name = model_name

        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
            """Save model at specified frequency."""
            if self.save_model_frequency and (epoch + 1) % self.save_model_frequency == 0:
                self.mlflow_tracker.log_model(self.model, f"{self.model_name}_epoch_{epoch + 1}")

        def on_train_end(self, logs: Optional[dict] = None):
            """Save final model."""
            if self.save_model_at_end:
                self.mlflow_tracker.log_model(self.model, f"{self.model_name}_final")


    # Re-export the base MLflowCallback for convenience
    MLflowCallback = BaseMLflowCallback

except ImportError:
    # MLflow not available, define dummy callbacks
    class MLflowCallback(keras.callbacks.Callback):
        """Dummy MLflow callback when MLflow is not available."""
        def __init__(self, *args, **kwargs):
            print("Warning: MLflow not available. Install with: pip install mlflow")

    class MLflowVisualizationCallback(keras.callbacks.Callback):
        """Dummy visualization callback when MLflow is not available."""
        def __init__(self, *args, **kwargs):
            print("Warning: MLflow not available. Install with: pip install mlflow")

    class MLflowModelCallback(keras.callbacks.Callback):
        """Dummy model callback when MLflow is not available."""
        def __init__(self, *args, **kwargs):
            print("Warning: MLflow not available. Install with: pip install mlflow")
