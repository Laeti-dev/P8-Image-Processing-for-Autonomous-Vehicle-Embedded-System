"""
MLflow tracking utilities for experiment management.

This module provides utilities to track experiments, log parameters, metrics,
and models using MLflow during training.
"""

import os
import mlflow
import mlflow.tensorflow
from pathlib import Path
import json
from datetime import datetime
import tensorflow as tf


class MLflowTracker:
    """
    MLflow experiment tracker for semantic segmentation training.

    This class handles experiment tracking, parameter logging, metric logging,
    and model versioning using MLflow.
    """

    def __init__(
        self,
        experiment_name="unet_cityscapes_segmentation",
        tracking_uri=None,
        artifact_location=None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (optional)
            artifact_location: Location to store artifacts (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.run_id = None
        self.experiment_id = None

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            print(f"✓ Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            print(f"✓ Using existing MLflow experiment: {experiment_name}")

    def start_run(self, run_name=None):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"training_run_{timestamp}"

        mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
        self.run_id = mlflow.active_run().info.run_id
        print(f"✓ Started MLflow run: {run_name} (ID: {self.run_id})")

        return self.run_id

    def log_params(self, params):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        # Flatten nested dictionaries for better MLflow organization
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            mlflow.log_param(key, value)

        print(f"✓ Logged {len(flat_params)} parameters to MLflow")

    def log_model_config(self, model_config):
        """
        Log model configuration details.

        Args:
            model_config: Dictionary containing model configuration
        """
        # Log model architecture details
        if 'input_shape' in model_config:
            input_shape = model_config['input_shape']
            mlflow.log_param("model.input_height", input_shape[0])
            mlflow.log_param("model.input_width", input_shape[1])
            mlflow.log_param("model.input_channels", input_shape[2])

        if 'n_classes' in model_config:
            mlflow.log_param("model.num_classes", model_config['n_classes'])

        if 'use_small' in model_config:
            mlflow.log_param("model.use_small_architecture", model_config['use_small'])

        if 'learning_rate' in model_config:
            mlflow.log_param("model.learning_rate", model_config['learning_rate'])

        # Log full config as artifact
        config_path = "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # Clean up temporary file

    def log_training_config(self, training_config):
        """
        Log training configuration details.

        Args:
            training_config: Dictionary containing training configuration
        """
        # Log training hyperparameters
        if 'batch_size' in training_config:
            mlflow.log_param("training.batch_size", training_config['batch_size'])

        if 'epochs' in training_config:
            mlflow.log_param("training.max_epochs", training_config['epochs'])

        if 'validation_split' in training_config:
            mlflow.log_param("training.validation_split", training_config['validation_split'])

        # Log augmentation config if present
        if 'augmentation' in training_config:
            aug_config = training_config['augmentation']
            mlflow.log_param("augmentation.type", aug_config.get('type', 'none'))

        # Log full config as artifact
        config_path = "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name="unet_model"):
        """
        Log TensorFlow/Keras model to MLflow.

        Args:
            model: Keras model to log
            model_name: Name for the model artifact
        """
        # Log model summary as text artifact
        summary_path = "model_summary.txt"
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)

        # Log model as MLflow model
        mlflow.tensorflow.log_model(model, model_name)
        print(f"✓ Logged model '{model_name}' to MLflow")

    def log_model_file(self, model_path, model_name="best_model"):
        """
        Log saved model file to MLflow.

        Args:
            model_path: Path to the saved model file
            model_name: Name for the model artifact
        """
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, model_name)
            print(f"✓ Logged model file '{model_path}' to MLflow as '{model_name}'")

    def log_visualization(self, image_path, artifact_name):
        """
        Log visualization image to MLflow.

        Args:
            image_path: Path to the image file
            artifact_name: Name for the artifact
        """
        if os.path.exists(image_path):
            mlflow.log_artifact(image_path, artifact_name)

    def log_training_history(self, history, plot_path=None):
        """
        Log training history and optionally save plots.

        Args:
            history: Keras training history object
            plot_path: Path to save training history plot
        """
        # Log final metrics
        if hasattr(history, 'history'):
            final_epoch = len(history.history['loss']) - 1

            for metric_name in history.history.keys():
                final_value = history.history[metric_name][final_epoch]
                mlflow.log_metric(f"final_{metric_name}", final_value)

        # Save history as JSON artifact
        history_path = "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        mlflow.log_artifact(history_path)
        os.remove(history_path)

        # Log plot if provided
        if plot_path and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, "training_history_plot")

    def log_system_info(self):
        """Log system and environment information."""
        import platform
        import tensorflow as tf

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
        }

        # Log GPU info if available
        if system_info["gpu_available"]:
            gpus = tf.config.list_physical_devices('GPU')
            system_info["gpu_count"] = len(gpus)
            system_info["gpu_names"] = [gpu.name for gpu in gpus]

        info_path = "system_info.json"
        with open(info_path, 'w') as f:
            json.dump(system_info, f, indent=2)
        mlflow.log_artifact(info_path)
        os.remove(info_path)

        print("✓ Logged system information to MLflow")

    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            print("✓ Ended MLflow run")
            self.run_id = None

    def _flatten_dict(self, d, prefix='', sep='.'):
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            prefix: Current prefix for keys
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)


class MLflowCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to log metrics to MLflow during training.
    """

    def __init__(self, tracker, log_frequency=1):
        """
        Initialize MLflow callback.

        Args:
            tracker: MLflowTracker instance
            log_frequency: Log metrics every N epochs
        """
        super().__init__()
        self.tracker = tracker
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs and (epoch + 1) % self.log_frequency == 0:
            # Add epoch number to metrics
            epoch_metrics = {f"epoch_{k}": v for k, v in logs.items()}
            epoch_metrics["epoch"] = epoch + 1

            self.tracker.log_metrics(epoch_metrics, step=epoch + 1)

    def on_train_end(self, logs=None):
        """Log final training metrics."""
        if logs:
            final_metrics = {f"final_{k}": v for k, v in logs.items()}
            self.tracker.log_metrics(final_metrics)


def setup_mlflow_tracking(
    experiment_name="unet_cityscapes_segmentation",
    tracking_uri=None,
    artifact_location=None
):
    """
    Convenience function to set up MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (optional)
        artifact_location: Location to store artifacts (optional)

    Returns:
        MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location
    )
