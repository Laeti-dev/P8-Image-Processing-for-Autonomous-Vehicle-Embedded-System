"""
Feature map extraction and visualization for segmentation models.

This module provides utilities to build a sub-model that outputs intermediate
layer activations (feature maps) and to plot them for U-Net, U-Net MobileNetV2,
and DeepLabV3 architectures. Use FeatureMapVisualizationCallback in
src.callbacks to visualize feature maps at key training steps.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras


# Default encoder layer names for models with MobileNetV2 backbone
# (U-Net MobileNet, DeepLabV3). Resolutions: 1/2, 1/4, 1/8, 1/16.
MOBILENET_FEATURE_LAYER_NAMES = [
    "block_1_expand_relu",   # 1/2
    "block_3_expand_relu",   # 1/4
    "block_6_expand_relu",   # 1/8
    "block_13_expand_relu",  # 1/16
]


def get_feature_map_model(
    model: keras.Model,
    layer_names: Optional[List[str]] = None,
) -> Tuple[keras.Model, List[str]]:
    """
    Build a sub-model that returns intermediate layer outputs (feature maps).

    Use this after training or loading your segmentation model. If layer_names
    is None, uses sensible defaults for MobileNetV2-based models (U-Net
    MobileNet, DeepLabV3); otherwise falls back to the first Conv2D-like
    layers found in the model.

    Args:
        model: Trained or loaded Keras model (e.g. from build_deeplabv3,
               build_unet_mobilenet, build_unet).
        layer_names: Optional list of layer names to extract. If None, tries
                     MOBILENET_FEATURE_LAYER_NAMES first, then first 8
                     Conv2D/activation layers.

    Returns:
        feature_model: keras.Model with inputs=model.input and
                      outputs=[layer1.output, layer2.output, ...].
        names_used: List of layer names actually used (for plotting labels).
    """
    if layer_names is not None:
        names_to_try = list(layer_names)
    else:
        names_to_try = list(MOBILENET_FEATURE_LAYER_NAMES)

    outputs = []
    names_used = []

    for name in names_to_try:
        try:
            layer = model.get_layer(name)
            outputs.append(layer.output)
            names_used.append(name)
        except (ValueError, AttributeError):
            continue

    if not outputs:
        # Fallback: use first Conv2D / Activation layers (avoid Input, etc.)
        for layer in model.layers:
            if len(outputs) >= 8:
                break
            name_lower = layer.name.lower()
            if "conv2d" in name_lower or "expand_relu" in name_lower or "activation" in name_lower:
                if "input" not in name_lower and hasattr(layer, "output"):
                    try:
                        outputs.append(layer.output)
                        names_used.append(layer.name)
                    except Exception:
                        continue

    if not outputs:
        raise ValueError(
            "Could not find any layer to extract. Pass layer_names explicitly, "
            "e.g. layer_names=['block_1_expand_relu', 'block_3_expand_relu']."
        )

    feature_model = keras.Model(inputs=model.input, outputs=outputs, name="feature_maps")
    return feature_model, names_used


def plot_feature_maps(
    feature_model: keras.Model,
    batch: Union[np.ndarray, tf.Tensor],
    layer_names: List[str],
    sample_idx: int = 0,
    max_channels: int = 16,
    figsize: Tuple[float, float] = (16, 12),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature maps for one sample: one row per layer, one subplot per channel.

    Args:
        feature_model: Model returned by get_feature_map_model.
        batch: Input batch of shape (batch_size, H, W, C). Only batch[sample_idx]
               is used for visualization.
        layer_names: List of layer names (same order as feature_model outputs).
        sample_idx: Index of the sample in the batch to visualize.
        max_channels: Maximum number of channels to plot per layer.
        figsize: Overall figure size (width, height).
        cmap: Matplotlib colormap for feature maps (e.g. 'viridis', 'gray').
        save_path: If set, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    if isinstance(batch, tf.Tensor):
        batch = batch.numpy()
    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]
    batch = batch[: sample_idx + 1]

    activations = feature_model.predict(batch, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    n_layers = len(activations)
    n_cols = max_channels
    n_rows = n_layers
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (name, act) in enumerate(zip(layer_names, activations)):
        if act.ndim == 3:
            act = act[np.newaxis, ...]
        feat = act[sample_idx]
        n_ch = min(feat.shape[-1], max_channels)
        for ch in range(n_cols):
            ax = axes[i, ch]
            if ch < n_ch:
                ax.imshow(feat[:, :, ch], cmap=cmap, aspect="auto")
                ax.set_title(f"{name[:20]} ch{ch}", fontsize=7)
            ax.axis("off")
    fig.suptitle(f"Feature maps (sample {sample_idx})", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
