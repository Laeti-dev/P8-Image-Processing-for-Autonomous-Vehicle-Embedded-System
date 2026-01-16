"""
Custom metrics and loss functions for semantic segmentation.

This module provides Dice coefficient, IoU (Intersection over Union),
and combined loss functions for multi-class segmentation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Optional


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Calculate Dice coefficient for multi-class segmentation.

    Args:
        y_true: Ground truth masks (batch, height, width) or one-hot encoded
        y_pred: Predicted masks (batch, height, width, n_classes) with softmax
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient (scalar)
    """
    # Convert y_true to one-hot if needed (create new variable to avoid modifying input)
    if len(y_true.shape) == 3:
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
    else:
        y_true_one_hot = y_true

    # Flatten tensors
    y_true_flat = tf.reshape(y_true_one_hot, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

    return dice


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Dice loss (1 - Dice coefficient).

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor

    Returns:
        Dice loss (scalar)
    """
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Calculate Intersection over Union (IoU) for multi-class segmentation.

    Args:
        y_true: Ground truth masks (batch, height, width) or one-hot encoded
        y_pred: Predicted masks (batch, height, width, n_classes) with softmax
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU coefficient (scalar)
    """
    # Convert y_true to one-hot if needed (create new variable to avoid modifying input)
    if len(y_true.shape) == 3:
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
    else:
        y_true_one_hot = y_true

    # Flatten tensors
    y_true_flat = tf.reshape(y_true_one_hot, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def iou_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    IoU loss (1 - IoU coefficient).

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor

    Returns:
        IoU loss (scalar)
    """
    return 1.0 - iou_coefficient(y_true, y_pred, smooth)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    """
    Combined categorical crossentropy and Dice loss.

    Args:
        y_true: Ground truth masks (batch, height, width) with integer labels
        y_pred: Predicted masks (batch, height, width, n_classes) with softmax
        alpha: Weight for categorical crossentropy (1-alpha for Dice loss)

    Returns:
        Combined loss (scalar)
    """
    # Ensure y_true is integer type
    y_true = tf.cast(y_true, tf.int32)

    # Sparse categorical crossentropy (accepts integer labels directly)
    # from_logits=False because y_pred already has softmax activation
    ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    ce_loss = tf.reduce_mean(ce_loss)

    # Dice loss: calculate directly to avoid gradient issues
    # Convert y_true to one-hot for Dice calculation
    n_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=n_classes)

    # Flatten tensors for Dice calculation
    y_true_flat = tf.reshape(y_true_one_hot, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Calculate Dice coefficient
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)
    dice_loss_val = 1.0 - dice

    # Combined loss (both are scalars)
    return alpha * ce_loss + (1 - alpha) * dice_loss_val


def mean_iou_per_class(y_true: tf.Tensor, y_pred: tf.Tensor, n_classes: int = 8) -> tf.Tensor:
    """
    Calculate mean IoU per class.

    Args:
        y_true: Ground truth masks (batch, height, width) with integer labels
        y_pred: Predicted masks (batch, height, width, n_classes) with softmax
        n_classes: Number of classes

    Returns:
        Mean IoU across all classes (scalar)
    """
    # Convert predictions to class indices
    y_pred_classes = tf.argmax(y_pred, axis=-1)

    # Convert to one-hot
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
    y_pred_one_hot = tf.one_hot(tf.cast(y_pred_classes, tf.int32), depth=n_classes)

    # Calculate IoU for each class
    ious = []
    for i in range(n_classes):
        y_true_class = y_true_one_hot[..., i]
        y_pred_class = y_pred_one_hot[..., i]

        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection

        iou = intersection / (union + 1e-6)
        ious.append(iou)

    # Return mean IoU
    return tf.reduce_mean(ious)


class DiceCoefficient(keras.metrics.Metric):
    """Keras metric wrapper for Dice coefficient."""

    def __init__(self, name="dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice = self.add_weight(name="dice", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        dice_val = dice_coefficient(y_true, y_pred)
        self.dice.assign_add(dice_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice / self.count

    def reset_state(self):
        self.dice.assign(0.0)
        self.count.assign(0.0)


class IoUCoefficient(keras.metrics.Metric):
    """Keras metric wrapper for IoU coefficient."""

    def __init__(self, name="iou_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou_val = iou_coefficient(y_true, y_pred)
        self.iou.assign_add(iou_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.iou / self.count

    def reset_state(self):
        self.iou.assign(0.0)
        self.count.assign(0.0)
