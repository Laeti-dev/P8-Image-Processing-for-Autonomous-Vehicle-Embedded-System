"""
U-Net architecture for semantic segmentation on Cityscapes dataset.

This module implements a U-Net model optimized for 8-category semantic
segmentation, with custom loss functions and metrics.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple, Optional


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    activation: str = "relu",
    padding: str = "same"
) -> tf.Tensor:
    """
    Create a convolutional block with two conv layers and batch normalization.

    Args:
        inputs: Input tensor
        filters: Number of filters
        kernel_size: Size of the convolution kernel
        activation: Activation function
        padding: Padding type

    Returns:
        Output tensor
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        kernel_initializer="he_normal"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    filters: int = 64,
    dropout: float = 0.5,
    activation: str = "softmax"
) -> keras.Model:
    """
    Build a U-Net model for semantic segmentation.

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        filters: Number of filters in the first layer (will be doubled in each level)
        dropout: Dropout rate in the bottleneck
        activation: Final activation function ('softmax' for multi-class)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder (downsampling path)
    # Level 1
    c1 = conv_block(inputs, filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Level 2
    c2 = conv_block(p1, filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Level 3
    c3 = conv_block(p2, filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Level 4
    c4 = conv_block(p3, filters * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, filters * 16)
    c5 = layers.Dropout(dropout)(c5)

    # Decoder (upsampling path)
    # Level 4
    u6 = layers.Conv2DTranspose(filters * 8, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, filters * 8)

    # Level 3
    u7 = layers.Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, filters * 4)

    # Level 2
    u8 = layers.Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, filters * 2)

    # Level 1
    u9 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, filters)

    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(c9)

    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net")

    return model


def build_unet_small(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    filters: int = 32,
    dropout: float = 0.5,
    activation: str = "softmax"
) -> keras.Model:
    """
    Build a smaller U-Net model for faster training (useful for testing).

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        filters: Number of filters in the first layer
        dropout: Dropout rate in the bottleneck
        activation: Final activation function

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, filters)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, filters * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = conv_block(p3, filters * 8)
    c4 = layers.Dropout(dropout)(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(filters * 4, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = conv_block(u5, filters * 4)

    u6 = layers.Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = conv_block(u6, filters * 2)

    u7 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = conv_block(u7, filters)

    # Output
    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(c7)

    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net-Small")

    return model


def build_unet_mobilenet(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    dropout: float = 0.5,
    activation: str = "softmax",
    alpha: float = 1.0,
    weights: Optional[str] = "imagenet",
    decoder_filters: int = 256
) -> keras.Model:
    """
    Build a U-Net model with MobileNetV2 backbone for semantic segmentation.

    This architecture uses MobileNetV2 as the encoder (downsampling path) and
    a custom decoder (upsampling path) with skip connections. This provides
    better feature extraction while maintaining efficiency.

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        dropout: Dropout rate in the decoder
        activation: Final activation function ('softmax' for multi-class)
        alpha: Width multiplier for MobileNet (controls model size)
               - 1.0: Full MobileNet (default)
               - 0.75: 75% of filters
               - 0.5: 50% of filters
               - 0.35: 35% of filters (smallest)
        weights: Pre-trained weights ('imagenet' or None)
        decoder_filters: Number of filters in the decoder layers

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Load MobileNetV2 as encoder (backbone)
    # MobileNetV2 expects inputs in range [-1, 1] or [0, 1]
    # We'll use include_top=False to get feature maps
    encoder = MobileNetV2(
        input_tensor=inputs,
        alpha=alpha,
        weights=weights,
        include_top=False
    )

    # Extract skip connections from MobileNet layers
    # MobileNetV2 structure:
    # - block_1_expand_relu: 1/2 resolution
    # - block_3_expand_relu: 1/4 resolution
    # - block_6_expand_relu: 1/8 resolution
    # - block_13_expand_relu: 1/16 resolution
    # - block_16_project: 1/32 resolution (bottleneck)

    skip_connection_names = [
        'block_1_expand_relu',   # 1/2
        'block_3_expand_relu',   # 1/4
        'block_6_expand_relu',   # 1/8
        'block_13_expand_relu',  # 1/16
    ]

    # Get skip connection layers
    skip_connections = []
    for layer_name in skip_connection_names:
        try:
            skip_connections.append(encoder.get_layer(layer_name).output)
        except ValueError:
            # If layer name doesn't exist, try to find by pattern
            # This handles potential version differences in layer naming
            raise ValueError(
                f"Layer '{layer_name}' not found in MobileNetV2. "
                f"Available layers: {[layer.name for layer in encoder.layers[-20:]]}"
            )

    # Bottleneck (output of MobileNet encoder)
    bottleneck = encoder.output

    # Decoder (upsampling path)
    # MobileNetV2 bottleneck is typically at 1/32 resolution
    # We need 5 upsampling steps total: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
    # Skip connections are at: 1/16, 1/8, 1/4, 1/2

    # First upsampling: bottleneck (1/32) -> 1/16 to match skip_connections[3]
    x = layers.Conv2DTranspose(
        decoder_filters, (2, 2), strides=(2, 2), padding="same"
    )(bottleneck)

    # Level 4: 1/16 -> 1/8 (with skip connection from 1/16)
    u1 = layers.concatenate([x, skip_connections[3]])
    c1 = conv_block(u1, decoder_filters)

    # Level 3: 1/8 -> 1/4
    u2 = layers.Conv2DTranspose(
        decoder_filters // 2, (2, 2), strides=(2, 2), padding="same"
    )(c1)
    u2 = layers.concatenate([u2, skip_connections[2]])
    c2 = conv_block(u2, decoder_filters // 2)

    # Level 2: 1/4 -> 1/2
    u3 = layers.Conv2DTranspose(
        decoder_filters // 4, (2, 2), strides=(2, 2), padding="same"
    )(c2)
    u3 = layers.concatenate([u3, skip_connections[1]])
    c3 = conv_block(u3, decoder_filters // 4)

    # Level 1: 1/2 -> 1/1 (full resolution)
    u4 = layers.Conv2DTranspose(
        decoder_filters // 8, (2, 2), strides=(2, 2), padding="same"
    )(c3)
    u4 = layers.concatenate([u4, skip_connections[0]])
    c4 = conv_block(u4, decoder_filters // 8)
    c4 = layers.Dropout(dropout)(c4)

    # Output layer (before resize)
    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(c4)

    # Final resize to ensure output matches input size exactly
    # The decoder reaches 1/2 resolution (128x256), so we need to upsample by 2x
    # Use Lambda with tf.image.resize to force the resize operation
    target_height, target_width = input_shape[0], input_shape[1]
    outputs = layers.Lambda(
        lambda x: tf.image.resize(x, size=[target_height, target_width], method='bilinear'),
        name='resize_to_input_size'
    )(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net-MobileNet")

    return model
