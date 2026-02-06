"""
U-Net and DeepLabV3 architectures for semantic segmentation on Cityscapes dataset.

This module implements U-Net and DeepLabV3 models optimized for 8-category semantic
segmentation, with custom loss functions and metrics.

Models available:
- U-Net: Classic encoder-decoder architecture with skip connections
- U-Net with MobileNetV2: U-Net using MobileNetV2 as encoder backbone
- U-Net with ResNet50: U-Net using ResNet50 as encoder backbone
- DeepLabV3: DeepLabV3 with MobileNetV2 or ResNet50 backbone and ASPP module
  (Based on TensorFlow Model Garden tutorial)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from
from typing import Tuple, Optional, List


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
    decoder_filters: int = 256,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Build a U-Net model with MobileNetV2 backbone for semantic segmentation.

    This architecture uses MobileNetV2 as the encoder (downsampling path) and
    a custom decoder (upsampling path) with skip connections. A built-in
    preprocessing layer converts inputs from [0, 1] to [-1, 1] as expected
    by MobileNetV2 pretrained on ImageNet.

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
        freeze_backbone: Whether to freeze encoder weights for transfer learning.
                        When True, only the decoder is trained initially.
                        Set to False for fine-tuning the entire model.

    Returns:
        Keras model

    Note:
        Feed images normalized to [0, 1] range (as done by CityscapesDataGenerator
        with normalize=True). The model handles the conversion to [-1, 1] internally.
    """
    inputs = layers.Input(shape=input_shape)

    # Preprocessing: MobileNetV2 pretrained on ImageNet expects inputs in [-1, 1].
    # Our data pipeline (CityscapesDataGenerator) normalizes images to [0, 1],
    # so we rescale here: [0, 1] -> [-1, 1] via x * 2.0 - 1.0
    preprocessed = layers.Lambda(
        lambda x: x * 2.0 - 1.0,
        name='mobilenet_preprocess'
    )(inputs)

    # Load MobileNetV2 as encoder (backbone)
    encoder = MobileNetV2(
        input_tensor=preprocessed,
        alpha=alpha,
        weights=weights,
        include_top=False
    )

    # Freeze backbone for transfer learning: train only decoder first,
    # then optionally unfreeze for fine-tuning with a lower learning rate.
    if freeze_backbone:
        encoder.trainable = False

    # Extract skip connections from MobileNet intermediate layers
    # MobileNetV2 structure (with input 512x512):
    # - block_1_expand_relu:  256x256 (1/2 resolution)
    # - block_3_expand_relu:  128x128 (1/4 resolution)
    # - block_6_expand_relu:   64x64  (1/8 resolution)
    # - block_13_expand_relu:  32x32  (1/16 resolution)
    # - encoder.output:        16x16  (1/32 resolution, bottleneck)

    skip_connection_names = [
        'block_1_expand_relu',   # 1/2
        'block_3_expand_relu',   # 1/4
        'block_6_expand_relu',   # 1/8
        'block_13_expand_relu',  # 1/16
    ]

    skip_connections = []
    for layer_name in skip_connection_names:
        try:
            skip_connections.append(encoder.get_layer(layer_name).output)
        except ValueError:
            raise ValueError(
                f"Layer '{layer_name}' not found in MobileNetV2. "
                f"Available layers: {[layer.name for layer in encoder.layers[-20:]]}"
            )

    # Bottleneck (output of MobileNet encoder) at 1/32 resolution
    bottleneck = encoder.output

    # Decoder (upsampling path)
    # 5 upsampling steps to return to full resolution:
    # 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
    # Skip connections at: 1/16, 1/8, 1/4, 1/2

    # Step 1: 1/32 -> 1/16 (with skip from block_13_expand_relu)
    x = layers.Conv2DTranspose(
        decoder_filters, (3, 3), strides=(2, 2), padding="same"
    )(bottleneck)
    x = layers.concatenate([x, skip_connections[3]])
    x = conv_block(x, decoder_filters)

    # Step 2: 1/16 -> 1/8 (with skip from block_6_expand_relu)
    x = layers.Conv2DTranspose(
        decoder_filters // 2, (3, 3), strides=(2, 2), padding="same"
    )(x)
    x = layers.concatenate([x, skip_connections[2]])
    x = conv_block(x, decoder_filters // 2)

    # Step 3: 1/8 -> 1/4 (with skip from block_3_expand_relu)
    x = layers.Conv2DTranspose(
        decoder_filters // 4, (3, 3), strides=(2, 2), padding="same"
    )(x)
    x = layers.concatenate([x, skip_connections[1]])
    x = conv_block(x, decoder_filters // 4)

    # Step 4: 1/4 -> 1/2 (with skip from block_1_expand_relu)
    x = layers.Conv2DTranspose(
        decoder_filters // 8, (3, 3), strides=(2, 2), padding="same"
    )(x)
    x = layers.concatenate([x, skip_connections[0]])
    x = conv_block(x, decoder_filters // 8)

    # Step 5: 1/2 -> 1/1 (full resolution, no skip connection at this level)
    x = layers.Conv2DTranspose(
        decoder_filters // 8, (3, 3), strides=(2, 2), padding="same"
    )(x)
    x = conv_block(x, decoder_filters // 8)
    x = layers.Dropout(dropout)(x)

    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net-MobileNet")

    return model


def build_unet_resnet34(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    dropout: float = 0.5,
    activation: str = "softmax",
    weights: Optional[str] = "imagenet",
    decoder_filters: int = 256
) -> keras.Model:
    """
    Build a U-Net model with ResNet50 backbone for semantic segmentation.

    Uses ResNet50 as the encoder (downsampling path) and a custom decoder
    (upsampling path) with skip connections. ResNet50 provides stronger
    feature extraction than MobileNetV2 at the cost of more parameters.

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        dropout: Dropout rate in the decoder
        activation: Final activation function ('softmax' for multi-class)
        weights: Pre-trained weights ('imagenet' or None)
        decoder_filters: Number of filters in the decoder layers

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Load ResNet50 as encoder (backbone)
    encoder = ResNet34(
        input_tensor=inputs,
        weights=weights,
        include_top=False
    )

    # ResNet50 structure (include_top=False):
    # - conv1_relu: 1/2 resolution
    # - conv2_block3_out: 1/4 resolution
    # - conv3_block4_out: 1/8 resolution
    # - conv4_block6_out: 1/16 resolution
    # - conv5_block3_out: 1/32 resolution (bottleneck)

    skip_connection_names = [
        "conv1_relu",        # 1/2
        "conv2_block3_out",  # 1/4
        "conv3_block4_out",  # 1/8
        "conv4_block6_out",  # 1/16
    ]

    skip_connections = []
    for layer_name in skip_connection_names:
        try:
            skip_connections.append(encoder.get_layer(layer_name).output)
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Layer '{layer_name}' not found in ResNet50. "
                f"Available: {[ly.name for ly in encoder.layers[-30:]]}"
            ) from e

    bottleneck = encoder.output

    # Decoder: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
    x = layers.Conv2DTranspose(
        decoder_filters, (2, 2), strides=(2, 2), padding="same"
    )(bottleneck)

    u1 = layers.concatenate([x, skip_connections[3]])
    c1 = conv_block(u1, decoder_filters)

    u2 = layers.Conv2DTranspose(
        decoder_filters // 2, (2, 2), strides=(2, 2), padding="same"
    )(c1)
    u2 = layers.concatenate([u2, skip_connections[2]])
    c2 = conv_block(u2, decoder_filters // 2)

    u3 = layers.Conv2DTranspose(
        decoder_filters // 4, (2, 2), strides=(2, 2), padding="same"
    )(c2)
    u3 = layers.concatenate([u3, skip_connections[1]])
    c3 = conv_block(u3, decoder_filters // 4)

    u4 = layers.Conv2DTranspose(
        decoder_filters // 8, (2, 2), strides=(2, 2), padding="same"
    )(c3)
    u4 = layers.concatenate([u4, skip_connections[0]])
    c4 = conv_block(u4, decoder_filters // 8)
    c4 = layers.Dropout(dropout)(c4)

    outputs = layers.Conv2D(n_classes, (1, 1), activation=activation)(c4)

    target_height, target_width = input_shape[0], input_shape[1]
    outputs = layers.Lambda(
        lambda x: tf.image.resize(x, size=[target_height, target_width], method='bilinear'),
        name='resize_to_input_size'
    )(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net-ResNet50")

    return model


def aspp_block(
    inputs: tf.Tensor,
    filters: int = 256,
    atrous_rates: List[int] = [6, 12, 18]
) -> tf.Tensor:
    """
    Atrous Spatial Pyramid Pooling (ASPP) module for DeepLabV3.

    The ASPP module captures multi-scale contextual information by applying
    atrous convolutions with different dilation rates in parallel.

    Args:
        inputs: Input tensor
        filters: Number of filters for each atrous convolution
        atrous_rates: List of dilation rates for atrous convolutions

    Returns:
        Concatenated output tensor from all ASPP branches
    """
    # Image pooling branch (global average pooling + upsampling)
    image_pool = layers.GlobalAveragePooling2D()(inputs)
    # Reshape to (batch, 1, 1, channels) - use -1 to infer channels
    image_pool = layers.Reshape((1, 1, -1))(image_pool)
    image_pool = layers.Conv2D(
        filters, 1, padding='same', kernel_initializer='he_normal'
    )(image_pool)
    image_pool = layers.BatchNormalization()(image_pool)
    image_pool = layers.Activation('relu')(image_pool)

    # Upsample to match input spatial dimensions
    # Use Lambda to get target size from inputs tensor
    image_pool = layers.Lambda(
        lambda x: tf.image.resize(
            x[0],
            tf.shape(x[1])[1:3],
            method='bilinear'
        ),
        name='aspp_image_pool_resize'
    )([image_pool, inputs])

    # Atrous convolution branches
    atrous_branches = [image_pool]

    # 1x1 convolution branch
    branch_1x1 = layers.Conv2D(
        filters, 1, padding='same', kernel_initializer='he_normal'
    )(inputs)
    branch_1x1 = layers.BatchNormalization()(branch_1x1)
    branch_1x1 = layers.Activation('relu')(branch_1x1)
    atrous_branches.append(branch_1x1)

    # Atrous convolutions with different rates
    for rate in atrous_rates:
        branch = layers.Conv2D(
            filters, 3,
            padding='same',
            dilation_rate=rate,
            kernel_initializer='he_normal'
        )(inputs)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Activation('relu')(branch)
        atrous_branches.append(branch)

    # Concatenate all branches
    output = layers.concatenate(atrous_branches, axis=-1)

    # Final projection
    output = layers.Conv2D(
        filters, 1, padding='same', kernel_initializer='he_normal'
    )(output)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)
    output = layers.Dropout(0.1)(output)

    return output


def build_deeplabv3(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    backbone: str = "mobilenetv2",
    alpha: float = 1.0,
    weights: Optional[str] = "imagenet",
    aspp_filters: int = 256,
    aspp_rates: List[int] = [6, 12, 18],
    activation: str = "softmax"
) -> keras.Model:
    """
    Build a DeepLabV3 model for semantic segmentation.

    DeepLabV3 uses Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale
    contextual information. Supports MobileNetV2 or ResNet50 as backbone encoder.

    Reference: https://www.tensorflow.org/tfmodels/vision/semantic_segmentation

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        backbone: Backbone architecture ('mobilenetv2' or 'resnet50')
        alpha: Width multiplier for MobileNet only (ignored for ResNet50)
        weights: Pre-trained weights ('imagenet' or None)
        aspp_filters: Number of filters in ASPP module
        aspp_rates: List of dilation rates for ASPP atrous convolutions
        activation: Final activation function ('softmax' for multi-class)

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Load backbone encoder
    if backbone == "mobilenetv2":
        encoder = MobileNetV2(
            input_tensor=inputs,
            alpha=alpha,
            weights=weights,
            include_top=False
        )
        encoder_output = encoder.output
    elif backbone == "resnet50":
        encoder = ResNet50(
            input_tensor=inputs,
            weights=weights,
            include_top=False
        )
        encoder_output = encoder.output
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. Use 'mobilenetv2' or 'resnet50'."
        )

    # Apply atrous convolution to increase feature map resolution
    # Use atrous convolution to maintain receptive field while increasing resolution
    # This is equivalent to output_stride=16 (instead of 32)
    # MobileNetV2 output is at 1/32 resolution, we want 1/16 for ASPP
    x = layers.Conv2D(
        aspp_filters, 3,
        padding='same',
        dilation_rate=2,
        kernel_initializer='he_normal'
    )(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply ASPP module
    x = aspp_block(x, filters=aspp_filters, atrous_rates=aspp_rates)

    # Decoder: upsample to original resolution using bilinear interpolation
    # DeepLabV3 typically uses bilinear upsampling (simpler and smoother than transpose conv)
    target_height, target_width = input_shape[0], input_shape[1]
    x = layers.Lambda(
        lambda img: tf.image.resize(img, size=[target_height, target_width], method='bilinear'),
        name='resize_to_input_size'
    )(x)

    # Output layer
    outputs = layers.Conv2D(n_classes, 1, activation=activation, name='predictions')(x)

    model_name = f"DeepLabV3-{backbone}"
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def build_deeplabv3_resnet50(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    weights: Optional[str] = "imagenet",
    aspp_filters: int = 256,
    aspp_rates: List[int] = [6, 12, 18],
    activation: str = "softmax"
) -> keras.Model:
    """
    Build a DeepLabV3 model with ResNet50 backbone (convenience wrapper).

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        aspp_filters: Number of filters in ASPP module
        aspp_rates: List of dilation rates for ASPP atrous convolutions
        activation: Final activation function ('softmax' for multi-class)

    Returns:
        Compiled Keras model
    """
    return build_deeplabv3(
        input_shape=input_shape,
        n_classes=n_classes,
        backbone="resnet50",
        alpha=1.0,
        weights=weights,
        aspp_filters=aspp_filters,
        aspp_rates=aspp_rates,
        activation=activation
    )


def build_deeplabv3_mobilenet(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    n_classes: int = 8,
    alpha: float = 1.0,
    weights: Optional[str] = "imagenet",
    aspp_filters: int = 256,
    activation: str = "softmax"
) -> keras.Model:
    """
    Build a DeepLabV3 model with MobileNetV2 backbone (convenience wrapper).

    This is a convenience function that calls build_deeplabv3 with MobileNetV2
    as the backbone, matching the TensorFlow Model Garden tutorial.

    Args:
        input_shape: Shape of input images (height, width, channels)
        n_classes: Number of output classes
        alpha: Width multiplier for MobileNet (controls model size)
        weights: Pre-trained weights ('imagenet' or None)
        aspp_filters: Number of filters in ASPP module
        activation: Final activation function ('softmax' for multi-class)

    Returns:
        Compiled Keras model
    """
    return build_deeplabv3(
        input_shape=input_shape,
        n_classes=n_classes,
        backbone="mobilenetv2",
        alpha=alpha,
        weights=weights,
        aspp_filters=aspp_filters,
        activation=activation
    )
