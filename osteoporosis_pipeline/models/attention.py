"""
models/attention.py
--------------------
Convolutional Block Attention Module (CBAM).
Used by both the knee-disease and anatomy classifiers.
"""

import tensorflow as tf
from tensorflow.keras import layers


def channel_attention(feature_map: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Squeeze-and-Excitation style channel attention.

    Parameters
    ----------
    feature_map : tf.Tensor  shape (B, H, W, C)
    ratio       : bottleneck ratio for the shared MLP

    Returns
    -------
    tf.Tensor  same shape as feature_map, channel-re-weighted
    """
    channel = feature_map.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(feature_map)   # (B, C)
    max_pool = layers.GlobalMaxPooling2D()(feature_map)        # (B, C)

    shared_dense_1 = layers.Dense(channel // ratio, activation="relu", use_bias=True)
    shared_dense_2 = layers.Dense(channel, use_bias=True)

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation("sigmoid")(attention)
    attention = layers.Reshape((1, 1, channel))(attention)     # broadcast over H, W

    return layers.Multiply()([feature_map, attention])


def spatial_attention(feature_map: tf.Tensor) -> tf.Tensor:
    """
    Spatial attention using average-pool and max-pool along the channel axis.

    Parameters
    ----------
    feature_map : tf.Tensor  shape (B, H, W, C)

    Returns
    -------
    tf.Tensor  same shape as feature_map, spatially re-weighted
    """
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1),
    )(feature_map)

    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1),
    )(feature_map)

    concat = layers.Concatenate()([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)

    return layers.Multiply()([feature_map, attention])


def cbam_block(feature_map: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Full CBAM: channel attention followed by spatial attention.

    Parameters
    ----------
    feature_map : tf.Tensor  shape (B, H, W, C)
    ratio       : bottleneck ratio for channel attention MLP

    Returns
    -------
    tf.Tensor  same shape, attention-refined
    """
    x = channel_attention(feature_map, ratio=ratio)
    x = spatial_attention(x)
    return x