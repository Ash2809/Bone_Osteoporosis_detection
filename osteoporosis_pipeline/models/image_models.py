"""
models/image_models.py
-----------------------
Keras model builders for:
  - Disease classifier  (ResNet50 + CBAM, 3-class softmax)
  - Anatomy classifier  (EfficientNetV2B0 + CBAM, 2-class softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from osteoporosis_pipeline.models.attention import cbam_block
from osteoporosis_pipeline import config


def build_disease_classifier(
    img_size: int = config.IMG_SIZE,
    num_classes: int = config.NUM_DISEASE_CLASSES,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    ResNet50 backbone with CBAM attention head for 3-class disease prediction
    (normal / osteopenia / osteoporosis).

    Parameters
    ----------
    img_size    : square input side length (pixels)
    num_classes : number of output classes
    freeze_base : whether to freeze the ResNet50 backbone weights

    Returns
    -------
    Compiled tf.keras.Model
    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = not freeze_base

    # augmentation
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )

    inp = layers.Input((img_size, img_size, 3))
    x   = augmentation(inp)
    x   = tf.keras.applications.resnet.preprocess_input(x * 255.0)  # dataset yields [0,1]
    x   = base(x, training=False)
    x   = cbam_block(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="disease_classifier")
    model.compile(
        optimizer=optimizers.Adam(config.LR_STAGE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_anatomy_classifier(
    img_size: int = config.IMG_SIZE,
    num_classes: int = config.NUM_ANATOMY_CLASSES,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    EfficientNetV2B0 backbone with CBAM for 2-class anatomy detection
    (knee / spine).

    Parameters
    ----------
    img_size    : square input side length (pixels)
    num_classes : number of output classes (2 → knee vs spine)
    freeze_base : whether to freeze the EfficientNet backbone

    Returns
    -------
    Compiled tf.keras.Model
    """
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet",
    )
    base.trainable = not freeze_base

    inp = layers.Input((img_size, img_size, 3))
    x   = base(inp, training=False)
    x   = cbam_block(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out, name="anatomy_classifier")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = 150) -> tf.keras.Model:
    """
    Unfreeze the last `n_layers` of the first sub-model (backbone) for fine-tuning.
    BatchNormalization layers are kept frozen to preserve running statistics.

    Parameters
    ----------
    model    : model returned by one of the builders above
    n_layers : how many layers from the top of the backbone to unfreeze

    Returns
    -------
    The same model with updated trainability and recompiled at a lower LR
    """
    backbone = model.layers[1]          # index 0 = Input, index 1 = backbone
    for layer in backbone.layers[-n_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(config.LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model