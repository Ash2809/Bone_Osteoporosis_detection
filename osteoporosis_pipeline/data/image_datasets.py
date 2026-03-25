"""
data/image_datasets.py
-----------------------
TensorFlow dataset loaders for:
  - Disease classification  (knee or spine, 3 classes)
  - Anatomy classification  (knee vs spine, 2 classes)
"""

import os
import tensorflow as tf

from osteoporosis_pipeline import config

AUTOTUNE = tf.data.AUTOTUNE


def _normalize(x: tf.Tensor, y: tf.Tensor):
    """Scale pixel values from [0, 255] → [0, 1]."""
    return tf.cast(x, tf.float32) / 255.0, y


def load_disease_dataset(
    base_dir: str,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    class_names: list[str] = config.CLASS_NAMES_DISEASE,
) -> tuple:
    """
    Load train / val / test splits for disease classification.

    Expected directory layout::

        base_dir/
          train/
            normal/
            osteopenia/
            osteoporosis/
          val/   (same sub-dirs)
          test/  (same sub-dirs)

    Parameters
    ----------
    base_dir   : root directory (KNEE_PATH or SPINE_PATH)
    img_size   : resize images to (img_size × img_size)
    batch_size : mini-batch size
    class_names: ordered list of class folder names

    Returns
    -------
    (train_ds, val_ds, test_ds) as prefetched tf.data.Dataset objects
    """
    kwargs = dict(
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "train"), shuffle=True, **kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "val"), shuffle=False, **kwargs
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "test"), shuffle=False, **kwargs
    )

    train_ds = train_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_normalize,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds  = test_ds.map(_normalize,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def load_anatomy_dataset(
    knee_dir: str = config.KNEE_DATA_DIR,
    spine_dir: str = config.SPINE_DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
) -> tuple:
    """
    Build a combined knee-vs-spine binary dataset by:
      1. Loading disease datasets for both anatomical sites.
      2. Replacing labels with one-hot anatomy labels (knee=0, spine=1).
      3. Concatenating the two streams.

    Parameters
    ----------
    knee_dir  : root directory for knee images
    spine_dir : root directory for spine images
    img_size  : square resize target
    batch_size: mini-batch size

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    train_k, val_k, test_k = load_disease_dataset(knee_dir,  img_size, batch_size)
    train_s, val_s, test_s = load_disease_dataset(spine_dir, img_size, batch_size)

    def _label_knee(x, _y):
        n = tf.shape(x)[0]
        return x, tf.one_hot(tf.zeros((n,), dtype=tf.int32), 2)

    def _label_spine(x, _y):
        n = tf.shape(x)[0]
        return x, tf.one_hot(tf.ones((n,), dtype=tf.int32), 2)

    train_k = train_k.map(_label_knee,  num_parallel_calls=AUTOTUNE)
    val_k   = val_k.map(_label_knee,    num_parallel_calls=AUTOTUNE)
    test_k  = test_k.map(_label_knee,   num_parallel_calls=AUTOTUNE)

    train_s = train_s.map(_label_spine, num_parallel_calls=AUTOTUNE)
    val_s   = val_s.map(_label_spine,   num_parallel_calls=AUTOTUNE)
    test_s  = test_s.map(_label_spine,  num_parallel_calls=AUTOTUNE)

    train = train_k.concatenate(train_s).prefetch(AUTOTUNE)
    val   = val_k.concatenate(val_s).prefetch(AUTOTUNE)
    test  = test_k.concatenate(test_s).prefetch(AUTOTUNE)

    return train, val, test