"""
training/train_image.py
------------------------
Training routines for the image-based classifiers:
  - Disease classifier (knee or spine)
  - Anatomy classifier (knee vs spine)

Usage (CLI)::

    python -m osteoporosis_pipeline.training.train_image --task disease --site knee
    python -m osteoporosis_pipeline.training.train_image --task anatomy
"""

from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from osteoporosis_pipeline import config
from osteoporosis_pipeline.data.image_datasets import load_disease_dataset, load_anatomy_dataset
from osteoporosis_pipeline.models.image_models import (
    build_disease_classifier,
    build_anatomy_classifier,
    unfreeze_top_layers,
)


def _make_callbacks(checkpoint_path: str) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def _compute_class_weights(train_ds: tf.data.Dataset) -> dict:
    """Iterate one pass over the training dataset to compute balanced class weights."""
    labels = [np.argmax(y.numpy()) for _, ys in train_ds.unbatch().batch(1024) for y in ys]
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_disease_classifier(
    data_dir: str,
    save_path: str,
    epochs: int = config.EPOCHS_IMAGE,
    fine_tune: bool = True,
) -> tf.keras.Model:
    """
    Full training pipeline for the disease classifier.

    Stage 1 : train the head with frozen backbone.
    Stage 2 : optionally unfreeze top backbone layers and fine-tune.

    Parameters
    ----------
    data_dir  : directory with train/ val/ test/ sub-folders
    save_path : file path to save the best model weights (.h5)
    epochs    : max epochs per stage
    fine_tune : whether to run stage-2 fine-tuning

    Returns
    -------
    Trained tf.keras.Model
    """
    print(f"\n[INFO] Loading disease dataset from: {data_dir}")
    train_ds, val_ds, test_ds = load_disease_dataset(data_dir)

    cw = _compute_class_weights(train_ds)
    print(f"[INFO] Class weights: {cw}")

    model = build_disease_classifier(freeze_base=True)

    # ── Stage 1 ──────────────────────────────────────────────
    print("\n[INFO] Stage 1: training head (frozen backbone) …")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=cw,
        callbacks=_make_callbacks(save_path),
    )

    # ── Stage 2 (fine-tune) ──────────────────────────────────
    if fine_tune:
        print("\n[INFO] Stage 2: fine-tuning top backbone layers …")
        model = unfreeze_top_layers(model, n_layers=50)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs // 2,
            class_weight=cw,
            callbacks=_make_callbacks(save_path),
        )

    print("\n[INFO] Final evaluation on test set:")
    model.evaluate(test_ds)

    return model


def train_anatomy_classifier(
    knee_dir: str = config.KNEE_DATA_DIR,
    spine_dir: str = config.SPINE_DATA_DIR,
    save_path: str = config.ANATOMY_MODEL_PATH,
    epochs: int = 12,
) -> tf.keras.Model:
    """
    Train the knee-vs-spine anatomy classifier.

    Parameters
    ----------
    knee_dir  : knee image root directory
    spine_dir : spine image root directory
    save_path : output model path
    epochs    : training epochs

    Returns
    -------
    Trained tf.keras.Model
    """
    print("\n[INFO] Loading anatomy dataset …")
    train_ds, val_ds, test_ds = load_anatomy_dataset(knee_dir, spine_dir)

    model = build_anatomy_classifier()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_make_callbacks(save_path),
    )

    print("\n[INFO] Anatomy test evaluation:")
    model.evaluate(test_ds)
    model.save(save_path)
    print(f"[INFO] Saved anatomy model to: {save_path}")

    return model


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image models")
    parser.add_argument("--task", choices=["disease", "anatomy"], required=True)
    parser.add_argument("--site", choices=["knee", "spine"], default="knee",
                        help="Which anatomical site to train disease model for")
    args = parser.parse_args()

    if args.task == "disease":
        data_dir  = config.KNEE_DATA_DIR if args.site == "knee" else config.SPINE_DATA_DIR
        save_path = config.KNEE_DISEASE_WEIGHTS if args.site == "knee" else config.SPINE_DISEASE_WEIGHTS
        train_disease_classifier(data_dir, save_path)
    else:
        train_anatomy_classifier()