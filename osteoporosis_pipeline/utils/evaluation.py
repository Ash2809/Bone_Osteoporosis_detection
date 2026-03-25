"""
utils/evaluation.py
--------------------
Evaluation helpers: classification reports, confusion matrices,
and training-curve plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

from osteoporosis_pipeline import config


def evaluate_image_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str] = config.CLASS_NAMES_DISEASE,
) -> dict:
    """
    Run the model on a test dataset and print a full classification report.

    Parameters
    ----------
    model       : trained Keras model
    test_ds     : test tf.data.Dataset yielding (images, one-hot labels)
    class_names : label names in class-index order

    Returns
    -------
    dict with keys 'accuracy', 'report', 'confusion_matrix'
    """
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds  = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"Accuracy : {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def plot_training_curves(
    history,
    title: str = "Training Curves",
    save_path: str | None = None,
) -> None:
    """
    Plot accuracy and loss curves from a Keras History object or a plain dict.

    Parameters
    ----------
    history   : tf.keras.callbacks.History or dict with keys
                'accuracy', 'val_accuracy', 'loss', 'val_loss'
    title     : figure title
    save_path : if given, save figure to this path instead of showing
    """
    h = history.history if hasattr(history, "history") else history

    epochs = range(1, len(h["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    ax1.plot(epochs, h["accuracy"],     marker="o", label="Train Accuracy")
    ax1.plot(epochs, h["val_accuracy"], marker="o", label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, h["loss"],     marker="o", label="Train Loss")
    ax2.plot(epochs, h["val_loss"], marker="o", label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Figure saved to: {save_path}")
    else:
        plt.show()