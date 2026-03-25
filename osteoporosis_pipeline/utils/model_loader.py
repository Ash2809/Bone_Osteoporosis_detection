"""
utils/model_loader.py
----------------------
Convenience functions for loading saved model artefacts
and constructing predictor objects ready for inference.
"""

from __future__ import annotations

import joblib
import tensorflow as tf
from tensorflow.keras import layers

from osteoporosis_pipeline import config
from osteoporosis_pipeline.models.attention import cbam_block
from osteoporosis_pipeline.models.image_models import build_disease_classifier
from osteoporosis_pipeline.inference.predictors import (
    XRayPredictor,
    TScorePredictor,
    BMDPredictor,
    MetaPredictor,
)


def load_disease_model(
    weights_path: str = config.KNEE_DISEASE_WEIGHTS,
    img_size: int = config.IMG_SIZE,
) -> tf.keras.Model:
    """
    Reconstruct the disease classifier architecture and load saved weights.

    Parameters
    ----------
    weights_path : path to the .h5 weights file
    img_size     : must match the size used during training

    Returns
    -------
    tf.keras.Model  (compiled, ready for inference)
    """
    model = build_disease_classifier(img_size=img_size, freeze_base=False)
    model.load_weights(weights_path)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[INFO] Disease model loaded from: {weights_path}")
    return model


def load_xray_predictor(
    weights_path: str = config.KNEE_DISEASE_WEIGHTS,
    img_size: int = config.IMG_SIZE,
) -> XRayPredictor:
    """Return an XRayPredictor wrapping the loaded disease classifier."""
    model = load_disease_model(weights_path, img_size)
    return XRayPredictor(model, img_size=img_size)


def load_tscore_predictor(
    model_path:    str = config.TSCORE_MODEL_PATH,
    scaler_path:   str = config.TSCORE_SCALER_PATH,
    features_path: str = config.TSCORE_FEATURES_PATH,
) -> TScorePredictor:
    """Return a TScorePredictor from saved joblib artefacts."""
    return TScorePredictor.from_files(model_path, scaler_path, features_path)


def load_bmd_predictor(model_path: str = config.BMD_MODEL_PATH) -> BMDPredictor:
    """Return a BMDPredictor from a saved joblib pipeline."""
    return BMDPredictor.from_file(model_path)


def load_meta_predictor(
    xray_weights:    str = config.KNEE_DISEASE_WEIGHTS,
    tscore_model:    str = config.TSCORE_MODEL_PATH,
    tscore_scaler:   str = config.TSCORE_SCALER_PATH,
    tscore_features: str = config.TSCORE_FEATURES_PATH,
    bmd_model:       str = config.BMD_MODEL_PATH,
) -> MetaPredictor:
    """
    Build and return a fully initialised MetaPredictor.

    All three sub-models are loaded from their saved artefacts.
    """
    xray_pred   = load_xray_predictor(xray_weights)
    tscore_pred = load_tscore_predictor(tscore_model, tscore_scaler, tscore_features)
    bmd_pred    = load_bmd_predictor(bmd_model)

    return MetaPredictor(xray_pred, tscore_pred, bmd_pred)