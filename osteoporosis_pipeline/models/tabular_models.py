"""
models/tabular_models.py
-------------------------
Builders for tabular (non-image) models:
  - MLP for multi-class disease classification from clinical features
  - Random Forest for T-score regression
  - Random Forest pipeline for BMD regression
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from osteoporosis_pipeline import config


# ─────────────────────────────────────────────────────────────
# Keras MLP classifier (clinical → diagnosis)
# ─────────────────────────────────────────────────────────────

def build_tabular_mlp(
    input_dim: int,
    num_classes: int = config.NUM_DISEASE_CLASSES,
) -> tf.keras.Model:
    """
    Two-hidden-layer MLP for tabular clinical feature classification.

    Parameters
    ----------
    input_dim   : number of input features (after preprocessing)
    num_classes : number of output classes

    Returns
    -------
    Compiled tf.keras.Model
    """
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="tabular_mlp")
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# Random Forest – T-score regression
# ─────────────────────────────────────────────────────────────

def build_tscore_rf(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = config.SEED,
) -> RandomForestRegressor:
    """
    Random Forest regressor for T-score prediction.

    Parameters
    ----------
    n_estimators : number of trees
    max_depth    : max tree depth (None = unlimited)
    random_state : reproducibility seed

    Returns
    -------
    sklearn RandomForestRegressor (unfitted)
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────
# Random Forest pipeline – BMD regression
# ─────────────────────────────────────────────────────────────

def build_bmd_rf_pipeline(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    n_estimators: int = 500,
    max_depth: int = 8,
    random_state: int = config.SEED,
) -> Pipeline:
    """
    sklearn Pipeline that one-hot encodes categorical columns, passes numerical
    columns through unchanged, then fits a Random Forest regressor for BMD
    prediction.

    Parameters
    ----------
    categorical_cols : list of categorical column names (e.g. ["medication"])
    numerical_cols   : list of numeric column names
    n_estimators     : RF trees
    max_depth        : RF max depth
    random_state     : seed

    Returns
    -------
    sklearn Pipeline (unfitted)
    """
    if categorical_cols is None:
        categorical_cols = ["medication"]
    if numerical_cols is None:
        numerical_cols = ["age", "sex", "fracture", "weight_kg", "height_m", "bmi", "waiting_time"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )
    return pipeline