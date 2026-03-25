"""
training/train_tabular.py
--------------------------
Training routines for tabular models:
  - T-score Random Forest regressor
  - BMD Random Forest regressor (sklearn Pipeline)
  - Clinical MLP classifier

Usage (CLI)::

    python -m osteoporosis_pipeline.training.train_tabular --model tscore
    python -m osteoporosis_pipeline.training.train_tabular --model bmd
    python -m osteoporosis_pipeline.training.train_tabular --model mlp
"""

from __future__ import annotations

import argparse
import pickle
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

from osteoporosis_pipeline import config
from osteoporosis_pipeline.data.tabular_datasets import (
    load_clinical_excel,
    load_tscore_dataset,
    load_bmd_dataset,
)
from osteoporosis_pipeline.models.tabular_models import (
    build_tabular_mlp,
    build_tscore_rf,
    build_bmd_rf_pipeline,
)


# ─────────────────────────────────────────────────────────────
# T-score regressor
# ─────────────────────────────────────────────────────────────

def train_tscore_model(
    excel_path: str = config.CLINICAL_XLSX,
    model_out:  str = config.TSCORE_MODEL_PATH,
    scaler_out: str = config.TSCORE_SCALER_PATH,
    features_out: str = config.TSCORE_FEATURES_PATH,
) -> None:
    """
    Train and persist the T-score Random Forest regressor.

    Artefacts saved
    ---------------
    - rf_tscore_model.joblib   : fitted RandomForestRegressor
    - rf_tscore_scaler.joblib  : fitted StandardScaler
    - rf_tscore_features.joblib: list of feature names (in training order)
    """
    print("\n[INFO] Loading T-score dataset …")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = (
        load_tscore_dataset(excel_path)
    )

    print(f"[INFO] Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

    rf = build_tscore_rf()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[RESULT] T-score RF  RMSE={rmse:.4f}  MAE={mean_absolute_error(y_test, y_pred):.4f}"
          f"  R²={r2_score(y_test, y_pred):.4f}")

    joblib.dump(rf,           model_out)
    joblib.dump(scaler,       scaler_out)
    joblib.dump(feature_names, features_out)
    print(f"[INFO] Saved → {model_out}, {scaler_out}, {features_out}")


# ─────────────────────────────────────────────────────────────
# BMD regressor
# ─────────────────────────────────────────────────────────────

def train_bmd_model(
    csv_path:  str = config.BMD_CSV,
    model_out: str = config.BMD_MODEL_PATH,
    meta_out:  str = config.BMD_META_PATH,
) -> None:
    """
    Train and persist the BMD Random Forest pipeline.

    Artefacts saved
    ---------------
    - bmd_rf_model.joblib : fitted sklearn Pipeline (preprocessor + RF)
    - bmd_meta.joblib     : dict with 'categorical_cols', 'numerical_cols'
    """
    print("\n[INFO] Loading BMD dataset …")
    X_train, X_test, y_train, y_test = load_bmd_dataset(csv_path)

    pipeline = build_bmd_rf_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[RESULT] BMD RF  RMSE={rmse:.4f}  MAE={mean_absolute_error(y_test, y_pred):.4f}"
          f"  R²={r2_score(y_test, y_pred):.4f}")

    meta = {
        "categorical_cols": ["medication"],
        "numerical_cols":   ["age", "sex", "fracture", "weight_kg", "height_m", "bmi", "waiting_time"],
    }
    joblib.dump(pipeline, model_out)
    joblib.dump(meta,     meta_out)
    print(f"[INFO] Saved → {model_out}, {meta_out}")


# ─────────────────────────────────────────────────────────────
# Clinical MLP classifier
# ─────────────────────────────────────────────────────────────

def train_tabular_mlp(
    excel_path:  str = config.CLINICAL_XLSX,
    model_out:   str = config.TABULAR_MODEL_SAVE,
    scaler_pkl:  str = config.TABULAR_SCALER_PKL,
    epochs:      int = config.EPOCHS_TABULAR,
) -> tf.keras.Model:
    """
    Train and persist the tabular MLP classifier for disease diagnosis.

    Artefacts saved
    ---------------
    - tabular_mlp.h5                   : Keras model
    - tabular_scaler_feature_names.pkl : scaler + feature_names dict
    """
    print("\n[INFO] Loading clinical Excel …")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = (
        load_clinical_excel(excel_path)
    )

    # Class weights
    y_idx    = np.argmax(y_train, axis=1)
    classes  = np.unique(y_idx)
    cw_vals  = compute_class_weight("balanced", classes=classes, y=y_idx)
    cw       = {int(c): float(w) for c, w in zip(classes, cw_vals)}
    print(f"[INFO] Class weights: {cw}")

    model = build_tabular_mlp(input_dim=X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print("\n[INFO] Training tabular MLP …")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=cw,
        callbacks=callbacks,
    )

    print("\n[INFO] Test evaluation:")
    model.evaluate(X_test, y_test)

    model.save(model_out)

    with open(scaler_pkl, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_names": feature_names}, f)

    print(f"[INFO] Saved → {model_out}, {scaler_pkl}")
    return model


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular models")
    parser.add_argument("--model", choices=["tscore", "bmd", "mlp"], required=True)
    args = parser.parse_args()

    if args.model == "tscore":
        train_tscore_model()
    elif args.model == "bmd":
        train_bmd_model()
    else:
        train_tabular_mlp()