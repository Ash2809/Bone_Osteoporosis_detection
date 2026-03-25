"""
inference/predictors.py
------------------------
Individual model predictors and the meta-model fusion layer.

Each predictor is a thin wrapper that:
  1. Accepts raw inputs (image path / feature dict)
  2. Preprocesses them
  3. Returns both a numeric prediction and a risk score in [0, 1]

The MetaPredictor combines all three to yield a final diagnosis.
"""

from __future__ import annotations

import numpy as np
import joblib
import tensorflow as tf

from osteoporosis_pipeline import config


# ─────────────────────────────────────────────────────────────
# X-ray / image predictor
# ─────────────────────────────────────────────────────────────

class XRayPredictor:
    """Wraps the disease-classification CNN for single-image inference."""

    def __init__(self, model: tf.keras.Model, img_size: int = config.IMG_SIZE):
        self.model    = model
        self.img_size = img_size

    def predict(self, image_path: str) -> tuple[float, np.ndarray]:
        """
        Classify a single X-ray image.

        Parameters
        ----------
        image_path : path to the image file

        Returns
        -------
        risk_score : float in {0.0, 0.5, 1.0}  (normal / osteopenia / osteoporosis)
        raw_probs  : np.ndarray of shape (3,)
        """
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_size, self.img_size)
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        raw_probs = self.model.predict(img, verbose=0)[0]
        cls       = int(np.argmax(raw_probs))
        risk_map  = {0: 0.0, 1: 0.5, 2: 1.0}
        return risk_map[cls], raw_probs


# ─────────────────────────────────────────────────────────────
# T-score predictor
# ─────────────────────────────────────────────────────────────

class TScorePredictor:
    """Wraps the Random Forest T-score regressor."""

    def __init__(self, model, scaler, feature_names: list[str]):
        self.model         = model
        self.scaler        = scaler
        # strip any stray whitespace from stored feature names
        self.feature_names = [f.strip() for f in feature_names]

    @classmethod
    def from_files(
        cls,
        model_path:    str = config.TSCORE_MODEL_PATH,
        scaler_path:   str = config.TSCORE_SCALER_PATH,
        features_path: str = config.TSCORE_FEATURES_PATH,
    ) -> "TScorePredictor":
        return cls(
            model         = joblib.load(model_path),
            scaler        = joblib.load(scaler_path),
            feature_names = joblib.load(features_path),
        )

    def predict(self, features_dict: dict) -> tuple[float, float]:
        """
        Predict the T-score for a patient.

        Parameters
        ----------
        features_dict : {feature_name: value, …}
                        keys must match the training feature names

        Returns
        -------
        tscore_value : float   (predicted DXA T-score)
        risk_score   : float in {0.0, 0.5, 1.0}
        """
        # strip whitespace from input keys to guard against mismatches
        features_dict = {k.strip(): v for k, v in features_dict.items()}
        ordered = [features_dict[col] for col in self.feature_names]

        X          = np.array([ordered])
        X          = self.scaler.transform(X)
        tscore_val = float(self.model.predict(X)[0])

        if tscore_val <= config.TSCORE_OSTEOPOROSIS:
            risk = 1.0
        elif tscore_val <= config.TSCORE_OSTEOPENIA:
            risk = 0.5
        else:
            risk = 0.0

        return tscore_val, risk


# ─────────────────────────────────────────────────────────────
# BMD predictor
# ─────────────────────────────────────────────────────────────

class BMDPredictor:
    """Wraps the Random Forest BMD pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_file(cls, model_path: str = config.BMD_MODEL_PATH) -> "BMDPredictor":
        return cls(pipeline=joblib.load(model_path))

    @staticmethod
    def prepare_features(bmd_dict: dict) -> list:
        """
        Convert a raw BMD input dictionary to the ordered numeric feature list
        expected by the pre-pipeline predictor (used when the model is a bare RF,
        not a sklearn Pipeline).

        Parameters
        ----------
        bmd_dict : dict with keys:
            age, sex ("F"/"M"), fracture ("no fracture"/"fracture"),
            weight_kg, height_cm, medication, waiting_time

        Returns
        -------
        list of 10 numeric values in the order the bare RF was trained on
        """
        sex_bin      = 1 if bmd_dict["sex"].lower() == "f" else 0
        fracture_bin = 0 if "no" in bmd_dict["fracture"].lower() else 1

        height_m = bmd_dict["height_cm"] / 100.0
        bmi      = bmd_dict["weight_kg"] / (height_m ** 2)

        med          = bmd_dict["medication"].lower().strip()
        med_anticonv = int(med == "anticonvulsant")
        med_glucocort = int(med == "glucocorticoids")
        med_none     = int(med == "no medication")

        return [
            med_anticonv,
            med_glucocort,
            med_none,
            bmd_dict["age"],
            sex_bin,
            fracture_bin,
            bmd_dict["weight_kg"],
            height_m,
            bmi,
            bmd_dict["waiting_time"],
        ]

    def predict(self, bmd_features) -> tuple[float, float]:
        """
        Predict BMD value and risk score.

        Parameters
        ----------
        bmd_features : list | np.ndarray | pd.DataFrame
            Either a raw feature list (bare RF) or a DataFrame (Pipeline).

        Returns
        -------
        bmd_value  : float
        risk_score : float in {0.0, 0.5, 1.0}
        """
        if isinstance(bmd_features, list):
            bmd_features = [bmd_features]   # bare RF expects 2-D
        bmd_val = float(self.pipeline.predict(bmd_features)[0])

        if bmd_val < config.BMD_OSTEOPOROSIS:
            risk = 1.0
        elif bmd_val < config.BMD_OSTEOPENIA:
            risk = 0.5
        else:
            risk = 0.0

        return bmd_val, risk


# ─────────────────────────────────────────────────────────────
# Meta-model fusion
# ─────────────────────────────────────────────────────────────

def fuse_risks(
    xray_risk:   float,
    tscore_risk: float,
    bmd_risk:    float,
    xray_weight:   float = config.XRAY_WEIGHT,
    tscore_weight: float = config.TSCORE_WEIGHT,
    bmd_weight:    float = config.BMD_WEIGHT,
) -> tuple[str, float]:
    """
    Weighted fusion of the three risk scores into a final diagnosis.

    Parameters
    ----------
    xray_risk, tscore_risk, bmd_risk : risk scores from the individual predictors
    *_weight : fusion weights (should sum to 1.0)

    Returns
    -------
    diagnosis   : str  "Osteoporosis" | "Osteopenia" | "Normal"
    final_risk  : float  combined risk score
    """
    final_risk = xray_weight * xray_risk + tscore_weight * tscore_risk + bmd_weight * bmd_risk

    if final_risk >= config.OSTEOPOROSIS_THRESHOLD:
        return "Osteoporosis", final_risk
    elif final_risk >= config.OSTEOPENIA_THRESHOLD:
        return "Osteopenia", final_risk
    else:
        return "Normal", final_risk


class MetaPredictor:
    """
    High-level interface that runs all three sub-models and fuses their outputs.

    Example
    -------
    >>> meta = MetaPredictor(xray_predictor, tscore_predictor, bmd_predictor)
    >>> result = meta.predict(image_path, tscore_dict, bmd_dict)
    >>> print(result["Final Diagnosis"])
    """

    def __init__(
        self,
        xray_predictor:   XRayPredictor,
        tscore_predictor: TScorePredictor,
        bmd_predictor:    BMDPredictor,
    ):
        self.xray    = xray_predictor
        self.tscore  = tscore_predictor
        self.bmd     = bmd_predictor

    def predict(
        self,
        image_path:   str,
        tscore_dict:  dict,
        bmd_dict:     dict,
    ) -> dict:
        """
        Run the full pipeline for a single patient.

        Parameters
        ----------
        image_path   : path to knee/spine X-ray image
        tscore_dict  : clinical features for T-score model
        bmd_dict     : raw BMD input dictionary (see BMDPredictor.prepare_features)

        Returns
        -------
        Structured result dict with sub-model outputs and final diagnosis.
        """
        xray_risk, xray_probs   = self.xray.predict(image_path)
        tscore_val, tscore_risk = self.tscore.predict(tscore_dict)
        bmd_features            = BMDPredictor.prepare_features(bmd_dict)
        bmd_val, bmd_risk       = self.bmd.predict(bmd_features)
        diagnosis, final_risk   = fuse_risks(xray_risk, tscore_risk, bmd_risk)

        return {
            "X-ray": {
                "raw_probabilities": xray_probs.tolist(),
                "risk_score":        xray_risk,
            },
            "T-score": {
                "predicted_value": tscore_val,
                "risk_score":      tscore_risk,
            },
            "BMD": {
                "predicted_value": bmd_val,
                "risk_score":      bmd_risk,
            },
            "Final Diagnosis": {
                "label":               diagnosis,
                "combined_risk_score": final_risk,
            },
        }