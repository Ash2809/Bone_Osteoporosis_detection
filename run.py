"""
meta_model.py
-------------
Full osteoporosis meta-model pipeline.

Loads three pre-trained models:
  1. X-ray CNN classifier        (ResNet50 + CBAM)
  2. T-score Random Forest       (tabular clinical features → T-score)
  3. BMD Random Forest           (patient demographics → BMD value)

Fuses their risk scores with weighted averaging to produce a final diagnosis:
  Normal / Osteopenia / Osteoporosis

Usage
-----
  # Run the built-in demo case:
  python meta_model.py

  # Import and call from another script:
  from meta_model import run_full_prediction
  result = run_full_prediction(image_path, tscore_dict, bmd_dict)
"""

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  – update these to match your local environment
# ─────────────────────────────────────────────────────────────────────────────

KNEE_WEIGHTS_PATH  = "/Users/aashutoshkumar/Documents/Projects/Capstone-Project/best_knee_disease_model.h5"
KNEE_TEST_DIR      = "/Users/aashutoshkumar/Documents/Data/Knee_/test"

TSCORE_MODEL_PATH    = r"tabular_model_output/rf_tscore_model.joblib"
TSCORE_SCALER_PATH   = r"tabular_model_output/rf_tscore_scaler.joblib"
TSCORE_FEATURES_PATH = r"tabular_model_output/rf_tscore_features.joblib"

BMD_MODEL_PATH = r"/Users/aashutoshkumar/Documents/Projects/Capstone-Project/bmd_model_output/bmd_rf_model.joblib"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE            = 224
CLASS_NAMES_DISEASE = ["normal", "osteopenia", "osteoporosis"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  X-RAY MODEL  –  architecture rebuild + weight loading
# ─────────────────────────────────────────────────────────────────────────────

def _cbam_block(feature_map, ratio=8):
    """Convolutional Block Attention Module (channel + spatial attention)."""
    channel = feature_map.shape[-1]

    # -- Channel attention --
    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    max_pool = layers.GlobalMaxPooling2D()(feature_map)

    mlp  = layers.Dense(channel // ratio, activation="relu")
    mlp2 = layers.Dense(channel)

    avg_out = mlp2(mlp(avg_pool))
    max_out = mlp2(mlp(max_pool))

    ch_att = layers.Add()([avg_out, max_out])
    ch_att = layers.Activation("sigmoid")(ch_att)
    ch_att = layers.Reshape((1, 1, channel))(ch_att)
    x = layers.Multiply()([feature_map, ch_att])

    # -- Spatial attention --
    avg_sp = layers.Lambda(
        lambda v: tf.reduce_mean(v, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1),
    )(x)
    max_sp = layers.Lambda(
        lambda v: tf.reduce_max(v, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1),
    )(x)

    concat  = layers.Concatenate()([avg_sp, max_sp])
    sp_att  = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    x       = layers.Multiply()([x, sp_att])

    return x


def _build_disease_classifier(img_size=IMG_SIZE):
    """Rebuild the ResNet50 + CBAM disease classifier (must match saved weights)."""
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inp = layers.Input((img_size, img_size, 3))
    x   = base(inp, training=False)
    x   = _cbam_block(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(3, activation="softmax")(x)

    return models.Model(inp, out)


def load_xray_model(weights_path=KNEE_WEIGHTS_PATH, img_size=IMG_SIZE):
    """
    Rebuild the disease classifier architecture and load saved weights.

    Parameters
    ----------
    weights_path : str   path to the .h5 weights file
    img_size     : int   must match the size used at training time

    Returns
    -------
    Compiled tf.keras.Model
    """
    model = _build_disease_classifier(img_size)
    model.load_weights(weights_path)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("X-ray model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(
    knee_weights=KNEE_WEIGHTS_PATH,
    tscore_model_path=TSCORE_MODEL_PATH,
    tscore_scaler_path=TSCORE_SCALER_PATH,
    tscore_features_path=TSCORE_FEATURES_PATH,
    bmd_model_path=BMD_MODEL_PATH,
):
    """
    Load and return all three sub-models.

    Returns
    -------
    xray_model   : tf.keras.Model
    tscore_model : sklearn RandomForestRegressor
    tscore_scaler: sklearn StandardScaler
    tscore_meta  : list[str]  – feature names in training order
    bmd_model    : sklearn RandomForestRegressor (or Pipeline)
    """
    xray_model = load_xray_model(knee_weights)

    tscore_model  = joblib.load(tscore_model_path)
    tscore_scaler = joblib.load(tscore_scaler_path)
    tscore_meta   = [col.strip() for col in joblib.load(tscore_features_path)]

    bmd_model = joblib.load(bmd_model_path)

    print("CLEANED T-score feature names:", tscore_meta)
    return xray_model, tscore_model, tscore_scaler, tscore_meta, bmd_model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INDIVIDUAL PREDICTORS
# ─────────────────────────────────────────────────────────────────────────────

def classify_xray(image_path, xray_model, img_size=IMG_SIZE):
    """
    Classify a single X-ray image.

    Parameters
    ----------
    image_path : str             path to the image
    xray_model : tf.keras.Model  loaded CNN
    img_size   : int

    Returns
    -------
    risk_score : float  0.0 (normal) | 0.5 (osteopenia) | 1.0 (osteoporosis)
    raw_probs  : np.ndarray shape (1, 3)
    """
    img  = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img  = tf.keras.preprocessing.image.img_to_array(img)
    img  = np.expand_dims(img, axis=0) / 255.0

    pred     = xray_model.predict(img, verbose=0)
    cls      = int(np.argmax(pred))
    risk_map = {0: 0.0, 1: 0.5, 2: 1.0}

    return risk_map[cls], pred


def predict_tscore(features_dict, tscore_model, tscore_scaler, tscore_meta):
    """
    Predict T-score from a clinical feature dictionary.

    Parameters
    ----------
    features_dict : dict  {feature_name: value}
    tscore_model  : fitted sklearn regressor
    tscore_scaler : fitted sklearn scaler
    tscore_meta   : list[str]  feature names in training order

    Returns
    -------
    tscore_value : float
    risk_score   : float  0.0 | 0.5 | 1.0
    """
    features_dict  = {k.strip(): v for k, v in features_dict.items()}
    ordered_values = [features_dict[col] for col in tscore_meta]

    X          = np.array([ordered_values])
    X          = tscore_scaler.transform(X)
    tscore_val = float(tscore_model.predict(X)[0])

    if tscore_val <= -2.5:
        risk = 1.0
    elif tscore_val <= -1.0:
        risk = 0.5
    else:
        risk = 0.0

    return tscore_val, risk


def prepare_bmd_features(bmd_dict):
    """
    Convert a raw BMD input dictionary to the ordered numeric feature list
    expected by the BMD Random Forest.

    Parameters
    ----------
    bmd_dict : dict  with keys:
        age, sex ("F"/"M"), fracture (str containing "no" or not),
        weight_kg, height_cm, medication (str), waiting_time

    Returns
    -------
    list of 10 floats/ints
    """
    sex_bin      = 1 if bmd_dict["sex"].lower() == "f" else 0
    fracture_bin = 0 if "no" in bmd_dict["fracture"].lower() else 1

    height_m = bmd_dict["height_cm"] / 100.0
    bmi      = bmd_dict["weight_kg"] / (height_m ** 2)

    med           = bmd_dict["medication"].lower().strip()
    med_anticonv  = int(med == "anticonvulsant")
    med_glucocort = int(med == "glucocorticoids")
    med_none      = int(med == "no medication")

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


def predict_bmd(bmd_features, bmd_model):
    """
    Predict BMD value from a numeric feature list.

    Parameters
    ----------
    bmd_features : list  output of prepare_bmd_features()
    bmd_model    : fitted sklearn model

    Returns
    -------
    bmd_value  : float
    risk_score : float  0.0 | 0.5 | 1.0
    """
    bmd_val = float(bmd_model.predict([bmd_features])[0])

    if bmd_val < 0.7:
        risk = 1.0
    elif bmd_val < 0.9:
        risk = 0.5
    else:
        risk = 0.0

    return bmd_val, risk


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FUSION  –  weighted risk combination → final diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def final_diagnosis(xray_risk, tscore_risk, bmd_risk,
                    xray_w=0.5, tscore_w=0.3, bmd_w=0.2):
    """
    Fuse three risk scores into a single diagnosis.

    Parameters
    ----------
    xray_risk, tscore_risk, bmd_risk : float  risk scores in [0, 1]
    xray_w, tscore_w, bmd_w          : float  weights (should sum to 1.0)

    Returns
    -------
    label      : str   "Osteoporosis" | "Osteopenia" | "Normal"
    final_risk : float combined risk score
    """
    final_risk = xray_w * xray_risk + tscore_w * tscore_risk + bmd_w * bmd_risk

    if final_risk >= 0.70:
        return "Osteoporosis", final_risk
    elif final_risk >= 0.35:
        return "Osteopenia", final_risk
    else:
        return "Normal", final_risk


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HIGH-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_prediction(image_path, tscore_dict, bmd_dict,
                        xray_model, tscore_model, tscore_scaler, tscore_meta, bmd_model):
    """
    Run the complete osteoporosis pipeline for a single patient.

    Parameters
    ----------
    image_path   : str   path to the X-ray image
    tscore_dict  : dict  clinical features for T-score model
    bmd_dict     : dict  raw patient data for BMD model
    xray_model, tscore_model, tscore_scaler, tscore_meta, bmd_model :
                         pre-loaded model objects from load_all_models()

    Returns
    -------
    dict with sub-model outputs and final diagnosis
    """
    # 1) X-ray
    xray_risk, xray_raw = classify_xray(image_path, xray_model)

    # 2) T-score
    tscore_value, tscore_risk = predict_tscore(
        tscore_dict, tscore_model, tscore_scaler, tscore_meta
    )

    # 3) BMD
    bmd_features_list    = prepare_bmd_features(bmd_dict)
    bmd_value, bmd_risk  = predict_bmd(bmd_features_list, bmd_model)

    # 4) Fusion
    diagnosis, final_risk = final_diagnosis(xray_risk, tscore_risk, bmd_risk)

    return {
        "X-ray": {
            "raw_prediction": xray_raw.tolist(),
            "risk_score":     xray_risk,
        },
        "T-score": {
            "predicted_value": tscore_value,
            "risk_score":      tscore_risk,
        },
        "BMD": {
            "predicted_value": bmd_value,
            "risk_score":      bmd_risk,
        },
        "Final Diagnosis": {
            "label":               diagnosis,
            "combined_risk_score": final_risk,
        },
    }


def test_single_case(image_path, clinical_features_dict, bmd_feature_list,
                     xray_model, tscore_model, tscore_scaler, tscore_meta, bmd_model):
    """
    Verbose single-case runner that prints intermediate results.

    Parameters
    ----------
    image_path            : str
    clinical_features_dict: dict  T-score model features
    bmd_feature_list      : list  already-prepared numeric BMD features
                            (output of prepare_bmd_features)
    *model args           : from load_all_models()

    Returns
    -------
    dict with all intermediate and final values
    """
    print("\n============== RUNNING TEST CASE ==============\n")

    xray_risk, xray_raw = classify_xray(image_path, xray_model)
    print("X-ray prediction raw:", xray_raw)
    print("X-ray risk score    :", xray_risk)

    tscore_pred, tscore_risk = predict_tscore(
        clinical_features_dict, tscore_model, tscore_scaler, tscore_meta
    )
    print("\nT-score predicted   :", tscore_pred)
    print("T-score risk        :", tscore_risk)

    bmd_pred, bmd_risk = predict_bmd(bmd_feature_list, bmd_model)
    print("\nBMD predicted       :", bmd_pred)
    print("BMD risk            :", bmd_risk)

    final_label, final_risk = final_diagnosis(xray_risk, tscore_risk, bmd_risk)
    print("\nFINAL DIAGNOSIS     :", final_label)
    print("Final combined risk :", final_risk)

    return {
        "final_label":  final_label,
        "final_risk":   final_risk,
        "xray_risk":    xray_risk,
        "tscore_pred":  tscore_pred,
        "tscore_risk":  tscore_risk,
        "bmd_pred":     bmd_pred,
        "bmd_risk":     bmd_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CNN EVALUATION  (on the full test set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_xray_model(xray_model, test_dir=KNEE_TEST_DIR,
                        img_size=IMG_SIZE, batch_size=16):
    """
    Evaluate the X-ray CNN on an image directory and print a full report.

    Parameters
    ----------
    xray_model : tf.keras.Model
    test_dir   : str  directory with class sub-folders
    img_size   : int
    batch_size : int

    Returns
    -------
    dict with 'accuracy', 'confusion_matrix', 'classification_report'
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds  = xray_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES_DISEASE)

    print("CNN Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": report}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO / ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

# Example patient data (from the original notebook cell 6)
EXAMPLE_TSCORE_DICT = {
    "Age": 58,
    "Gender_bin": 0,
    "height  (meter)": 1.58,
    "Weight (KG)": 62,
    "Smoker_bin": 0,
    "Alcoholic_bin": 0,
    "Diabetic_bin": 0,
    "Hypothyroidism_bin": 1,
    "Number of Pregnancies": 2,
    "Seizer Disorder_bin": 0,
    "Estrogen Use_bin": 1,
    "Family History of Osteoporosis_bin": 1,
    "Joint Pain:_bin": 1,
    "Dialysis:_bin": 0,
    "Menopause Age": 51,
    "Maximum Walking distance (km)": 1.5,
    "BMI:": 24.83,
    "history_of_fracture": 0,
}

EXAMPLE_BMD_DICT = {
    "id": 1001,
    "age": 58,
    "sex": "F",
    "fracture": "no fracture",
    "weight_kg": 62,
    "height_cm": 158,
    "medication": "No medication",
    "waiting_time": 18,
}

EXAMPLE_IMAGE_PATH = (
    "/Users/aashutoshkumar/Documents/Data/Knee_/train/osteopenia/Osteopenia 8.jpg"
)


if __name__ == "__main__":
    # Load all models
    xray_model, tscore_model, tscore_scaler, tscore_meta, bmd_model = load_all_models()

    # Run full prediction on the demo patient
    result = run_full_prediction(
        image_path   = EXAMPLE_IMAGE_PATH,
        tscore_dict  = EXAMPLE_TSCORE_DICT,
        bmd_dict     = EXAMPLE_BMD_DICT,
        xray_model   = xray_model,
        tscore_model = tscore_model,
        tscore_scaler= tscore_scaler,
        tscore_meta  = tscore_meta,
        bmd_model    = bmd_model,
    )

    import json
    print("\n===== PREDICTION RESULT =====")
    print(json.dumps(result, indent=2))

    # Evaluate CNN on the full test set
    evaluate_xray_model(xray_model, test_dir=KNEE_TEST_DIR)