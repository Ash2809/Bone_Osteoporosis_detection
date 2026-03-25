"""
data/tabular_datasets.py
-------------------------
Preprocessing pipelines for:
  - Clinical Excel → multi-class classification dataset (diagnosis)
  - Clinical Excel → T-score regression dataset
  - BMD CSV → regression dataset
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from osteoporosis_pipeline import config


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

_BINARY_MAP = {
    "yes": 1, "y": 1, "true": 1, "1": 1,
    "no": 0,  "n": 0, "false": 0, "0": 0,
    "male": 1, "female": 0,
}


def _encode_binary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add a `<col>_bin` column and drop the original."""
    df[col + "_bin"] = (
        df[col].astype(str).str.strip().str.lower().map(_BINARY_MAP).fillna(0)
    )
    return df.drop(columns=[col])


# ─────────────────────────────────────────────────────────────
# Clinical Excel → classification dataset
# ─────────────────────────────────────────────────────────────

def load_clinical_excel(
    path: str = config.CLINICAL_XLSX,
    label_column: str = "Diagnosis",
    test_size: float = 0.175,
    val_size: float = 0.175,
    random_state: int = config.SEED,
) -> tuple:
    """
    Load and preprocess the clinical Excel for 3-class disease classification.

    Processing steps:
      1. Drop identifier / free-text columns
      2. Map diagnosis labels to integer indices
      3. Parse numeric columns
      4. One-hot encode low-cardinality categoricals
      5. Impute missing values with column mean
      6. StandardScaler normalisation
      7. Stratified train / val / test split

    Parameters
    ----------
    path         : path to the Excel file
    label_column : name of the diagnosis column
    test_size    : proportion for test set
    val_size     : proportion for validation set (of the full dataset)
    random_state : reproducibility seed

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names
    where y_* are one-hot categorical arrays.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # 1. Drop identifier / free-text columns
    drop_cols = ["S.No", "Patient Id", "Medical History", "Occupation ", "Daily Eating habits"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2. Clean and encode labels
    df = df.dropna(subset=[label_column])
    df[label_column] = df[label_column].str.lower().str.strip()
    label_map = {"normal": 0, "osteopenia": 1, "osteoporosis": 2}
    df[label_column] = df[label_column].map(label_map)
    df = df.dropna(subset=[label_column])

    # 3. Parse numeric columns
    numeric_cols = [
        "Age", "Menopause Age", "height  (meter)", "Weight (KG) ",
        "Number of Pregnancies", "Maximum Walking distance (km)",
        "T-score Value", "Z-Score Value", "BMI: ",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. One-hot encode low-cardinality object columns
    small_cat = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() <= 6]
    df = pd.get_dummies(df, columns=small_cat)

    # 5. Fill remaining NaN with column means
    df = df.fillna(df.mean(numeric_only=True))

    y = df[label_column].values.astype(int)
    X = df.drop(columns=[label_column]).select_dtypes(include=[np.number]).values
    feature_names = df.drop(columns=[label_column]).select_dtypes(include=[np.number]).columns.tolist()

    # 6. Normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 7. Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size / (test_size + val_size),
        stratify=y_temp,
        random_state=random_state,
    )

    y_train = to_categorical(y_train, 3)
    y_val   = to_categorical(y_val, 3)
    y_test  = to_categorical(y_test, 3)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names


# ─────────────────────────────────────────────────────────────
# Clinical Excel → T-score regression dataset
# ─────────────────────────────────────────────────────────────

def load_tscore_dataset(
    path: str = config.CLINICAL_XLSX,
    random_state: int = config.SEED,
) -> tuple:
    """
    Load and preprocess the clinical Excel for T-score regression.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Detect T-score column
    target_col = _find_tscore_column(df)
    df = df.dropna(subset=[target_col])

    # Drop identifiers and leakage columns
    drop_cols = ["S.No", "Patient Id", "Occupation", "Diagnosis", "Z-Score Value", "Site"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fracture history → binary
    if "History of Fracture" in df.columns:
        df["history_of_fracture"] = df["History of Fracture"].apply(
            lambda x: 0 if str(x).strip().lower() in ["no", "none", "nan"] else 1
        )
        df = df.drop(columns=["History of Fracture"])

    # Binary encode yes/no, gender columns
    binary_cols = [
        "Smoker", "Alcoholic", "Diabetic", "Hypothyroidism",
        "Seizer Disorder", "Estrogen Use", "Family History of Osteoporosis",
        "Gender", "Joint Pain:", "Dialysis:",
    ]
    for col in binary_cols:
        if col in df.columns:
            df = _encode_binary(df, col)

    # Drop remaining free-text
    if "Medical History" in df.columns:
        df = df.drop(columns=["Medical History"])

    df = df.fillna(method="ffill").fillna(0)
    num_df = df.select_dtypes(include=[np.number])

    y = num_df[target_col].values
    X = num_df.drop(columns=[target_col]).values
    feature_names = num_df.drop(columns=[target_col]).columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=random_state)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names


def _find_tscore_column(df: pd.DataFrame) -> str:
    """Return the first column name that looks like a T-score column."""
    candidates = [
        "T-score Value", "T score", "T-score", "Tscore",
        "T score Value", "Tscore Value", "T-Score Value", "T Score Value", "T-Score",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if re.search(r"\bt[- ]?score\b", c, flags=re.I):
            return c
    raise ValueError(
        "Cannot locate T-score column. Available columns:\n" + ", ".join(df.columns.tolist())
    )


# ─────────────────────────────────────────────────────────────
# BMD CSV → regression dataset
# ─────────────────────────────────────────────────────────────

def load_bmd_dataset(
    path: str = config.BMD_CSV,
    random_state: int = config.SEED,
) -> tuple:
    """
    Load and preprocess the BMD CSV for BMD regression.

    Expected columns: id, age, sex, fracture, weight_kg, height_cm,
                      medication, waiting_time, bmd

    Returns
    -------
    X_df    : pd.DataFrame with categorical + numeric columns (for pipeline)
    y       : np.ndarray of BMD values
    X_train, X_test, y_train, y_test  (80/20 split)
    """
    df = pd.read_csv(path)

    df["sex"]      = df["sex"].map({"M": 0, "F": 1})
    df["fracture"] = df["fracture"].apply(lambda x: 0 if "no" in str(x).lower() else 1)
    df["height_m"] = df["height_cm"] / 100.0
    df["bmi"]      = df["weight_kg"] / (df["height_m"] ** 2)

    categorical = ["medication"]
    numerical   = ["age", "sex", "fracture", "weight_kg", "height_m", "bmi", "waiting_time"]

    X = df[categorical + numerical]
    y = df["bmd"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    return X_train, X_test, y_train, y_test