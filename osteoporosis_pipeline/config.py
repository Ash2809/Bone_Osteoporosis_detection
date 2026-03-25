# ──────────────────────────────────────────────
# Data paths
# ──────────────────────────────────────────────
KNEE_DATA_DIR  = "/Users/aashutoshkumar/Documents/Data/Knee_"
SPINE_DATA_DIR = "/Users/aashutoshkumar/Documents/Data/Spine"
CLINICAL_XLSX  = "/Users/aashutoshkumar/Documents/Projects/Capstone-Project/patient details.xlsx"
BMD_CSV        = "bmd.csv"

# ──────────────────────────────────────────────
# Saved model / artefact paths
# ──────────────────────────────────────────────
KNEE_DISEASE_WEIGHTS  = "/Users/aashutoshkumar/Documents/Projects/Capstone-Project/best_knee_disease_model.h5"
SPINE_DISEASE_WEIGHTS = "best_spine_disease_model.h5"
ANATOMY_MODEL_PATH    = "anatomy_classifier.h5"
IMAGE_MODEL_SAVE      = "/Users/aashutoshkumar/Documents/Projects/Bone_Osteoporosis_detection/osteoporosis_pipeline/artifacts/knee_cbam_resnet50.h5"
TABULAR_MODEL_SAVE    = "tabular_mlp.h5"

TSCORE_MODEL_PATH    = "tabular_model_output/rf_tscore_model.joblib"
TSCORE_SCALER_PATH   = "tabular_model_output/rf_tscore_scaler.joblib"
TSCORE_FEATURES_PATH = "tabular_model_output/rf_tscore_features.joblib"

BMD_MODEL_PATH = "/Users/aashutoshkumar/Documents/Projects/Capstone-Project/bmd_model_output/bmd_rf_model.joblib"
BMD_META_PATH  = "bmd_meta.joblib"

TABULAR_SCALER_PKL = "tabular_scaler_feature_names.pkl"

# ──────────────────────────────────────────────
# Dataset / model hyper-parameters
# ──────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
SEED       = 42

CLASS_NAMES_DISEASE = ["normal", "osteopenia", "osteoporosis"]
CLASS_NAMES_ANATOMY = ["knee", "spine"]
NUM_DISEASE_CLASSES = len(CLASS_NAMES_DISEASE)
NUM_ANATOMY_CLASSES = len(CLASS_NAMES_ANATOMY)

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
EPOCHS_IMAGE   = 30
EPOCHS_TABULAR = 40
LR_STAGE1      = 1e-4
LR_FINETUNE    = 5e-6
PATIENCE       = 10

# ──────────────────────────────────────────────
# Meta-model fusion weights
# ──────────────────────────────────────────────
XRAY_WEIGHT   = 0.5
TSCORE_WEIGHT = 0.3
BMD_WEIGHT    = 0.2

ENSEMBLE_IMG_WEIGHT = 0.6
ENSEMBLE_TAB_WEIGHT = 0.4

# ──────────────────────────────────────────────
# Diagnosis thresholds
# ──────────────────────────────────────────────
OSTEOPOROSIS_THRESHOLD = 0.70
OSTEOPENIA_THRESHOLD   = 0.35

TSCORE_OSTEOPOROSIS = -2.5
TSCORE_OSTEOPENIA   = -1.0

BMD_OSTEOPOROSIS = 0.7
BMD_OSTEOPENIA   = 0.9