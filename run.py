"""
run.py
------
Top-level entry point demonstrating the full inference pipeline.

Usage::

    python run.py --image /path/to/knee.jpg

All model paths and data paths are configured in config.py.
"""

import argparse
import json

from osteoporosis_pipeline.utils.model_loader import load_meta_predictor


# ─────────────────────────────────────────────────────────────
# Example patient data  (replace with real input)
# ─────────────────────────────────────────────────────────────

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
    "age": 58,
    "sex": "F",
    "fracture": "no fracture",
    "weight_kg": 62,
    "height_cm": 158,
    "medication": "No medication",
    "waiting_time": 18,
}


def main():
    parser = argparse.ArgumentParser(description="Osteoporosis detection pipeline")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    args = parser.parse_args()

    print("\n[INFO] Loading all models …")
    meta = load_meta_predictor()

    print("\n[INFO] Running prediction …")
    result = meta.predict(
        image_path=args.image,
        tscore_dict=EXAMPLE_TSCORE_DICT,
        bmd_dict=EXAMPLE_BMD_DICT,
    )

    print("\n" + "=" * 50)
    print("  DIAGNOSIS RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()