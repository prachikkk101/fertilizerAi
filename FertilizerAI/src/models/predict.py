"""Fertilizer AI Prediction Module"""

from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "exported"

CLASSIFIER_PATH = MODEL_DIR / "fertilizer_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "fertilizer_regressor.pkl"


def load_models():
    """Load the trained classification and regression models."""

    if not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_PATH}"
        )

    if not REGRESSOR_PATH.is_file():
        raise FileNotFoundError(
            f"Regressor model not found: {REGRESSOR_PATH}"
        )

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    return classifier, regressor


def predict_fertilizer(input_data):
    """Predict fertilizer and mixing parameters."""

    classifier, regressor = load_models()

    input_df = pd.DataFrame([input_data])

    fertilizer_prediction = classifier.predict(input_df)[0]

    print("\nRecommended Fertilizer:")
    print(fertilizer_prediction)

    input_df["Primary_Fertilizer"] = fertilizer_prediction

    mixing_prediction = regressor.predict(input_df)[0]

    if len(mixing_prediction) < 5:
        raise ValueError(
            "The regression model must return at least five values: "
            "fertilizer amount, water, Pump1, Pump2, and Pump3."
        )

    return {
        "Recommended_Fertilizer": str(fertilizer_prediction),
        "Primary_Amount_kg_per_hectare": round(float(mixing_prediction[0]), 2),
        "Water_Liters": round(float(mixing_prediction[1]), 2),
        "Pump1_ml": round(float(mixing_prediction[2]), 2),
        "Pump2_ml": round(float(mixing_prediction[3]), 2),
        "Pump3_ml": round(float(mixing_prediction[4]), 2),
    }


if __name__ == "__main__":
    sample_input = {
        "District_Name": "Vellore",
        "Soil_color": "Red",
        "Nitrogen": 80,
        "Phosphorus": 40,
        "Potassium": 50,
        "pH": 6.5,
        "Rainfall": 900,
        "Temperature": 28,
        "Crop": "Rice",
        "Growth_Stage": "Vegetative",
    }

    prediction = predict_fertilizer(sample_input)

    print("\n========== AI RESULT ==========")
    for key, value in prediction.items():
        print(f"{key}: {value}")
