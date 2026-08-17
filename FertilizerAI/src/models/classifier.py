"""Auto-generated file"""
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/featured_dataset.csv"

MODEL_PATH = "models/classifier.pkl"

PREPROCESSOR_PATH = "data/encoders/classifier_preprocessor.pkl"


def train_classifier():

    df = pd.read_csv(DATA_PATH)

    features = [
        "District_Name",
        "Soil_color",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH",
        "Rainfall",
        "Temperature",
        "Crop",
        "Growth_Stage"
    ]

    target = "Primary_Fertilizer"

    X = df[features]

    y = df[target]

    categorical = [
        "District_Name",
        "Soil_color",
        "Crop",
        "Growth_Stage"
    ]

    numerical = [
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH",
        "Rainfall",
        "Temperature"
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical),
            ("cat", categorical_transformer, categorical)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples :", X_train.shape[0])
    print("Testing samples  :", X_test.shape[0])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy")
    print(f"{accuracy:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/encoders", exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\nClassifier Saved Successfully")


if __name__ == "__main__":
    train_classifier()