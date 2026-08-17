import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np


# ==============================
# PATHS
# ==============================

DATA_PATH = "data/processed/featured_dataset.csv"

MODEL_PATH = "models/best_classifier.pkl"

OUTPUT_PATH = "outputs/figures"



# ==============================
# SHAP CLASSIFIER EXPLANATION
# ==============================


def explain_classifier():


    print("Loading Model...")


    pipeline = joblib.load(
        MODEL_PATH
    )


    df = pd.read_csv(
        DATA_PATH
    )



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



    X = df[features]



    # Sample data for SHAP

    X_sample = X.sample(

        100,

        random_state=42

    )



    print("Transforming Data...")



    # Get preprocessing pipeline

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]



    # Get classifier model

    model = pipeline.named_steps[
        "classifier"
    ]



    # ==============================
    # ENCODE FEATURES
    # ==============================


    X_encoded = preprocessor.transform(
        X_sample
    )



    # Convert sparse matrix

    if hasattr(
        X_encoded,
        "toarray"
    ):

        X_encoded = X_encoded.toarray()



    # Force numeric type

    X_encoded = X_encoded.astype(
        np.float64
    )



    print(
        "Encoded Shape:",
        X_encoded.shape
    )



    # ==============================
    # FEATURE NAMES
    # ==============================


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )



    X_encoded_df = pd.DataFrame(

        X_encoded,

        columns=feature_names

    )



    print(
        "Creating SHAP Explanation..."
    )



    # ==============================
    # SHAP
    # ==============================


    explainer = shap.TreeExplainer(
        model
    )



    shap_values = explainer.shap_values(
        X_encoded_df
    )



    print(
        "SHAP Generated"
    )



    os.makedirs(

        OUTPUT_PATH,

        exist_ok=True

    )



    # ==============================
    # HANDLE SHAP OUTPUT
    # ==============================


    if isinstance(
        shap_values,
        list
    ):

        # Multiclass old SHAP format

        shap_values_plot = shap_values[0]


    elif len(shap_values.shape) == 3:

        # New SHAP multiclass format

        shap_values_plot = shap_values[:, :, 0]


    else:

        shap_values_plot = shap_values



    print(
        "Generating Plot..."
    )



    plt.figure(
        figsize=(12,8)
    )


    shap.summary_plot(

        shap_values_plot,

        X_encoded_df,

        show=False

    )



    plt.tight_layout()



    plt.savefig(

        f"{OUTPUT_PATH}/classifier_shap.png",

        dpi=300,

        bbox_inches="tight"

    )



    plt.close()



    print(
        "SHAP Plot Saved Successfully"
    )




# ==============================


if __name__ == "__main__":

    explain_classifier()