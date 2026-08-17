import os
import json
import joblib
import pandas as pd


CLASSIFIER_PATH = "models/best_classifier.pkl"

REGRESSOR_PATH = "models/best_regressor.pkl"


EXPORT_FOLDER = "models/exported"



def export_models():


    print("Starting Model Export...")


    os.makedirs(
        EXPORT_FOLDER,
        exist_ok=True
    )


    # ==================================
    # Copy Classifier
    # ==================================


    classifier = joblib.load(
        CLASSIFIER_PATH
    )


    joblib.dump(

        classifier,

        f"{EXPORT_FOLDER}/fertilizer_classifier.pkl"

    )



    print(
        "Classifier Exported"
    )



    # ==================================
    # Copy Regressor
    # ==================================


    regressor = joblib.load(

        REGRESSOR_PATH

    )


    joblib.dump(

        regressor,

        f"{EXPORT_FOLDER}/fertilizer_regressor.pkl"

    )


    print(
        "Regressor Exported"
    )



    # ==================================
    # Save Feature Information
    # ==================================


    classifier_features = [

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



    regressor_features = [

        "District_Name",
        "Soil_color",

        "Nitrogen",
        "Phosphorus",
        "Potassium",

        "pH",

        "Rainfall",
        "Temperature",

        "Crop",
        "Growth_Stage",

        "Primary_Fertilizer"

    ]



    features = {


        "classifier_features":
        classifier_features,


        "regressor_features":
        regressor_features

    }



    with open(

        f"{EXPORT_FOLDER}/feature_columns.json",

        "w"

    ) as file:


        json.dump(

            features,

            file,

            indent=4

        )



    print(
        "Feature information saved"
    )



    # ==================================
    # Model Metadata
    # ==================================


    metadata = {


        "project":
        "AI Based Fertilizer Mixing System",


        "models":

        {


            "classifier":
            "RandomForestClassifier",


            "regressor":
            "RandomForestRegressor"

        },


        "version":
        "1.0",


        "purpose":
        "Fertilizer recommendation and automated mixing"

    }



    with open(

        f"{EXPORT_FOLDER}/model_metadata.json",

        "w"

    ) as file:


        json.dump(

            metadata,

            file,

            indent=4

        )



    print(
        "Metadata saved"
    )



    print(
        "\nModel Export Completed"
    )




if __name__ == "__main__":

    export_models()