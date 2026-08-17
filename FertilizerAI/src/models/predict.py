"""Auto-generated file"""
import joblib
import pandas as pd


# ==============================
# MODEL PATHS
# ==============================

CLASSIFIER_PATH = "models/exported/fertilizer_classifier.pkl"

REGRESSOR_PATH = "models/exported/fertilizer_regressor.pkl"



# ==============================
# LOAD MODELS
# ==============================

def load_models():

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    regressor = joblib.load(
        REGRESSOR_PATH
    )

    return classifier, regressor



# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_fertilizer(input_data):


    classifier, regressor = load_models()



    # Convert input dictionary
    # into dataframe

    input_df = pd.DataFrame(
        [input_data]
    )



    # ------------------------------
    # STEP 1:
    # Predict Fertilizer
    # ------------------------------

    fertilizer_prediction = classifier.predict(
        input_df
    )[0]



    print(
        "\nRecommended Fertilizer:"
    )

    print(
        fertilizer_prediction
    )



    # Add fertilizer output
    # for regression model

    input_df["Primary_Fertilizer"] = (
        fertilizer_prediction
    )



    # ------------------------------
    # STEP 2:
    # Predict Mixing Parameters
    # ------------------------------

    mixing_prediction = regressor.predict(
        input_df
    )[0]



    result = {


        "Recommended_Fertilizer":
        fertilizer_prediction,


        "Primary_Amount_kg_per_hectare":
        round(
            mixing_prediction[0],
            2
        ),


        "Water_Liters":
        round(
            mixing_prediction[1],
            2
        ),


        "Pump1_ml":
        round(
            mixing_prediction[2],
            2
        ),


        "Pump2_ml":
        round(
            mixing_prediction[3],
            2
        ),


        "Pump3_ml":
        round(
            mixing_prediction[4],
            2
        )

    }



    return result




# ==============================
# TEST PREDICTION
# ==============================


if __name__ == "__main__":



    sample_input = {


        "District_Name":
        "Vellore",


        "Soil_color":
        "Red",


        "Nitrogen":
        80,


        "Phosphorus":
        40,


        "Potassium":
        50,


        "pH":
        6.5,


        "Rainfall":
        900,


        "Temperature":
        28,


        "Crop":
        "Rice",


        "Growth_Stage":
        "Vegetative"


    }



    prediction = predict_fertilizer(
        sample_input
    )



    print(
        "\n========== AI RESULT =========="
    )


    for key,value in prediction.items():

        print(
            f"{key}: {value}"
        )