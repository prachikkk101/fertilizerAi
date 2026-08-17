"""Fertilizer AI - Regression Model Training"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.multioutput import MultiOutputRegressor


# ==============================
# PATHS
# ==============================

DATA_PATH = "data/processed/featured_dataset.csv"

MODEL_PATH = "models/regressor.pkl"



def train_regressor():


    # ==============================
    # LOAD DATA
    # ==============================

    print("\nLoading Dataset...")

    df = pd.read_csv(DATA_PATH)


    print("Dataset Shape:", df.shape)



    # ==============================
    # INPUT FEATURES
    # ==============================


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
        "Growth_Stage",

        "Primary_Fertilizer"

    ]



    X = df[features]



    # ==============================
    # TARGET OUTPUTS
    # ==============================


    targets = [

        "Primary_Amount_kg_per_hectare",

        "Water_Liters",

        "Pump1_ml",
        "Pump2_ml",
        "Pump3_ml"

    ]


    y = df[targets]



    # ==============================
    # COLUMN TYPES
    # ==============================


    categorical = [

        "District_Name",

        "Soil_color",

        "Crop",

        "Growth_Stage",

        "Primary_Fertilizer"

    ]


    numerical = [

        "Nitrogen",

        "Phosphorus",

        "Potassium",

        "pH",

        "Rainfall",

        "Temperature"

    ]



    # ==============================
    # PREPROCESSING
    # ==============================


    numerical_pipeline = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )

    ])



    categorical_pipeline = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),


        (
            "encoder",

            OneHotEncoder(

                handle_unknown="ignore"

            )

        )

    ])




    preprocessor = ColumnTransformer([


        (
            "numerical",

            numerical_pipeline,

            numerical

        ),


        (
            "categorical",

            categorical_pipeline,

            categorical

        )

    ])




    # ==============================
    # RANDOM FOREST REGRESSOR
    # ==============================


    regressor = MultiOutputRegressor(

        RandomForestRegressor(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        )

    )




    # COMPLETE PIPELINE


    pipeline = Pipeline([


        (
            "preprocessor",

            preprocessor

        ),


        (
            "model",

            regressor

        )

    ])




    # ==============================
    # TRAIN TEST SPLIT
    # ==============================


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )



    print("\nTraining Samples:", X_train.shape)

    print("Testing Samples :", X_test.shape)



    # ==============================
    # TRAIN MODEL
    # ==============================


    print("\nTraining Regression Model...\n")


    pipeline.fit(

        X_train,

        y_train

    )



    # ==============================
    # PREDICTION
    # ==============================


    predictions = pipeline.predict(

        X_test

    )



    # ==============================
    # EVALUATION
    # ==============================


    print("\n========== Regression Results ==========\n")



    for i, col in enumerate(targets):


        actual = y_test.iloc[:, i]


        predicted = predictions[:, i]



        mae = mean_absolute_error(

            actual,

            predicted

        )



        # FIXED RMSE

        mse = mean_squared_error(

            actual,

            predicted

        )


        rmse = mse ** 0.5



        r2 = r2_score(

            actual,

            predicted

        )



        print("--------------------------------")

        print("Target:", col)

        print("MAE  :", round(mae,4))

        print("RMSE :", round(rmse,4))

        print("R2   :", round(r2,4))



    # ==============================
    # SAVE MODEL
    # ==============================


    os.makedirs(

        "models",

        exist_ok=True

    )



    joblib.dump(

        pipeline,

        MODEL_PATH

    )



    print("\n================================")

    print("Regression Model Saved Successfully")

    print("Location:", MODEL_PATH)

    print("================================")





if __name__ == "__main__":

    train_regressor()