"""Fertilizer AI - Model Evaluation"""


import os
import json
import joblib
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# ==========================================
# PATHS
# ==========================================


DATA_PATH = "data/processed/featured_dataset.csv"


CLASSIFIER_PATH = "models/best_classifier.pkl"


REGRESSOR_PATH = "models/best_regressor.pkl"





# ==========================================
# CLASSIFIER EVALUATION
# ==========================================


def evaluate_classifier():


    print("\n========== CLASSIFIER EVALUATION ==========\n")


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




    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )




    model = joblib.load(

        CLASSIFIER_PATH

    )




    prediction = model.predict(

        X_test

    )




    accuracy = accuracy_score(

        y_test,

        prediction

    )




    report = classification_report(

        y_test,

        prediction,

        output_dict=True,

        zero_division=0

    )



    print("Accuracy:", round(accuracy,4))


    print(

        classification_report(

            y_test,

            prediction,

            zero_division=0

        )

    )




    # ==============================
    # CONFUSION MATRIX
    # ==============================


    cm = confusion_matrix(

        y_test,

        prediction

    )



    os.makedirs(

        "outputs/figures",

        exist_ok=True

    )



    plt.figure(

        figsize=(10,8)

    )



    plt.imshow(cm)



    plt.title(

        "Fertilizer Classification Confusion Matrix"

    )



    plt.xlabel(

        "Predicted"

    )



    plt.ylabel(

        "Actual"

    )



    plt.colorbar()



    plt.tight_layout()



    plt.savefig(

        "outputs/figures/confusion_matrix.png"

    )



    plt.close()




    # ==============================
    # SAVE METRICS
    # ==============================


    os.makedirs(

        "outputs/metrics",

        exist_ok=True

    )




    metrics = {

        "accuracy": accuracy,

        "classification_report": report

    }



    with open(

        "outputs/metrics/classifier_metrics.json",

        "w"

    ) as file:


        json.dump(

            metrics,

            file,

            indent=4

        )




    print("\nClassifier Metrics Saved")







# ==========================================
# REGRESSOR EVALUATION
# ==========================================



def evaluate_regressor():


    print("\n========== REGRESSOR EVALUATION ==========\n")



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

        "Growth_Stage",

        "Primary_Fertilizer"

    ]




    targets = [

        "Primary_Amount_kg_per_hectare",

        "Water_Liters",

        "Pump1_ml",

        "Pump2_ml",

        "Pump3_ml"

    ]




    X = df[features]

    y = df[targets]





    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )




    model = joblib.load(

        REGRESSOR_PATH

    )




    prediction = model.predict(

        X_test

    )




    results = {}




    for i,target in enumerate(targets):



        actual = y_test.iloc[:,i]


        predicted = prediction[:,i]




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





        results[target] = {


            "MAE": mae,

            "RMSE": rmse,

            "R2": r2

        }




        print("--------------------------------")


        print(target)


        print("MAE :", round(mae,4))


        print("RMSE:", round(rmse,4))


        print("R2  :", round(r2,4))





    os.makedirs(

        "outputs/metrics",

        exist_ok=True

    )




    with open(

        "outputs/metrics/regressor_metrics.json",

        "w"

    ) as file:


        json.dump(

            results,

            file,

            indent=4

        )




    print("\nRegressor Metrics Saved")






# ==========================================
# MAIN
# ==========================================


if __name__ == "__main__":


    print(

        """

        1. Evaluate Classifier

        2. Evaluate Regressor

        """

    )



    choice = input(

        "Enter choice: "

    )




    if choice == "1":


        evaluate_classifier()



    elif choice == "2":


        evaluate_regressor()



    else:


        print(

            "Invalid Choice"

        )