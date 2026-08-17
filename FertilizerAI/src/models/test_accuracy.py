import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report
)



MODEL_PATH="models/best_classifier.pkl"



def evaluate(dataset_path):


    df=pd.read_csv(dataset_path)


    features=[

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


    X=df[features]


    y=df["Primary_Fertilizer"]



    model=joblib.load(
        MODEL_PATH
    )



    prediction=model.predict(X)



    accuracy=accuracy_score(
        y,
        prediction
    )


    print("\nDataset:",dataset_path)

    print(
        "Accuracy:",
        accuracy*100,
        "%"
    )


    print(
        classification_report(
            y,
            prediction
        )
    )




if __name__=="__main__":


    evaluate(
        "data/processed/splits/train.csv"
    )


    evaluate(
        "data/processed/splits/validation.csv"
    )


    evaluate(
        "data/processed/splits/test.csv"
    )