import pandas as pd
import joblib
import os


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)


from sklearn.pipeline import Pipeline


from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import OneHotEncoder


from sklearn.impute import SimpleImputer


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


from sklearn.metrics import (
    accuracy_score,
    r2_score
)



DATA_PATH = "data/processed/featured_dataset.csv"



def create_preprocessor(
        categorical,
        numerical
):


    numeric_pipeline = Pipeline([

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



    return ColumnTransformer([

        (
            "num",
            numeric_pipeline,
            numerical
        ),

        (
            "cat",
            categorical_pipeline,
            categorical
        )

    ])




# ==========================================
# CLASSIFIER TUNING
# ==========================================


def tune_classifier():


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



    preprocessor = create_preprocessor(
        categorical,
        numerical
    )



    model = RandomForestClassifier()



    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            model
        )

    ])



    params = {


        "classifier__n_estimators":

        [
            100,
            300,
            500
        ],



        "classifier__max_depth":

        [
            None,
            10,
            20,
            30
        ],



        "classifier__min_samples_split":

        [
            2,
            5,
            10
        ]

    }



    X_train,X_test,y_train,y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    grid = GridSearchCV(

        pipeline,

        params,

        cv=5,

        scoring="accuracy",

        n_jobs=-1,

        verbose=2

    )



    grid.fit(

        X_train,

        y_train

    )



    print("\nBEST PARAMETERS")

    print(
        grid.best_params_
    )



    prediction = grid.predict(

        X_test

    )


    accuracy = accuracy_score(

        y_test,

        prediction

    )


    print(
        "\nAccuracy:",
        accuracy
    )



    os.makedirs(
        "models",
        exist_ok=True
    )



    joblib.dump(

        grid.best_estimator_,

        "models/best_classifier.pkl"

    )


    print(
        "Best Classifier Saved"
    )





# ==========================================
# REGRESSOR TUNING
# ==========================================


def tune_regressor():


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



    X=df[features]

    y=df[targets]



    categorical=[

        "District_Name",
        "Soil_color",
        "Crop",
        "Growth_Stage",
        "Primary_Fertilizer"

    ]



    numerical=[

        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH",
        "Rainfall",
        "Temperature"

    ]



    preprocessor=create_preprocessor(

        categorical,

        numerical

    )



    model=RandomForestRegressor()



    pipeline=Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            model
        )

    ])




    params={


        "regressor__n_estimators":

        [
            100,
            300,
            500
        ],



        "regressor__max_depth":

        [
            None,
            10,
            20
        ]

    }




    X_train,X_test,y_train,y_test=train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )



    grid=GridSearchCV(

        pipeline,

        params,

        cv=5,

        scoring="r2",

        n_jobs=-1,

        verbose=2

    )



    grid.fit(

        X_train,

        y_train

    )



    print(
        "\nBEST PARAMETERS"
    )

    print(
        grid.best_params_
    )



    prediction=grid.predict(

        X_test

    )



    score=r2_score(

        y_test,

        prediction

    )



    print(
        "\nR2 Score:",
        score
    )



    joblib.dump(

        grid.best_estimator_,

        "models/best_regressor.pkl"

    )


    print(
        "Best Regressor Saved"
    )




if __name__=="__main__":


    print(
        """
        1. Tune Classifier
        2. Tune Regressor
        """
    )


    choice=input(
        "Enter choice: "
    )


    if choice=="1":

        tune_classifier()


    elif choice=="2":

        tune_regressor()


    else:

        print(
            "Invalid Choice"
        )