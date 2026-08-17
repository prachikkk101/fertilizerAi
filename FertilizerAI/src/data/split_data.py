"""Auto-generated file"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


DATA_PATH = "data/processed/featured_dataset.csv"


def split_dataset():

    df = pd.read_csv(DATA_PATH)


    # First split
    train, temp = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["Primary_Fertilizer"]
    )


    # Validation + Test split

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["Primary_Fertilizer"]
    )


    os.makedirs(
        "data/processed/splits",
        exist_ok=True
    )


    train.to_csv(
        "data/processed/splits/train.csv",
        index=False
    )


    val.to_csv(
        "data/processed/splits/validation.csv",
        index=False
    )


    test.to_csv(
        "data/processed/splits/test.csv",
        index=False
    )


    print("Dataset Split Complete")


    print(
        "Train:",
        len(train)
    )

    print(
        "Validation:",
        len(val)
    )

    print(
        "Test:",
        len(test)
    )



if __name__=="__main__":

    split_dataset()