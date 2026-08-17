import pandas as pd


RAW_PATH = "data/raw/fertilizer_full_recipe_dataset.csv"

PROCESSED_PATH = "data/processed/cleaned_dataset.csv"



def clean_data():

    df = pd.read_csv(RAW_PATH)


    print("Original Shape:")
    print(df.shape)



    # Handle missing categorical values

    categorical_columns = [
        "Secondary_Fertilizer",
        "Micronutrient"
    ]


    for col in categorical_columns:

        df[col] = df[col].fillna("None")



    # Remove unnecessary spaces

    for col in df.select_dtypes(
        include="object"
    ).columns:

        df[col] = df[col].str.strip()



    # Remove duplicates

    df.drop_duplicates(
        inplace=True
    )



    print("\nAfter Cleaning:")
    print(df.shape)



    # Save processed dataset

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )


    print(
        "\nSaved:",
        PROCESSED_PATH
    )



if __name__ == "__main__":

    clean_data()