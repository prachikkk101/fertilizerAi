import pandas as pd


def perform_eda(df):

    print("\n========== INFO ==========")

    print(df.info())


    print("\n========== STATISTICS ==========")

    print(df.describe())


    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())


    print("\n========== DUPLICATES ==========")

    print(df.duplicated().sum())



if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/fertilizer_full_recipe_dataset.csv"
    )

    perform_eda(df)
