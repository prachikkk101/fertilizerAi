import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

if __name__=="__main__":
    load_dataset(
        "data/raw/fertilizer_full_recipe_dataset.csv"
    )
