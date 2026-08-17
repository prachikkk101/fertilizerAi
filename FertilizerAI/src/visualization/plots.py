import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv(
    "data/raw/fertilizer_full_recipe_dataset.csv"
)


target = "Pump1_ml"


plt.figure(figsize=(10,5))


df[target].value_counts().sort_index().plot(
    kind="bar"
)


plt.title(
    "Pump 1 Fertilizer Quantity Distribution"
)


plt.xlabel(
    "Pump1 ml"
)


plt.ylabel(
    "Count"
)


plt.tight_layout()


os.makedirs(
    "outputs/figures",
    exist_ok=True
)


plt.savefig(
    "outputs/figures/pump1_distribution.png"
)


plt.show()