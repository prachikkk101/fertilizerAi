import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/cleaned_dataset.csv"
OUTPUT_PATH = "data/processed/featured_dataset.csv"


def feature_engineering():

    # Load cleaned dataset
    df = pd.read_csv(INPUT_PATH)

    print("Original Shape:", df.shape)

    # ==========================================================
    # 1. Total NPK
    # ==========================================================
    df["Total_NPK"] = (
        df["Nitrogen"] +
        df["Phosphorus"] +
        df["Potassium"]
    )

    # ==========================================================
    # 2. Nutrient Ratios
    # ==========================================================
    df["N_P_Ratio"] = df["Nitrogen"] / (df["Phosphorus"] + 1)

    df["N_K_Ratio"] = df["Nitrogen"] / (df["Potassium"] + 1)

    df["P_K_Ratio"] = df["Phosphorus"] / (df["Potassium"] + 1)

    # ==========================================================
    # 3. Average Nutrient Level
    # ==========================================================
    df["Average_NPK"] = (
        df["Nitrogen"] +
        df["Phosphorus"] +
        df["Potassium"]
    ) / 3

    # ==========================================================
    # 4. Overall Deficiency Score
    # ==========================================================
    df["Total_Deficiency"] = (
        df["N_Deficiency"] +
        df["P_Deficiency"] +
        df["K_Deficiency"]
    )

    # ==========================================================
    # 5. Temperature-Rainfall Index
    # ==========================================================
    df["Weather_Index"] = (
        df["Temperature"] *
        df["Rainfall"]
    )

    # ==========================================================
    # 6. Water per kg Fertilizer
    # ==========================================================
    total_fertilizer = (
        df["Primary_Amount_kg_per_hectare"] +
        df["Secondary_Amount_kg_per_hectare"] +
        df["Micronutrient_kg_per_hectare"]
    )

    df["Water_per_kg"] = (
        df["Water_Liters"] /
        (total_fertilizer + 1)
    )

    # ==========================================================
    # 7. Pump Total
    # ==========================================================
    df["Total_Pump_ml"] = (
        df["Pump1_ml"] +
        df["Pump2_ml"] +
        df["Pump3_ml"]
    )

    # ==========================================================
    # 8. Pump Balance
    # ==========================================================
    df["Pump1_Percentage"] = (
        df["Pump1_ml"] /
        (df["Total_Pump_ml"] + 1)
    )

    df["Pump2_Percentage"] = (
        df["Pump2_ml"] /
        (df["Total_Pump_ml"] + 1)
    )

    df["Pump3_Percentage"] = (
        df["Pump3_ml"] /
        (df["Total_Pump_ml"] + 1)
    )

    # ==========================================================
    # 9. Categorical Combination
    # ==========================================================
    df["Crop_Soil"] = (
        df["Crop"] +
        "_" +
        df["Soil_color"]
    )

    df["Crop_Growth"] = (
        df["Crop"] +
        "_" +
        df["Growth_Stage"]
    )

    # ==========================================================
    # 10. EC Efficiency
    # ==========================================================
    df["EC_per_Water"] = (
        df["Target_EC_dS_m"] /
        (df["Water_Liters"] + 1)
    )

    print("New Shape:", df.shape)

    print("\nNew Features Added:")

    new_columns = [
        "Total_NPK",
        "N_P_Ratio",
        "N_K_Ratio",
        "P_K_Ratio",
        "Average_NPK",
        "Total_Deficiency",
        "Weather_Index",
        "Water_per_kg",
        "Total_Pump_ml",
        "Pump1_Percentage",
        "Pump2_Percentage",
        "Pump3_Percentage",
        "Crop_Soil",
        "Crop_Growth",
        "EC_per_Water"
    ]

    print(new_columns)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved Successfully")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    feature_engineering()