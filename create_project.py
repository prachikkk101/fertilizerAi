import os

# ==========================================================
# AI Fertilizer Mixing System Project Structure Generator
# ==========================================================

PROJECT_NAME = "FertilizerAI"

folders = [

    # Root folders
    "api",
    "configs",
    "data",
    "docs",
    "logs",
    "models",
    "notebooks",
    "outputs",
    "src",
    "tests",

    # Data
    "data/raw",
    "data/processed",
    "data/external",
    "data/encoders",

    # Outputs
    "outputs/figures",
    "outputs/reports",
    "outputs/predictions",
    "outputs/metrics",

    # Source
    "src/data",
    "src/features",
    "src/models",
    "src/pipelines",
    "src/utils",
    "src/visualization",
]

files = [

    # Root
    "README.md",
    "requirements.txt",
    ".gitignore",

    # Configs
    "configs/config.yaml",
    "configs/params.yaml",

    # Logs
    "logs/training.log",

    # API
    "api/main.py",
    "api/routes.py",
    "api/schemas.py",

    # Notebooks
    "notebooks/01_EDA.ipynb",
    "notebooks/02_Preprocessing.ipynb",
    "notebooks/03_Model_Training.ipynb",
    "notebooks/04_Model_Evaluation.ipynb",

    # Source -> data
    "src/data/load_data.py",
    "src/data/clean_data.py",
    "src/data/split_data.py",

    # Source -> features
    "src/features/encoding.py",
    "src/features/feature_engineering.py",
    "src/features/feature_selection.py",

    # Source -> models
    "src/models/classifier.py",
    "src/models/regressor.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",

    # Pipelines
    "src/pipelines/train_pipeline.py",
    "src/pipelines/predict_pipeline.py",

    # Utils
    "src/utils/config.py",
    "src/utils/logger.py",
    "src/utils/metrics.py",

    # Visualization
    "src/visualization/eda.py",
    "src/visualization/plots.py",
    "src/visualization/shap_plots.py",

    # Tests
    "tests/test_data.py",
    "tests/test_models.py",
]


def create_structure():

    os.makedirs(PROJECT_NAME, exist_ok=True)

    for folder in folders:
        os.makedirs(os.path.join(PROJECT_NAME, folder), exist_ok=True)

    for file in files:

        filepath = os.path.join(PROJECT_NAME, file)

        directory = os.path.dirname(filepath)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:

                if filepath.endswith(".py"):
                    f.write('"""Auto-generated file"""\n')

                elif filepath.endswith(".md"):
                    f.write(f"# {PROJECT_NAME}\n")

                elif filepath.endswith(".yaml"):
                    f.write("# Configuration\n")

                elif filepath.endswith(".gitignore"):
                    f.write(
"""__pycache__/
*.pyc
*.pkl
*.joblib
.ipynb_checkpoints/
.env
venv/
logs/
"""
                    )

    print("=" * 55)
    print(" Project structure created successfully!")
    print("=" * 55)
    print(f"\nLocation:\n{os.path.abspath(PROJECT_NAME)}")


if __name__ == "__main__":
    create_structure()