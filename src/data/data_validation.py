
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "employee_attrition.csv"


def validate_data():

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE RECORDS ==========")
    print(df.duplicated().sum())

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df.describe())

    print("\n========== CATEGORICAL SUMMARY ==========")
    print(df.describe(include="object"))

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["Attrition"].value_counts())

    print("\n========== TARGET PERCENTAGE ==========")
    print(df["Attrition"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    validate_data()
