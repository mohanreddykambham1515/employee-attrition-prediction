
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "employee_attrition.csv"

PROCESSED_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "employee_attrition_cleaned.csv"
)


def build_features():

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", df.shape)

    # 1. Remove duplicate records
    df = df.drop_duplicates()

    # 2. Remove irrelevant columns
    irrelevant_columns = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    df = df.drop(
        columns=irrelevant_columns,
        errors="ignore"
    )

    # 3. Handle missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum().sum())

    # 4. Encode target variable
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    # 5. Feature engineering

    # Total years of experience
    df["CareerYears"] = (
        df["TotalWorkingYears"] -
        df["YearsAtCompany"]
    )

    # Promotion waiting period
    df["YearsSinceLastPromotion"] = (
        df["YearsSinceLastPromotion"]
    )

    # Years in current role
    df["RoleExperienceRatio"] = (
        df["YearsInCurrentRole"] /
        (df["YearsAtCompany"] + 1)
    )

    # 6. Validate target
    if df["Attrition"].isnull().any():
        raise ValueError("Target contains invalid values")

    # 7. Save processed dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\n========== FEATURE ENGINEERING COMPLETE ==========")
    print("Processed shape:", df.shape)
    print("Saved to:", PROCESSED_DATA_PATH)

    print("\nTarget distribution:")
    print(df["Attrition"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    return df


if __name__ == "__main__":
    build_features()
