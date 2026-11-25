
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "employee_attrition_cleaned.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"


def split_data():

    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    # First split: 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Save datasets
    datasets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    for name, data in datasets.items():
        data.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    print("\n========== DATA SPLIT COMPLETE ==========")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    print("\n========== TARGET DISTRIBUTION ==========")

    print("Training:")
    print(y_train.value_counts(normalize=True))

    print("\nValidation:")
    print(y_val.value_counts(normalize=True))

    print("\nTesting:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    split_data()
