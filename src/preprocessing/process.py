
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "X_train.csv"


def create_preprocessor():

    # Load training features
    X_train = pd.read_csv(DATA_PATH)

    # Identify numerical columns
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("Numerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


if __name__ == "__main__":

    preprocessor = create_preprocessor()

    X_train = pd.read_csv(DATA_PATH)

    X_train_transformed = preprocessor.fit_transform(X_train)

    print("\n========== PREPROCESSING COMPLETE ==========")

    print("Original shape:", X_train.shape)

    print("Transformed shape:", X_train_transformed.shape)
