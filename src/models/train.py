
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing.process import create_preprocessor


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_logistic_regression():

    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    # Load validation data
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Create complete ML pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    # Train model
    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    # Validation predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluation
    print("\n========== VALIDATION RESULTS ==========")

    print("Accuracy:", accuracy_score(y_val, y_pred))

    print("Precision:", precision_score(y_val, y_pred))

    print("Recall:", recall_score(y_val, y_pred))

    print("F1 Score:", f1_score(y_val, y_pred))

    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save complete pipeline
    model_path = MODEL_DIR / "logistic_regression.pkl"

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")
    print("Location:", model_path)

    return model


if __name__ == "__main__":
    train_logistic_regression()
