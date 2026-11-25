
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }


def compare_models():

    # Load validation data for model selection
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    # Load trained models
    logistic_model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    random_forest_model = joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )

    # Evaluate models on the same validation data
    results = {
        "Logistic Regression": evaluate_model(
            logistic_model, X_val, y_val
        ),
        "Random Forest": evaluate_model(
            random_forest_model, X_val, y_val
        )
    }

    comparison = pd.DataFrame(results).T

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.round(4))

    # Save report
    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv"
    )

    print("\nComparison report saved successfully!")

    return comparison


if __name__ == "__main__":
    compare_models()
