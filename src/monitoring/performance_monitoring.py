
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def monitor_performance():

    # Load model
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    threshold = float(
        (MODEL_DIR / "best_threshold.txt").read_text()
    )

    # Load labeled evaluation data
    X = pd.read_csv(DATA_DIR / "X_test.csv")

    y_actual = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze("columns")

    # Generate predictions
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(
            y_actual, predictions
        ),
        "precision": precision_score(
            y_actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_actual, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_actual, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_actual, probabilities
        ),
        "pr_auc": average_precision_score(
            y_actual, probabilities
        )
    }

    print("\n========== MODEL PERFORMANCE MONITORING ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "performance_monitoring.csv",
        index=False
    )

    print("\nPerformance report saved!")


if __name__ == "__main__":
    monitor_performance()
