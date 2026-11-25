
import pandas as pd
import joblib
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def optimize_threshold():

    # Load validation data
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    # Load tuned model
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    # Predicted probabilities
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    # Test different thresholds
    thresholds = np.arange(0.10, 0.91, 0.05)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        results.append({
            "threshold": round(threshold, 2),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    # Create comparison dataframe
    results_df = pd.DataFrame(results)

    print("\n========== THRESHOLD RESULTS ==========")
    print(results_df.round(4).to_string(index=False))

    # Find threshold with best F1
    best_index = results_df["f1_score"].idxmax()

    best_threshold = results_df.loc[
        best_index, "threshold"
    ]

    print("\nBest threshold:", best_threshold)

    # Predictions using best threshold
    best_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\n========== BEST THRESHOLD METRICS ==========")

    print(
        "Precision:",
        precision_score(
            y_val, best_predictions, zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_val, best_predictions, zero_division=0
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_val, best_predictions, zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, best_predictions))

    # Save threshold results
    results_df.to_csv(
        REPORT_DIR / "threshold_comparison.csv",
        index=False
    )

    # Save selected threshold
    threshold_path = MODEL_DIR / "best_threshold.txt"

    threshold_path.write_text(str(best_threshold))

    print("\nThreshold saved successfully!")


if __name__ == "__main__":
    optimize_threshold()
