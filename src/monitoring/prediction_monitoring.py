
import pandas as pd
import joblib

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def monitor_predictions():

    # Load model
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    threshold = float(
        (MODEL_DIR / "best_threshold.txt").read_text()
    )

    # Load reference and current datasets
    X_reference = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    X_current = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    # Generate predictions
    reference_probabilities = model.predict_proba(
        X_reference
    )[:, 1]

    current_probabilities = model.predict_proba(
        X_current
    )[:, 1]

    reference_predictions = (
        reference_probabilities >= threshold
    ).astype(int)

    current_predictions = (
        current_probabilities >= threshold
    ).astype(int)

    # Calculate positive prediction rates
    reference_rate = reference_predictions.mean() * 100

    current_rate = current_predictions.mean() * 100

    print("\n========== PREDICTION MONITORING ==========")

    print(
        f"Reference attrition prediction rate: {reference_rate:.2f}%"
    )

    print(
        f"Current attrition prediction rate: {current_rate:.2f}%"
    )

    print(
        f"Reference average probability: {reference_probabilities.mean():.4f}"
    )

    print(
        f"Current average probability: {current_probabilities.mean():.4f}"
    )

    # Save monitoring report
    report = pd.DataFrame({
        "Dataset": ["Reference", "Current"],
        "Positive_Prediction_Rate": [
            reference_rate,
            current_rate
        ],
        "Average_Probability": [
            reference_probabilities.mean(),
            current_probabilities.mean()
        ]
    })

    report.to_csv(
        REPORT_DIR / "prediction_monitoring.csv",
        index=False
    )

    print("\nPrediction monitoring report saved!")


if __name__ == "__main__":
    monitor_predictions()
