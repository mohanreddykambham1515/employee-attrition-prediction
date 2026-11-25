
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def final_evaluation():

    # Load unseen test dataset
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    # Load selected model
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    # Load threshold selected using validation data
    threshold = float(
        (MODEL_DIR / "best_threshold.txt").read_text().strip()
    )

    print("Test dataset shape:", X_test.shape)
    print("Selected threshold:", threshold)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply selected threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, y_pred
        )
    }

    print("\n========== FINAL TEST METRICS ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Attrition", "Attrition"],
            zero_division=0
        )
    )

    print("\n========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # Save evaluation report
    report = {
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": cm.tolist()
    }

    report_path = REPORT_DIR / "final_evaluation.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nFinal evaluation report saved!")
    print("Location:", report_path)


if __name__ == "__main__":
    final_evaluation()
