
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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


def train_random_forest():

    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    # Load validation data
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Create Random Forest pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # Train model
    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluation
    print("\n========== RANDOM FOREST RESULTS ==========")

    print("Accuracy:", accuracy_score(y_val, y_pred))

    print("Precision:", precision_score(y_val, y_pred, zero_division=0))

    print("Recall:", recall_score(y_val, y_pred, zero_division=0))

    print("F1 Score:", f1_score(y_val, y_pred, zero_division=0))

    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save model
    model_path = MODEL_DIR / "random_forest.pkl"

    joblib.dump(model, model_path)

    print("\nRandom Forest saved successfully!")
    print("Location:", model_path)

    return model


if __name__ == "__main__":
    train_random_forest()
