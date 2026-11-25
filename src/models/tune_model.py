
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score
)

from src.preprocessing.process import create_preprocessor


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def tune_random_forest():

    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    # Load validation data
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    # Create complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # Hyperparameter search space
    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [5, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample"
        ]
    }

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")

    # Fit only on training data
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\n========== BEST PARAMETERS ==========")
    print(search.best_params_)

    print("\nBest CV F1 Score:")
    print(search.best_score_)

    # Evaluate on validation data
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    print("\n========== VALIDATION RESULTS ==========")
    print("F1 Score:", f1_score(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Save best model
    joblib.dump(
        best_model,
        MODEL_DIR / "best_random_forest.pkl"
    )

    print("\nBest model saved successfully!")


if __name__ == "__main__":
    tune_random_forest()
