
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
DATA_DIR = ROOT_DIR / "data" / "processed"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_feature_importance():

    # Load trained model
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    # Extract preprocessing and classifier
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Get importance scores
    importance_scores = classifier.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_scores
    })

    # Sort features
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== TOP 20 IMPORTANT FEATURES ==========")
    print(importance_df.head(20).to_string(index=False))

    # Save results
    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    # Visualize top 20 features
    top_features = importance_df.head(20).sort_values(
        "Importance"
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nFeature importance report saved!")


if __name__ == "__main__":
    calculate_feature_importance()
