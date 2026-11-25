
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def explain_model():

    # Load model and validation data
    model = joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

    X_val = pd.read_csv(DATA_DIR / "X_val.csv")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform validation features
    X_transformed = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    # Use a small sample for explanation
    if hasattr(X_transformed, "toarray"):
        X_sample = X_transformed[:100].toarray()
    else:
        X_sample = X_transformed[:100]

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # Summary plot
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP explanation completed!")
    print("Saved:", REPORT_DIR / "shap_summary.png")


if __name__ == "__main__":
    explain_model()
