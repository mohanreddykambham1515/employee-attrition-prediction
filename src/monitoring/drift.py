
import numpy as np
import pandas as pd

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_psi(reference, current, bins=10):

    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna()

    if reference.empty or current.empty:
        return np.nan

    # Build bins using reference distribution
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(edges) < 2:
        return 0.0

    # Extend outer boundaries
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(
        reference, bins=edges
    )[0]

    current_counts = np.histogram(
        current, bins=edges
    )[0]

    # Calculate proportions with smoothing
    epsilon = 0.0001

    reference_pct = (
        reference_counts + epsilon
    ) / (reference_counts.sum() + epsilon * len(reference_counts))

    current_pct = (
        current_counts + epsilon
    ) / (current_counts.sum() + epsilon * len(current_counts))

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_categorical_drift(reference, current):

    reference = reference.fillna("__MISSING__").astype(str)
    current = current.fillna("__MISSING__").astype(str)

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    reference_dist = (
        reference.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    current_dist = (
        current.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    # Total Variation Distance
    drift_score = 0.5 * np.abs(
        reference_dist - current_dist
    ).sum()

    return float(drift_score)


def monitor_drift():

    # Reference training data
    reference = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    # Current data: test data used as a monitoring demonstration
    current = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    results = []

    for column in reference.columns:

        if pd.api.types.is_numeric_dtype(reference[column]):

            score = calculate_psi(
                reference[column],
                current[column]
            )

            metric = "PSI"

            # Common heuristic thresholds
            if pd.isna(score):
                status = "Insufficient data"
            elif score < 0.10:
                status = "No significant drift"
            elif score < 0.25:
                status = "Moderate drift"
            else:
                status = "High drift"

        else:

            score = calculate_categorical_drift(
                reference[column],
                current[column]
            )

            metric = "Total Variation Distance"

            if score < 0.10:
                status = "Low drift"
            elif score < 0.25:
                status = "Moderate drift"
            else:
                status = "High drift"

        results.append({
            "Feature": column,
            "Metric": metric,
            "Drift Score": score,
            "Status": status
        })

    report = pd.DataFrame(results)

    print("\n========== DATA DRIFT REPORT ==========")

    print(
        report.sort_values(
            "Drift Score",
            ascending=False
        ).to_string(index=False)
    )

    report.to_csv(
        REPORT_DIR / "data_drift_report.csv",
        index=False
    )

    print("\nDrift report saved successfully!")


if __name__ == "__main__":
    monitor_drift()
