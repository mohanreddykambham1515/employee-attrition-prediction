
import pandas as pd
import joblib

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "best_random_forest.pkl"
THRESHOLD_PATH = ROOT_DIR / "models" / "best_threshold.txt"


def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError("Model threshold not found")

    model = joblib.load(MODEL_PATH)

    threshold = float(THRESHOLD_PATH.read_text().strip())

    return model, threshold


def predict_employee(employee_data):

    model, threshold = load_artifacts()

    # Convert incoming employee data to DataFrame
    employee_df = pd.DataFrame([employee_data])

    # Validate input columns
    expected_columns = list(model.feature_names_in_)

    missing_columns = set(expected_columns) - set(employee_df.columns)
    extra_columns = set(employee_df.columns) - set(expected_columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(extra_columns)}"
        )

    # Maintain training feature order
    employee_df = employee_df[expected_columns]

    # Predict probability
    probability = model.predict_proba(employee_df)[0][1]

    # Apply selected threshold
    prediction = int(probability >= threshold)

    # Business-defined illustrative risk categories
    if probability < 0.30:
        risk_category = "Low"
    elif probability < 0.60:
        risk_category = "Medium"
    else:
        risk_category = "High"

    result = {
        "prediction": "Yes" if prediction == 1 else "No",
        "attrition_probability": round(float(probability), 4),
        "risk_category": risk_category,
        "decision_threshold": threshold
    }

    return result


if __name__ == "__main__":

    # Example employee record
    employee_data = {
        "Age": 32,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1100,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Medical",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
        "CareerYears": 3,
        "RoleExperienceRatio": 0.5
    }

    result = predict_employee(employee_data)

    print("\n========== EMPLOYEE ATTRITION PREDICTION ==========")

    for key, value in result.items():
        print(f"{key}: {value}")
