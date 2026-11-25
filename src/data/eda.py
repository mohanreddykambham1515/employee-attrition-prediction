
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "employee_attrition.csv"

REPORT_DIR = ROOT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")


# 1. Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.savefig(REPORT_DIR / "attrition_distribution.png")
plt.close()


# 2. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Attrition", kde=True)
plt.title("Age Distribution by Attrition")
plt.savefig(REPORT_DIR / "age_distribution.png")
plt.close()


# 3. Monthly income distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyIncome", hue="Attrition", kde=True)
plt.title("Monthly Income by Attrition")
plt.savefig(REPORT_DIR / "income_distribution.png")
plt.close()


# 4. Attrition by overtime
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Attrition by Overtime")
plt.savefig(REPORT_DIR / "overtime_attrition.png")
plt.close()


# 5. Attrition by department
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Department", hue="Attrition")
plt.title("Attrition by Department")
plt.xticks(rotation=15)
plt.savefig(REPORT_DIR / "department_attrition.png")
plt.close()


# 6. Numerical correlation
numerical_df = df.select_dtypes(include="number")

plt.figure(figsize=(16, 12))
sns.heatmap(
    numerical_df.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Numerical Feature Correlation")
plt.savefig(REPORT_DIR / "correlation_matrix.png")
plt.close()


print("EDA completed successfully!")
print(f"Charts saved in: {REPORT_DIR}")
