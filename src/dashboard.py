
import streamlit as st
import pandas as pd
import json

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT_DIR / "reports"


st.set_page_config(
    page_title="Employee Attrition ML Dashboard",
    page_icon="📊",
    layout="wide"
)


st.title("Employee Attrition Prediction Dashboard")

st.caption(
    "Machine Learning Model Performance and Production Monitoring"
)


def load_csv(filename):

    path = REPORT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return None


def load_json(filename):

    path = REPORT_DIR / filename

    if path.exists():
        with open(path) as file:
            return json.load(file)

    return None


# Sidebar
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "Model Performance",
        "Data Drift",
        "Prediction Monitoring",
        "Feature Importance"
    ]
)


# Load reports
evaluation = load_json("final_evaluation.json")

drift = load_csv("data_drift_report.csv")

prediction = load_csv("prediction_monitoring.csv")

importance = load_csv("feature_importance.csv")

performance = load_csv("performance_monitoring.csv")


# Overview
if page == "Overview":

    st.header("Model Overview")

    if evaluation:

        metrics = evaluation["metrics"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.3f}"
        )

        col2.metric(
            "Precision",
            f"{metrics['Precision']:.3f}"
        )

        col3.metric(
            "Recall",
            f"{metrics['Recall']:.3f}"
        )

        col4.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.3f}"
        )

        st.subheader("Confusion Matrix")

        cm = pd.DataFrame(
            evaluation["confusion_matrix"],
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        st.dataframe(cm, use_container_width=True)

    else:
        st.warning("Final evaluation report not found.")

    st.info(
        "Model: Tuned Random Forest | "
        "Task: Binary Classification"
    )


# Model Performance
elif page == "Model Performance":

    st.header("Model Performance")

    if performance is not None:

        st.dataframe(
            performance,
            use_container_width=True
        )

        st.bar_chart(
            performance.T,
            height=350
        )

    else:
        st.warning("Performance report not found.")


# Data Drift
elif page == "Data Drift":

    st.header("Data Drift Monitoring")

    if drift is not None:

        st.dataframe(
            drift,
            use_container_width=True
        )

        st.subheader("Feature Drift Scores")

        chart_data = drift.set_index("Feature")[
            "Drift Score"
        ]

        st.bar_chart(chart_data)

    else:
        st.warning("Data drift report not found.")


# Prediction Monitoring
elif page == "Prediction Monitoring":

    st.header("Prediction Monitoring")

    if prediction is not None:

        st.dataframe(
            prediction,
            use_container_width=True
        )

        st.subheader("Positive Prediction Rate")

        st.bar_chart(
            prediction.set_index("Dataset")[
                "Positive_Prediction_Rate"
            ]
        )

        st.subheader("Average Prediction Probability")

        st.bar_chart(
            prediction.set_index("Dataset")[
                "Average_Probability"
            ]
        )

    else:
        st.warning("Prediction monitoring report not found.")


# Feature Importance
elif page == "Feature Importance":

    st.header("Feature Importance")

    if importance is not None:

        top_features = importance.head(20)

        st.dataframe(
            top_features,
            use_container_width=True
        )

        st.subheader("Top 20 Important Features")

        st.bar_chart(
            top_features.set_index("Feature")[
                "Importance"
            ]
        )

    else:
        st.warning("Feature importance report not found.")
