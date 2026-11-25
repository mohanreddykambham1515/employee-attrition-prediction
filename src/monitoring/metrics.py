
from prometheus_client import Counter, Histogram

# Total prediction requests
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Prediction errors
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

# Prediction latency
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction API response time"
)

# Predicted positive cases
ATTRITION_PREDICTIONS = Counter(
    "attrition_predictions_total",
    "Number of predicted attrition cases"
)
