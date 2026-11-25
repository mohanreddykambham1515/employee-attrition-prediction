
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.predict import predict_employee
from src.monitoring.logger import logger
from src.monitoring.metrics import (
    PREDICTION_REQUESTS,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    ATTRITION_PREDICTIONS
)


app = FastAPI(
    title="Employee Attrition Prediction API",
    version="2.0.0"
)


class EmployeeInput(BaseModel):

    Age: int = Field(ge=18, le=100)
    BusinessTravel: str
    DailyRate: float
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: float
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: float
    MonthlyRate: float
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: float
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    CareerYears: int
    RoleExperienceRatio: float


@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict")
def predict(data: EmployeeInput):

    start_time = time.perf_counter()

    PREDICTION_REQUESTS.inc()

    try:
        employee_data = (
            data.model_dump()
            if hasattr(data, "model_dump")
            else data.dict()
        )

        result = predict_employee(employee_data)

        if result["prediction"] == "Yes":
            ATTRITION_PREDICTIONS.inc()

        logger.info(
            "Prediction completed: prediction=%s risk=%s",
            result["prediction"],
            result["risk_category"]
        )

        return {
            "status": "success",
            "prediction": result
        }

    except (FileNotFoundError, ValueError) as error:

        PREDICTION_ERRORS.inc()

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        PREDICTION_ERRORS.inc()

        logger.exception("Unexpected prediction error")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )

    finally:

        duration = time.perf_counter() - start_time

        PREDICTION_LATENCY.observe(duration)
