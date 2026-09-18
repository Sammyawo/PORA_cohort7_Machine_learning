import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

MODEL_PATH = "model/model.json"

# Load the trained model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API"
)

VALID_GEOGRAPHIES = {
    "France",
    "Germany",
    "Spain"
}

# Must match FEATURE_ORDER in model_pipeline.py
FEATURE_ORDER = [
    "CreditScore",
    "Geography_Germany",
    "Geography_Spain",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


class CustomerFeatures(BaseModel):
    credit_score: int = Field(..., json_schema_extra={"example":515})
    geography: str = Field(..., json_schema_extra={"example":"France"})
    age: int = Field(..., json_schema_extra={"example":42})
    gender: str = Field(..., json_schema_extra={"example":"Female"})
    tenure: int = Field(..., json_schema_extra={"example":2})
    balance: float = Field(..., json_schema_extra={"example":120.01})
    num_of_products: int = Field(..., json_schema_extra={"example":1})
    has_credit_card: int = Field(..., json_schema_extra={"example":1})
    is_active_member: int = Field(..., json_schema_extra={"example":1})
    estimated_salary: float = Field(..., json_schema_extra={"example":12002.00})

def build_feature_row(customer: CustomerFeatures) -> pd.DataFrame:
    if customer.geography not in VALID_GEOGRAPHIES:
        raise ValueError(
            f"Geography must be one of {sorted(VALID_GEOGRAPHIES)}, got {customer.geography}"
        )

    gender_encoded = 0 if customer.gender.strip().lower() == "female" else 1

    row = {
        "CreditScore": customer.credit_score,
        "Geography_Germany": 1 if customer.geography == "Germany" else 0,
        "Geography_Spain": 1 if customer.geography == "Spain" else 0,
        "Gender": gender_encoded,
        "Age": customer.age,
        "Tenure": customer.tenure,
        "Balance": customer.balance,
        "NumOfProducts": customer.num_of_products,
        "HasCrCard": customer.has_credit_card,
        "IsActiveMember": customer.is_active_member,
        "EstimatedSalary": customer.estimated_salary
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)



@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/api/predict")
def predict(customer: CustomerFeatures):
    try:
        input_df = build_feature_row(customer)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        churn_status = "leave" if prediction == 1 else "stay"

        return {
            "churn_status": churn_status,
            "churn_value": prediction,
            "churn_probability": round(probability, 4),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8055, reload=True)