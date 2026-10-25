import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import TransactionRequest, PredictionResponse
from app.decision_engine import get_decision, get_risk_band, get_reason_codes
from app.database import create_tables, insert_scored_transaction

from src.config import MODEL_PATH, SCALER_PATH


app = FastAPI(
    title="Transaction Risk Scoring and Fraud Decision Engine",
    description="Scores card transactions for fraud risk and recommends approve, review, or decline.",
    version="1.0.0",
)


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

create_tables()


@app.get("/")
def home():
    return {
        "message": "Transaction Risk Scoring and Fraud Decision Engine is running."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    transaction_dict = transaction.model_dump()

    input_df = pd.DataFrame([transaction_dict])

    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

    fraud_probability = float(model.predict_proba(input_df)[0][1])

    risk_band = get_risk_band(fraud_probability)
    decision = get_decision(fraud_probability)
    reason_codes = get_reason_codes(transaction_dict, fraud_probability)

    insert_scored_transaction(
        transaction_time=transaction.Time,
        amount=transaction.Amount,
        fraud_probability=fraud_probability,
        risk_band=risk_band,
        decision=decision,
    )

    return PredictionResponse(
        fraud_probability=round(fraud_probability, 6),
        risk_band=risk_band,
        decision=decision,
        reason_codes=reason_codes,
    )