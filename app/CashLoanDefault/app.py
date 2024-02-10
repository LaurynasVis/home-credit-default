import pickle
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI


# Pydantic classes for input and output
class CashLoanDefault(BaseModel):
    code_gender: str
    flag_own_car: str
    amt_credit: float
    amt_goods_price: float
    name_education_type: str
    days_employed: float
    ext_source_1: float
    ext_source_2: float
    ext_source_3: float
    max_bureau_days_credit_enddate: float
    min_installments_payments_amt_payment: float
    std_installments_payments_days_instalment: float
    mean_previous_application_min_installments_payments_amt_payment: float
    max_pos_cash_balance_previous_application_days_last_due: float
    mode_installments_payments_previous_application_product_combination: str


class PredictionOut(BaseModel):
    request_status: str


# Load the model
model = pickle.load(open("cash_loan_default-0.1.0.pkl", "rb"))

# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Cash loan default classification", "model_version": 0.1}


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: CashLoanDefault):
    cust_df = pd.DataFrame([payload.model_dump()])
    cat = [
        "code_gender", 
        "flag_own_car", 
        "name_education_type", 
        "mode_installments_payments_previous_application_product_combination"
        ]
    for col in cat:
        cust_df[col] = cust_df[col].astype("category")
    preds = model.predict_proba(cust_df)[0, 1]
    if preds > 0.08:
        preds = "Risk of loan default"
    else:
        preds = "The loan most likely will be repaid"
    result = {"request_status": preds}
    return result
