import random
from locust import HttpUser, task, constant_throughput

test_applications = [
    {
        "code_gender": "M",
        "flag_own_car": "Y",
        "amt_credit": 1125000.0,
        "amt_goods_price": 1129500.0,
        "name_education_type": "Higher education",
        "days_employed": -1188.0,
        "ext_source_1": 0.311267,
        "ext_source_2": 0.650442,
        "ext_source_3": 0.139376,
        "max_bureau_days_credit_enddate": 593.0,
        "min_installments_payments_amt_payment": 5.445000,
        "std_installments_payments_days_instalment": 667.460022,
        "mean_previous_application_min_installments_payments_amt_payment": 2696.985107,
        "max_pos_cash_balance_previous_application_days_last_due": 365243.0,
        "mode_installments_payments_previous_application_product_combination": "Cash X-Sell: low",
    },
    {
        "code_gender": "F",
        "flag_own_car": "N",
        "amt_credit": 450000.0,
        "amt_goods_price": 850000.0,
        "name_education_type": "Secondary / secondary special",
        "days_employed": -1500.0,
        "ext_source_1": 0.428532,
        "ext_source_2": 0.720115,
        "ext_source_3": 0.189765,
        "max_bureau_days_credit_enddate": -1019.0,
        "min_installments_payments_amt_payment": 1435.454956,
        "std_installments_payments_days_instalment": 750.134033,
        "mean_previous_application_min_installments_payments_amt_payment": 238.184998,
        "max_pos_cash_balance_previous_application_days_last_due": -414.0,
        "mode_installments_payments_previous_application_product_combination": "POS mobile with interest",
    },
]


class BankLoan(HttpUser):
    wait_time = constant_throughput(1)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=random.choice(test_applications),
            timeout=1,
        )
