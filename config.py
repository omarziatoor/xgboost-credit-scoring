DATA_PATH = "credit_data.csv"
MODEL_PATH = "models/xgb_credit_model.json"

TARGET = "defaulted"

NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "account_age_years",
    "total_loans",
    "total_loan_outstanding",
    "overdue_loans",
    "loan_overdue_ratio",
    "debt_to_income_ratio",
    "monthly_transactions",
    "avg_transaction_amount",
    "credit_utilization",
    "foreign_transaction_ratio",
    "last_activity_days",
    "bureau_risk_flag",
]

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Score mapping (industry-standard style: higher score = lower risk)
SCORE_MIN = 300
SCORE_MAX = 850
