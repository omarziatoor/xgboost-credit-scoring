"""
Generates a synthetic credit applicant dataset with a REAL, learnable
relationship between features and a binary default outcome.

This exists because a genuine credit scoring model requires historical
outcome labels (did the customer default or not) — there is no way to
validly train or evaluate a scoring model without them. If you have real
historical loan outcome data, replace this script's output with that
data instead; keep the same column names or update config.py accordingly.

The default probability here is generated from a hidden logistic function
of the features plus noise, so the "ground truth" relationship is known
and the model's job is to recover it — a standard way to sanity-check a
credit scoring pipeline before ever pointing it at real data.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_dataset(n_customers: int = 20000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_customers)
    annual_income = rng.lognormal(mean=10.8, sigma=0.55, size=n_customers).clip(15_000, 500_000)
    account_age_years = rng.exponential(scale=4, size=n_customers).clip(0, 30)
    total_loans = rng.poisson(lam=1.4, size=n_customers)
    total_loan_outstanding = (rng.exponential(scale=8000, size=n_customers) * (total_loans > 0)).clip(0)
    overdue_loans = rng.binomial(total_loans, p=0.12)
    monthly_transactions = rng.poisson(lam=18, size=n_customers)
    avg_transaction_amount = rng.lognormal(mean=6.5, sigma=0.7, size=n_customers).clip(50, 20_000)
    credit_utilization = rng.beta(2, 5, size=n_customers)  # skewed toward lower utilization
    foreign_transaction_ratio = rng.beta(1.2, 8, size=n_customers)
    last_activity_days = rng.exponential(scale=15, size=n_customers).clip(0, 365)
    risk_flag_from_bureau = rng.binomial(1, p=0.15, size=n_customers)  # e.g. prior delinquency flag

    debt_to_income_ratio = total_loan_outstanding / (annual_income + 1)
    loan_overdue_ratio = overdue_loans / (total_loans + 1)

    # --- Hidden "true" relationship used only to generate labels ---
    # Coefficients are on STANDARDIZED features so no single raw-scale
    # feature can dominate purely due to units (the bug in the original
    # heuristic project). This is the thing a real model has to recover.
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    logit = (
        -3.3
        + 1.6 * z(debt_to_income_ratio)
        + 1.3 * z(loan_overdue_ratio)
        + 1.1 * z(credit_utilization)
        + 0.9 * risk_flag_from_bureau
        + 0.5 * z(foreign_transaction_ratio)
        - 0.7 * z(np.log1p(annual_income))
        - 0.5 * z(account_age_years)
        - 0.3 * z(monthly_transactions)
        + 0.2 * z(last_activity_days)
        + rng.normal(0, 0.6, n_customers)  # irreducible noise — no model gets this perfect
    )
    default_prob = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, default_prob)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "age": age,
        "annual_income": annual_income.round(2),
        "account_age_years": account_age_years.round(2),
        "total_loans": total_loans,
        "total_loan_outstanding": total_loan_outstanding.round(2),
        "overdue_loans": overdue_loans,
        "loan_overdue_ratio": loan_overdue_ratio.round(4),
        "debt_to_income_ratio": debt_to_income_ratio.round(4),
        "monthly_transactions": monthly_transactions,
        "avg_transaction_amount": avg_transaction_amount.round(2),
        "credit_utilization": credit_utilization.round(4),
        "foreign_transaction_ratio": foreign_transaction_ratio.round(4),
        "last_activity_days": last_activity_days.round(1),
        "bureau_risk_flag": risk_flag_from_bureau,
        "defaulted": defaulted,  # <-- the real outcome label a model needs
    })
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("credit_data.csv", index=False)
    print(f"Generated {len(data)} rows -> credit_data.csv")
    print(f"Default rate: {data['defaulted'].mean():.2%}")
