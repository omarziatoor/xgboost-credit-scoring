"""
SHAP-based explainability for the XGBoost model.

Gradient boosting is less inherently interpretable than logistic
regression, which matters in lending: many jurisdictions require being
able to explain *why* an applicant received a given decision or score
(adverse action reasons). SHAP values recover per-feature, per-prediction
attributions, closing most of that gap.
"""
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import config


def explain():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)

    test_df = pd.read_csv("test_split.csv")
    X_test = test_df[config.NUMERICAL_FEATURES]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Global feature importance (which features matter most overall)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved results/shap_summary.png")

    # Per-customer explanation for the first test row (adverse-action style)
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("results/shap_single_customer.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved results/shap_single_customer.png (explanation for one customer's score)")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    explain()
