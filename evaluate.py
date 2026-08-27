"""
Evaluates the trained model on the held-out test set using metrics that
are actually appropriate for a PROBABILITY-of-default classifier:

- AUC-ROC: overall ranking quality (can the model separate defaulters
  from non-defaulters across all thresholds?)
- KS statistic: the standard credit-industry metric — max separation
  between the cumulative distributions of good vs. bad accounts
- Calibration: are predicted probabilities close to actual observed
  default rates? (critical if the output is used to price risk)

This replaces the mismatched roc_auc_score/log_loss-on-a-regressor
approach from the original heuristic pipeline.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

import config


def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic — standard credit scoring metric.
    Measures the max gap between cumulative % of goods vs. bads captured
    as the score threshold moves. Higher is better; >0.3 is often
    considered acceptable in practice, >0.4 good."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks


def evaluate():
    # Use the calibrated model — see train.py for why raw XGBoost
    # probabilities under scale_pos_weight are not well-calibrated.
    model = joblib.load("models/calibrated_model.joblib")

    test_df = pd.read_csv("test_split.csv")
    X_test = test_df[config.NUMERICAL_FEATURES]
    y_test = test_df[config.TARGET]

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ks = ks_statistic(y_test, y_prob)

    print("=== Evaluation on held-out test set ===")
    print(f"AUC-ROC:       {auc:.4f}")
    print(f"KS statistic:  {ks:.4f}")
    print(f"Default rate (actual): {y_test.mean():.2%}")
    print(f"Default rate (predicted avg): {y_prob.mean():.2%}")

    # --- Calibration plot ---
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability of default")
    plt.ylabel("Observed default rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/calibration_curve.png", dpi=120)
    print("Saved results/calibration_curve.png")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=120)
    print("Saved results/roc_curve.png")

    return {"auc": auc, "ks": ks}


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    evaluate()
