"""
Converts model output (probability of default) into a conventional
300-850 style credit score.

This is how real scorecards work: the MODEL predicts a probability;
the SCORE is a separate, monotonic transformation of that probability
chosen for interpretability. This is different from training a
regressor to directly predict a score value (what the original
heuristic-based project did), which has no probabilistic grounding.

Higher score = lower risk (standard convention).
"""
import numpy as np
import pandas as pd
import joblib

import config


def probability_to_score(prob_default: np.ndarray,
                          score_min: int = config.SCORE_MIN,
                          score_max: int = config.SCORE_MAX) -> np.ndarray:
    """
    Simple log-odds linear mapping: lower default probability -> higher score.
    Uses a logit transform so the score scale behaves sensibly near the
    extremes (0 and 1 probability), rather than a naive linear inversion.
    """
    eps = 1e-6
    prob_default = np.clip(prob_default, eps, 1 - eps)
    log_odds_good = -np.log(prob_default / (1 - prob_default))  # log-odds of NOT defaulting

    # Min-max scale the log-odds into [score_min, score_max]
    lo_min, lo_max = log_odds_good.min(), log_odds_good.max()
    scaled = (log_odds_good - lo_min) / (lo_max - lo_min + eps)
    score = score_min + scaled * (score_max - score_min)
    return score.round(0).astype(int)


def score_customers(input_csv: str, output_csv: str = "scored_customers.csv"):
    model = joblib.load("models/calibrated_model.joblib")

    df = pd.read_csv(input_csv)
    X = df[config.NUMERICAL_FEATURES]

    prob_default = model.predict_proba(X)[:, 1]
    scores = probability_to_score(prob_default)

    df["probability_of_default"] = prob_default.round(4)
    df["credit_score"] = scores
    df.to_csv(output_csv, index=False)
    print(f"Scored {len(df)} customers -> {output_csv}")
    print(df[["customer_id", "probability_of_default", "credit_score"]].head(10))


if __name__ == "__main__":
    score_customers(input_csv="test_split.csv")
