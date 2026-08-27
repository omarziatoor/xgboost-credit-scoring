"""
Trains an XGBoost binary classifier to predict probability of default (PD),
using a REAL outcome label (`defaulted`) — not a hand-written heuristic.

This is the key structural difference from a heuristic-scoring pipeline:
the model here learns from historical outcomes, so its accuracy can
actually be measured against ground truth (see evaluate.py).
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
import joblib

import config


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    X = df[["customer_id"] + config.NUMERICAL_FEATURES]
    y = df[config.TARGET]
    return X, y


def train():
    X, y = load_data()

    # 60/20/20 train/val/test split. Val is used for early stopping,
    # test is held out entirely until evaluate.py.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=config.RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y_temp
    )

    # customer_id is kept only for output/reporting — never used as a
    # model feature (it carries no predictive signal and would be a
    # form of leakage/overfitting to identity if included).
    id_test = X_test["customer_id"]
    X_train = X_train[config.NUMERICAL_FEATURES]
    X_val = X_val[config.NUMERICAL_FEATURES]
    X_test_features = X_test[config.NUMERICAL_FEATURES]

    # Class imbalance handling — defaults are typically a minority class
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=config.RANDOM_SEED,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation AUC: {model.best_score:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_model(config.MODEL_PATH)

    # --- Probability calibration ---
    # scale_pos_weight improves ranking (AUC) under class imbalance, but
    # skews raw predicted probabilities away from true observed rates.
    # Since the scorecard step needs genuinely calibrated probabilities
    # (not just correct ranking), we fit an isotonic calibration layer
    # on the validation set, on top of the already-trained model.
    if _HAS_FROZEN:
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    else:
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_val, y_val)
    joblib.dump(calibrated_model, "models/calibrated_model.joblib")

    # Persist test split (with customer_id restored for reporting) so
    # evaluate.py and scorecard.py use the exact same held-out data
    test_out = X_test_features.copy()
    test_out.insert(0, "customer_id", id_test)
    test_out[config.TARGET] = y_test
    test_out.to_csv("test_split.csv", index=False)

    print(f"Model saved to {config.MODEL_PATH}")
    print("Calibrated model saved to models/calibrated_model.joblib")
    print(f"Held-out test set saved to test_split.csv ({len(X_test_features)} rows)")


if __name__ == "__main__":
    train()
