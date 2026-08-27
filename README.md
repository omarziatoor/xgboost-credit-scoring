# XGBoost Credit Scoring Model

A properly validated credit scoring pipeline: predicts probability of default (PD) with XGBoost, calibrates those probabilities, converts them into a conventional 300–850 score, and explains individual predictions with SHAP.

## Why this project exists

A separate credit-scoring project I reviewed used a hand-picked linear formula to *generate* its own training labels, then trained a model to approximate that formula — meaning the model's "accuracy" said nothing about real creditworthiness, since there was no real outcome data anywhere in the pipeline. This project is built the other way around: a model is only as good as the ground truth it learns from.

Since no real, labeled historical loan-outcome dataset was available, this project uses a **synthetic dataset with a known, hidden data-generating process** (`generate_synthetic_data.py`) — the point is to demonstrate and validate a *correct pipeline structure*, not to claim real-world predictive accuracy. Swapping in a real historical dataset (with a genuine `defaulted` outcome column) is a drop-in replacement — no other code needs to change.

## Pipeline

1. **`generate_synthetic_data.py`** — generates a synthetic applicant dataset where default risk is a real (but noisy) function of the features, with a genuine binary `defaulted` outcome label
2. **`train.py`** — trains an XGBoost classifier on a proper train/validation/test split, with early stopping and class-imbalance handling, then applies isotonic probability calibration on top
3. **`evaluate.py`** — evaluates on a held-out test set using metrics appropriate for a PD model: AUC-ROC, the KS statistic (the standard credit-industry separation metric), and a calibration curve
4. **`scorecard.py`** — converts calibrated default probabilities into a conventional 300–850 score via a log-odds transform (the way real scorecards derive a score from a model, rather than training a regressor to directly output a score)
5. **`explain.py`** — SHAP-based global feature importance and per-customer explanations, for the kind of "why was this applicant scored this way" explainability real lending decisions require

## Key design decisions (and why)

- **Binary classification on a real outcome, not regression on a heuristic.** The target is `defaulted` (0/1) — the model predicts probability of default, which is then mapped to a score. This is the industry-standard structure.
- **Calibration matters as much as ranking.** `scale_pos_weight` (used to handle class imbalance) improves AUC but skews raw predicted probabilities — see `results/calibration_curve.png` before vs. after isotonic calibration was added.
- **Evaluation uses credit-scoring-appropriate metrics** — AUC, KS statistic, calibration — not accuracy or metrics mismatched to the model type.
- **SHAP explainability** — gradient boosting is a black box by default; SHAP recovers per-feature, per-prediction attribution, which matters for adverse-action explainability in lending.

## Results (on the synthetic dataset)

- **AUC-ROC**: ~0.92 on held-out test data
- **KS statistic**: ~0.68 (values above 0.4 are generally considered good separation in credit scoring)
- Calibration curve closely tracks the diagonal after isotonic calibration (see `results/calibration_curve.png`)

These numbers reflect a synthetic dataset with a known, moderately noisy signal — they are a validation of the pipeline's correctness, not a claim about real-world credit risk prediction.

## Setup

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
python train.py
python evaluate.py
python scorecard.py
python explain.py
```

## Using real data instead

Replace `credit_data.csv` with a real historical dataset containing:
- the feature columns listed in `config.py` (`NUMERICAL_FEATURES`) — or update that list to match your real features
- a genuine binary outcome column named `defaulted` (or update `config.TARGET`)

No other code changes are required — the rest of the pipeline is agnostic to where the data came from.
