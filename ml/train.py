"""
ml/train.py
-----------
Trains and compares multiple models for NCAA tournament game prediction.

Goal: beat KenPom ordinal rank as a standalone predictor.

Models compared:
  - LogisticRegression        (L2 regularized — strong baseline for log-loss)
  - LogisticRegression_C0.1   (heavier regularization)
  - Ridge_calibrated          (fast Ridge with Platt scaling)
  - RandomForestClassifier    (non-linear, can capture seed×efficiency interaction)
  - LightGBM                  (gradient boosting, fast, excellent on tabular data)
  - XGBoost                   (alternative gradient boosting)

Evaluation metrics (log-loss primary, then AUC, Brier score):
  - Log-loss  (primary — penalizes confident wrong predictions hard)
  - AUC-ROC   (overall ranking quality)
  - Brier score (mean squared error of probabilities — calibration-sensitive)

Cross-validation strategy: temporal holdout by season.
  Trains on seasons 2003 .. (test_season - 1), evaluates on test_season.
  This respects temporal data ordering — no lookahead bias.

Experiment results are appended to data/experiments.jsonl for comparison
across runs. No external tracking tool required.

Usage:
    uv run python ml/train.py
    uv run python ml/train.py --test-season 2024
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLS, build_training_features

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAS_XGB = False


EXPERIMENTS_FILE = Path(__file__).parent.parent / "data" / "experiments.jsonl"
MODEL_DIR         = Path(__file__).parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Experiment logging (simple JSON Lines — no MLflow dependency needed)
# ---------------------------------------------------------------------------

def _log_run(model_name: str, test_season: int, metrics: dict, params: dict) -> None:
    """Append a single experiment run to data/experiments.jsonl."""
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":   datetime.now().isoformat(),
        "model":       model_name,
        "test_season": test_season,
        "metrics":     metrics,
        "params":      params,
    }
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_experiment_history() -> pd.DataFrame:
    """Load all past experiment runs from data/experiments.jsonl."""
    if not EXPERIMENTS_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "timestamp":   r["timestamp"],
                "model":       r["model"],
                "test_season": r["test_season"],
                **{f"metric_{k}": v for k, v in r["metrics"].items()},
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    """
    Return a dict of {name: sklearn-compatible model} to evaluate.

    RidgeClassifier doesn't support predict_proba natively, so we wrap
    it with Platt scaling via CalibratedClassifierCV.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]),
        "LogisticRegression_C0.1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=42)),
        ]),
        "Ridge_calibrated": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                RidgeClassifier(alpha=1.0), cv=5, method="sigmoid"
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
        ),
    }
    if HAS_LGB:
        assert lgb is not None
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, verbose=-1
        )
    if HAS_XGB:
        assert xgb is not None
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42, verbosity=0
        )
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Compute log-loss, AUC, and Brier score for a fitted model.
    Returns: {"log_loss": float, "auc": float, "brier": float}
    """
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "log_loss": float(log_loss(y_test, proba)),
        "auc":      float(roc_auc_score(y_test, proba)),
        "brier":    float(brier_score_loss(y_test, proba)),
    }


def baseline_kenpom(X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Compute metrics for the KenPom-rank-only naive baseline.

    kenpom_rank_diff < 0 means team A is ranked better.
    We convert via a logistic transformation (scale=50 calibrated to upset rates).
    This is the bar we need to beat.
    """
    from scipy.special import expit
    rank_diff = X_test["kenpom_rank_diff"].fillna(0)
    proba = expit(-rank_diff / 50.0)
    return {
        "log_loss": float(log_loss(y_test, proba)),
        "auc":      float(roc_auc_score(y_test, proba)),
        "brier":    float(brier_score_loss(y_test, proba)),
    }


# ---------------------------------------------------------------------------
# Train / evaluate pipeline
# ---------------------------------------------------------------------------

def run_training(test_season: int = 2025) -> pd.DataFrame:
    """
    Train all models on seasons < test_season, evaluate on test_season.
    Saves best model to data/models/best.pkl.
    Returns a DataFrame comparing model metrics.
    """
    print("Loading features from Kaggle data...")
    all_features = build_training_features(min_season=2003, max_season=test_season)

    train_df = all_features[all_features["Season"] < test_season]
    test_df  = all_features[all_features["Season"] == test_season]

    print(
        f"Train: {len(train_df)} games ({train_df['Season'].nunique()} seasons) | "
        f"Test: {len(test_df)} games (season {test_season})"
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    models  = _build_models()
    results = []

    # Baseline: KenPom rank only
    baseline_metrics = baseline_kenpom(X_test, y_test)
    results.append({"model": "KenPom_baseline", **baseline_metrics})
    _log_run("KenPom_baseline", test_season, baseline_metrics, {})
    print(f"  KenPom baseline:  log-loss={baseline_metrics['log_loss']:.4f}  "
          f"AUC={baseline_metrics['auc']:.4f}  Brier={baseline_metrics['brier']:.4f}")

    best_model = None
    best_logloss = float("inf")

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results.append({"model": name, **metrics})
        _log_run(name, test_season, metrics, {"features": str(FEATURE_COLS)})

        print(f"    log-loss={metrics['log_loss']:.4f}  "
              f"AUC={metrics['auc']:.4f}  Brier={metrics['brier']:.4f}")

        if metrics["log_loss"] < best_logloss:
            best_logloss = metrics["log_loss"]
            best_model   = (name, model)

    results_df = (
        pd.DataFrame(results)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )

    print(f"\n{'='*65}")
    print(f"  Results: test season {test_season}  (lower log-loss = better)")
    print(f"{'='*65}")
    print(results_df.to_string(index=False))

    if best_model is not None:
        print(f"\nBest model: {best_model[0]}  log-loss={best_logloss:.4f}  "
              f"(KenPom baseline: {baseline_metrics['log_loss']:.4f})")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "best.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model[1], f)
        print(f"Saved best model → {model_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tournament prediction models.")
    parser.add_argument(
        "--test-season", type=int, default=2025,
        help="Season used as test set; all prior seasons train."
    )
    args = parser.parse_args()
    run_training(test_season=args.test_season)
