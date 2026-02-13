"""
ML training pipeline for the F1 Race Intelligence Engine.

Trains two classification models:
  1. Win probability (binary: did driver win the race?)
  2. Podium probability (binary: finished in top 3?)

Uses XGBoost with Platt scaling for calibrated probability outputs.
Models are saved as versioned .joblib artifacts.

Usage:
    python -m ml.train
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from data.database import SessionLocal, init_db
from data.models import FeatureRow, Race

# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "driver_form",
    "driver_win_rate",
    "driver_podium_rate",
    "driver_dnf_rate",
    "team_strength",
    "team_reliability",
    "track_affinity",
    "circuit_type_street",
    "circuit_type_hybrid",
    "grid_position",
    "qualifying_delta_sec",
    "rain_probability",
    "era_v8",
    "era_hybrid_v6",
    "era_ground_effect",
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_feature_dataframe() -> pd.DataFrame:
    """Load features from DB into a pandas DataFrame, sorted by race date."""
    init_db()
    db = SessionLocal()
    try:
        rows = (
            db.query(FeatureRow, Race.race_date, Race.season_year, Race.round)
            .join(Race, FeatureRow.race_id == Race.id)
            .order_by(Race.race_date)
            .all()
        )

        if not rows:
            print("  ⚠ No feature rows found. Run data ingestion and feature engineering first.")
            sys.exit(1)

        records = []
        for feat, race_date, season_year, rnd in rows:
            record = {
                "race_id": feat.race_id,
                "driver_id": feat.driver_id,
                "race_date": race_date,
                "season_year": season_year,
                "round": rnd,
            }
            for col in FEATURE_COLUMNS:
                record[col] = getattr(feat, col)

            # Targets
            record["is_winner"] = feat.is_winner
            record["is_podium"] = feat.is_podium
            record["is_dnf"] = feat.is_dnf
            record["finished_position"] = feat.finished_position
            records.append(record)

        df = pd.DataFrame(records)

        # Convert booleans to int
        bool_cols = [
            "circuit_type_street", "circuit_type_hybrid",
            "era_v8", "era_hybrid_v6", "era_ground_effect",
            "is_winner", "is_podium", "is_dnf",
        ]
        for col in bool_cols:
            df[col] = df[col].astype(int)

        print(f"  Loaded {len(df)} feature rows spanning {df['season_year'].nunique()} seasons")
        return df
    finally:
        db.close()


# ─── Training ─────────────────────────────────────────────────────────────────

def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature values with sensible defaults."""
    defaults = {
        "driver_form": 12.0,       # midfield assumption
        "driver_win_rate": 0.0,
        "driver_podium_rate": 0.0,
        "driver_dnf_rate": 0.1,
        "team_strength": 5.0,
        "team_reliability": 0.85,
        "track_affinity": 10.0,
        "grid_position": 15,
        "qualifying_delta_sec": 3.0,
        "rain_probability": 0.1,
    }
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


def train_model(
    target: str,
    model_name: str,
    df: pd.DataFrame,
    test_seasons: int = 2,
) -> dict:
    """
    Train and evaluate a binary classifier.

    Args:
        target: Target column name ('is_winner' or 'is_podium')
        model_name: Name for the saved model file
        df: Feature DataFrame
        test_seasons: Number of most recent seasons for test set

    Returns:
        Dictionary of evaluation metrics
    """
    df = _fill_missing(df.copy())

    # Time-based train/test split (no data leakage)
    all_seasons = sorted(df["season_year"].unique())
    train_seasons = all_seasons[:-test_seasons]
    test_season_list = all_seasons[-test_seasons:]

    train_mask = df["season_year"].isin(train_seasons)
    test_mask = df["season_year"].isin(test_season_list)

    X_train = df.loc[train_mask, FEATURE_COLUMNS].values
    y_train = df.loc[train_mask, target].values
    X_test = df.loc[test_mask, FEATURE_COLUMNS].values
    y_test = df.loc[test_mask, target].values

    print(f"\n{'─'*60}")
    print(f"  Training: {model_name}")
    print(f"  Train seasons: {train_seasons[0]}–{train_seasons[-1]} ({len(y_train)} samples)")
    print(f"  Test seasons:  {test_season_list[0]}–{test_season_list[-1]} ({len(y_test)} samples)")
    print(f"  Positive rate:  Train={y_train.mean():.3f}  Test={y_test.mean():.3f}")
    print(f"{'─'*60}")

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Train XGBoost
    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Calibrate probabilities using Platt scaling
    calibrated = CalibratedClassifierCV(xgb, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # Evaluate
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "log_loss": log_loss(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else None,
    }

    print(f"\n  📊 Evaluation Metrics:")
    print(f"     Log Loss:    {metrics['log_loss']:.4f}")
    print(f"     Brier Score: {metrics['brier_score']:.4f}")
    if metrics["roc_auc"]:
        print(f"     ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance
    importances = xgb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"  🔑 Top Feature Importances:")
    for i in sorted_idx[:8]:
        print(f"     {FEATURE_COLUMNS[i]:25s} {importances[i]:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"{model_name}_v{timestamp}.joblib"
    latest_path = MODEL_DIR / f"{model_name}_latest.joblib"

    joblib.dump({
        "model": calibrated,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "trained_at": timestamp,
        "train_seasons": list(train_seasons),
        "test_seasons": list(test_season_list),
    }, model_path)

    # Also save as 'latest'
    joblib.dump({
        "model": calibrated,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "trained_at": timestamp,
        "train_seasons": list(train_seasons),
        "test_seasons": list(test_season_list),
    }, latest_path)

    print(f"\n  💾 Model saved: {model_path}")
    print(f"  💾 Latest link: {latest_path}")

    return metrics


# ─── Main ──────────────────────────────────────────────────────────────────────

def train_all():
    """Train all ML models."""
    df = load_feature_dataframe()

    print(f"\n{'='*60}")
    print(f"  F1 RACE INTELLIGENCE — ML TRAINING PIPELINE")
    print(f"{'='*60}")

    results = {}

    # Win probability model
    results["win"] = train_model(
        target="is_winner",
        model_name="win_model",
        df=df,
    )

    # Podium probability model
    results["podium"] = train_model(
        target="is_podium",
        model_name="podium_model",
        df=df,
    )

    # DNF probability model
    results["dnf"] = train_model(
        target="is_dnf",
        model_name="dnf_model",
        df=df,
    )

    print(f"\n{'='*60}")
    print(f"  ✅ ALL MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*60}")

    return results


def main():
    train_all()


if __name__ == "__main__":
    main()
