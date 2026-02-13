"""
ML inference module for the F1 Race Intelligence Engine.

Loads trained models and generates predictions for a given race setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"


def _load_model(model_name: str) -> dict | None:
    """Load the latest version of a model."""
    path = MODEL_DIR / f"{model_name}_latest.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_probabilities(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate win, podium, and DNF probabilities for a set of drivers.

    Args:
        features_df: DataFrame with FEATURE_COLUMNS indexed per driver

    Returns:
        DataFrame with columns: driver_id, win_prob, podium_prob, dnf_prob,
                                 expected_position
    """
    win_bundle = _load_model("win_model")
    podium_bundle = _load_model("podium_model")
    dnf_bundle = _load_model("dnf_model")

    if not all([win_bundle, podium_bundle, dnf_bundle]):
        raise FileNotFoundError(
            "One or more models not found. Run `python -m ml.train` first."
        )

    feature_cols = win_bundle["feature_columns"]
    X = features_df[feature_cols].values

    # Fill NaN with defaults
    X = np.nan_to_num(X, nan=0.0)

    win_probs = win_bundle["model"].predict_proba(X)[:, 1]
    podium_probs = podium_bundle["model"].predict_proba(X)[:, 1]
    dnf_probs = dnf_bundle["model"].predict_proba(X)[:, 1]

    # Estimate expected position from probabilities
    # Higher win/podium prob → lower (better) expected position
    n_drivers = len(features_df)
    composite_score = (
        0.5 * (1 - win_probs) +
        0.3 * (1 - podium_probs) +
        0.2 * dnf_probs
    )
    # Rank by composite score → expected position
    rank_order = np.argsort(composite_score)
    expected_positions = np.empty(n_drivers, dtype=float)
    for rank, idx in enumerate(rank_order):
        expected_positions[idx] = rank + 1

    result = pd.DataFrame({
        "driver_id": features_df["driver_id"].values if "driver_id" in features_df.columns else range(n_drivers),
        "win_prob": np.round(win_probs, 4),
        "podium_prob": np.round(podium_probs, 4),
        "dnf_prob": np.round(dnf_probs, 4),
        "expected_position": expected_positions,
    })

    # Normalize win probabilities to sum to ~1
    win_sum = result["win_prob"].sum()
    if win_sum > 0:
        result["win_prob"] = np.round(result["win_prob"] / win_sum, 4)

    return result.sort_values("expected_position").reset_index(drop=True)


def get_model_info() -> dict:
    """Return metadata about currently loaded models."""
    info = {}
    for name in ["win_model", "podium_model", "dnf_model"]:
        bundle = _load_model(name)
        if bundle:
            info[name] = {
                "trained_at": bundle.get("trained_at"),
                "metrics": bundle.get("metrics"),
                "train_seasons": bundle.get("train_seasons"),
                "test_seasons": bundle.get("test_seasons"),
            }
        else:
            info[name] = None
    return info
