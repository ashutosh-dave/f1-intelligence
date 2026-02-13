"""
Ensemble combiner — merges ML predictions with simulator outputs.

Produces final race outcome probabilities using a weighted combination
of the ML model predictions and the Monte Carlo simulation frequencies.
"""

from __future__ import annotations

from typing import Optional

from backend.app.config import settings


def combine_predictions(
    ml_predictions: dict[int, dict],
    sim_results: dict,
    ml_weight: float | None = None,
    sim_weight: float | None = None,
) -> list[dict]:
    """
    Combine ML model predictions with simulation results.

    Args:
        ml_predictions: {driver_id: {win_prob, podium_prob, dnf_prob, expected_position}}
        sim_results: simulator AggregatedResults.to_dict() output
        ml_weight: weight for ML predictions (default from config)
        sim_weight: weight for simulation results (default from config)

    Returns:
        List of combined driver prediction dicts, sorted by expected position
    """
    w_ml = ml_weight or settings.ml_weight
    w_sim = sim_weight or settings.sim_weight

    # Normalize weights
    total_w = w_ml + w_sim
    w_ml /= total_w
    w_sim /= total_w

    # Build sim lookup
    sim_lookup = {}
    for driver in sim_results.get("drivers", []):
        sim_lookup[driver["driver_id"]] = driver

    combined = []
    all_driver_ids = set(ml_predictions.keys()) | set(sim_lookup.keys())

    for did in all_driver_ids:
        ml = ml_predictions.get(did, {})
        sim = sim_lookup.get(did, {})

        # Weighted combination
        win_prob = (
            w_ml * ml.get("win_prob", 0) +
            w_sim * sim.get("win_probability", 0)
        )
        podium_prob = (
            w_ml * ml.get("podium_prob", 0) +
            w_sim * sim.get("podium_probability", 0)
        )
        dnf_prob = (
            w_ml * ml.get("dnf_prob", 0) +
            w_sim * sim.get("dnf_rate", 0)
        )
        expected_pos = (
            w_ml * ml.get("expected_position", 15) +
            w_sim * sim.get("expected_position", 15)
        )

        combined.append({
            "driver_id": did,
            "name": sim.get("name", ml.get("name", f"Driver {did}")),
            "win_probability": round(win_prob, 4),
            "podium_probability": round(podium_prob, 4),
            "dnf_probability": round(dnf_prob, 4),
            "expected_position": round(expected_pos, 2),
            "position_std": sim.get("position_std"),
            "position_distribution": sim.get("position_distribution", {}),
            # Source breakdown
            "ml_win_prob": round(ml.get("win_prob", 0), 4),
            "sim_win_prob": round(sim.get("win_probability", 0), 4),
        })

    # Normalize win probabilities
    win_total = sum(d["win_probability"] for d in combined)
    if win_total > 0:
        for d in combined:
            d["win_probability"] = round(d["win_probability"] / win_total, 4)

    combined.sort(key=lambda d: d["expected_position"])
    return combined
