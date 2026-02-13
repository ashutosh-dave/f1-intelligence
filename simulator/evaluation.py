"""
Evaluation and calibration tools for the Monte Carlo simulator.

Provides utilities to:
  1. Compare simulated probability distributions against historical
     race results.
  2. Measure calibration (predicted vs observed win rates).
  3. Sweep simulation parameters and quantify output sensitivity.
  4. Report variance / entropy metrics across simulations.

All functions accept the structured SimulationOutput schema and
historical actual results as simple dicts/lists.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from simulator.schemas import DriverResult, SimulationOutput


# ─── Historical Comparison ──────────────────────────────────────────────────────

@dataclass
class HistoricalComparison:
    """Comparison of simulator output vs actual race result."""
    driver_id: int
    driver_name: str
    predicted_position: float    # expected finish from simulation
    actual_position: int         # actual race finish
    position_error: float        # predicted − actual
    win_predicted: float         # simulated win probability
    actually_won: bool           # did they win for real?
    podium_predicted: float      # simulated podium probability
    actually_podium: bool        # did they finish top 3?


def compare_to_historical(
    sim_output: SimulationOutput,
    actual_results: dict[int, int],
) -> list[HistoricalComparison]:
    """
    Compare simulation output to actual race finishing positions.

    Args:
        sim_output:      SimulationOutput from the engine.
        actual_results:  Mapping of driver_id → actual finishing position.

    Returns:
        List of per-driver comparison records, sorted by actual position.
    """
    comparisons = []
    for driver in sim_output.drivers:
        actual = actual_results.get(driver.driver_id)
        if actual is None:
            continue
        comparisons.append(HistoricalComparison(
            driver_id=driver.driver_id,
            driver_name=driver.name,
            predicted_position=driver.expected_position,
            actual_position=actual,
            position_error=driver.expected_position - actual,
            win_predicted=driver.win_probability,
            actually_won=(actual == 1),
            podium_predicted=driver.podium_probability,
            actually_podium=(actual <= 3),
        ))
    comparisons.sort(key=lambda c: c.actual_position)
    return comparisons


def compute_comparison_metrics(
    comparisons: list[HistoricalComparison],
) -> dict[str, float]:
    """
    Compute aggregate metrics from a list of comparisons.

    Metrics:
      - MAE:            Mean absolute error of expected vs actual position.
      - RMSE:           Root mean squared error.
      - rank_correlation: Spearman rank correlation.
      - winner_correct: Did the highest-probability driver actually win?
      - top3_overlap:   Fraction of predicted top-3 that appeared in actual top-3.
    """
    if not comparisons:
        return {}

    n = len(comparisons)
    errors = [abs(c.position_error) for c in comparisons]
    sq_errors = [c.position_error ** 2 for c in comparisons]
    mae = sum(errors) / n
    rmse = (sum(sq_errors) / n) ** 0.5

    # Spearman rank correlation
    predicted_ranks = sorted(comparisons, key=lambda c: c.predicted_position)
    pred_rank_map = {c.driver_id: i + 1 for i, c in enumerate(predicted_ranks)}
    actual_rank_map = {c.driver_id: c.actual_position for c in comparisons}

    d_sq_sum = 0
    common = [did for did in pred_rank_map if did in actual_rank_map]
    nn = len(common)
    for did in common:
        d = pred_rank_map[did] - actual_rank_map[did]
        d_sq_sum += d * d
    if nn > 1:
        rank_corr = 1 - (6 * d_sq_sum) / (nn * (nn ** 2 - 1))
    else:
        rank_corr = 0.0

    # Winner check
    best_predicted = max(comparisons, key=lambda c: c.win_predicted)
    winner_correct = best_predicted.actually_won

    # Top-3 overlap
    pred_top3_ids = {c.driver_id for c in sorted(comparisons, key=lambda c: c.predicted_position)[:3]}
    actual_top3_ids = {c.driver_id for c in comparisons if c.actual_position <= 3}
    top3_overlap = len(pred_top3_ids & actual_top3_ids) / 3.0 if actual_top3_ids else 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "rank_correlation": round(rank_corr, 4),
        "winner_correct": winner_correct,
        "top3_overlap": round(top3_overlap, 4),
    }


# ─── Calibration ────────────────────────────────────────────────────────────────

@dataclass
class CalibrationBucket:
    """One bin of the calibration curve."""
    predicted_low: float
    predicted_high: float
    predicted_mean: float
    observed_rate: float
    n_samples: int


def calibration_report(
    predictions: list[tuple[float, bool]],
    n_buckets: int = 10,
) -> list[CalibrationBucket]:
    """
    Compute calibration buckets for a binary outcome.

    Args:
        predictions:  List of (predicted_probability, actual_outcome).
        n_buckets:    Number of bins.

    Returns:
        List of CalibrationBuckets from lowest to highest predicted probability.
    """
    if not predictions:
        return []

    sorted_preds = sorted(predictions, key=lambda x: x[0])
    bucket_size = max(1, len(sorted_preds) // n_buckets)
    buckets = []

    for i in range(0, len(sorted_preds), bucket_size):
        chunk = sorted_preds[i:i + bucket_size]
        probs = [p for p, _ in chunk]
        outcomes = [o for _, o in chunk]
        buckets.append(CalibrationBucket(
            predicted_low=min(probs),
            predicted_high=max(probs),
            predicted_mean=sum(probs) / len(probs),
            observed_rate=sum(outcomes) / len(outcomes),
            n_samples=len(chunk),
        ))
    return buckets


# ─── Sensitivity Analysis ───────────────────────────────────────────────────────

@dataclass
class SensitivityResult:
    """Result of sweeping a single parameter."""
    parameter_name: str
    values_tested: list[float]
    win_probs_by_driver: dict[int, list[float]]  # driver_id → [prob at each value]
    expected_pos_by_driver: dict[int, list[float]]


def sensitivity_analysis(
    simulator_factory,
    param_name: str,
    param_values: list[float],
) -> SensitivityResult:
    """
    Sweep a config parameter and record how outputs change.

    Args:
        simulator_factory:  Callable(param_value) → RaceSimulator.
                            The caller is responsible for constructing the
                            simulator with the varied parameter.
        param_name:         Name of the parameter being swept.
        param_values:       List of values to test.

    Returns:
        SensitivityResult with per-driver metrics at each parameter value.
    """
    win_probs: dict[int, list[float]] = defaultdict(list)
    expected_pos: dict[int, list[float]] = defaultdict(list)

    for val in param_values:
        sim = simulator_factory(val)
        results = sim.run()
        for did in results.driver_ids:
            win_probs[did].append(results.win_probability[did])
            expected_pos[did].append(results.expected_position[did])

    return SensitivityResult(
        parameter_name=param_name,
        values_tested=param_values,
        win_probs_by_driver=dict(win_probs),
        expected_pos_by_driver=dict(expected_pos),
    )


# ─── Variance Report ────────────────────────────────────────────────────────────

@dataclass
class VarianceReport:
    """Aggregate variance and entropy metrics."""
    avg_position_std: float
    max_position_std: float
    min_position_std: float
    win_entropy: float              # Shannon entropy of win distribution
    effective_competitors: float    # exp(entropy): how many "real" contenders


def variance_report(sim_output: SimulationOutput) -> VarianceReport:
    """
    Compute variance and entropy metrics from simulation output.

    The 'effective competitors' metric (exp of entropy) tells you
    how competitive the race is:  1.0 = one dominant driver,
    20.0 = perfectly competitive.
    """
    stds = [d.position_std for d in sim_output.drivers]
    win_probs = [d.win_probability for d in sim_output.drivers if d.win_probability > 0]

    if win_probs:
        entropy = -sum(p * math.log2(p) for p in win_probs)
        effective = 2 ** entropy
    else:
        entropy = 0.0
        effective = 1.0

    return VarianceReport(
        avg_position_std=round(sum(stds) / max(len(stds), 1), 3),
        max_position_std=round(max(stds) if stds else 0, 3),
        min_position_std=round(min(stds) if stds else 0, 3),
        win_entropy=round(entropy, 4),
        effective_competitors=round(effective, 2),
    )
