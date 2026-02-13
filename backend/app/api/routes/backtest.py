"""
Backtesting API endpoint.

GET /backtest — runs evaluation across historical seasons and returns metrics.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from scipy import stats
from sqlalchemy.orm import Session

from backend.app.schemas.prediction import (
    BacktestMetrics,
    BacktestResponse,
)
from data.database import get_db
from data.models import (
    Constructor, Driver, FeatureRow,
    Race, RaceResult, Weather,
)
from simulator.config import SimulationConfig
from simulator.engine import DriverSetup, RaceSimulator

router = APIRouter()


def _compute_race_metrics(
    actual_positions: list[int | None],
    predicted_probs: list[float],
    predicted_positions: list[float],
    actual_wins: list[int],
) -> dict:
    """Compute evaluation metrics for a single race."""
    # Filter out DNFs for position metrics
    valid = [
        (ap, pp, prob)
        for ap, pp, prob in zip(actual_positions, predicted_positions, predicted_probs)
        if ap is not None
    ]

    if not valid:
        return {}

    actual_pos = [v[0] for v in valid]
    pred_pos = [v[1] for v in valid]

    # Log loss (binary cross-entropy)
    eps = 1e-15
    probs_clipped = [max(eps, min(1 - eps, p)) for p in predicted_probs]
    log_loss_val = -np.mean([
        y * np.log(p) + (1 - y) * np.log(1 - p)
        for y, p in zip(actual_wins, probs_clipped)
    ])

    # Brier score
    brier = np.mean([
        (p - y) ** 2
        for y, p in zip(actual_wins, predicted_probs)
    ])

    # Top-3 accuracy
    pred_top3 = set(
        did for did, pos in zip(range(len(pred_pos)), pred_pos)
        if pos <= 3
    )
    actual_top3 = set(
        did for did, pos in zip(range(len(actual_pos)), actual_pos)
        if pos <= 3
    )
    top3_overlap = len(pred_top3 & actual_top3)
    top3_acc = top3_overlap / max(len(actual_top3), 1)

    # Rank correlation (Spearman)
    if len(actual_pos) >= 3:
        corr, _ = stats.spearmanr(actual_pos, pred_pos)
    else:
        corr = 0.0

    return {
        "log_loss": float(log_loss_val),
        "brier_score": float(brier),
        "top3_accuracy": float(top3_acc),
        "rank_correlation": float(corr) if not np.isnan(corr) else 0.0,
    }


@router.get("/backtest", response_model=BacktestResponse)
async def backtest(
    seasons: str = Query("2023", description="Comma-separated season years"),
    n_simulations: int = Query(1000, ge=100, le=10000),
    db: Session = Depends(get_db),
):
    """
    Run backtesting across historical seasons.

    For each race in the specified seasons, runs simulation and compares
    predicted outcomes to actual results.
    """
    season_list = [int(s.strip()) for s in seasons.split(",")]

    per_season_results = []
    all_metrics = {"log_loss": [], "brier_score": [], "top3_accuracy": [], "rank_correlation": []}

    for year in season_list:
        races = (
            db.query(Race)
            .filter_by(season_year=year)
            .order_by(Race.round)
            .all()
        )

        if not races:
            continue

        season_metrics = {
            "log_loss": [], "brier_score": [],
            "top3_accuracy": [], "rank_correlation": [],
        }

        for race in races:
            results = db.query(RaceResult).filter_by(race_id=race.id).all()
            if not results:
                continue

            weather = db.query(Weather).filter_by(race_id=race.id).first()

            # Build driver setups
            base_pace = 90.0
            driver_setups = []
            actual_positions = []
            actual_wins = []

            for res in results:
                driver = db.query(Driver).get(res.driver_id)
                pace_offset = (res.grid - 1) * 0.12

                feature = (
                    db.query(FeatureRow)
                    .filter_by(race_id=race.id, driver_id=res.driver_id)
                    .first()
                )
                dnf_prob = feature.driver_dnf_rate if feature and feature.driver_dnf_rate else 0.05
                name = f"{driver.first_name} {driver.last_name}" if driver else f"Driver {res.driver_id}"

                driver_setups.append(DriverSetup(
                    driver_id=res.driver_id,
                    name=name,
                    grid_position=res.grid,
                    base_pace=base_pace + pace_offset,
                    dnf_probability=dnf_prob,
                ))
                actual_positions.append(res.position)
                actual_wins.append(1 if res.position == 1 else 0)

            # Run simulation
            config = SimulationConfig(
                n_simulations=n_simulations,
                total_laps=race.total_laps or 55,
            )
            rain_prob = weather.rain_probability if weather else 0.0
            simulator = RaceSimulator(driver_setups, config, rain_probability=rain_prob)
            agg = simulator.run()

            # Extract predictions
            predicted_probs = [agg.win_probability.get(d.driver_id, 0) for d in driver_setups]
            predicted_positions = [agg.expected_position.get(d.driver_id, 15) for d in driver_setups]

            # Compute metrics
            metrics = _compute_race_metrics(
                actual_positions, predicted_probs, predicted_positions, actual_wins
            )

            for key in season_metrics:
                if key in metrics:
                    season_metrics[key].append(metrics[key])

        # Season averages
        if season_metrics["log_loss"]:
            season_avg = {
                k: float(np.mean(v)) for k, v in season_metrics.items() if v
            }
            per_season_results.append(BacktestMetrics(
                season=year,
                n_races=len(races),
                log_loss=season_avg.get("log_loss", 0),
                brier_score=season_avg.get("brier_score", 0),
                top3_accuracy=season_avg.get("top3_accuracy", 0),
                rank_correlation=season_avg.get("rank_correlation", 0),
            ))

            for k, v in season_avg.items():
                all_metrics[k].append(v)

    # Overall averages
    overall = {
        k: round(float(np.mean(v)), 4) if v else 0.0
        for k, v in all_metrics.items()
    }

    return BacktestResponse(
        seasons=season_list,
        overall_metrics=overall,
        per_season=per_season_results,
    )
