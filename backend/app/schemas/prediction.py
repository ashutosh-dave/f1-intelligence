"""
Pydantic schemas for prediction API requests and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for POST /predict."""
    season_year: int = Field(..., ge=2010, le=2030, description="Season year")
    round_number: int = Field(..., ge=1, le=30, description="Race round number")

    class Config:
        json_schema_extra = {
            "example": {
                "season_year": 2023,
                "round_number": 5,
            }
        }


class SimulateRequest(BaseModel):
    """Query parameters for GET /simulate."""
    season_year: int = Field(..., ge=2010, le=2030)
    round_number: int = Field(..., ge=1, le=30)
    n_simulations: int = Field(5000, ge=100, le=20000)

    class Config:
        json_schema_extra = {
            "example": {
                "season_year": 2023,
                "round_number": 5,
                "n_simulations": 5000,
            }
        }


class BacktestRequest(BaseModel):
    """Query parameters for GET /backtest."""
    seasons: list[int] = Field(default=[2023], description="Seasons to backtest")


# ─── Response Schemas ──────────────────────────────────────────────────────────

class DriverPrediction(BaseModel):
    """Prediction for a single driver."""
    driver_id: int
    driver_name: str
    driver_code: Optional[str] = None
    constructor: Optional[str] = None

    # Probabilities
    win_probability: float = Field(ge=0, le=1)
    podium_probability: float = Field(ge=0, le=1)
    dnf_probability: float = Field(ge=0, le=1)
    expected_position: float
    position_std: Optional[float] = None

    # Source breakdown
    ml_win_prob: Optional[float] = None
    sim_win_prob: Optional[float] = None

    # Position distribution (position → probability)
    position_distribution: Optional[dict[str, float]] = None


class PredictResponse(BaseModel):
    """Response for POST /predict."""
    season_year: int
    round_number: int
    race_name: Optional[str] = None
    circuit_name: Optional[str] = None
    predictions: list[DriverPrediction]
    model_info: Optional[dict] = None


class SimulateResponse(BaseModel):
    """Response for GET /simulate."""
    season_year: int
    round_number: int
    race_name: Optional[str] = None
    n_simulations: int
    avg_safety_cars_per_race: float
    drivers: list[DriverPrediction]


class BacktestMetrics(BaseModel):
    """Evaluation metrics for a single season."""
    season: int
    n_races: int
    log_loss: float
    brier_score: float
    top3_accuracy: float
    rank_correlation: float


class BacktestResponse(BaseModel):
    """Response for GET /backtest."""
    seasons: list[int]
    overall_metrics: dict
    per_season: list[BacktestMetrics]
