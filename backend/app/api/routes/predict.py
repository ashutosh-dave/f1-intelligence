"""
Prediction API endpoint.

POST /predict — generates race outcome predictions for a given race.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.schemas.prediction import (
    DriverPrediction,
    PredictRequest,
    PredictResponse,
)
from backend.app.ensemble.combiner import combine_predictions
from data.database import get_db
from data.models import (
    Circuit, Constructor, Driver, FeatureRow,
    Qualifying, Race, RaceResult, Weather,
)
from simulator.config import SimulationConfig
from simulator.schemas import (
    DriverInput, MLPriors, RaceInput, TrackProfile, WeatherConditions,
)
from simulator.engine import RaceSimulator

router = APIRouter()


def _build_driver_inputs(
    db: Session,
    race: Race,
    results: list[RaceResult],
    ml_preds: dict[int, dict] | None = None,
) -> list[DriverInput]:
    """Build DriverInput objects from race results, qualifying, and ML predictions."""
    inputs = []

    for res in results:
        driver = db.query(Driver).get(res.driver_id)
        constructor = db.query(Constructor).get(res.constructor_id) if res.constructor_id else None

        # Qualifying data for pace
        quali = (
            db.query(Qualifying)
            .filter_by(race_id=race.id, driver_id=res.driver_id)
            .first()
        )
        grid = quali.position if quali else res.grid

        # DNF rate from features
        feature = (
            db.query(FeatureRow)
            .filter_by(race_id=race.id, driver_id=res.driver_id)
            .first()
        )
        dnf_rate = feature.driver_dnf_rate if feature and feature.driver_dnf_rate else 0.05

        name = f"{driver.first_name} {driver.last_name}" if driver else f"Driver {res.driver_id}"

        # ML priors (if available)
        ml_prior = None
        if ml_preds and res.driver_id in ml_preds:
            mp = ml_preds[res.driver_id]
            ml_prior = MLPriors(
                win_probability=mp.get("win_prob", 0.0),
                podium_probability=mp.get("podium_prob", 0.0),
                dnf_probability=mp.get("dnf_prob", 0.05),
                predicted_position=mp.get("expected_position", 10.0),
            )

        inputs.append(DriverInput(
            driver_id=res.driver_id,
            name=name,
            driver_code=driver.code if driver else "",
            constructor=constructor.name if constructor else "",
            grid_position=grid,
            team_reliability=1.0 - dnf_rate,
            ml_priors=ml_prior,
        ))

    return inputs


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Generate race outcome predictions.

    Combines ML model predictions with Monte Carlo simulation results
    using the ensemble combiner.  ML predictions are passed into the
    simulator as priors via the DriverInput schema.
    """
    # Look up the race
    race = (
        db.query(Race)
        .filter_by(season_year=request.season_year, round=request.round_number)
        .first()
    )
    if not race:
        raise HTTPException(
            status_code=404,
            detail=f"Race not found: {request.season_year} round {request.round_number}"
        )

    circuit = db.query(Circuit).get(race.circuit_id)
    weather_row = db.query(Weather).filter_by(race_id=race.id).first()

    results = db.query(RaceResult).filter_by(race_id=race.id).all()
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this race")

    # ── ML Predictions ──
    ml_preds: dict[int, dict] = {}
    features = db.query(FeatureRow).filter_by(race_id=race.id).all()
    if features:
        try:
            import pandas as pd
            from ml.predict import predict_probabilities
            from ml.train import FEATURE_COLUMNS

            feature_records = []
            for f in features:
                record = {"driver_id": f.driver_id}
                for col in FEATURE_COLUMNS:
                    record[col] = getattr(f, col)
                feature_records.append(record)

            df = pd.DataFrame(feature_records)
            pred_df = predict_probabilities(df)

            for _, row in pred_df.iterrows():
                ml_preds[int(row["driver_id"])] = {
                    "win_prob": float(row["win_prob"]),
                    "podium_prob": float(row["podium_prob"]),
                    "dnf_prob": float(row["dnf_prob"]),
                    "expected_position": float(row["expected_position"]),
                }
        except Exception:
            pass

    # ── Build structured input with ML priors ──
    driver_inputs = _build_driver_inputs(db, race, results, ml_preds)

    rain_prob = weather_row.rain_probability if weather_row else 0.0
    track = TrackProfile(
        name=circuit.name if circuit else "Unknown",
        total_laps=race.total_laps or 55,
    )
    weather_cond = WeatherConditions(
        rain_probability=rain_prob,
        conditions="wet" if rain_prob > 0.5 else "dry",
    )
    race_input = RaceInput(
        drivers=driver_inputs,
        track=track,
        weather=weather_cond,
        season_year=request.season_year,
        round_number=request.round_number,
        race_name=race.name,
    )

    # ── Monte Carlo Simulation ──
    config = SimulationConfig(n_simulations=2000)
    simulator = RaceSimulator.from_race_input(race_input, config)
    sim_results = simulator.run().to_dict()

    # ── Ensemble ──
    if ml_preds:
        combined = combine_predictions(ml_preds, sim_results)
    else:
        combined = sim_results.get("drivers", [])

    # Build response
    predictions = []
    for d in combined:
        did = d.get("driver_id")
        driver = db.query(Driver).get(did) if did else None
        constructor_id = None
        for res in results:
            if res.driver_id == did:
                constructor_id = res.constructor_id
                break
        constructor = db.query(Constructor).get(constructor_id) if constructor_id else None

        predictions.append(DriverPrediction(
            driver_id=did,
            driver_name=d.get("name", "Unknown"),
            driver_code=driver.code if driver else None,
            constructor=constructor.name if constructor else None,
            win_probability=d.get("win_probability", 0),
            podium_probability=d.get("podium_probability", 0),
            dnf_probability=d.get("dnf_probability", d.get("dnf_rate", 0)),
            expected_position=d.get("expected_position", 20),
            position_std=d.get("position_std"),
            ml_win_prob=d.get("ml_win_prob"),
            sim_win_prob=d.get("sim_win_prob"),
            position_distribution=d.get("position_distribution"),
        ))

    return PredictResponse(
        season_year=request.season_year,
        round_number=request.round_number,
        race_name=race.name,
        circuit_name=circuit.name if circuit else None,
        predictions=predictions,
    )
