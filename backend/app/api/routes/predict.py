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
from simulator.engine import DriverSetup, RaceSimulator

router = APIRouter()


def _build_driver_setups(
    db: Session,
    race: Race,
    results: list[RaceResult],
) -> list[DriverSetup]:
    """Build DriverSetup objects from race results and qualifying data."""
    setups = []
    base_pace = 90.0  # base lap time in seconds (normalized)

    for res in results:
        driver = db.query(Driver).get(res.driver_id)
        constructor = db.query(Constructor).get(res.constructor_id)

        # Estimate pace from grid position (rough correlation)
        pace_offset = (res.grid - 1) * 0.12  # ~0.12s per grid slot

        # Get qualifying delta for more precise pace
        quali = (
            db.query(Qualifying)
            .filter_by(race_id=race.id, driver_id=res.driver_id)
            .first()
        )
        if quali:
            # Use qualifying data for pace if available
            pace_offset = (quali.position - 1) * 0.1

        # Get DNF probability from features if available
        feature = (
            db.query(FeatureRow)
            .filter_by(race_id=race.id, driver_id=res.driver_id)
            .first()
        )
        dnf_prob = feature.driver_dnf_rate if feature and feature.driver_dnf_rate else 0.05

        name = f"{driver.first_name} {driver.last_name}" if driver else f"Driver {res.driver_id}"

        setups.append(DriverSetup(
            driver_id=res.driver_id,
            name=name,
            grid_position=res.grid,
            base_pace=base_pace + pace_offset,
            dnf_probability=dnf_prob,
        ))

    return setups


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Generate race outcome predictions.

    Combines ML model predictions with Monte Carlo simulation results
    using the ensemble combiner.
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
    weather = db.query(Weather).filter_by(race_id=race.id).first()

    # Get race results (we need the grid for simulation)
    results = db.query(RaceResult).filter_by(race_id=race.id).all()
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this race")

    # ── ML Predictions ──
    features = db.query(FeatureRow).filter_by(race_id=race.id).all()
    ml_preds = {}
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
            # ML models not trained yet — continue with simulation only
            pass

    # ── Monte Carlo Simulation ──
    driver_setups = _build_driver_setups(db, race, results)
    rain_prob = weather.rain_probability if weather else 0.0

    config = SimulationConfig(
        n_simulations=2000,  # fewer for API responsiveness
        total_laps=race.total_laps or 55,
    )
    simulator = RaceSimulator(driver_setups, config, rain_probability=rain_prob)
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
