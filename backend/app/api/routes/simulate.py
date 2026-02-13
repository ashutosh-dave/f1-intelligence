"""
Simulation API endpoint.

GET /simulate — runs Monte Carlo simulation and returns finishing distributions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.schemas.prediction import DriverPrediction, SimulateResponse
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


@router.get("/simulate", response_model=SimulateResponse)
async def simulate(
    season_year: int = Query(..., ge=2010, le=2030),
    round_number: int = Query(..., ge=1, le=30),
    n_simulations: int = Query(5000, ge=100, le=20000),
    seed: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Run Monte Carlo race simulation.

    Returns probability distributions of finishing positions
    for each driver in the specified race, using the modular
    simulation engine.
    """
    race = (
        db.query(Race)
        .filter_by(season_year=season_year, round=round_number)
        .first()
    )
    if not race:
        raise HTTPException(
            status_code=404,
            detail=f"Race not found: {season_year} round {round_number}"
        )

    circuit = db.query(Circuit).get(race.circuit_id)
    weather = db.query(Weather).filter_by(race_id=race.id).first()

    results = db.query(RaceResult).filter_by(race_id=race.id).all()
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this race")

    # Build track profile
    track = TrackProfile(
        name=circuit.name if circuit else "Unknown",
        total_laps=race.total_laps or 55,
        pit_loss_sec=22.0,
    )

    # Build weather conditions
    rain_prob = weather.rain_probability if weather else 0.0
    weather_cond = WeatherConditions(
        rain_probability=rain_prob,
        conditions="wet" if rain_prob > 0.5 else "dry",
    )

    # Build rich driver inputs
    driver_inputs = []
    for res in results:
        driver = db.query(Driver).get(res.driver_id)
        constructor = db.query(Constructor).get(res.constructor_id) if res.constructor_id else None

        feature = (
            db.query(FeatureRow)
            .filter_by(race_id=race.id, driver_id=res.driver_id)
            .first()
        )
        dnf_rate = feature.driver_dnf_rate if feature and feature.driver_dnf_rate else 0.05

        name = f"{driver.first_name} {driver.last_name}" if driver else f"Driver {res.driver_id}"

        driver_inputs.append(DriverInput(
            driver_id=res.driver_id,
            name=name,
            driver_code=driver.code if driver else "",
            constructor=constructor.name if constructor else "",
            grid_position=res.grid,
            team_reliability=1.0 - dnf_rate,
        ))

    # Build structured input
    race_input = RaceInput(
        drivers=driver_inputs,
        track=track,
        weather=weather_cond,
        season_year=season_year,
        round_number=round_number,
        race_name=race.name,
    )

    # Run simulation
    config = SimulationConfig(n_simulations=n_simulations, seed=seed)
    simulator = RaceSimulator.from_race_input(race_input, config)
    agg = simulator.run()
    result_dict = agg.to_dict()

    # Build response
    drivers = []
    for d in result_dict["drivers"]:
        drivers.append(DriverPrediction(
            driver_id=d["driver_id"],
            driver_name=d.get("name", "Unknown"),
            driver_code=d.get("driver_code") or None,
            constructor=d.get("constructor") or None,
            win_probability=d["win_probability"],
            podium_probability=d["podium_probability"],
            dnf_probability=d["dnf_rate"],
            expected_position=d["expected_position"],
            position_std=d.get("position_std"),
            position_distribution=d.get("position_distribution"),
        ))

    return SimulateResponse(
        season_year=season_year,
        round_number=round_number,
        race_name=race.name,
        n_simulations=n_simulations,
        avg_safety_cars_per_race=result_dict["avg_safety_cars_per_race"],
        drivers=drivers,
    )
