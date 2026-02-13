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
from simulator.engine import DriverSetup, RaceSimulator

router = APIRouter()


@router.get("/simulate", response_model=SimulateResponse)
async def simulate(
    season_year: int = Query(..., ge=2010, le=2030),
    round_number: int = Query(..., ge=1, le=30),
    n_simulations: int = Query(5000, ge=100, le=20000),
    db: Session = Depends(get_db),
):
    """
    Run Monte Carlo race simulation.

    Returns probability distributions of finishing positions
    for each driver in the specified race.
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

    # Build driver setups
    base_pace = 90.0
    driver_setups = []

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

    # Run simulation
    config = SimulationConfig(
        n_simulations=n_simulations,
        total_laps=race.total_laps or 55,
    )
    rain_prob = weather.rain_probability if weather else 0.0
    simulator = RaceSimulator(driver_setups, config, rain_probability=rain_prob)
    agg = simulator.run()
    result_dict = agg.to_dict()

    # Build response
    drivers = []
    for d in result_dict["drivers"]:
        did = d["driver_id"]
        driver = db.query(Driver).get(did)
        constructor_id = None
        for res in results:
            if res.driver_id == did:
                constructor_id = res.constructor_id
                break
        constructor = db.query(Constructor).get(constructor_id) if constructor_id else None

        drivers.append(DriverPrediction(
            driver_id=did,
            driver_name=d.get("name", "Unknown"),
            driver_code=driver.code if driver else None,
            constructor=constructor.name if constructor else None,
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
