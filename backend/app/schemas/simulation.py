"""
Pydantic schemas for simulation API.
"""

from pydantic import BaseModel
from typing import Optional


class SimulationDriverResult(BaseModel):
    """Simulation result for a single driver."""
    driver_id: int
    name: str
    win_probability: float
    podium_probability: float
    expected_position: float
    position_std: float
    dnf_rate: float
    position_distribution: dict[str, float]


class SimulationResponse(BaseModel):
    """Response for the simulation endpoint."""
    n_simulations: int
    avg_safety_cars_per_race: float
    drivers: list[SimulationDriverResult]
