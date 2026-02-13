"""
Simulation configuration and parameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuration for the Monte Carlo race simulator."""

    # Number of simulation iterations
    n_simulations: int = 5000

    # Race parameters
    total_laps: int = 55

    # Pace variance (seconds per lap, Gaussian σ)
    pace_sigma: float = 0.35

    # Pit stop parameters
    min_stops: int = 1
    max_stops: int = 3
    pit_loss_seconds: float = 22.0       # time lost in pit lane
    pit_loss_sigma: float = 2.0          # variance in pit time

    # Tire degradation (seconds per lap increase)
    tire_deg_soft: float = 0.08
    tire_deg_medium: float = 0.05
    tire_deg_hard: float = 0.03

    # Tire stint lengths (laps)
    soft_stint_range: tuple[int, int] = (12, 20)
    medium_stint_range: tuple[int, int] = (20, 30)
    hard_stint_range: tuple[int, int] = (28, 40)

    # Safety car
    safety_car_rate: float = 1.2         # expected safety cars per race (Poisson λ)
    safety_car_laps: int = 4             # laps under safety car
    safety_car_pace: float = 1.8         # seconds/lap slower under SC

    # Reliability DNF probability (per race, overridden by ML model if available)
    base_dnf_probability: float = 0.05

    # Weather impact
    wet_pace_penalty: float = 3.5        # seconds/lap slower in wet
    wet_variance_multiplier: float = 2.0  # more variance in wet

    # Random seed (None = random)
    seed: Optional[int] = None
