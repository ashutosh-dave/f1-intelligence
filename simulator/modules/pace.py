"""
Pace modeling module.

Responsible for:
  1. Computing a driver's base lap time from grid position, team
     strength, ML priors, and track affinity.
  2. Sampling per-lap time with Gaussian noise, tire degradation,
     weather multiplier, and safety-car pace override.

Distributions used:
  - Base pace offset:  linear mapping from grid and ML position.
  - Lap noise:         Normal(0, σ), σ scaled by track type and weather.
  - Tire degradation:  linear ramp per lap on current compound.
"""

from __future__ import annotations

from simulator.config import SimulationConfig
from simulator.schemas import DriverInput, TrackProfile, WeatherConditions

import numpy as np


# ─── Base Pace ──────────────────────────────────────────────────────────────────

def compute_base_pace(
    driver: DriverInput,
    config: SimulationConfig,
    track: TrackProfile,
) -> float:
    """
    Derive a driver's base lap time (seconds) from multiple signals.

    The base pace 90.0 s is a normalised reference.  Slower drivers
    get a positive offset (larger lap time).

    Signals blended:
      - Grid position: 0.12 s per position behind pole.
      - ML predicted position: 0.10 s per predicted position (when available).
      - Team strength: top-team bonus up to −0.3 s.
      - Track affinity: bonus/penalty up to ±0.15 s.
    """
    BASE_LAP_TIME = 90.0
    GRID_SECONDS_PER_SLOT = 0.12
    ML_SECONDS_PER_SLOT = 0.10

    grid_offset = (driver.grid_position - 1) * GRID_SECONDS_PER_SLOT

    if driver.ml_priors is not None:
        ml_offset = (driver.ml_priors.predicted_position - 1) * ML_SECONDS_PER_SLOT
        offset = (
            config.grid_pace_weight * grid_offset
            + config.ml_pace_weight * ml_offset
        )
    else:
        offset = grid_offset

    # Team strength bonus (top team → negative offset)
    team_bonus = -0.3 * driver.team_strength

    # Track affinity (positive affinity → small bonus)
    affinity_bonus = -0.15 * driver.track_affinity

    return BASE_LAP_TIME + offset + team_bonus + affinity_bonus


# ─── Lap Time Sampling ─────────────────────────────────────────────────────────

def compute_pace_sigma(
    config: SimulationConfig,
    weather: WeatherConditions,
) -> float:
    """
    Compute the effective pace variance (σ) for this race,
    accounting for track type and weather.
    """
    sigma = config.pace_sigma * config.pace_sigma_track_multiplier

    if weather.conditions == "wet":
        sigma *= config.wet_variance_multiplier
    elif weather.conditions in ("damp", "storm"):
        sigma *= config.damp_variance_multiplier

    return sigma


def sample_lap_time(
    base_pace: float,
    sigma: float,
    tire_compound: str,
    tire_age: int,
    config: SimulationConfig,
    weather: WeatherConditions,
    is_safety_car: bool,
    is_vsc: bool,
    driver: DriverInput,
    rng: np.random.RandomState,
) -> float:
    """
    Sample a single lap time.

    Components:
      1. base_pace + Gaussian noise
      2. Tire degradation (linear with age)
      3. Weather penalty
      4. Safety car override
      5. Driver consistency modifier
    """
    # 1. Gaussian noise (scaled by driver consistency — 1.0=avg, 0=perfect)
    consistency_factor = 0.5 + 0.5 * (1.0 - driver.consistency)
    noise = rng.normal(0, sigma * consistency_factor)
    lap_time = base_pace + noise

    # 2. Tire degradation
    deg_rate = tire_degradation_rate(tire_compound, config)
    lap_time += deg_rate * tire_age

    # 3. Weather penalty
    if weather.conditions == "wet":
        lap_time += config.wet_pace_penalty * driver.rain_skill
    elif weather.conditions == "damp":
        lap_time += config.damp_pace_penalty * driver.rain_skill
    elif weather.conditions == "storm":
        lap_time += config.wet_pace_penalty * 1.5 * driver.rain_skill

    # 4. Safety car: everyone runs at SC pace
    if is_safety_car:
        lap_time = base_pace + config.safety_car_pace
    elif is_vsc:
        lap_time = base_pace + config.safety_car_pace * 0.5

    # 5. Floor: never faster than 95% of base pace
    return max(lap_time, base_pace * 0.95)


# ─── Tire Degradation ──────────────────────────────────────────────────────────

def tire_degradation_rate(compound: str, config: SimulationConfig) -> float:
    """Return per-lap degradation rate (seconds) for a tire compound."""
    rates = {
        "soft": config.tire_deg_soft,
        "medium": config.tire_deg_medium,
        "hard": config.tire_deg_hard,
        "intermediate": config.tire_deg_intermediate,
        "wet": config.tire_deg_wet,
    }
    return rates.get(compound, config.tire_deg_medium)
