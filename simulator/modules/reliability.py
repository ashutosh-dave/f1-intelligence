"""
Reliability / DNF modeling module.

Determines whether a driver retires from the race on any given lap,
using a composite probability derived from:
  - Team reliability score
  - Track-specific risk factor
  - Weather risk factor
  - ML prior DNF probability

Distribution:
  Per-lap DNF is modelled as an independent Bernoulli trial with
  probability  p_race / total_laps  (uniformly spread).  A bathtub
  curve bias is optionally applied: cars are more likely to fail on
  lap 1 (formation stress) and near the end (thermal fatigue).

Limitations:
  - No correlation between drivers on the same team (a shared
    engine failure mode is not modelled).
  - No progressive degradation leading to failure.
"""

from __future__ import annotations

import random

from simulator.config import SimulationConfig
from simulator.schemas import DriverInput, TrackProfile, WeatherConditions


def compute_race_dnf_probability(
    driver: DriverInput,
    config: SimulationConfig,
    track: TrackProfile,
    weather: WeatherConditions,
) -> float:
    """
    Compute the total per-race DNF probability for a driver.

    Factors:
      - base_dnf_probability from driver (ML + team reliability blend)
      - track DNF multiplier
      - weather risk factor
    Clamped to [0, 0.5] to remain realistic.
    """
    base = driver.base_dnf_probability

    # Track risk
    base *= config.track_dnf_multiplier

    # Weather risk
    base *= weather.weather_risk_factor

    # Wet conditions push DNF rate up further
    if weather.conditions in ("wet", "storm"):
        base *= config.weather_dnf_multiplier

    return min(max(base, 0.0), 0.5)


def sample_dnf(
    driver: DriverInput,
    lap: int,
    total_laps: int,
    race_dnf_prob: float,
    rng: random.Random,
) -> bool:
    """
    Determine if a driver DNFs on this specific lap.

    Uses a bathtub curve: slightly elevated risk on lap 1 (start
    chaos) and final 10% of laps (thermal/mechanical fatigue).

    Returns:
        True if the driver retires on this lap.
    """
    # Uniform per-lap base
    per_lap = race_dnf_prob / total_laps

    # Bathtub bias
    if lap == 1:
        per_lap *= 3.0          # First-lap incidents
    elif lap > total_laps * 0.9:
        per_lap *= 1.5          # Late-race fatigue

    return rng.random() < per_lap
