"""
Structured input/output schemas for the Monte Carlo simulator.

Provides rich typed dataclasses for race setup, driver configuration,
track characteristics, weather, and simulation results — replacing
the flat DriverSetup with a comprehensive, ML-integrated schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Track Profile ─────────────────────────────────────────────────────────────

@dataclass
class TrackProfile:
    """
    Circuit characteristics that influence simulation parameters.

    Attributes:
        name:           Human-readable circuit name.
        circuit_type:   One of 'permanent', 'street', 'hybrid'.
        total_laps:     Race distance in laps.
        lap_length_km:  Track length in kilometres.
        pit_loss_sec:   Time lost for a pit stop (track-specific).
        overtaking_difficulty:
            0.0 (easy, e.g. Monza) to 1.0 (very hard, e.g. Monaco).
        safety_car_rate:
            Expected safety cars per race (Poisson λ), track-specific.
        degradation_multiplier:
            Multiplier on tyre degradation rates (1.0 = baseline).
        drs_zones:      Number of DRS zones.
    """
    name: str = "Default Circuit"
    circuit_type: str = "permanent"       # permanent | street | hybrid
    total_laps: int = 55
    lap_length_km: float = 5.3
    pit_loss_sec: float = 22.0
    overtaking_difficulty: float = 0.4
    safety_car_rate: float = 1.2
    degradation_multiplier: float = 1.0
    drs_zones: int = 2

    @property
    def pace_sigma_multiplier(self) -> float:
        """Street circuits → more variance; permanent → baseline."""
        return {
            "street": 1.5,
            "hybrid": 1.2,
            "permanent": 1.0,
        }.get(self.circuit_type, 1.0)

    @property
    def first_lap_chaos_factor(self) -> float:
        """Position-shuffle probability on lap 1 scales with overtaking difficulty."""
        return 0.3 + 0.3 * self.overtaking_difficulty  # 0.3 – 0.6


# ─── Weather ───────────────────────────────────────────────────────────────────

@dataclass
class WeatherConditions:
    """
    Environmental conditions for the race.

    Attributes:
        rain_probability:  Chance of rain during the race [0, 1].
        temperature_c:     Ambient temperature in Celsius.
        humidity_pct:      Relative humidity [0, 100].
        wind_speed_kph:    Wind speed in km/h.
        conditions:        Categorical: 'dry', 'damp', 'wet', 'storm'.
    """
    rain_probability: float = 0.0
    temperature_c: float = 25.0
    humidity_pct: float = 50.0
    wind_speed_kph: float = 10.0
    conditions: str = "dry"

    @property
    def weather_risk_factor(self) -> float:
        """Composite risk multiplier for reliability & pace."""
        base = 1.0
        if self.conditions == "wet":
            base = 1.8
        elif self.conditions == "damp":
            base = 1.3
        elif self.conditions == "storm":
            base = 2.5
        # High temperature and humidity add marginal risk
        if self.temperature_c > 35:
            base *= 1.1
        return base


# ─── ML Priors ─────────────────────────────────────────────────────────────────

@dataclass
class MLPriors:
    """
    ML model prediction outputs used as prior signals in simulation.

    All probabilities are in [0, 1].
    """
    win_probability: float = 0.0
    podium_probability: float = 0.0
    dnf_probability: float = 0.05
    predicted_position: float = 10.0   # expected finish from ML model


# ─── Driver Input ──────────────────────────────────────────────────────────────

@dataclass
class DriverInput:
    """
    Rich driver configuration for the simulation.

    Combines qualifying data, team performance, track affinity,
    and ML model outputs into a single structured input.
    """
    driver_id: int
    name: str
    driver_code: str = ""
    constructor: str = ""

    # Race setup
    grid_position: int = 10
    qualifying_delta_sec: float = 0.0    # gap to pole in qualifying

    # Team signals
    team_strength: float = 0.5           # [0, 1] percentile
    team_reliability: float = 0.95       # [0, 1] probability of finishing

    # Driver skill
    rain_skill: float = 1.0              # multiplier, <1 = better in rain
    track_affinity: float = 0.0          # [-1, 1], positive = strong at track
    consistency: float = 0.5             # [0, 1], higher = more consistent

    # ML priors (optional)
    ml_priors: Optional[MLPriors] = None

    # Strategy priors
    preferred_strategy_stops: Optional[int] = None   # if team tends 1-stop, 2-stop, etc.

    @property
    def base_dnf_probability(self) -> float:
        """
        Composite per-race DNF probability from team reliability + ML prior.
        ML prior gets 60% weight when available.
        """
        base = 1.0 - self.team_reliability
        if self.ml_priors:
            return 0.4 * base + 0.6 * self.ml_priors.dnf_probability
        return base


# ─── Race Input ────────────────────────────────────────────────────────────────

@dataclass
class RaceInput:
    """
    Complete structured input for a race simulation.

    Bundles all driver, track, and weather data needed by the engine.
    """
    drivers: list[DriverInput]
    track: TrackProfile = field(default_factory=TrackProfile)
    weather: WeatherConditions = field(default_factory=WeatherConditions)

    # Optional metadata
    season_year: int = 2023
    round_number: int = 1
    race_name: str = ""

    def validate(self) -> None:
        """Raise ValueError if the input is malformed."""
        if not self.drivers:
            raise ValueError("At least one driver is required.")
        ids = [d.driver_id for d in self.drivers]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate driver IDs detected.")
        grids = [d.grid_position for d in self.drivers]
        if len(grids) != len(set(grids)):
            raise ValueError("Duplicate grid positions detected.")


# ─── Simulation Output ────────────────────────────────────────────────────────

@dataclass
class DriverResult:
    """Per-driver simulation output."""
    driver_id: int
    name: str
    driver_code: str = ""
    constructor: str = ""

    # Probability metrics
    win_probability: float = 0.0
    podium_probability: float = 0.0
    top10_probability: float = 0.0
    dnf_probability: float = 0.0

    # Position metrics
    expected_position: float = 10.0
    position_std: float = 3.0
    median_position: float = 10.0

    # Distribution: {position_str: probability}
    position_distribution: dict[str, float] = field(default_factory=dict)

    # Confidence interval (e.g. 90%)
    ci_lower: float = 1.0
    ci_upper: float = 20.0


@dataclass
class SimulationOutput:
    """Complete output of all Monte Carlo iterations."""
    n_simulations: int
    drivers: list[DriverResult]

    # System-level metrics
    avg_safety_cars_per_race: float = 0.0
    avg_dnfs_per_race: float = 0.0

    # Confidence / quality metrics
    convergence_score: float = 0.0      # 0–1, how stable the estimates are
    entropy: float = 0.0                # Shannon entropy of win distribution

    # Metadata
    race_name: str = ""
    season_year: int = 0
    round_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        drivers_list = []
        for d in sorted(self.drivers, key=lambda x: x.expected_position):
            drivers_list.append({
                "driver_id": d.driver_id,
                "name": d.name,
                "driver_code": d.driver_code,
                "constructor": d.constructor,
                "win_probability": round(d.win_probability, 4),
                "podium_probability": round(d.podium_probability, 4),
                "top10_probability": round(d.top10_probability, 4),
                "dnf_rate": round(d.dnf_probability, 4),
                "expected_position": round(d.expected_position, 2),
                "position_std": round(d.position_std, 2),
                "median_position": round(d.median_position, 2),
                "ci_lower": round(d.ci_lower, 1),
                "ci_upper": round(d.ci_upper, 1),
                "position_distribution": d.position_distribution,
            })
        return {
            "n_simulations": self.n_simulations,
            "avg_safety_cars_per_race": round(self.avg_safety_cars_per_race, 2),
            "avg_dnfs_per_race": round(self.avg_dnfs_per_race, 2),
            "convergence_score": round(self.convergence_score, 4),
            "entropy": round(self.entropy, 4),
            "race_name": self.race_name,
            "season_year": self.season_year,
            "round_number": self.round_number,
            "drivers": drivers_list,
        }
