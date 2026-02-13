"""
Simulation configuration and parameters.

Supports construction from Python, dictionaries, and YAML files.
All parameters are tunable and documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SimulationConfig:
    """
    Configuration for the Monte Carlo race simulator.

    All numeric parameters have sensible defaults calibrated against
    real F1 race data (2018–2024).
    """

    # ── Iteration count ─────────────────────────────────────────────────────
    n_simulations: int = 5000

    # ── Race parameters ─────────────────────────────────────────────────────
    total_laps: int = 55

    # ── Pace variance (seconds per lap, Gaussian σ) ─────────────────────────
    pace_sigma: float = 0.35
    pace_sigma_track_multiplier: float = 1.0   # overridden from TrackProfile

    # ── ML prior blending ───────────────────────────────────────────────────
    ml_pace_weight: float = 0.5       # how much ML predicted position influences base pace
    grid_pace_weight: float = 0.5     # complement (grid → pace mapping weight)

    # ── Pit stop parameters ─────────────────────────────────────────────────
    min_stops: int = 1
    max_stops: int = 3
    pit_loss_seconds: float = 22.0         # time lost in pit lane
    pit_loss_sigma: float = 2.0            # variance in pit time
    pit_under_sc_bonus: float = 8.0        # time saved pitting under SC vs green

    # ── Tire degradation (seconds per lap increase) ─────────────────────────
    tire_deg_soft: float = 0.08
    tire_deg_medium: float = 0.05
    tire_deg_hard: float = 0.03
    tire_deg_intermediate: float = 0.06
    tire_deg_wet: float = 0.04

    # ── Tire stint lengths (laps) ───────────────────────────────────────────
    soft_stint_range: tuple[int, int] = (12, 20)
    medium_stint_range: tuple[int, int] = (20, 30)
    hard_stint_range: tuple[int, int] = (28, 40)
    intermediate_stint_range: tuple[int, int] = (15, 25)
    wet_stint_range: tuple[int, int] = (10, 20)

    # ── Safety car ──────────────────────────────────────────────────────────
    safety_car_rate: float = 1.2           # expected SCs per race (Poisson λ)
    safety_car_laps: int = 4               # laps under safety car
    safety_car_pace: float = 1.8           # seconds/lap slower under SC
    vsc_rate: float = 0.8                  # expected VSCs per race
    vsc_laps: int = 2                      # laps under VSC
    gap_compression_factor: float = 0.6    # how much gaps compress under SC (0=none, 1=full)

    # ── Reliability / DNF ───────────────────────────────────────────────────
    base_dnf_probability: float = 0.05     # per-race baseline
    weather_dnf_multiplier: float = 1.5    # wet weather multiplier on DNF rate
    track_dnf_multiplier: float = 1.0      # track-specific override

    # ── Weather impact ──────────────────────────────────────────────────────
    wet_pace_penalty: float = 3.5          # seconds/lap slower in wet
    wet_variance_multiplier: float = 2.0   # more variance in wet
    damp_pace_penalty: float = 1.5
    damp_variance_multiplier: float = 1.4

    # ── First-lap modeling ──────────────────────────────────────────────────
    first_lap_incident_rate: float = 0.15  # probability of a first-lap incident
    first_lap_position_shuffle: float = 0.3  # base shuffle probability

    # ── Parallelization ─────────────────────────────────────────────────────
    n_workers: int = 1                     # 1 = sequential, >1 = parallel
    chunk_size: int = 500                  # simulations per parallel chunk

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: Optional[int] = None
    deterministic: bool = False            # alias: sets seed=42 if True and seed is None

    def __post_init__(self):
        """Apply deterministic alias and clamp values."""
        if self.deterministic and self.seed is None:
            self.seed = 42
        self.n_simulations = max(1, self.n_simulations)
        self.total_laps = max(1, self.total_laps)
        self.n_workers = max(1, self.n_workers)

    # ── Factory methods ─────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """
        Create config from a flat dictionary.

        Unknown keys are silently ignored so that partial overrides work.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        # Handle tuple fields stored as lists in JSON/YAML
        for key in ['soft_stint_range', 'medium_stint_range', 'hard_stint_range',
                     'intermediate_stint_range', 'wet_stint_range']:
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> SimulationConfig:
        """Load configuration from a YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        from dataclasses import asdict
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        for key in ['soft_stint_range', 'medium_stint_range', 'hard_stint_range',
                     'intermediate_stint_range', 'wet_stint_range']:
            if key in d and isinstance(d[key], tuple):
                d[key] = list(d[key])
        return d

    def apply_track_profile(self, track) -> SimulationConfig:
        """
        Return a new config with track-specific overrides applied.

        Does not mutate self.
        """
        from copy import copy
        c = copy(self)
        c.total_laps = track.total_laps
        c.pit_loss_seconds = track.pit_loss_sec
        c.safety_car_rate = track.safety_car_rate
        c.pace_sigma_track_multiplier = track.pace_sigma_multiplier
        c.first_lap_position_shuffle = track.first_lap_chaos_factor
        # Scale tire deg by track multiplier
        c.tire_deg_soft *= track.degradation_multiplier
        c.tire_deg_medium *= track.degradation_multiplier
        c.tire_deg_hard *= track.degradation_multiplier
        return c
