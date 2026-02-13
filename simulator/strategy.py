"""
Pit strategy model for the Monte Carlo simulator.

Generates realistic pit stop strategies with tire compound selection
and stint length variation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from simulator.config import SimulationConfig


@dataclass
class PitStop:
    """A single pit stop event."""
    lap: int
    tire_compound: str          # "soft" | "medium" | "hard"
    pit_time_seconds: float     # total time lost


@dataclass
class Strategy:
    """Complete race strategy for a driver."""
    starting_tire: str
    stops: list[PitStop]
    total_stops: int


# Common F1 strategy patterns (compound sequences)
STRATEGY_PATTERNS = [
    # 1-stop strategies
    ["medium", "hard"],
    ["soft", "hard"],
    ["soft", "medium"],
    ["hard", "medium"],
    # 2-stop strategies
    ["soft", "medium", "hard"],
    ["soft", "hard", "medium"],
    ["medium", "soft", "hard"],
    ["soft", "medium", "medium"],
    ["medium", "hard", "soft"],
    # 3-stop strategies (aggressive)
    ["soft", "soft", "medium", "hard"],
    ["soft", "medium", "soft", "hard"],
]

# Strategy weights (likelihood of each pattern)
STRATEGY_WEIGHTS = [
    0.20, 0.15, 0.15, 0.05,   # 1-stop
    0.12, 0.08, 0.06, 0.05, 0.04,  # 2-stop
    0.05, 0.05,  # 3-stop
]


def _stint_length(compound: str, config: SimulationConfig, rng: random.Random) -> int:
    """Generate a realistic stint length for a given tire compound."""
    ranges = {
        "soft": config.soft_stint_range,
        "medium": config.medium_stint_range,
        "hard": config.hard_stint_range,
    }
    lo, hi = ranges.get(compound, (15, 25))
    return rng.randint(lo, hi)


def _tire_degradation(compound: str, config: SimulationConfig) -> float:
    """Return the per-lap degradation rate for a compound."""
    rates = {
        "soft": config.tire_deg_soft,
        "medium": config.tire_deg_medium,
        "hard": config.tire_deg_hard,
    }
    return rates.get(compound, config.tire_deg_medium)


def generate_strategy(
    config: SimulationConfig,
    rng: random.Random,
    total_laps: int | None = None,
) -> Strategy:
    """
    Generate a random but valid pit strategy.

    Selects a strategy pattern, then distributes the laps across
    stints, placing pit stops accordingly.
    """
    laps = total_laps or config.total_laps

    # Pick strategy pattern
    pattern = rng.choices(STRATEGY_PATTERNS, weights=STRATEGY_WEIGHTS, k=1)[0]
    starting_tire = pattern[0]
    n_stops = len(pattern) - 1

    # Distribute laps across stints
    stint_targets = []
    for compound in pattern:
        target = _stint_length(compound, config, rng)
        stint_targets.append(target)

    # Scale to fit total race laps
    total_stint = sum(stint_targets)
    scale = laps / total_stint
    stints = [max(5, int(s * scale)) for s in stint_targets]

    # Adjust last stint to exactly fill race
    stints[-1] = laps - sum(stints[:-1])
    if stints[-1] < 3:
        stints[-2] -= (3 - stints[-1])
        stints[-1] = 3

    # Create pit stops
    stops = []
    current_lap = 0
    for i in range(n_stops):
        current_lap += stints[i]
        pit_time = max(
            18.0,
            rng.gauss(config.pit_loss_seconds, config.pit_loss_sigma)
        )
        stops.append(PitStop(
            lap=current_lap,
            tire_compound=pattern[i + 1],
            pit_time_seconds=round(pit_time, 2),
        ))

    return Strategy(
        starting_tire=starting_tire,
        stops=stops,
        total_stops=n_stops,
    )
