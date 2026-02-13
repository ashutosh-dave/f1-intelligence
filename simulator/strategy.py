"""
Pit strategy model for the Monte Carlo simulator.

Generates realistic pit stop strategies with tire compound selection,
stint length variation, and support for wet-weather compounds and
reactive pitting under safety car.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulator.config import SimulationConfig
from simulator.schemas import TrackProfile, WeatherConditions


@dataclass
class PitStop:
    """A single pit stop event."""
    lap: int
    tire_compound: str          # soft | medium | hard | intermediate | wet
    pit_time_seconds: float     # total time lost


@dataclass
class Strategy:
    """Complete race strategy for a driver."""
    starting_tire: str
    stops: list[PitStop]
    total_stops: int


# Common F1 strategy patterns (compound sequences)
DRY_STRATEGY_PATTERNS = [
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

DRY_STRATEGY_WEIGHTS = [
    0.20, 0.15, 0.15, 0.05,           # 1-stop
    0.12, 0.08, 0.06, 0.05, 0.04,     # 2-stop
    0.05, 0.05,                        # 3-stop
]

# Wet-weather strategy patterns
WET_STRATEGY_PATTERNS = [
    ["intermediate", "intermediate"],
    ["wet", "intermediate"],
    ["intermediate", "soft"],          # drying track
    ["wet", "intermediate", "soft"],   # rain → drying
    ["intermediate", "medium"],        # late drying
]

WET_STRATEGY_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]


def _stint_length(
    compound: str,
    config: SimulationConfig,
    rng: random.Random,
) -> int:
    """Generate a realistic stint length for a given tire compound."""
    ranges = {
        "soft": config.soft_stint_range,
        "medium": config.medium_stint_range,
        "hard": config.hard_stint_range,
        "intermediate": config.intermediate_stint_range,
        "wet": config.wet_stint_range,
    }
    lo, hi = ranges.get(compound, (15, 25))
    return rng.randint(lo, hi)


def _tire_degradation(compound: str, config: SimulationConfig) -> float:
    """Return the per-lap degradation rate for a compound."""
    rates = {
        "soft": config.tire_deg_soft,
        "medium": config.tire_deg_medium,
        "hard": config.tire_deg_hard,
        "intermediate": config.tire_deg_intermediate,
        "wet": config.tire_deg_wet,
    }
    return rates.get(compound, config.tire_deg_medium)


def generate_strategy(
    config: SimulationConfig,
    rng: random.Random,
    total_laps: int | None = None,
    weather: WeatherConditions | None = None,
    preferred_stops: int | None = None,
) -> Strategy:
    """
    Generate a random but valid pit strategy.

    Selects a strategy pattern based on weather conditions, then
    distributes laps across stints.

    Args:
        config:           Simulation configuration.
        rng:              Random number generator.
        total_laps:       Override for race length (else use config).
        weather:          Weather conditions — selects wet patterns if rainy.
        preferred_stops:  Optional team preference for stop count.
    """
    laps = total_laps or config.total_laps

    # Select pattern pool based on weather
    is_wet = weather and weather.conditions in ("wet", "damp", "storm")

    if is_wet:
        patterns = WET_STRATEGY_PATTERNS
        weights = WET_STRATEGY_WEIGHTS
    else:
        patterns = DRY_STRATEGY_PATTERNS
        weights = DRY_STRATEGY_WEIGHTS

    # If team has a preferred stop count, bias toward matching patterns
    if preferred_stops is not None:
        adjusted_weights = []
        for pat, w in zip(patterns, weights):
            n_stops = len(pat) - 1
            if n_stops == preferred_stops:
                adjusted_weights.append(w * 2.0)  # double weight
            else:
                adjusted_weights.append(w)
        weights = adjusted_weights

    # Pick strategy pattern
    pattern = rng.choices(patterns, weights=weights, k=1)[0]
    starting_tire = pattern[0]
    n_stops = len(pattern) - 1

    # Distribute laps across stints
    stint_targets = [_stint_length(compound, config, rng) for compound in pattern]

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
            rng.gauss(config.pit_loss_seconds, config.pit_loss_sigma),
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


def should_pit_under_sc(
    current_tire_age: int,
    stops_done: int,
    strategy: Strategy,
    config: SimulationConfig,
    rng: random.Random,
) -> PitStop | None:
    """
    Decide whether to make a reactive pit stop under safety car.

    Teams will opportunistically pit if:
      - Tyre age is high (> 60% of expected stint).
      - There are remaining planned stops not yet done.

    Returns:
        A PitStop to execute, or None.
    """
    remaining_stops = strategy.total_stops - stops_done
    if remaining_stops <= 0:
        return None

    # Find the next planned stop
    planned = [s for s in strategy.stops if s.lap > 0]
    if stops_done < len(planned):
        next_stop = planned[stops_done]
    else:
        return None

    # Heuristic: pit if tire age > 12 or random chance on fresh tires
    if current_tire_age > 12 or rng.random() < 0.25:
        reduced_time = max(
            18.0,
            config.pit_loss_seconds - config.pit_under_sc_bonus
            + rng.gauss(0, config.pit_loss_sigma * 0.5),
        )
        return PitStop(
            lap=0,  # will be set to current lap by engine
            tire_compound=next_stop.tire_compound,
            pit_time_seconds=round(reduced_time, 2),
        )
    return None
