"""
Safety car modeling module.

Generates safety car (SC) and virtual safety car (VSC) events
for a simulated race, and provides helper functions to apply
their effects on race state.

Distribution:
  - Number of SCs per race:  Poisson(λ = safety_car_rate)
  - Number of VSCs per race: Poisson(λ = vsc_rate)
  - SC start lap:            Uniform(1, total_laps − sc_duration)

Effects modelled:
  - Gap compression among running positions.
  - Pit window opportunity (bonus for pitting under SC).
  - Elevated position volatility on SC restart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from simulator.config import SimulationConfig
from simulator.schemas import TrackProfile, WeatherConditions


@dataclass
class SafetyCarEvent:
    """A single SC or VSC event."""
    start_lap: int
    end_lap: int
    is_vsc: bool = False    # True = virtual safety car (lighter impact)


def generate_safety_car_events(
    config: SimulationConfig,
    track: TrackProfile,
    weather: WeatherConditions,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> list[SafetyCarEvent]:
    """
    Generate all SC and VSC events for a race.

    SC frequency is track-specific (from config, which has been
    adjusted via apply_track_profile).  Wet weather adds +0.5
    to the Poisson rate.
    """
    total_laps = config.total_laps
    events: list[SafetyCarEvent] = []
    occupied: set[int] = set()

    # Adjust rates for weather
    sc_rate = config.safety_car_rate
    vsc_rate = config.vsc_rate
    if weather.conditions in ("wet", "storm"):
        sc_rate += 0.5
        vsc_rate += 0.3
    elif weather.conditions == "damp":
        sc_rate += 0.2

    # Sample SC events
    n_sc = int(np_rng.poisson(sc_rate))
    for _ in range(n_sc):
        duration = config.safety_car_laps
        max_start = max(1, total_laps - duration)
        start = rng.randint(1, max_start)

        # Avoid overlapping existing events
        laps = set(range(start, min(start + duration, total_laps + 1)))
        if laps & occupied:
            continue
        occupied |= laps

        events.append(SafetyCarEvent(
            start_lap=start,
            end_lap=min(start + duration - 1, total_laps),
            is_vsc=False,
        ))

    # Sample VSC events
    n_vsc = int(np_rng.poisson(vsc_rate))
    for _ in range(n_vsc):
        duration = config.vsc_laps
        max_start = max(1, total_laps - duration)
        start = rng.randint(1, max_start)

        laps = set(range(start, min(start + duration, total_laps + 1)))
        if laps & occupied:
            continue
        occupied |= laps

        events.append(SafetyCarEvent(
            start_lap=start,
            end_lap=min(start + duration - 1, total_laps),
            is_vsc=True,
        ))

    events.sort(key=lambda e: e.start_lap)
    return events


def build_sc_lap_lookup(events: list[SafetyCarEvent]) -> dict[int, SafetyCarEvent | None]:
    """
    Build a lap → event mapping for O(1) lookup during simulation.

    Returns: dict[lap_number, SafetyCarEvent_or_None]
    """
    lookup: dict[int, SafetyCarEvent | None] = {}
    for event in events:
        for lap in range(event.start_lap, event.end_lap + 1):
            lookup[lap] = event
    return lookup


def is_restart_lap(lap: int, events: list[SafetyCarEvent]) -> bool:
    """
    Check if this lap is the first green-flag lap after an SC period.

    Restart laps have elevated position volatility.
    """
    for event in events:
        if lap == event.end_lap + 1 and not event.is_vsc:
            return True
    return False
