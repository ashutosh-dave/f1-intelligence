"""
Ranking / classification module.

Responsible for:
  1. Determining the finishing order from driver states after all laps.
  2. Modelling first-lap incidents (position shuffles and DNFs).
  3. Modelling restart volatility after safety car periods.

Assumptions:
  - DNF drivers are classified behind all finishers, in order
    of the lap on which they retired (later retirement = higher place).
  - First-lap incidents are modelled as random adjacent swaps
    with a configurable probability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriverState:
    """Mutable state of a driver during a single simulation."""
    driver_id: int
    name: str
    total_time: float = 0.0
    current_lap: int = 0
    current_tire: str = "medium"
    tire_age: int = 0
    stops_done: int = 0
    is_dnf: bool = False
    dnf_lap: Optional[int] = None


def classify_finishing_order(states: list[DriverState]) -> tuple[list[int], list[int]]:
    """
    Classify drivers into finishing order.

    Returns:
        (finishing_order, dnf_ids)
        finishing_order: driver_ids in P1 → Pn order.
        dnf_ids:         driver_ids that retired.
    """
    finished = [s for s in states if not s.is_dnf]
    dnf_states = [s for s in states if s.is_dnf]

    # Finishers sorted by total race time
    finished.sort(key=lambda s: s.total_time)

    # DNFs sorted by retirement lap (later retirement = better classification)
    dnf_states.sort(key=lambda s: -(s.dnf_lap or 0))

    order = [s.driver_id for s in finished] + [s.driver_id for s in dnf_states]
    dnf_ids = [s.driver_id for s in dnf_states]
    return order, dnf_ids


def apply_first_lap_shuffle(
    states: list[DriverState],
    shuffle_probability: float,
    incident_rate: float,
    rng: random.Random,
) -> list[DriverState]:
    """
    Apply first-lap chaos: random adjacent swaps in the running order.

    Models the turbulence of turn-1 and opening lap, where cars are
    tightly bunched and overtakes / position losses are common.

    Optionally triggers a first-lap incident (DNF for a random driver
    in the lower half of the grid).

    Args:
        states:              List of DriverStates sorted by grid position.
        shuffle_probability: Probability of swapping any adjacent pair.
        incident_rate:       Probability of a first-lap DNF.
        rng:                 Random number generator.

    Returns:
        The (potentially mutated) states list — same objects, reordered.
    """
    # Sort by grid (via total_time proxy — all zeros at lap 0, so use id order)
    n = len(states)

    # Adjacent swaps
    for i in range(n - 1):
        if rng.random() < shuffle_probability:
            states[i], states[i + 1] = states[i + 1], states[i]

    # First-lap incident: random car from bottom half DNFs
    if rng.random() < incident_rate:
        candidates = states[n // 2:]  # lower half of grid
        if candidates:
            victim = rng.choice(candidates)
            victim.is_dnf = True
            victim.dnf_lap = 1

    return states


def apply_restart_volatility(
    states: list[DriverState],
    volatility: float,
    rng: random.Random,
) -> list[DriverState]:
    """
    Apply position volatility after a safety car restart.

    Models the bunched-up restart where drivers can gain or lose
    1–2 positions through braking/traction differences.

    Args:
        states:     Running drivers sorted by current position.
        volatility: Base probability of a swap (0–1).
        rng:        Random number generator.
    """
    running = [s for s in states if not s.is_dnf]
    for i in range(len(running) - 1):
        if rng.random() < volatility * 0.5:
            # Small time swap: trailing driver gains a tiny advantage
            delta = rng.uniform(0.05, 0.2)
            running[i + 1].total_time -= delta
    return states
