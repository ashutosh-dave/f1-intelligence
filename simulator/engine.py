"""
Monte Carlo Race Simulation Engine.

Simulates F1 races stochastically by sampling:
  - Driver pace variance (Gaussian noise per lap)
  - Pit stop timing and execution
  - Tire degradation effects
  - Mechanical/reliability DNFs
  - Safety car events (Poisson-distributed)
  - Weather impact

Produces probability distributions of finishing positions
across N simulation iterations.

Usage:
    from simulator.engine import RaceSimulator
    sim = RaceSimulator(drivers, config)
    results = sim.run()
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simulator.config import SimulationConfig
from simulator.strategy import Strategy, generate_strategy, _tire_degradation


# ─── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class DriverSetup:
    """Input configuration for a driver in the simulation."""
    driver_id: int
    name: str
    grid_position: int
    base_pace: float              # base lap time in seconds
    dnf_probability: float = 0.05 # per-race DNF probability
    rain_skill: float = 1.0       # multiplier (1.0 = neutral, <1 = better in rain)


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
    final_position: Optional[int] = None


@dataclass
class SimulationResult:
    """Result of a single simulation iteration."""
    finishing_order: list[int]     # driver_ids in finishing order
    dnf_drivers: list[int]        # driver_ids that DNF'd
    safety_car_laps: list[int]    # laps where safety car was deployed


@dataclass
class AggregatedResults:
    """Aggregated results across all simulation iterations."""
    driver_ids: list[int]
    driver_names: dict[int, str]
    n_simulations: int

    # Position distributions: driver_id → {position: count}
    position_counts: dict[int, dict[int, int]]

    # Derived probabilities
    win_probability: dict[int, float]
    podium_probability: dict[int, float]
    expected_position: dict[int, float]
    position_std: dict[int, float]
    dnf_rate: dict[int, float]

    # Summary of safety car frequency
    avg_safety_cars: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        drivers = []
        for did in self.driver_ids:
            # Build position distribution
            pos_dist = {}
            total = sum(self.position_counts[did].values())
            for pos, count in sorted(self.position_counts[did].items()):
                pos_dist[str(pos)] = round(count / total, 4)

            drivers.append({
                "driver_id": did,
                "name": self.driver_names.get(did, f"Driver {did}"),
                "win_probability": round(self.win_probability.get(did, 0), 4),
                "podium_probability": round(self.podium_probability.get(did, 0), 4),
                "expected_position": round(self.expected_position.get(did, 99), 2),
                "position_std": round(self.position_std.get(did, 0), 2),
                "dnf_rate": round(self.dnf_rate.get(did, 0), 4),
                "position_distribution": pos_dist,
            })

        # Sort by expected position
        drivers.sort(key=lambda d: d["expected_position"])

        return {
            "n_simulations": self.n_simulations,
            "avg_safety_cars_per_race": round(self.avg_safety_cars, 2),
            "drivers": drivers,
        }


# ─── Simulation Engine ────────────────────────────────────────────────────────

class RaceSimulator:
    """Monte Carlo F1 race simulation engine."""

    def __init__(
        self,
        drivers: list[DriverSetup],
        config: SimulationConfig | None = None,
        rain_probability: float = 0.0,
    ):
        self.drivers = drivers
        self.config = config or SimulationConfig()
        self.rain_probability = rain_probability
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.RandomState(self.config.seed)

    def _simulate_single_race(self) -> SimulationResult:
        """Run a single race simulation."""
        config = self.config
        total_laps = config.total_laps

        # Determine weather for this simulation
        is_wet = self.rng.random() < self.rain_probability

        # Generate safety car events (Poisson)
        n_safety_cars = self.np_rng.poisson(config.safety_car_rate)
        safety_car_laps_set = set()
        for _ in range(n_safety_cars):
            sc_start = self.rng.randint(1, max(1, total_laps - config.safety_car_laps))
            for lap in range(sc_start, min(sc_start + config.safety_car_laps, total_laps + 1)):
                safety_car_laps_set.add(lap)

        # Initialize driver states
        states: list[DriverState] = []
        strategies: dict[int, Strategy] = {}

        for driver in self.drivers:
            state = DriverState(
                driver_id=driver.driver_id,
                name=driver.name,
            )
            states.append(state)

            # Generate pit strategy
            strategy = generate_strategy(config, self.rng, total_laps)
            state.current_tire = strategy.starting_tire
            strategies[driver.driver_id] = strategy

        # Simulate lap by lap
        for lap in range(1, total_laps + 1):
            is_safety_car = lap in safety_car_laps_set

            for i, state in enumerate(states):
                if state.is_dnf:
                    continue

                driver = self.drivers[i]
                strategy = strategies[driver.driver_id]

                # ── Check for reliability DNF ──
                if self.rng.random() < (driver.dnf_probability / total_laps):
                    state.is_dnf = True
                    state.dnf_lap = lap
                    state.total_time += 999999  # penalty
                    continue

                # ── Base lap time ──
                base_time = driver.base_pace

                # ── Pace variance (Gaussian noise) ──
                sigma = config.pace_sigma
                if is_wet:
                    sigma *= config.wet_variance_multiplier

                pace_noise = self.np_rng.normal(0, sigma)
                lap_time = base_time + pace_noise

                # ── Weather penalty ──
                if is_wet:
                    wet_penalty = config.wet_pace_penalty * driver.rain_skill
                    lap_time += wet_penalty

                # ── Tire degradation ──
                deg_rate = _tire_degradation(state.current_tire, config)
                tire_deg_penalty = deg_rate * state.tire_age
                lap_time += tire_deg_penalty
                state.tire_age += 1

                # ── Safety car ──
                if is_safety_car:
                    lap_time = base_time + config.safety_car_pace

                # ── Pit stops ──
                for stop in strategy.stops:
                    if stop.lap == lap:
                        lap_time += stop.pit_time_seconds
                        state.current_tire = stop.tire_compound
                        state.tire_age = 0
                        state.stops_done += 1
                        break

                # ── Grid position advantage (first lap) ──
                if lap == 1:
                    # Front-row advantage on first lap
                    position_factor = (driver.grid_position - 1) * 0.15
                    lap_time += position_factor

                state.total_time += max(lap_time, base_time * 0.95)  # floor
                state.current_lap = lap

        # ── Determine finishing order ──
        finished = [s for s in states if not s.is_dnf]
        dnf_drivers_list = [s for s in states if s.is_dnf]

        finished.sort(key=lambda s: s.total_time)

        finishing_order = []
        for pos, state in enumerate(finished, 1):
            state.final_position = pos
            finishing_order.append(state.driver_id)

        # DNFs get positions after all finishers
        for pos, state in enumerate(dnf_drivers_list, len(finished) + 1):
            state.final_position = pos
            finishing_order.append(state.driver_id)

        return SimulationResult(
            finishing_order=finishing_order,
            dnf_drivers=[s.driver_id for s in dnf_drivers_list],
            safety_car_laps=sorted(safety_car_laps_set),
        )

    def run(self) -> AggregatedResults:
        """
        Run N Monte Carlo simulations and aggregate results.

        Returns:
            AggregatedResults with position distributions and probabilities
        """
        n = self.config.n_simulations
        driver_ids = [d.driver_id for d in self.drivers]
        driver_names = {d.driver_id: d.name for d in self.drivers}
        n_drivers = len(driver_ids)

        # Accumulators
        position_counts: dict[int, dict[int, int]] = {
            did: defaultdict(int) for did in driver_ids
        }
        win_counts: dict[int, int] = defaultdict(int)
        podium_counts: dict[int, int] = defaultdict(int)
        position_sums: dict[int, float] = defaultdict(float)
        position_sq_sums: dict[int, float] = defaultdict(float)
        dnf_counts: dict[int, int] = defaultdict(int)
        total_safety_cars = 0

        for _ in range(n):
            result = self._simulate_single_race()
            total_safety_cars += len(result.safety_car_laps) / max(self.config.safety_car_laps, 1)

            for pos, did in enumerate(result.finishing_order, 1):
                position_counts[did][pos] += 1
                position_sums[did] += pos
                position_sq_sums[did] += pos * pos

                if pos == 1:
                    win_counts[did] += 1
                if pos <= 3:
                    podium_counts[did] += 1

            for did in result.dnf_drivers:
                dnf_counts[did] += 1

        # Compute final probabilities
        win_prob = {did: win_counts[did] / n for did in driver_ids}
        podium_prob = {did: podium_counts[did] / n for did in driver_ids}
        expected_pos = {did: position_sums[did] / n for did in driver_ids}
        dnf_rate = {did: dnf_counts[did] / n for did in driver_ids}

        # Standard deviation of position
        position_std = {}
        for did in driver_ids:
            mean = expected_pos[did]
            mean_sq = position_sq_sums[did] / n
            variance = max(0, mean_sq - mean * mean)
            position_std[did] = variance ** 0.5

        return AggregatedResults(
            driver_ids=driver_ids,
            driver_names=driver_names,
            n_simulations=n,
            position_counts={did: dict(pc) for did, pc in position_counts.items()},
            win_probability=win_prob,
            podium_probability=podium_prob,
            expected_position=expected_pos,
            position_std=position_std,
            dnf_rate=dnf_rate,
            avg_safety_cars=total_safety_cars / n,
        )
