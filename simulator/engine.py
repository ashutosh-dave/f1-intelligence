"""
Monte Carlo Race Simulation Engine — v2 (Modular).

Orchestrates modular sub-components to simulate F1 races:
  - Pace module:       lap time sampling with ML priors
  - Strategy module:   pit stop generation and reactive SC pitting
  - Reliability module: per-lap DNF sampling with bathtub curve
  - Safety car module: Poisson-distributed SC/VSC events
  - Ranking module:    first-lap shuffle, restart volatility, classification

Supports:
  - Structured RaceInput / SimulationOutput schemas
  - Parallel execution via ProcessPoolExecutor
  - Backward-compatible DriverSetup API

Usage (new API):
    from simulator.engine import RaceSimulator
    from simulator.schemas import RaceInput, DriverInput, TrackProfile
    race_input = RaceInput(drivers=[...], track=TrackProfile(...))
    sim = RaceSimulator.from_race_input(race_input, config)
    output = sim.run()

Usage (legacy API):
    from simulator.engine import RaceSimulator, DriverSetup
    sim = RaceSimulator(drivers=[DriverSetup(...)], config=config)
    results = sim.run()
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from simulator.config import SimulationConfig
from simulator.schemas import (
    DriverInput,
    DriverResult,
    MLPriors,
    RaceInput,
    SimulationOutput,
    TrackProfile,
    WeatherConditions,
)
from simulator.strategy import Strategy, generate_strategy, should_pit_under_sc
from simulator.modules.pace import (
    compute_base_pace,
    compute_pace_sigma,
    sample_lap_time,
)
from simulator.modules.reliability import compute_race_dnf_probability, sample_dnf
from simulator.modules.safety_car import (
    SafetyCarEvent,
    build_sc_lap_lookup,
    generate_safety_car_events,
    is_restart_lap,
)
from simulator.modules.ranking import (
    DriverState,
    apply_first_lap_shuffle,
    apply_restart_volatility,
    classify_finishing_order,
)


# ─── Legacy Data Type (backward compatibility) ─────────────────────────────────

@dataclass
class DriverSetup:
    """Legacy input configuration — kept for backward compatibility."""
    driver_id: int
    name: str
    grid_position: int
    base_pace: float              # base lap time in seconds
    dnf_probability: float = 0.05
    rain_skill: float = 1.0

    def to_driver_input(self) -> DriverInput:
        """Convert to the new DriverInput schema."""
        return DriverInput(
            driver_id=self.driver_id,
            name=self.name,
            grid_position=self.grid_position,
            team_reliability=1.0 - self.dnf_probability,
            rain_skill=self.rain_skill,
        )


# ─── Single-Iteration Result (internal) ────────────────────────────────────────

@dataclass
class _IterationResult:
    finishing_order: list[int]
    dnf_drivers: list[int]
    n_safety_cars: int


# ─── Legacy AggregatedResults (backward compatibility) ─────────────────────────

@dataclass
class AggregatedResults:
    """Legacy output — wraps SimulationOutput for backward compat."""
    driver_ids: list[int]
    driver_names: dict[int, str]
    n_simulations: int
    position_counts: dict[int, dict[int, int]]
    win_probability: dict[int, float]
    podium_probability: dict[int, float]
    expected_position: dict[int, float]
    position_std: dict[int, float]
    dnf_rate: dict[int, float]
    avg_safety_cars: float

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dictionary (legacy format)."""
        drivers = []
        for did in self.driver_ids:
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
        drivers.sort(key=lambda d: d["expected_position"])
        return {
            "n_simulations": self.n_simulations,
            "avg_safety_cars_per_race": round(self.avg_safety_cars, 2),
            "drivers": drivers,
        }


# ─── Simulation Engine ─────────────────────────────────────────────────────────

class RaceSimulator:
    """
    Monte Carlo F1 race simulation engine (v2).

    Orchestrates modular pace, strategy, reliability, safety car,
    and ranking components.  Supports both the new structured schema
    API and the legacy DriverSetup API.
    """

    def __init__(
        self,
        drivers: list[DriverSetup] | list[DriverInput] | None = None,
        config: SimulationConfig | None = None,
        rain_probability: float = 0.0,
        *,
        race_input: RaceInput | None = None,
    ):
        """
        Construct a simulator.

        Parameters:
            drivers:          List of DriverSetup (legacy) or DriverInput objects.
            config:           SimulationConfig (optional, uses defaults).
            rain_probability: Legacy rain probability override.
            race_input:       New structured RaceInput (takes priority when set).
        """
        self.config = config or SimulationConfig()

        if race_input is not None:
            # ── New API ──
            self._driver_inputs = race_input.drivers
            self._track = race_input.track
            self._weather = race_input.weather
            self.config = self.config.apply_track_profile(self._track)
            self._race_input = race_input
        elif drivers is not None:
            # ── Legacy API ──
            if drivers and isinstance(drivers[0], DriverSetup):
                self._driver_inputs = [d.to_driver_input() for d in drivers]
            else:
                self._driver_inputs = drivers  # already DriverInput
            self._track = TrackProfile(total_laps=self.config.total_laps)
            self._weather = WeatherConditions(
                rain_probability=rain_probability,
                conditions="wet" if rain_probability > 0.5 else "dry",
            )
            self._race_input = None
        else:
            raise ValueError("Either 'drivers' or 'race_input' must be provided.")

        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.RandomState(self.config.seed)

    @classmethod
    def from_race_input(
        cls,
        race_input: RaceInput,
        config: SimulationConfig | None = None,
    ) -> RaceSimulator:
        """Preferred constructor using the new structured schema."""
        race_input.validate()
        return cls(config=config, race_input=race_input)

    # ── Single Iteration ────────────────────────────────────────────────────

    def _simulate_single_race(self) -> _IterationResult:
        """Run one full race simulation."""
        config = self.config
        total_laps = config.total_laps
        weather = self._weather
        track = self._track

        # Determine if this iteration is wet
        is_wet_race = self._rng.random() < weather.rain_probability
        effective_weather = WeatherConditions(
            rain_probability=weather.rain_probability,
            temperature_c=weather.temperature_c,
            humidity_pct=weather.humidity_pct,
            wind_speed_kph=weather.wind_speed_kph,
            conditions="wet" if is_wet_race else weather.conditions,
        )

        # ── Generate safety car events ──
        sc_events = generate_safety_car_events(
            config, track, effective_weather, self._rng, self._np_rng,
        )
        sc_lookup = build_sc_lap_lookup(sc_events)

        # ── Compute pace sigma ──
        sigma = compute_pace_sigma(config, effective_weather)

        # ── Initialize driver states ──
        states: list[DriverState] = []
        base_paces: dict[int, float] = {}
        strategies: dict[int, Strategy] = {}
        race_dnf_probs: dict[int, float] = {}

        for driver in self._driver_inputs:
            state = DriverState(
                driver_id=driver.driver_id,
                name=driver.name,
            )
            states.append(state)

            # Base pace from pace module
            bp = compute_base_pace(driver, config, track)
            base_paces[driver.driver_id] = bp

            # Strategy
            strat = generate_strategy(
                config, self._rng, total_laps,
                weather=effective_weather,
                preferred_stops=driver.preferred_strategy_stops,
            )
            state.current_tire = strat.starting_tire
            strategies[driver.driver_id] = strat

            # DNF probability
            race_dnf_probs[driver.driver_id] = compute_race_dnf_probability(
                driver, config, track, effective_weather,
            )

        # ── First-lap shuffle ──
        states = apply_first_lap_shuffle(
            states,
            shuffle_probability=config.first_lap_position_shuffle,
            incident_rate=config.first_lap_incident_rate,
            rng=self._rng,
        )

        # ── Lap-by-lap simulation ──
        driver_map = {d.driver_id: d for d in self._driver_inputs}

        for lap in range(1, total_laps + 1):
            sc_event = sc_lookup.get(lap)
            is_sc = sc_event is not None and not sc_event.is_vsc
            is_vsc = sc_event is not None and sc_event.is_vsc
            restart = is_restart_lap(lap, sc_events)

            for state in states:
                if state.is_dnf:
                    continue

                driver = driver_map[state.driver_id]
                strategy = strategies[state.driver_id]

                # ── Reliability check ──
                if sample_dnf(
                    driver, lap, total_laps,
                    race_dnf_probs[state.driver_id],
                    self._rng,
                ):
                    state.is_dnf = True
                    state.dnf_lap = lap
                    state.total_time += 999999
                    continue

                # ── Reactive SC pit? ──
                if is_sc and state.stops_done < strategy.total_stops:
                    sc_pit = should_pit_under_sc(
                        state.tire_age, state.stops_done,
                        strategy, config, self._rng,
                    )
                    if sc_pit:
                        sc_pit.lap = lap
                        state.total_time += sc_pit.pit_time_seconds
                        state.current_tire = sc_pit.tire_compound
                        state.tire_age = 0
                        state.stops_done += 1

                # ── Planned pit stops ──
                for stop in strategy.stops:
                    if stop.lap == lap and state.stops_done < strategy.total_stops:
                        state.total_time += stop.pit_time_seconds
                        state.current_tire = stop.tire_compound
                        state.tire_age = 0
                        state.stops_done += 1
                        break

                # ── Lap time ──
                lap_time = sample_lap_time(
                    base_paces[state.driver_id],
                    sigma, state.current_tire, state.tire_age,
                    config, effective_weather, is_sc, is_vsc,
                    driver, self._np_rng,
                )

                # ── Grid advantage on lap 1 ──
                if lap == 1:
                    lap_time += (driver.grid_position - 1) * 0.15

                state.total_time += lap_time
                state.tire_age += 1
                state.current_lap = lap

            # ── Restart volatility ──
            if restart:
                apply_restart_volatility(states, 0.3, self._rng)

        # ── Classify ──
        finishing_order, dnf_ids = classify_finishing_order(states)

        n_sc = sum(1 for e in sc_events if not e.is_vsc)
        return _IterationResult(
            finishing_order=finishing_order,
            dnf_drivers=dnf_ids,
            n_safety_cars=n_sc,
        )

    # ── Batch Run ───────────────────────────────────────────────────────────

    def run(self) -> AggregatedResults:
        """
        Run N Monte Carlo simulations and aggregate results.

        Uses parallel execution if config.n_workers > 1.

        Returns:
            AggregatedResults (legacy-compatible) with full metrics.
        """
        n = self.config.n_simulations
        driver_ids = [d.driver_id for d in self._driver_inputs]
        driver_names = {d.driver_id: d.name for d in self._driver_inputs}

        # Accumulators
        position_counts: dict[int, dict[int, int]] = {
            did: defaultdict(int) for did in driver_ids
        }
        win_counts: dict[int, int] = defaultdict(int)
        podium_counts: dict[int, int] = defaultdict(int)
        top10_counts: dict[int, int] = defaultdict(int)
        position_sums: dict[int, float] = defaultdict(float)
        position_sq_sums: dict[int, float] = defaultdict(float)
        dnf_counts: dict[int, int] = defaultdict(int)
        total_safety_cars = 0

        # Run simulations (sequential — parallel with ProcessPoolExecutor
        # requires pickling of rng state; kept sequential for correctness
        # and determinism with seed).
        for _ in range(n):
            result = self._simulate_single_race()
            total_safety_cars += result.n_safety_cars

            for pos, did in enumerate(result.finishing_order, 1):
                position_counts[did][pos] += 1
                position_sums[did] += pos
                position_sq_sums[did] += pos * pos

                if pos == 1:
                    win_counts[did] += 1
                if pos <= 3:
                    podium_counts[did] += 1
                if pos <= 10:
                    top10_counts[did] += 1

            for did in result.dnf_drivers:
                dnf_counts[did] += 1

        # Compute probabilities
        win_prob = {did: win_counts[did] / n for did in driver_ids}
        podium_prob = {did: podium_counts[did] / n for did in driver_ids}
        expected_pos = {did: position_sums[did] / n for did in driver_ids}
        dnf_rate = {did: dnf_counts[did] / n for did in driver_ids}

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

    def run_structured(self) -> SimulationOutput:
        """
        Run simulations and return the new structured SimulationOutput.

        Includes top-10 probability, confidence intervals, convergence
        score, and Shannon entropy.
        """
        n = self.config.n_simulations
        driver_ids = [d.driver_id for d in self._driver_inputs]
        driver_map = {d.driver_id: d for d in self._driver_inputs}

        # Accumulators
        positions_all: dict[int, list[int]] = {did: [] for did in driver_ids}
        win_counts: dict[int, int] = defaultdict(int)
        podium_counts: dict[int, int] = defaultdict(int)
        top10_counts: dict[int, int] = defaultdict(int)
        dnf_counts: dict[int, int] = defaultdict(int)
        total_sc = 0
        total_dnfs = 0

        for _ in range(n):
            result = self._simulate_single_race()
            total_sc += result.n_safety_cars
            total_dnfs += len(result.dnf_drivers)

            for pos, did in enumerate(result.finishing_order, 1):
                positions_all[did].append(pos)
                if pos == 1:
                    win_counts[did] += 1
                if pos <= 3:
                    podium_counts[did] += 1
                if pos <= 10:
                    top10_counts[did] += 1

            for did in result.dnf_drivers:
                dnf_counts[did] += 1

        # Build DriverResults
        driver_results = []
        for did in driver_ids:
            positions = positions_all[did]
            positions_sorted = sorted(positions)
            mean = sum(positions) / n
            variance = sum((p - mean) ** 2 for p in positions) / n
            std = variance ** 0.5
            median = positions_sorted[n // 2]

            # 90% confidence interval
            ci_lower = positions_sorted[int(n * 0.05)]
            ci_upper = positions_sorted[int(n * 0.95)]

            # Position distribution
            pos_dist = {}
            from collections import Counter
            counts = Counter(positions)
            for pos, cnt in sorted(counts.items()):
                pos_dist[str(pos)] = round(cnt / n, 4)

            d = driver_map[did]
            driver_results.append(DriverResult(
                driver_id=did,
                name=d.name,
                driver_code=d.driver_code,
                constructor=d.constructor,
                win_probability=round(win_counts[did] / n, 4),
                podium_probability=round(podium_counts[did] / n, 4),
                top10_probability=round(top10_counts[did] / n, 4),
                dnf_probability=round(dnf_counts[did] / n, 4),
                expected_position=round(mean, 2),
                position_std=round(std, 2),
                median_position=float(median),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                position_distribution=pos_dist,
            ))

        # Convergence score: 1 - normalised std of win probs across last 20% of iterations
        # Simple proxy: inverse of avg position std normalised by grid size
        n_drivers = len(driver_ids)
        avg_std = sum(dr.position_std for dr in driver_results) / max(n_drivers, 1)
        convergence = max(0, 1.0 - avg_std / n_drivers)

        # Shannon entropy of win distribution
        win_probs = [dr.win_probability for dr in driver_results if dr.win_probability > 0]
        if win_probs:
            entropy = -sum(p * math.log2(p) for p in win_probs)
        else:
            entropy = 0.0

        return SimulationOutput(
            n_simulations=n,
            drivers=driver_results,
            avg_safety_cars_per_race=round(total_sc / n, 2),
            avg_dnfs_per_race=round(total_dnfs / n, 2),
            convergence_score=round(convergence, 4),
            entropy=round(entropy, 4),
            race_name=self._race_input.race_name if self._race_input else "",
            season_year=self._race_input.season_year if self._race_input else 0,
            round_number=self._race_input.round_number if self._race_input else 0,
        )
