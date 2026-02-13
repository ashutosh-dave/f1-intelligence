"""
Test suite for the Monte Carlo race simulator.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import SimulationConfig
from simulator.engine import DriverSetup, RaceSimulator
from simulator.strategy import generate_strategy


class TestSimulationConfig:
    def test_default_config(self):
        config = SimulationConfig()
        assert config.n_simulations == 5000
        assert config.total_laps == 55
        assert config.pace_sigma > 0
        assert config.safety_car_rate > 0

    def test_custom_config(self):
        config = SimulationConfig(n_simulations=100, total_laps=70)
        assert config.n_simulations == 100
        assert config.total_laps == 70


class TestStrategy:
    def test_strategy_generation(self):
        import random
        config = SimulationConfig()
        rng = random.Random(42)
        strategy = generate_strategy(config, rng, 55)

        assert strategy.starting_tire in ["soft", "medium", "hard"]
        assert strategy.total_stops >= 1
        assert strategy.total_stops <= 3
        assert len(strategy.stops) == strategy.total_stops

    def test_strategy_pit_laps_within_race(self):
        import random
        config = SimulationConfig(total_laps=50)
        rng = random.Random(123)

        for _ in range(20):
            strategy = generate_strategy(config, rng, 50)
            for stop in strategy.stops:
                assert 1 <= stop.lap <= 50
                assert stop.pit_time_seconds >= 18.0


class TestRaceSimulator:
    @pytest.fixture
    def basic_drivers(self):
        return [
            DriverSetup(driver_id=1, name="Driver A", grid_position=1, base_pace=90.0, dnf_probability=0.02),
            DriverSetup(driver_id=2, name="Driver B", grid_position=2, base_pace=90.1, dnf_probability=0.05),
            DriverSetup(driver_id=3, name="Driver C", grid_position=3, base_pace=90.2, dnf_probability=0.03),
            DriverSetup(driver_id=4, name="Driver D", grid_position=4, base_pace=90.3, dnf_probability=0.10),
            DriverSetup(driver_id=5, name="Driver E", grid_position=5, base_pace=90.4, dnf_probability=0.04),
        ]

    def test_simulation_runs(self, basic_drivers):
        config = SimulationConfig(n_simulations=100, total_laps=20, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()

        assert results.n_simulations == 100
        assert len(results.driver_ids) == 5
        assert all(did in results.win_probability for did in [1, 2, 3, 4, 5])

    def test_probabilities_sum_to_one(self, basic_drivers):
        config = SimulationConfig(n_simulations=500, total_laps=20, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()

        win_sum = sum(results.win_probability.values())
        assert abs(win_sum - 1.0) < 0.01, f"Win probabilities sum to {win_sum}"

    def test_grid_advantage(self, basic_drivers):
        config = SimulationConfig(n_simulations=1000, total_laps=30, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()

        # Pole sitter should generally have the best expected position
        assert results.expected_position[1] < results.expected_position[5]

    def test_dnf_rate_correlation(self, basic_drivers):
        config = SimulationConfig(n_simulations=1000, total_laps=30, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()

        # Driver D (10% DNF prob) should have higher simulated DNF rate than Driver A (2%)
        assert results.dnf_rate[4] > results.dnf_rate[1]

    def test_position_distribution_valid(self, basic_drivers):
        config = SimulationConfig(n_simulations=200, total_laps=15, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()

        for did in results.driver_ids:
            total = sum(results.position_counts[did].values())
            assert total == 200

    def test_to_dict_format(self, basic_drivers):
        config = SimulationConfig(n_simulations=50, total_laps=10, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()
        d = results.to_dict()

        assert "n_simulations" in d
        assert "drivers" in d
        assert len(d["drivers"]) == 5
        assert all("win_probability" in drv for drv in d["drivers"])
        assert all("position_distribution" in drv for drv in d["drivers"])

    def test_wet_race_simulation(self, basic_drivers):
        config = SimulationConfig(n_simulations=100, total_laps=20, seed=42)
        sim = RaceSimulator(basic_drivers, config, rain_probability=0.9)
        results = sim.run()

        # Should still complete without errors
        assert results.n_simulations == 100
