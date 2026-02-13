"""
Comprehensive test suite for the modular Monte Carlo race simulator.

Covers:
  - Configuration (defaults, from_dict, track overrides)
  - Schemas (RaceInput validation, DriverInput properties)
  - Individual modules (pace, reliability, safety car, ranking)
  - Strategy generation (dry, wet, SC-reactive)
  - Full engine (legacy API, new structured API)
  - Evaluation tools (comparison, calibration, variance)
  - Backward compatibility
"""

import os
import sys
import random
import math

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import SimulationConfig
from simulator.schemas import (
    DriverInput, DriverResult, MLPriors, RaceInput,
    SimulationOutput, TrackProfile, WeatherConditions,
)
from simulator.engine import DriverSetup, RaceSimulator, AggregatedResults
from simulator.strategy import generate_strategy, should_pit_under_sc
from simulator.modules.pace import compute_base_pace, compute_pace_sigma, sample_lap_time, tire_degradation_rate
from simulator.modules.reliability import compute_race_dnf_probability, sample_dnf
from simulator.modules.safety_car import generate_safety_car_events, build_sc_lap_lookup, is_restart_lap
from simulator.modules.ranking import DriverState, classify_finishing_order, apply_first_lap_shuffle
from simulator.evaluation import (
    compare_to_historical, compute_comparison_metrics,
    calibration_report, variance_report,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

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

    def test_from_dict(self):
        data = {
            "n_simulations": 200,
            "pace_sigma": 0.5,
            "unknown_key": "ignored",
        }
        config = SimulationConfig.from_dict(data)
        assert config.n_simulations == 200
        assert config.pace_sigma == 0.5

    def test_from_dict_tuple_fields(self):
        data = {"soft_stint_range": [10, 18]}
        config = SimulationConfig.from_dict(data)
        assert config.soft_stint_range == (10, 18)

    def test_deterministic_flag(self):
        config = SimulationConfig(deterministic=True)
        assert config.seed == 42

    def test_apply_track_profile(self):
        config = SimulationConfig()
        track = TrackProfile(
            total_laps=78,
            pit_loss_sec=25.0,
            safety_car_rate=2.0,
            circuit_type="street",
            degradation_multiplier=1.2,
        )
        applied = config.apply_track_profile(track)
        assert applied.total_laps == 78
        assert applied.pit_loss_seconds == 25.0
        assert applied.safety_car_rate == 2.0
        assert applied.pace_sigma_track_multiplier == 1.5
        # Original is not mutated
        assert config.total_laps == 55

    def test_to_dict(self):
        config = SimulationConfig(n_simulations=99)
        d = config.to_dict()
        assert d["n_simulations"] == 99
        assert isinstance(d["soft_stint_range"], list)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_track_profile_sigma_multiplier(self):
        assert TrackProfile(circuit_type="street").pace_sigma_multiplier == 1.5
        assert TrackProfile(circuit_type="permanent").pace_sigma_multiplier == 1.0

    def test_weather_risk_factor(self):
        dry = WeatherConditions(conditions="dry")
        assert dry.weather_risk_factor == 1.0
        wet = WeatherConditions(conditions="wet")
        assert wet.weather_risk_factor > 1.5

    def test_driver_input_dnf_probability(self):
        d = DriverInput(driver_id=1, name="Test", team_reliability=0.90)
        assert abs(d.base_dnf_probability - 0.10) < 1e-6

    def test_driver_input_ml_blended_dnf(self):
        ml = MLPriors(dnf_probability=0.20)
        d = DriverInput(driver_id=1, name="Test", team_reliability=0.90, ml_priors=ml)
        # 0.4 * 0.10 + 0.6 * 0.20 = 0.16
        assert abs(d.base_dnf_probability - 0.16) < 1e-6

    def test_race_input_validation(self):
        drivers = [
            DriverInput(driver_id=1, name="A", grid_position=1),
            DriverInput(driver_id=2, name="B", grid_position=2),
        ]
        race = RaceInput(drivers=drivers)
        race.validate()  # should not raise

    def test_race_input_duplicate_ids(self):
        drivers = [
            DriverInput(driver_id=1, name="A", grid_position=1),
            DriverInput(driver_id=1, name="B", grid_position=2),
        ]
        race = RaceInput(drivers=drivers)
        with pytest.raises(ValueError, match="Duplicate driver IDs"):
            race.validate()


# ═══════════════════════════════════════════════════════════════════════════════
#  PACE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaceModule:
    def test_base_pace_pole(self):
        driver = DriverInput(driver_id=1, name="P1", grid_position=1, team_strength=0.5)
        config = SimulationConfig()
        track = TrackProfile()
        pace = compute_base_pace(driver, config, track)
        assert 89.0 < pace < 91.0

    def test_base_pace_back_of_grid(self):
        d_front = DriverInput(driver_id=1, name="P1", grid_position=1)
        d_back = DriverInput(driver_id=2, name="P20", grid_position=20)
        config = SimulationConfig()
        track = TrackProfile()
        assert compute_base_pace(d_front, config, track) < compute_base_pace(d_back, config, track)

    def test_ml_priors_affect_pace(self):
        ml = MLPriors(predicted_position=1.0)
        d_ml = DriverInput(driver_id=1, name="ML", grid_position=10, ml_priors=ml)
        d_no = DriverInput(driver_id=2, name="No", grid_position=10)
        config = SimulationConfig()
        track = TrackProfile()
        # ML driver with predicted P1 should be faster than grid-only P10
        assert compute_base_pace(d_ml, config, track) < compute_base_pace(d_no, config, track)

    def test_sigma_wet_higher(self):
        config = SimulationConfig()
        dry_sigma = compute_pace_sigma(config, WeatherConditions(conditions="dry"))
        wet_sigma = compute_pace_sigma(config, WeatherConditions(conditions="wet"))
        assert wet_sigma > dry_sigma

    def test_tire_degradation_rates(self):
        config = SimulationConfig()
        assert tire_degradation_rate("soft", config) > tire_degradation_rate("hard", config)
        assert tire_degradation_rate("intermediate", config) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  RELIABILITY MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestReliabilityModule:
    def test_wet_increases_dnf(self):
        driver = DriverInput(driver_id=1, name="Test", team_reliability=0.90)
        config = SimulationConfig()
        track = TrackProfile()
        dry_prob = compute_race_dnf_probability(driver, config, track, WeatherConditions(conditions="dry"))
        wet_prob = compute_race_dnf_probability(driver, config, track, WeatherConditions(conditions="wet"))
        assert wet_prob > dry_prob

    def test_dnf_clamped(self):
        driver = DriverInput(driver_id=1, name="Test", team_reliability=0.01)
        config = SimulationConfig(weather_dnf_multiplier=10.0, track_dnf_multiplier=10.0)
        track = TrackProfile()
        prob = compute_race_dnf_probability(driver, config, track, WeatherConditions(conditions="storm"))
        assert prob <= 0.5

    def test_sample_dnf_deterministic(self):
        driver = DriverInput(driver_id=1, name="Test", team_reliability=0.0)
        rng = random.Random(42)
        # With 100% DNF prob over 50 laps, should DNF eventually
        any_dnf = any(sample_dnf(driver, lap, 50, 1.0, rng) for lap in range(1, 51))
        assert any_dnf


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFETY CAR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyCarModule:
    def test_events_within_race(self):
        config = SimulationConfig(safety_car_rate=3.0, total_laps=50, seed=42)
        track = TrackProfile(total_laps=50, safety_car_rate=3.0)
        weather = WeatherConditions()
        events = generate_safety_car_events(config, track, weather, random.Random(42), np.random.RandomState(42))
        for e in events:
            assert 1 <= e.start_lap <= 50
            assert e.end_lap <= 50

    def test_events_non_overlapping(self):
        config = SimulationConfig(safety_car_rate=5.0, vsc_rate=3.0, total_laps=60, seed=1)
        track = TrackProfile(total_laps=60, safety_car_rate=5.0)
        weather = WeatherConditions()
        events = generate_safety_car_events(config, track, weather, random.Random(1), np.random.RandomState(1))
        occupied = set()
        for e in events:
            laps = set(range(e.start_lap, e.end_lap + 1))
            assert not (laps & occupied), "Events overlap!"
            occupied |= laps

    def test_sc_lookup(self):
        from simulator.modules.safety_car import SafetyCarEvent
        events = [SafetyCarEvent(start_lap=10, end_lap=13, is_vsc=False)]
        lookup = build_sc_lap_lookup(events)
        assert lookup.get(10) is not None
        assert lookup.get(12) is not None
        assert lookup.get(14) is None

    def test_restart_detection(self):
        from simulator.modules.safety_car import SafetyCarEvent
        events = [SafetyCarEvent(start_lap=10, end_lap=13, is_vsc=False)]
        assert is_restart_lap(14, events)
        assert not is_restart_lap(13, events)
        assert not is_restart_lap(15, events)


# ═══════════════════════════════════════════════════════════════════════════════
#  RANKING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankingModule:
    def test_classify_order(self):
        states = [
            DriverState(driver_id=1, name="A", total_time=5000),
            DriverState(driver_id=2, name="B", total_time=5005),
            DriverState(driver_id=3, name="C", total_time=0, is_dnf=True, dnf_lap=30),
        ]
        order, dnfs = classify_finishing_order(states)
        assert order == [1, 2, 3]
        assert dnfs == [3]

    def test_dnf_ordering_by_lap(self):
        states = [
            DriverState(driver_id=1, name="A", total_time=0, is_dnf=True, dnf_lap=5),
            DriverState(driver_id=2, name="B", total_time=0, is_dnf=True, dnf_lap=40),
            DriverState(driver_id=3, name="C", total_time=5000),
        ]
        order, _ = classify_finishing_order(states)
        # C finishes, then B (retired lap 40) beats A (retired lap 5)
        assert order == [3, 2, 1]

    def test_first_lap_shuffle(self):
        rng = random.Random(42)
        states = [DriverState(driver_id=i, name=f"D{i}") for i in range(1, 11)]
        original_order = [s.driver_id for s in states]
        apply_first_lap_shuffle(states, shuffle_probability=0.9, incident_rate=0.0, rng=rng)
        new_order = [s.driver_id for s in states]
        assert new_order != original_order  # high probability of shuffle


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategy:
    def test_dry_strategy(self):
        config = SimulationConfig()
        rng = random.Random(42)
        strategy = generate_strategy(config, rng, 55)
        assert strategy.starting_tire in ["soft", "medium", "hard"]
        assert 1 <= strategy.total_stops <= 3

    def test_wet_strategy(self):
        config = SimulationConfig()
        rng = random.Random(42)
        weather = WeatherConditions(conditions="wet")
        strategy = generate_strategy(config, rng, 55, weather=weather)
        # Wet strategies should include wet-weather compounds
        compounds = [strategy.starting_tire] + [s.tire_compound for s in strategy.stops]
        wet_compounds = {"intermediate", "wet"}
        assert any(c in wet_compounds for c in compounds)

    def test_pit_laps_within_race(self):
        config = SimulationConfig(total_laps=50)
        rng = random.Random(123)
        for _ in range(20):
            strategy = generate_strategy(config, rng, 50)
            for stop in strategy.stops:
                assert 1 <= stop.lap <= 50
                assert stop.pit_time_seconds >= 18.0

    def test_preferred_stops(self):
        config = SimulationConfig()
        rng = random.Random(42)
        # Generate many strategies with 1-stop preference
        one_stop_count = 0
        for _ in range(100):
            s = generate_strategy(config, rng, 55, preferred_stops=1)
            if s.total_stops == 1:
                one_stop_count += 1
        # Should have more 1-stop strategies than natural rate (~55%)
        assert one_stop_count > 40

    def test_sc_reactive_pit(self):
        config = SimulationConfig()
        rng = random.Random(42)
        strategy = generate_strategy(config, rng, 55)
        # With high tire age, should recommend pitting
        pit = should_pit_under_sc(20, 0, strategy, config, rng)
        assert pit is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE (LEGACY API)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyEngine:
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
        assert abs(win_sum - 1.0) < 0.01

    def test_grid_advantage(self, basic_drivers):
        config = SimulationConfig(n_simulations=1000, total_laps=30, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()
        assert results.expected_position[1] < results.expected_position[5]

    def test_dnf_rate_correlation(self, basic_drivers):
        config = SimulationConfig(n_simulations=1000, total_laps=30, seed=42)
        sim = RaceSimulator(basic_drivers, config)
        results = sim.run()
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
        assert results.n_simulations == 100


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE (NEW STRUCTURED API)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredEngine:
    @pytest.fixture
    def race_input(self):
        drivers = [
            DriverInput(
                driver_id=i, name=f"Driver {chr(64+i)}", driver_code=f"D{i}",
                constructor=f"Team {i}", grid_position=i,
                team_reliability=0.95 - (i * 0.01),
                ml_priors=MLPriors(predicted_position=float(i), win_probability=0.2 / i),
            )
            for i in range(1, 11)
        ]
        return RaceInput(
            drivers=drivers,
            track=TrackProfile(name="Silverstone", circuit_type="permanent", total_laps=52),
            weather=WeatherConditions(rain_probability=0.1),
            race_name="British Grand Prix",
            season_year=2023,
            round_number=10,
        )

    def test_structured_run(self, race_input):
        config = SimulationConfig(n_simulations=200, seed=42)
        sim = RaceSimulator.from_race_input(race_input, config)
        output = sim.run_structured()
        assert output.n_simulations == 200
        assert len(output.drivers) == 10
        assert output.race_name == "British Grand Prix"

    def test_structured_output_fields(self, race_input):
        config = SimulationConfig(n_simulations=300, seed=42)
        sim = RaceSimulator.from_race_input(race_input, config)
        output = sim.run_structured()
        for d in output.drivers:
            assert 0 <= d.win_probability <= 1
            assert 0 <= d.podium_probability <= 1
            assert 0 <= d.top10_probability <= 1
            assert d.position_std >= 0
            assert d.ci_lower <= d.expected_position <= d.ci_upper

    def test_structured_to_dict(self, race_input):
        config = SimulationConfig(n_simulations=100, seed=42)
        sim = RaceSimulator.from_race_input(race_input, config)
        output = sim.run_structured()
        d = output.to_dict()
        assert "convergence_score" in d
        assert "entropy" in d
        assert "avg_dnfs_per_race" in d
        assert len(d["drivers"]) == 10

    def test_convergence_score(self, race_input):
        config = SimulationConfig(n_simulations=500, seed=42)
        sim = RaceSimulator.from_race_input(race_input, config)
        output = sim.run_structured()
        assert 0 <= output.convergence_score <= 1

    def test_street_circuit_more_variance(self):
        """Street circuit should produce more variable outcomes."""
        drivers = [
            DriverInput(driver_id=i, name=f"D{i}", grid_position=i)
            for i in range(1, 6)
        ]
        perm_input = RaceInput(
            drivers=drivers,
            track=TrackProfile(circuit_type="permanent", total_laps=30),
        )
        street_input = RaceInput(
            drivers=drivers,
            track=TrackProfile(circuit_type="street", total_laps=30),
        )
        config = SimulationConfig(n_simulations=500, seed=42)
        perm_out = RaceSimulator.from_race_input(perm_input, config).run_structured()
        street_out = RaceSimulator.from_race_input(street_input, config).run_structured()

        perm_avg_std = sum(d.position_std for d in perm_out.drivers) / 5
        street_avg_std = sum(d.position_std for d in street_out.drivers) / 5
        assert street_avg_std > perm_avg_std


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    @pytest.fixture
    def sim_output(self):
        return SimulationOutput(
            n_simulations=1000,
            drivers=[
                DriverResult(driver_id=1, name="A", win_probability=0.5, podium_probability=0.8, expected_position=2.0, position_std=1.5),
                DriverResult(driver_id=2, name="B", win_probability=0.3, podium_probability=0.6, expected_position=3.5, position_std=2.0),
                DriverResult(driver_id=3, name="C", win_probability=0.2, podium_probability=0.4, expected_position=5.0, position_std=3.0),
            ],
        )

    def test_compare_to_historical(self, sim_output):
        actuals = {1: 1, 2: 3, 3: 5}
        comps = compare_to_historical(sim_output, actuals)
        assert len(comps) == 3
        assert comps[0].actual_position == 1
        assert comps[0].actually_won is True

    def test_comparison_metrics(self, sim_output):
        actuals = {1: 1, 2: 3, 3: 5}
        comps = compare_to_historical(sim_output, actuals)
        metrics = compute_comparison_metrics(comps)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "rank_correlation" in metrics
        assert metrics["winner_correct"] is True

    def test_calibration_report(self):
        preds = [(0.1, False), (0.2, False), (0.3, True), (0.7, True), (0.8, True), (0.9, True)]
        buckets = calibration_report(preds, n_buckets=2)
        assert len(buckets) >= 2
        assert all(b.n_samples > 0 for b in buckets)

    def test_variance_report(self, sim_output):
        report = variance_report(sim_output)
        assert report.avg_position_std > 0
        assert report.win_entropy > 0
        assert report.effective_competitors > 1.0
