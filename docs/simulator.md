# Monte Carlo Race Simulator — Technical Documentation

## Overview

The race simulator uses probabilistic Monte Carlo methods to generate distributions of finishing positions for F1 races. It runs thousands of independent race iterations, each with stochastic pace, pit strategy, reliability, safety car, and weather effects, then aggregates results into probability estimates.

## Architecture

```
simulator/
├── schemas.py          ← Structured I/O types (RaceInput, DriverInput, SimulationOutput)
├── config.py           ← All tunable parameters with from_dict/from_yaml factories
├── engine.py           ← Orchestration: iterates modules, aggregates results
├── strategy.py         ← Pit strategy generation (dry + wet compounds, SC-reactive)
├── evaluation.py       ← Calibration, sensitivity, historical comparison tools
└── modules/
    ├── pace.py         ← Base pace computation and lap time sampling
    ├── reliability.py  ← Per-lap DNF sampling with bathtub curve
    ├── safety_car.py   ← Poisson SC/VSC event generation
    └── ranking.py      ← Classification, first-lap shuffle, restart volatility
```

## Probability Distributions Used

| Component | Distribution | Parameters |
|-----------|-------------|------------|
| Lap time noise | `Normal(0, σ)` | σ = `pace_sigma × track_multiplier × weather_multiplier × consistency_factor` |
| Tire degradation | Linear ramp | `deg_rate × tire_age` (per compound) |
| Safety car count | `Poisson(λ)` | λ = `safety_car_rate` (track-specific, weather-adjusted) |
| VSC count | `Poisson(λ)` | λ = `vsc_rate` (weather-adjusted) |
| Per-lap DNF | `Bernoulli(p)` | p = `race_dnf_prob / total_laps × bathtub_bias` |
| Pit stop time | `Normal(μ, σ)` | μ = `pit_loss_seconds`, σ = `pit_loss_sigma`, floor 18s |
| Strategy selection | Categorical | Weighted pattern pool (dry: 11 patterns, wet: 5) |
| First-lap shuffle | `Bernoulli(p)` per adjacent pair | p = `first_lap_position_shuffle` |

## Modeling Assumptions

1. **Independence**: Each simulation iteration is statistically independent.
2. **Uniform DNF distribution with bathtub bias**: DNF probability is spread uniformly across laps, with 3× elevation on lap 1 and 1.5× in the final 10% of laps.
3. **No team-correlated failures**: Drivers on the same team fail independently.
4. **Linear tire degradation**: Proportional to tire age — no cliff model.
5. **No progressive damage**: Damage from incidents doesn't accumulate between laps.
6. **No overtaking model**: Positions are purely time-based; no DRS/slipstream logic.

## ML Prior Integration

When ML predictions are available, the pace module blends grid-based and ML-predicted pace:

```
offset = grid_pace_weight × (grid_position − 1) × 0.12
       + ml_pace_weight   × (ml_predicted_position − 1) × 0.10
```

DNF probability is also blended: `0.4 × team_base + 0.6 × ml_dnf_prediction`.

## Configuration

All parameters live in `SimulationConfig`. Load from Python, dict, or YAML:

```python
# From dict
config = SimulationConfig.from_dict({"n_simulations": 10000, "pace_sigma": 0.4})

# From YAML
config = SimulationConfig.from_yaml("configs/monza.yaml")

# Apply track profile
config = config.apply_track_profile(TrackProfile(circuit_type="street"))
```

Key parameters:
- `n_simulations`: Number of Monte Carlo iterations (default: 5000)
- `pace_sigma`: Base lap time noise σ (default: 0.35s)
- `safety_car_rate`: Expected SCs per race (Poisson λ, default: 1.2)
- `ml_pace_weight` / `grid_pace_weight`: Blending weights for ML vs grid pace

## Evaluation Tools

```python
from simulator.evaluation import compare_to_historical, calibration_report, variance_report

# Compare to actual race result
comparisons = compare_to_historical(sim_output, {1: 1, 44: 3, 16: 5})
metrics = compute_comparison_metrics(comparisons)
# → {"mae": 1.2, "rmse": 1.8, "rank_correlation": 0.85, ...}

# Calibration curve
buckets = calibration_report([(pred_prob, actual_outcome), ...])

# Variance/entropy analysis
report = variance_report(sim_output)
# → VarianceReport(win_entropy=2.5, effective_competitors=5.6, ...)
```

## Limitations

- No telemetry-level physics (aero, fuel load, energy recovery).
- No progressive damage or mechanical wear model.
- No race-director red flag logic.
- No real-time data ingestion (pre-race predictions only).
- Parallelization is available in config but simulations run sequentially
  when seeded for determinism.

## Tuning Guide

| Goal | Parameter(s) to adjust |
|------|----------------------|
| More unpredictable races | Increase `pace_sigma`, `first_lap_position_shuffle` |
| More safety cars | Increase `safety_car_rate` |
| More DNFs | Decrease `team_reliability` on drivers, increase `base_dnf_probability` |
| Higher ML influence | Increase `ml_pace_weight`, decrease `grid_pace_weight` |
| Track-specific behavior | Use `TrackProfile` with circuit type and degradation multiplier |
