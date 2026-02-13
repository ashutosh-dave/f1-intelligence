# F1 Race Intelligence Engine — Architecture

## System Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Frontend   │────▶│  FastAPI      │────▶│  PostgreSQL / DB │
│  (React)     │◀────│  Backend      │◀────│                  │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────┴───────┐
                    │              │
              ┌─────▼────┐  ┌─────▼──────┐
              │ ML Models │  │ Simulator  │
              │ (XGBoost) │  │ (Monte     │
              │           │  │  Carlo)    │
              └───────────┘  └────────────┘
```

## Layer Details

### Data Layer (`data/`)
- **models.py** — SQLAlchemy ORM (9 tables)
- **database.py** — Engine config (Postgres/SQLite)
- **ingest.py** — Jolpica-F1 API ingestion (2010–2025, Ergast-compatible)
- **features.py** — 15 ML features per driver per race

### ML Layer (`ml/`)
- **train.py** — XGBoost + Platt scaling (win/podium/DNF)
- **predict.py** — Inference with normalized probabilities

### Simulator (`simulator/`)
- **schemas.py** — Structured I/O types (RaceInput, DriverInput, SimulationOutput)
- **engine.py** — Modular Monte Carlo orchestration (pace, pits, DNF, safety car)
- **modules/** — pace.py, reliability.py, safety_car.py, ranking.py
- **strategy.py** — Pit strategy generation (dry + wet + SC-reactive)
- **config.py** — Tunable parameters with from_dict/from_yaml factories
- **evaluation.py** — Calibration, sensitivity analysis, historical comparison

### Backend (`backend/`)
- **main.py** — FastAPI app
- **ensemble/combiner.py** — Weighted ML + simulation merger (40/60)
- **api/routes/** — POST /predict, GET /simulate, GET /backtest

### Frontend (`frontend/`)
- Vite + React 18 + Recharts
- Tabs: Predictions | Simulation | Backtest
- F1-themed dark design system

## Data Flow

1. **Ingest** → Jolpica-F1 API → DB tables
2. **Features** → Compute from DB → features table
3. **Train** → Feature table → XGBoost → .joblib models
4. **Predict** → API receives race config → ML inference + Simulation → Ensemble → Response
5. **Display** → React dashboard renders charts and cards
