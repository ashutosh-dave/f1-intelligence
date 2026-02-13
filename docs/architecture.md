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
- **ingest.py** — Ergast API ingestion (2010–2024)
- **features.py** — 15 ML features per driver per race

### ML Layer (`ml/`)
- **train.py** — XGBoost + Platt scaling (win/podium/DNF)
- **predict.py** — Inference with normalized probabilities

### Simulator (`simulator/`)
- **engine.py** — Lap-by-lap Monte Carlo (pace, pits, DNF, safety car)
- **strategy.py** — Pit strategy generation
- **config.py** — Tunable simulation parameters

### Backend (`backend/`)
- **main.py** — FastAPI app
- **ensemble/combiner.py** — Weighted ML + simulation merger (40/60)
- **api/routes/** — POST /predict, GET /simulate, GET /backtest

### Frontend (`frontend/`)
- Vite + React 18 + Recharts
- Tabs: Predictions | Simulation | Backtest
- F1-themed dark design system

## Data Flow

1. **Ingest** → Ergast API → DB tables
2. **Features** → Compute from DB → features table
3. **Train** → Feature table → XGBoost → .joblib models
4. **Predict** → API receives race config → ML inference + Simulation → Ensemble → Response
5. **Display** → React dashboard renders charts and cards
