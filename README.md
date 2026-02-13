# 🏎️ F1 Race Intelligence Engine

A production-grade motorsport analytics system that predicts Formula 1 race outcomes using historical data, machine learning, and Monte Carlo simulation.

## Features

- **Win/Podium Probability Prediction** — ML-powered race outcome forecasting
- **Monte Carlo Race Simulation** — Stochastic modeling with pace variance, pit strategy, reliability, safety cars
- **REST API** — FastAPI backend with `/predict`, `/simulate`, `/backtest` endpoints
- **React Dashboard** — Interactive visualizations of predictions and distributions
- **Historical Backtesting** — Evaluate model performance across past seasons

## Architecture

```
f1-intelligence/
├── backend/        # FastAPI REST API
├── data/           # Database models, ingestion, feature engineering
├── ml/             # ML training and inference
├── simulator/      # Monte Carlo race simulator
├── frontend/       # React dashboard (Vite)
├── infra/          # Docker, Postgres config
├── tests/          # Test suite
└── docs/           # Architecture documentation
```

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Set up database
docker-compose -f infra/docker-compose.yml up -d postgres

# 3. Ingest historical data
python -m data.ingest

# 4. Train models
python -m ml.train

# 5. Start API
uvicorn backend.app.main:app --reload

# 6. Start frontend
cd frontend && npm install && npm run dev
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, SQLAlchemy |
| ML | scikit-learn, XGBoost, joblib |
| Database | PostgreSQL / SQLite (dev) |
| Frontend | React, Vite, Recharts |
| Infra | Docker, Docker Compose |

## Data Sources

- [Ergast Developer API](http://ergast.com/mrd/) — Historical F1 race data (2010–2024)
- Synthetic weather generation (extensible to real weather APIs)

## License

GNU General Public License v3.0
