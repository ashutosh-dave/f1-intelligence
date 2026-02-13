"""
F1 Race Intelligence Engine — FastAPI Application.

Main entry point for the backend API server.

Run with:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.api.routes import predict, simulate, backtest
from data.database import init_db

# ─── App Initialization ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade F1 race outcome prediction system combining "
        "machine learning models with Monte Carlo race simulation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ────────────────────────────────────────────────────────────────────

app.include_router(predict.router, tags=["Predictions"])
app.include_router(simulate.router, tags=["Simulation"])
app.include_router(backtest.router, tags=["Backtesting"])


# ─── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    init_db()


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "simulate": "GET /simulate",
            "backtest": "GET /backtest",
            "health": "GET /health",
        },
    }
