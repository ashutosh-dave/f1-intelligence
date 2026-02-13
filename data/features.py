"""
Feature engineering module for the F1 Race Intelligence Engine.

Computes per-driver-per-race feature vectors and persists them to the
`features` table. These features are consumed by the ML training pipeline.

Usage:
    python -m data.features                # Compute features for all races
    python -m data.features --year 2023    # Single season
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session
from tqdm import tqdm

from data.database import SessionLocal, init_db
from data.models import (
    Circuit, Constructor, Driver, FeatureRow,
    Qualifying, Race, RaceResult, Season, Weather,
)


# ─── Feature Computation Helpers ──────────────────────────────────────────────

def _driver_form(
    db: Session,
    driver_id: int,
    race_date,
    lookback: int = 5,
) -> Optional[float]:
    """
    Weighted average finishing position over the last `lookback` races.
    More recent races get higher weight (exponential decay).
    Returns None if no prior results.
    """
    prior_results = (
        db.query(RaceResult.position)
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.driver_id == driver_id,
            RaceResult.position.isnot(None),
            Race.race_date < race_date,
        )
        .order_by(Race.race_date.desc())
        .limit(lookback)
        .all()
    )

    if not prior_results:
        return None

    positions = [r[0] for r in prior_results]
    # Exponential decay weights: most recent = highest weight
    weights = [0.5 ** i for i in range(len(positions))]
    total_weight = sum(weights)
    return sum(p * w for p, w in zip(positions, weights)) / total_weight


def _driver_career_stats(
    db: Session,
    driver_id: int,
    race_date,
) -> tuple[float, float]:
    """Return (win_rate, podium_rate) based on all prior races."""
    prior = (
        db.query(RaceResult.position)
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.driver_id == driver_id,
            Race.race_date < race_date,
        )
        .all()
    )

    if not prior:
        return 0.0, 0.0

    total = len(prior)
    wins = sum(1 for (p,) in prior if p == 1)
    podiums = sum(1 for (p,) in prior if p is not None and p <= 3)
    return wins / total, podiums / total


def _driver_dnf_rate(
    db: Session,
    driver_id: int,
    race_date,
    lookback: int = 10,
) -> Optional[float]:
    """DNF rate over the last `lookback` races."""
    prior = (
        db.query(RaceResult.is_dnf)
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.driver_id == driver_id,
            Race.race_date < race_date,
        )
        .order_by(Race.race_date.desc())
        .limit(lookback)
        .all()
    )

    if not prior:
        return None

    return sum(1 for (dnf,) in prior if dnf) / len(prior)


def _team_strength(
    db: Session,
    constructor_id: int,
    season_year: int,
    race_round: int,
) -> Optional[float]:
    """Average points per race for the constructor in the current season (prior rounds)."""
    result = (
        db.query(func.avg(RaceResult.points))
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.constructor_id == constructor_id,
            Race.season_year == season_year,
            Race.round < race_round,
        )
        .scalar()
    )
    return float(result) if result is not None else None


def _team_reliability(
    db: Session,
    constructor_id: int,
    race_date,
    lookback: int = 20,
) -> Optional[float]:
    """1 - DNF rate for the constructor over its last `lookback` results."""
    prior = (
        db.query(RaceResult.is_dnf)
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.constructor_id == constructor_id,
            Race.race_date < race_date,
        )
        .order_by(Race.race_date.desc())
        .limit(lookback)
        .all()
    )

    if not prior:
        return None

    dnf_rate = sum(1 for (dnf,) in prior if dnf) / len(prior)
    return 1.0 - dnf_rate


def _track_affinity(
    db: Session,
    driver_id: int,
    circuit_id: int,
    race_date,
) -> Optional[float]:
    """Driver's average finishing position at this specific circuit."""
    prior = (
        db.query(func.avg(RaceResult.position))
        .join(Race, RaceResult.race_id == Race.id)
        .filter(
            RaceResult.driver_id == driver_id,
            Race.circuit_id == circuit_id,
            RaceResult.position.isnot(None),
            Race.race_date < race_date,
        )
        .scalar()
    )
    return float(prior) if prior is not None else None


def _qualifying_delta(
    db: Session,
    race_id: int,
    driver_id: int,
) -> Optional[float]:
    """
    Gap to pole position in qualifying (seconds).
    Uses the best qualifying time available (Q3 > Q2 > Q1).
    """
    all_quali = (
        db.query(Qualifying)
        .filter(Qualifying.race_id == race_id)
        .all()
    )

    if not all_quali:
        return None

    def _best_time(q: Qualifying) -> Optional[float]:
        """Extract the best qualifying time in seconds."""
        for time_str in [q.q3_time, q.q2_time, q.q1_time]:
            if time_str:
                parsed = _parse_time(time_str)
                if parsed is not None:
                    return parsed
        return None

    times = {}
    for q in all_quali:
        t = _best_time(q)
        if t is not None:
            times[q.driver_id] = t

    if not times:
        return None

    pole_time = min(times.values())
    driver_time = times.get(driver_id)

    if driver_time is None:
        return None

    return round(driver_time - pole_time, 4)


def _parse_time(time_str: str) -> Optional[float]:
    """Parse a qualifying time string like '1:23.456' to seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None


# ─── Main Feature Pipeline ────────────────────────────────────────────────────

def compute_features_for_race(db: Session, race: Race) -> int:
    """
    Compute and persist feature rows for all drivers in a given race.
    Returns the number of feature rows created.
    """
    # Skip if features already exist for this race
    existing = db.query(FeatureRow).filter_by(race_id=race.id).count()
    if existing > 0:
        return 0

    results = (
        db.query(RaceResult)
        .filter_by(race_id=race.id)
        .all()
    )

    if not results:
        return 0

    # Fetch race metadata
    circuit = db.query(Circuit).get(race.circuit_id)
    season = db.query(Season).get(race.season_year)
    weather = db.query(Weather).filter_by(race_id=race.id).first()

    # Regulation era flags
    era_v8 = season.regulation_era == "v8_era" if season else False
    era_hybrid = season.regulation_era == "hybrid_v6_era" if season else False
    era_ge = season.regulation_era == "ground_effect_era" if season else False

    count = 0
    for result in results:
        # Qualifying data
        quali = (
            db.query(Qualifying)
            .filter_by(race_id=race.id, driver_id=result.driver_id)
            .first()
        )

        # Compute features
        form = _driver_form(db, result.driver_id, race.race_date)
        win_rate, podium_rate = _driver_career_stats(db, result.driver_id, race.race_date)
        dnf_rate = _driver_dnf_rate(db, result.driver_id, race.race_date)
        t_strength = _team_strength(db, result.constructor_id, race.season_year, race.round)
        t_reliability = _team_reliability(db, result.constructor_id, race.race_date)
        t_affinity = _track_affinity(db, result.driver_id, race.circuit_id, race.race_date)
        q_delta = _qualifying_delta(db, race.id, result.driver_id)

        feature = FeatureRow(
            race_id=race.id,
            driver_id=result.driver_id,
            constructor_id=result.constructor_id,

            # Driver features
            driver_form=round(form, 4) if form is not None else None,
            driver_win_rate=round(win_rate, 4),
            driver_podium_rate=round(podium_rate, 4),
            driver_dnf_rate=round(dnf_rate, 4) if dnf_rate is not None else None,

            # Team features
            team_strength=round(t_strength, 4) if t_strength is not None else None,
            team_reliability=round(t_reliability, 4) if t_reliability is not None else None,

            # Track features
            track_affinity=round(t_affinity, 2) if t_affinity is not None else None,
            circuit_type_street=(circuit.circuit_type == "street") if circuit else False,
            circuit_type_hybrid=(circuit.circuit_type == "hybrid") if circuit else False,

            # Qualifying features
            grid_position=result.grid,
            qualifying_delta_sec=q_delta,

            # Weather
            rain_probability=weather.rain_probability if weather else 0.0,

            # Regulation era
            era_v8=era_v8,
            era_hybrid_v6=era_hybrid,
            era_ground_effect=era_ge,

            # Target labels
            finished_position=result.position,
            is_winner=(result.position == 1),
            is_podium=(result.position is not None and result.position <= 3),
            is_dnf=result.is_dnf,
        )
        db.add(feature)
        count += 1

    db.commit()
    return count


def compute_all_features(year: Optional[int] = None) -> None:
    """Compute features for all races (or a specific season)."""
    init_db()
    db = SessionLocal()
    try:
        query = db.query(Race).order_by(Race.season_year, Race.round)
        if year:
            query = query.filter(Race.season_year == year)

        races = query.all()
        if not races:
            print("  ⚠ No races found. Run data ingestion first.")
            return

        total_features = 0
        for race in tqdm(races, desc="  Computing features", unit="race"):
            n = compute_features_for_race(db, race)
            total_features += n

        print(f"\n  ✅ Features computed: {total_features} rows")
        print(f"  Total feature rows in DB: {db.query(FeatureRow).count()}")
    finally:
        db.close()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Intelligence — Feature Engineering"
    )
    parser.add_argument("--year", type=int, help="Compute features for a single season")
    args = parser.parse_args()
    compute_all_features(args.year)


if __name__ == "__main__":
    main()
