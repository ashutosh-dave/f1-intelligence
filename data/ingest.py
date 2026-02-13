"""
Data ingestion pipeline for the F1 Race Intelligence Engine.

Fetches historical F1 data from the Jolpica-F1 API (Ergast-compatible)
for seasons 2010–present and persists it into the database.

Data sources:
  - Jolpica-F1:  https://api.jolpi.ca/ergast/f1  (historical + current, Ergast drop-in)
  - OpenF1:      https://api.openf1.org           (real-time telemetry, no signup)
  - Hyprace:     https://developers.hyprace.com   (live data, low latency)

Usage:
    python -m data.ingest              # Ingest all seasons 2010–2025
    python -m data.ingest --year 2024  # Ingest a single season
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from datetime import date, datetime
from typing import Any, Optional

import requests
from sqlalchemy.orm import Session
from tqdm import tqdm

from data.database import SessionLocal, init_db
from data.models import (
    Circuit, Constructor, Driver, Qualifying,
    Race, RaceResult, Season, Weather,
)

# ─── Constants ─────────────────────────────────────────────────────────────────

# Primary data source: Jolpica-F1 (Ergast-compatible drop-in replacement)
# The original Ergast API (ergast.com) is defunct and the domain is for sale.
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Legacy alias — kept so downstream references still resolve
ERGAST_BASE = JOLPICA_BASE

START_YEAR = 2010
END_YEAR = 2025
REQUEST_DELAY = 0.5  # polite rate limiting (seconds)

REGULATION_ERAS = {
    range(2010, 2014): "v8_era",
    range(2014, 2022): "hybrid_v6_era",
    range(2022, 2030): "ground_effect_era",
}

# Circuit type classification (hand-curated)
STREET_CIRCUITS = {
    "monaco", "baku", "jeddah", "marina_bay", "albert_park",
    "vegas", "miami", "villeneuve",
}
HYBRID_CIRCUITS = {
    "albert_park", "sochi", "hanoi",
}

# Known wet races (partial list for synthetic weather seeding)
KNOWN_WET_RACES = {
    (2010, 4), (2010, 7), (2010, 14), (2011, 6), (2011, 7),
    (2012, 5), (2012, 7), (2013, 4), (2014, 4), (2014, 17),
    (2015, 8), (2016, 6), (2016, 19), (2017, 3), (2017, 15),
    (2018, 10), (2019, 11), (2019, 20), (2020, 6), (2020, 11),
    (2020, 14), (2021, 2), (2021, 10), (2021, 15), (2021, 16),
    (2022, 3), (2022, 4), (2023, 6), (2023, 9), (2023, 20),
    (2024, 6), (2024, 13),
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    """Make a GET request to the Jolpica-F1 / Ergast-compatible API with retries."""
    params = params or {}
    params["limit"] = 1000
    for attempt in range(3):
        try:
            resp = requests.get(url + ".json", params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f"  ⚠ Retry {attempt+1}/3 for {url}: {exc}")
            time.sleep(2 ** attempt)
    return {}


def _get_era(year: int) -> str:
    """Return the regulation era for a given year."""
    for year_range, era in REGULATION_ERAS.items():
        if year in year_range:
            return era
    return "unknown"


def _classify_circuit(ref: str) -> str:
    """Return circuit type: street, hybrid, or permanent."""
    if ref in STREET_CIRCUITS:
        return "street"
    if ref in HYBRID_CIRCUITS:
        return "hybrid"
    return "permanent"


def _parse_date(date_str: str | None) -> date | None:
    """Parse an ISO date string."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_lap_time_to_sec(time_str: str | None) -> float | None:
    """Convert 'M:SS.mmm' or 'SS.mmm' to seconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _seed_weather(race_id: int, year: int, rnd: int, circuit_ref: str) -> Weather:
    """
    Generate synthetic weather for a race.
    Uses a deterministic seed so results are reproducible.
    """
    seed = int(hashlib.md5(f"{year}-{rnd}-{circuit_ref}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    is_wet = (year, rnd) in KNOWN_WET_RACES
    if is_wet:
        rain_prob = rng.uniform(0.5, 1.0)
        conditions = rng.choice(["damp", "wet"])
        temp = rng.uniform(12.0, 22.0)
    else:
        rain_prob = rng.uniform(0.0, 0.25)
        conditions = "dry"
        temp = rng.uniform(18.0, 38.0)

    return Weather(
        race_id=race_id,
        temp_c=round(temp, 1),
        humidity_pct=round(rng.uniform(30, 90), 1),
        rain_probability=round(rain_prob, 3),
        wind_speed_kph=round(rng.uniform(5, 35), 1),
        conditions=conditions,
    )


# ─── Ingestion Functions ──────────────────────────────────────────────────────

def _get_or_create_circuit(db: Session, circ_data: dict) -> Circuit:
    """Find or create a Circuit record."""
    ref = circ_data.get("circuitId", "")
    existing = db.query(Circuit).filter_by(ref=ref).first()
    if existing:
        return existing

    loc = circ_data.get("Location", {})
    circuit = Circuit(
        ref=ref,
        name=circ_data.get("circuitName", ref),
        country=loc.get("country", "Unknown"),
        locality=loc.get("locality"),
        lat=float(loc["lat"]) if loc.get("lat") else None,
        lng=float(loc["long"]) if loc.get("long") else None,
        circuit_type=_classify_circuit(ref),
    )
    db.add(circuit)
    db.flush()
    return circuit


def _get_or_create_driver(db: Session, drv_data: dict) -> Driver:
    """Find or create a Driver record."""
    ref = drv_data.get("driverId", "")
    existing = db.query(Driver).filter_by(ref=ref).first()
    if existing:
        return existing

    driver = Driver(
        ref=ref,
        code=drv_data.get("code"),
        first_name=drv_data.get("givenName", ""),
        last_name=drv_data.get("familyName", ""),
        nationality=drv_data.get("nationality"),
        dob=_parse_date(drv_data.get("dateOfBirth")),
        number=int(drv_data["permanentNumber"]) if drv_data.get("permanentNumber") else None,
    )
    db.add(driver)
    db.flush()
    return driver


def _get_or_create_constructor(db: Session, con_data: dict) -> Constructor:
    """Find or create a Constructor record."""
    ref = con_data.get("constructorId", "")
    existing = db.query(Constructor).filter_by(ref=ref).first()
    if existing:
        return existing

    constructor = Constructor(
        ref=ref,
        name=con_data.get("name", ref),
        nationality=con_data.get("nationality"),
    )
    db.add(constructor)
    db.flush()
    return constructor


def _is_dnf(status: str) -> bool:
    """Determine if a race status indicates a DNF."""
    finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps",
                         "+4 Laps", "+5 Laps", "+6 Laps", "+7 Laps",
                         "+8 Laps", "+9 Laps", "+10 Laps", "+11 Laps",
                         "+12 Laps"}
    return status not in finished_statuses


def ingest_season(db: Session, year: int) -> None:
    """Ingest all races, results, and qualifying for a single season."""
    print(f"\n{'='*60}")
    print(f"  Ingesting season {year} ({_get_era(year)})")
    print(f"{'='*60}")

    # ── Season record ──
    existing_season = db.query(Season).filter_by(year=year).first()
    if not existing_season:
        season = Season(year=year, regulation_era=_get_era(year))
        db.add(season)
        db.flush()
    else:
        season = existing_season

    # ── Fetch race schedule ──
    data = _get(f"{ERGAST_BASE}/{year}")
    races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races_data:
        print(f"  ⚠ No races found for {year}")
        return

    for race_data in tqdm(races_data, desc=f"  Races {year}", unit="race"):
        rnd = int(race_data["round"])

        # Skip if already ingested
        existing_race = db.query(Race).filter_by(
            season_year=year, round=rnd
        ).first()
        if existing_race:
            continue

        # ── Circuit ──
        circuit = _get_or_create_circuit(db, race_data.get("Circuit", {}))

        # ── Race ──
        race = Race(
            season_year=year,
            round=rnd,
            circuit_id=circuit.id,
            name=race_data.get("raceName", f"Round {rnd}"),
            race_date=_parse_date(race_data.get("date")),
        )
        db.add(race)
        db.flush()

        # ── Weather (synthetic) ──
        weather = _seed_weather(race.id, year, rnd, circuit.ref)
        db.add(weather)

        # ── Race Results ──
        results_data = _get(f"{ERGAST_BASE}/{year}/{rnd}/results")
        results_list = (
            results_data.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [{}])[0]
            .get("Results", [])
        )

        max_laps = 0
        for res in results_list:
            driver = _get_or_create_driver(db, res.get("Driver", {}))
            constructor = _get_or_create_constructor(db, res.get("Constructor", {}))
            status = res.get("status", "Unknown")
            laps = int(res.get("laps", 0))
            max_laps = max(max_laps, laps)

            pos_text = res.get("positionText", "")
            pos = int(res["position"]) if res.get("position") and res["position"].isdigit() else None

            fl = res.get("FastestLap", {})

            result = RaceResult(
                race_id=race.id,
                driver_id=driver.id,
                constructor_id=constructor.id,
                grid=int(res.get("grid", 0)),
                position=pos,
                position_text=pos_text,
                points=float(res.get("points", 0)),
                laps=laps,
                status=status,
                fastest_lap_rank=int(fl["rank"]) if fl.get("rank") else None,
                fastest_lap_time=fl.get("Time", {}).get("time"),
                is_dnf=_is_dnf(status),
            )
            db.add(result)

        # Update total laps
        race.total_laps = max_laps

        # ── Qualifying Results ──
        quali_data = _get(f"{ERGAST_BASE}/{year}/{rnd}/qualifying")
        quali_list = (
            quali_data.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [{}])[0]
            .get("QualifyingResults", [])
        )

        for q in quali_list:
            driver = _get_or_create_driver(db, q.get("Driver", {}))
            constructor = _get_or_create_constructor(db, q.get("Constructor", {}))

            quali = Qualifying(
                race_id=race.id,
                driver_id=driver.id,
                constructor_id=constructor.id,
                position=int(q.get("position", 0)),
                q1_time=q.get("Q1"),
                q2_time=q.get("Q2"),
                q3_time=q.get("Q3"),
            )
            db.add(quali)

        db.commit()

    print(f"  ✓ Season {year} complete ({len(races_data)} races)")


def ingest_all(start: int = START_YEAR, end: int = END_YEAR) -> None:
    """Ingest all seasons from start to end (inclusive)."""
    init_db()
    db = SessionLocal()
    try:
        for year in range(start, end + 1):
            ingest_season(db, year)
        print(f"\n{'='*60}")
        print(f"  ✅ Ingestion complete: {start}–{end}")
        print(f"{'='*60}")

        # Summary stats
        print(f"  Circuits:     {db.query(Circuit).count()}")
        print(f"  Drivers:      {db.query(Driver).count()}")
        print(f"  Constructors: {db.query(Constructor).count()}")
        print(f"  Races:        {db.query(Race).count()}")
        print(f"  Results:      {db.query(RaceResult).count()}")
        print(f"  Qualifying:   {db.query(Qualifying).count()}")
        print(f"  Weather:      {db.query(Weather).count()}")
    finally:
        db.close()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Race Intelligence — Data Ingestion")
    parser.add_argument("--year", type=int, help="Ingest a single season")
    parser.add_argument("--start", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end", type=int, default=END_YEAR, help="End year")
    args = parser.parse_args()

    if args.year:
        init_db()
        db = SessionLocal()
        try:
            ingest_season(db, args.year)
        finally:
            db.close()
    else:
        ingest_all(args.start, args.end)


if __name__ == "__main__":
    main()
