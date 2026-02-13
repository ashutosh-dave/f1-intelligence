"""
SQLAlchemy ORM models for the F1 Race Intelligence Engine.

Tables:
  - circuits      Track metadata
  - drivers       Driver master records
  - constructors  Team master records
  - seasons       Season + regulation era mapping
  - races         Individual Grand Prix events
  - race_results  Finishing results per driver per race
  - qualifying    Qualifying session times
  - weather       Race-day weather conditions
  - features      Pre-computed feature vectors for ML
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, Date, DateTime,
    ForeignKey, UniqueConstraint, Boolean, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data.database import Base


# ─── Track / Circuit ───────────────────────────────────────────────────────────

class Circuit(Base):
    __tablename__ = "circuits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    locality: Mapped[str] = mapped_column(String(100), nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    altitude: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    circuit_length_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    circuit_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="street | permanent | hybrid"
    )

    races: Mapped[list[Race]] = relationship(back_populates="circuit")


# ─── Driver ────────────────────────────────────────────────────────────────────

class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    nationality: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dob: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    results: Mapped[list[RaceResult]] = relationship(back_populates="driver")
    qualifying_results: Mapped[list[Qualifying]] = relationship(back_populates="driver")


# ─── Constructor / Team ────────────────────────────────────────────────────────

class Constructor(Base):
    __tablename__ = "constructors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    nationality: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    results: Mapped[list[RaceResult]] = relationship(back_populates="constructor")


# ─── Season ────────────────────────────────────────────────────────────────────

class Season(Base):
    __tablename__ = "seasons"

    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    regulation_era: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="v8_era | hybrid_v6_era | ground_effect_era"
    )
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    races: Mapped[list[Race]] = relationship(back_populates="season")


# ─── Race (Grand Prix) ────────────────────────────────────────────────────────

class Race(Base):
    __tablename__ = "races"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season_year: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.year"), nullable=False
    )
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    circuit_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("circuits.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    race_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    total_laps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("season_year", "round", name="uq_race_season_round"),
    )

    season: Mapped[Season] = relationship(back_populates="races")
    circuit: Mapped[Circuit] = relationship(back_populates="races")
    results: Mapped[list[RaceResult]] = relationship(back_populates="race")
    qualifying_results: Mapped[list[Qualifying]] = relationship(back_populates="race")
    weather: Mapped[Optional[Weather]] = relationship(
        back_populates="race", uselist=False
    )
    feature_rows: Mapped[list[FeatureRow]] = relationship(back_populates="race")


# ─── Race Result ───────────────────────────────────────────────────────────────

class RaceResult(Base):
    __tablename__ = "race_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("races.id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drivers.id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("constructors.id"), nullable=False
    )
    grid: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="NULL if DNF"
    )
    position_text: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    points: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    laps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(100), nullable=False, default="Finished"
    )
    fastest_lap_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fastest_lap_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    is_dnf: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_result_race_driver"),
    )

    race: Mapped[Race] = relationship(back_populates="results")
    driver: Mapped[Driver] = relationship(back_populates="results")
    constructor: Mapped[Constructor] = relationship(back_populates="results")


# ─── Qualifying ────────────────────────────────────────────────────────────────

class Qualifying(Base):
    __tablename__ = "qualifying"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("races.id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drivers.id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("constructors.id"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    q1_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    q2_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    q3_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_quali_race_driver"),
    )

    race: Mapped[Race] = relationship(back_populates="qualifying_results")
    driver: Mapped[Driver] = relationship(back_populates="qualifying_results")
    constructor: Mapped[Constructor] = relationship()


# ─── Weather ───────────────────────────────────────────────────────────────────

class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("races.id"), unique=True, nullable=False
    )
    temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rain_probability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    wind_speed_kph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions: Mapped[str] = mapped_column(
        String(50), nullable=False, default="dry",
        comment="dry | damp | wet"
    )

    race: Mapped[Race] = relationship(back_populates="weather")


# ─── Feature Row (pre-computed for ML) ─────────────────────────────────────────

class FeatureRow(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("races.id"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drivers.id"), nullable=False
    )
    constructor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("constructors.id"), nullable=False
    )

    # ── driver features ──
    driver_form: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Weighted avg finish pos over last 5 races"
    )
    driver_win_rate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Career win rate"
    )
    driver_podium_rate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Career podium rate"
    )
    driver_dnf_rate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Recent DNF rate (last 10 races)"
    )

    # ── team features ──
    team_strength: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Avg points per race in current season"
    )
    team_reliability: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="1 - team DNF rate (last 20 races)"
    )

    # ── track features ──
    track_affinity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Driver avg finish at this circuit"
    )
    circuit_type_street: Mapped[bool] = mapped_column(Boolean, default=False)
    circuit_type_hybrid: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── qualifying features ──
    grid_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    qualifying_delta_sec: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Gap to pole in seconds"
    )

    # ── weather features ──
    rain_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── regulation era one-hot ──
    era_v8: Mapped[bool] = mapped_column(Boolean, default=False)
    era_hybrid_v6: Mapped[bool] = mapped_column(Boolean, default=False)
    era_ground_effect: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── target variables (labels) ──
    finished_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_winner: Mapped[bool] = mapped_column(Boolean, default=False)
    is_podium: Mapped[bool] = mapped_column(Boolean, default=False)
    is_dnf: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_feature_race_driver"),
    )

    race: Mapped[Race] = relationship(back_populates="feature_rows")
    driver: Mapped[Driver] = relationship()
    constructor: Mapped[Constructor] = relationship()
