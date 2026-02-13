"""
Test suite for the data ingestion and feature engineering pipeline.
"""

import os
import sys
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import Base
from data.models import (
    Circuit, Constructor, Driver, FeatureRow,
    Qualifying, Race, RaceResult, Season, Weather,
)


@pytest.fixture
def db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def seeded_db(db):
    """Seed the database with sample data."""
    # Season
    season = Season(year=2023, regulation_era="ground_effect_era")
    db.add(season)

    # Circuit
    circuit = Circuit(
        ref="silverstone", name="Silverstone Circuit",
        country="UK", circuit_type="permanent",
        lat=52.0786, lng=-1.0169,
    )
    db.add(circuit)

    # Drivers
    d1 = Driver(ref="max_verstappen", code="VER", first_name="Max", last_name="Verstappen")
    d2 = Driver(ref="lewis_hamilton", code="HAM", first_name="Lewis", last_name="Hamilton")
    d3 = Driver(ref="charles_leclerc", code="LEC", first_name="Charles", last_name="Leclerc")
    db.add_all([d1, d2, d3])

    # Constructor
    c1 = Constructor(ref="red_bull", name="Red Bull Racing")
    c2 = Constructor(ref="mercedes", name="Mercedes")
    c3 = Constructor(ref="ferrari", name="Ferrari")
    db.add_all([c1, c2, c3])
    db.flush()

    # Race
    race = Race(
        season_year=2023, round=10, circuit_id=circuit.id,
        name="British Grand Prix", race_date=date(2023, 7, 9),
        total_laps=52,
    )
    db.add(race)
    db.flush()

    # Results
    r1 = RaceResult(
        race_id=race.id, driver_id=d1.id, constructor_id=c1.id,
        grid=1, position=1, points=25, laps=52, status="Finished", is_dnf=False,
    )
    r2 = RaceResult(
        race_id=race.id, driver_id=d2.id, constructor_id=c2.id,
        grid=3, position=2, points=18, laps=52, status="Finished", is_dnf=False,
    )
    r3 = RaceResult(
        race_id=race.id, driver_id=d3.id, constructor_id=c3.id,
        grid=5, position=None, points=0, laps=30, status="Engine", is_dnf=True,
    )
    db.add_all([r1, r2, r3])

    # Weather
    weather = Weather(
        race_id=race.id, temp_c=22.5, humidity_pct=55,
        rain_probability=0.1, wind_speed_kph=15, conditions="dry",
    )
    db.add(weather)

    db.commit()
    return db


class TestModels:
    """Test ORM model creation and relationships."""

    def test_circuit_creation(self, db):
        circuit = Circuit(ref="monza", name="Monza", country="Italy", circuit_type="permanent")
        db.add(circuit)
        db.commit()
        assert db.query(Circuit).count() == 1
        assert circuit.id is not None

    def test_driver_creation(self, db):
        driver = Driver(ref="test_driver", code="TST", first_name="Test", last_name="Driver")
        db.add(driver)
        db.commit()
        assert driver.id is not None
        assert driver.code == "TST"

    def test_race_relationships(self, seeded_db):
        race = seeded_db.query(Race).first()
        assert race is not None
        assert race.name == "British Grand Prix"
        assert len(race.results) == 3
        assert race.weather is not None
        assert race.circuit.name == "Silverstone Circuit"

    def test_dnf_status(self, seeded_db):
        results = seeded_db.query(RaceResult).all()
        dnf_results = [r for r in results if r.is_dnf]
        assert len(dnf_results) == 1
        assert dnf_results[0].status == "Engine"

    def test_season_era(self, seeded_db):
        season = seeded_db.query(Season).first()
        assert season.regulation_era == "ground_effect_era"


class TestFeatures:
    """Test feature row schema."""

    def test_feature_creation(self, seeded_db):
        race = seeded_db.query(Race).first()
        driver = seeded_db.query(Driver).first()
        constructor = seeded_db.query(Constructor).first()

        feature = FeatureRow(
            race_id=race.id,
            driver_id=driver.id,
            constructor_id=constructor.id,
            driver_form=2.5,
            team_strength=12.0,
            grid_position=1,
            era_ground_effect=True,
            is_winner=True,
            is_podium=True,
            is_dnf=False,
        )
        seeded_db.add(feature)
        seeded_db.commit()

        loaded = seeded_db.query(FeatureRow).first()
        assert loaded.driver_form == 2.5
        assert loaded.is_winner is True
        assert loaded.era_ground_effect is True
