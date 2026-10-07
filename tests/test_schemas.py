"""Tests for clinic schemas and SQLite persistence."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.db import load_clinic_config, save_clinic_config
from app.schemas import BookingRequest, ClinicConfig, ConsultationEvent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"


def load_sample_config_data() -> dict:
    """Read the sample clinic configuration as a dictionary."""
    return json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))


def test_sample_config_loads_and_validates() -> None:
    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert config.clinic_name == "Al Noor Family Clinic"
    assert config.available_days == ["Mon", "Tue", "Wed", "Thu", "Sat"]


def test_work_end_before_work_start_is_rejected() -> None:
    config = load_sample_config_data()
    config["work_start"] = "21:00:00"
    config["work_end"] = "16:00:00"

    with pytest.raises(ValidationError, match="work_end must be after work_start"):
        ClinicConfig.model_validate(config)


def test_high_wait_threshold_below_acceptable_wait_is_rejected() -> None:
    config = load_sample_config_data()
    config["acceptable_wait_min"] = 20
    config["high_wait_threshold_min"] = 15

    with pytest.raises(ValidationError, match="high_wait_threshold_min"):
        ClinicConfig.model_validate(config)


def test_break_outside_working_hours_is_rejected() -> None:
    config = load_sample_config_data()
    config["breaks"][0] = {"start": "15:30:00", "end": "15:45:00"}

    with pytest.raises(ValidationError, match="must lie within working hours"):
        ClinicConfig.model_validate(config)


def test_consultation_event_with_end_before_start_is_rejected() -> None:
    with pytest.raises(ValidationError, match="end must be after start"):
        ConsultationEvent(
            appointment_id=1,
            start="2026-10-08T17:00:00",
            end="2026-10-08T16:45:00",
        )


def test_booking_request_with_past_date_is_rejected() -> None:
    with pytest.raises(ValidationError, match="appointment_date must be after today"):
        BookingRequest(
            patient_name="Test Patient",
            patient_phone="+962-7-9000-0000",
            appointment_date=date.today() - timedelta(days=1),
        )


def test_clinic_config_database_round_trip(tmp_path: Path) -> None:
    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    database_path = str(tmp_path / "clinic.sqlite")

    save_clinic_config(config, path=database_path)

    assert load_clinic_config(path=database_path) == config
