"""Tests for rule-based clinic scheduling and appointment event recording."""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from app.db import get_appointment, save_clinic_config
from app.schemas import (
    ArrivalEvent,
    BookingRequest,
    ClinicConfig,
    ConsultationEvent,
)
from app.scheduler import (
    ClinicClosedError,
    DuplicateBookingError,
    FullyBookedError,
    book_appointment,
    cancel_appointment,
    classify_wait,
    day_abbrev,
    estimate_wait_rules,
    generate_slots,
    get_service_duration,
    is_peak,
    record_arrival,
    record_consultation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"


def next_open_day(config: ClinicConfig) -> date:
    """Find the next future date when the configured clinic is open."""
    candidate = date.today() + timedelta(days=1)
    while day_abbrev(candidate) not in config.available_days:
        candidate += timedelta(days=1)
    return candidate


def next_closed_day(config: ClinicConfig) -> date:
    """Find the next future date when the configured clinic is closed."""
    candidate = date.today() + timedelta(days=1)
    while day_abbrev(candidate) in config.available_days:
        candidate += timedelta(days=1)
    return candidate


def booking_request(
    appointment_date: date, phone: str = "+962-7-9000-0000"
) -> BookingRequest:
    """Build a valid booking request for a future clinic date."""
    return BookingRequest(
        patient_name="Test Patient",
        patient_phone=phone,
        appointment_date=appointment_date,
    )


@pytest.fixture
def scheduler_setup(tmp_path: Path) -> tuple[ClinicConfig, str]:
    """Save the sample config to an isolated temporary database."""
    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    database_path = str(tmp_path / "scheduler.sqlite")
    save_clinic_config(config, path=database_path)
    return config, database_path


def test_generate_slots_respects_hours_break_and_interval(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup
    slots = generate_slots(config, next_open_day(config))
    interval = config.appointment_duration_min + config.buffer_min

    assert slots[0] == config.work_start
    assert all(
        datetime.combine(date.min, slot)
        + timedelta(minutes=config.appointment_duration_min)
        <= datetime.combine(date.min, config.work_end)
        for slot in slots
    )
    assert all(
        not (
            slot < clinic_break.end
            and (
                datetime.combine(date.min, slot)
                + timedelta(minutes=config.appointment_duration_min)
            ).time()
            > clinic_break.start
        )
        for slot in slots
        for clinic_break in config.breaks
    )
    assert all(
        int(
            (
                datetime.combine(date.min, slot)
                - datetime.combine(date.min, config.work_start)
            ).total_seconds()
            // 60
        )
        % interval
        == 0
        for slot in slots
    )
    assert any(
        (later.hour * 60 + later.minute) - (earlier.hour * 60 + earlier.minute)
        == interval
        for earlier, later in zip(slots, slots[1:])
    )


def test_generate_slots_returns_empty_on_closed_day(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup

    assert generate_slots(config, next_closed_day(config)) == []


def test_is_peak_uses_inclusive_start_and_exclusive_end(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup

    assert is_peak(config, time(18, 0))
    assert not is_peak(config, time(16, 30))
    assert not is_peak(config, time(20, 0))


@pytest.mark.parametrize(
    ("wait_min", "expected"),
    [(15, "acceptable"), (16, "moderate"), (29, "moderate"), (30, "high")],
)
def test_classify_wait_uses_threshold_boundaries(
    scheduler_setup: tuple[ClinicConfig, str],
    wait_min: int,
    expected: str,
) -> None:
    config, _ = scheduler_setup

    assert classify_wait(wait_min, config) == expected


def test_estimate_wait_includes_same_slot_patients_and_peak_extra(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup

    assert estimate_wait_rules(2, config, time(18, 0)) == 45
    assert estimate_wait_rules(2, config, time(16, 30)) == 40


def test_first_booking_has_first_queue_position(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    response = book_appointment(
        booking_request(next_open_day(config)), path=database_path
    )

    assert response.queue_number == 1
    assert response.patients_before == 0
    assert response.method == "rules"


def test_second_booking_counts_the_first_active_booking(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    appointment_date = next_open_day(config)

    first = book_appointment(booking_request(appointment_date), path=database_path)
    second = book_appointment(
        booking_request(appointment_date, "+962-7-9000-0001"),
        path=database_path,
    )

    assert first.queue_number == 1
    assert second.queue_number == 2
    assert second.patients_before == 1


def test_recommended_arrival_is_early_but_not_before_work_start(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    appointment_date = next_open_day(config)
    book_appointment(
        booking_request(appointment_date), path=database_path
    )
    response = book_appointment(
        booking_request(appointment_date, "+962-7-9000-0001"),
        path=database_path,
    )

    assert response.recommended_arrival_time < response.expected_consultation_time
    assert response.recommended_arrival_time >= config.work_start


def test_booking_on_closed_day_is_rejected(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup

    with pytest.raises(ClinicClosedError):
        book_appointment(
            booking_request(next_closed_day(config)), path=database_path
        )


def test_duplicate_active_phone_booking_is_rejected(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    request = booking_request(next_open_day(config))
    book_appointment(request, path=database_path)

    with pytest.raises(DuplicateBookingError):
        book_appointment(request, path=database_path)


def test_daily_appointment_limit_is_enforced(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    limited_config = config.model_copy(
        update={"max_appointments_per_day": 1}
    )
    save_clinic_config(limited_config, path=database_path)
    appointment_date = next_open_day(config)
    book_appointment(booking_request(appointment_date), path=database_path)

    with pytest.raises(FullyBookedError):
        book_appointment(
            booking_request(appointment_date, "+962-7-9000-0001"),
            path=database_path,
        )


def test_arrival_and_consultation_events_update_record(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    response = book_appointment(
        booking_request(next_open_day(config)), path=database_path
    )
    booked_record = get_appointment(response.appointment_id, path=database_path)
    assert booked_record is not None
    scheduled = datetime.combine(
        booked_record.appointment_date, booked_record.scheduled_time
    )
    actual_arrival = scheduled + timedelta(minutes=7)
    record = record_arrival(
        ArrivalEvent(
            appointment_id=response.appointment_id,
            actual_arrival=actual_arrival,
        ),
        path=database_path,
    )
    consult_start = actual_arrival + timedelta(minutes=12)
    record = record_consultation(
        ConsultationEvent(
            appointment_id=response.appointment_id,
            start=consult_start,
            end=consult_start + timedelta(minutes=18),
        ),
        path=database_path,
    )

    assert record.late_minutes == 7
    assert record.consult_duration_min == 18
    assert record.actual_wait_min == 12
    assert record.outcome == "completed"


def test_consultation_before_arrival_is_rejected(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    response = book_appointment(
        booking_request(next_open_day(config)), path=database_path
    )
    consult_start = datetime.combine(
        response.appointment_date, response.expected_consultation_time
    )

    with pytest.raises(ValueError, match="arrival must be recorded"):
        record_consultation(
            ConsultationEvent(
                appointment_id=response.appointment_id,
                start=consult_start,
                end=consult_start + timedelta(minutes=15),
            ),
            path=database_path,
        )


def test_cancellation_frees_daily_capacity_and_queue_count(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    save_clinic_config(
        config.model_copy(update={"max_appointments_per_day": 1}),
        path=database_path,
    )
    appointment_date = next_open_day(config)
    first = book_appointment(
        booking_request(appointment_date), path=database_path
    )
    cancel_appointment(first.appointment_id, path=database_path)

    replacement = book_appointment(
        booking_request(appointment_date, "+962-7-9000-0001"),
        path=database_path,
    )

    assert replacement.queue_number == 1
    assert replacement.patients_before == 0


def test_cancelled_booking_does_not_prevent_same_phone_rebooking(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, database_path = scheduler_setup
    request = booking_request(next_open_day(config))
    first = book_appointment(request, path=database_path)
    cancel_appointment(first.appointment_id, path=database_path)

    replacement = book_appointment(request, path=database_path)

    assert replacement.queue_number == 1


def test_unknown_service_name_is_rejected(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup

    with pytest.raises(ValueError, match="Unknown service"):
        get_service_duration(config, "Unlisted service")


def test_service_duration_lookup_is_case_insensitive(
    scheduler_setup: tuple[ClinicConfig, str],
) -> None:
    config, _ = scheduler_setup

    assert get_service_duration(config, "GENERAL CONSULTATION") == 15
