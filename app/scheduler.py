"""Rule-based appointment scheduling and wait-time recommendations."""

from datetime import date, datetime, time, timedelta, tzinfo
from typing import Literal

from app.db import (
    DEFAULT_DB_PATH,
    get_appointment,
    get_next_appointment_id,
    has_active_booking,
    insert_appointment,
    list_appointments_by_date,
    load_clinic_config,
    update_arrival,
    update_consultation,
    update_outcome,
)
from app.schemas import (
    AppointmentRecord,
    ArrivalEvent,
    BookingRequest,
    BookingResponse,
    ClinicConfig,
    ConsultationEvent,
)


ARRIVE_EARLY_MIN = 10
PEAK_EXTRA_WAIT_MIN = 5


class ClinicClosedError(Exception):
    """Raised when booking is requested for a clinic's closed day."""


class FullyBookedError(Exception):
    """Raised when no daily capacity or appointment slots remain."""


class DuplicateBookingError(Exception):
    """Raised when the same phone already has an active booking that day."""


class ConfigNotFoundError(Exception):
    """Raised when no clinic configuration has been saved."""


def day_abbrev(d: date) -> str:
    """Return the three-letter weekday abbreviation for a date."""
    return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[d.weekday()]


def add_minutes(t: time, minutes: int) -> time:
    """Add minutes to a time without wrapping across midnight."""
    try:
        result = datetime.combine(date.min, t) + timedelta(minutes=minutes)
    except OverflowError as exc:
        raise ValueError("Time arithmetic cannot cross midnight") from exc
    if result.date() != date.min:
        raise ValueError("Time arithmetic cannot cross midnight")
    return result.time()


def generate_slots(config: ClinicConfig, d: date) -> list[time]:
    """Generate appointment start times that fit working hours and breaks."""
    if day_abbrev(d) not in config.available_days:
        return []

    interval = config.appointment_duration_min + config.buffer_min
    slots: list[time] = []
    current = datetime.combine(d, config.work_start)
    working_end = datetime.combine(d, config.work_end)
    while current + timedelta(minutes=config.appointment_duration_min) <= working_end:
        slot_end = current + timedelta(minutes=config.appointment_duration_min)
        overlaps_break = any(
            current.time() < clinic_break.end
            and slot_end.time() > clinic_break.start
            for clinic_break in config.breaks
        )
        if not overlaps_break:
            slots.append(current.time())
        current += timedelta(minutes=interval)
    return slots


def is_peak(config: ClinicConfig, t: time) -> bool:
    """Return whether a time falls inside any configured peak range."""
    return any(peak.start <= t < peak.end for peak in config.peak_hours)


def get_service_duration(
    config: ClinicConfig, service_name: str | None
) -> int:
    """Return a named service duration or the default appointment duration."""
    if service_name is None:
        return config.appointment_duration_min
    for service in config.services:
        if service.name.casefold() == service_name.casefold():
            return service.duration_min
    raise ValueError(f"Unknown service: {service_name}")


def estimate_wait_rules(
    patients_ahead_in_slot: int, config: ClinicConfig, scheduled_time: time
) -> int:
    """Estimate wait from patients sharing the slot and peak-period demand."""
    return (
        patients_ahead_in_slot
        * (config.appointment_duration_min + config.buffer_min)
        + (PEAK_EXTRA_WAIT_MIN if is_peak(config, scheduled_time) else 0)
    )


def classify_wait(
    wait_min: int, config: ClinicConfig
) -> Literal["acceptable", "moderate", "high"]:
    """Classify a wait using the clinic's configured thresholds."""
    if wait_min <= config.acceptable_wait_min:
        return "acceptable"
    if wait_min >= config.high_wait_threshold_min:
        return "high"
    return "moderate"


def _require_config(path: str) -> ClinicConfig:
    config = load_clinic_config(path=path)
    if config is None:
        raise ConfigNotFoundError("No clinic configuration has been saved")
    return config


def book_appointment(
    req: BookingRequest, path: str = DEFAULT_DB_PATH
) -> BookingResponse:
    """Book the first available slot using cold-start rule-based estimates."""
    config = _require_config(path)
    if day_abbrev(req.appointment_date) not in config.available_days:
        raise ClinicClosedError(
            f"The clinic is closed on {day_abbrev(req.appointment_date)}"
        )
    get_service_duration(config, req.service)

    appointments = list_appointments_by_date(req.appointment_date, path=path)
    active_appointments = [
        appointment
        for appointment in appointments
        if appointment.outcome != "cancelled"
    ]
    if has_active_booking(req.patient_phone, req.appointment_date, path=path):
        raise DuplicateBookingError(
            "This phone number already has an active booking on that date"
        )
    if len(active_appointments) >= config.max_appointments_per_day:
        raise FullyBookedError("The clinic has reached its daily booking limit")

    slot = next(
        (
            candidate
            for candidate in generate_slots(config, req.appointment_date)
            if sum(
                appointment.scheduled_time == candidate
                for appointment in active_appointments
            )
            < config.max_patients_per_slot
        ),
        None,
    )
    if slot is None:
        raise FullyBookedError("There are no appointment slots available")

    patients_before = len(active_appointments)
    queue_number = patients_before + 1
    patients_ahead_in_slot = sum(
        appointment.scheduled_time == slot for appointment in active_appointments
    )
    estimated_wait_min = estimate_wait_rules(
        patients_ahead_in_slot, config, slot
    )
    expected_consultation_time = add_minutes(slot, estimated_wait_min)
    expected_minutes = (
        expected_consultation_time.hour * 60
        + expected_consultation_time.minute
    )
    work_start_minutes = config.work_start.hour * 60 + config.work_start.minute
    if expected_minutes - ARRIVE_EARLY_MIN < work_start_minutes:
        recommended_arrival_time = config.work_start
    else:
        recommended_arrival_time = add_minutes(
            expected_consultation_time, -ARRIVE_EARLY_MIN
        )

    record = AppointmentRecord(
        appointment_id=get_next_appointment_id(path=path),
        booking_timestamp=datetime.now(),
        appointment_date=req.appointment_date,
        day_of_week=day_abbrev(req.appointment_date),
        scheduled_time=slot,
        queue_number=queue_number,
        patients_before=patients_before,
        bookings_that_day=queue_number,
        is_peak=is_peak(config, slot),
        recommended_arrival=recommended_arrival_time,
        predicted_wait_min=estimated_wait_min,
        predicted_by="rules",
        outcome="pending",
    )
    insert_appointment(record, path=path, patient_phone=req.patient_phone)

    return BookingResponse(
        appointment_id=record.appointment_id,
        appointment_date=record.appointment_date,
        queue_number=record.queue_number,
        patients_before=record.patients_before,
        expected_consultation_time=expected_consultation_time,
        recommended_arrival_time=recommended_arrival_time,
        estimated_wait_min=estimated_wait_min,
        method="rules",
    )


def _scheduled_datetime(
    record: AppointmentRecord, timezone: tzinfo | None = None
) -> datetime:
    """Combine an appointment date and time, optionally adopting a timezone."""
    return datetime.combine(
        record.appointment_date, record.scheduled_time, tzinfo=timezone
    )


def record_arrival(
    event: ArrivalEvent, path: str = DEFAULT_DB_PATH
) -> AppointmentRecord:
    """Record arrival and whole-minute lateness for an existing booking."""
    record = get_appointment(event.appointment_id, path=path)
    if record is None:
        raise ValueError(f"Appointment {event.appointment_id} does not exist")
    scheduled = _scheduled_datetime(record, event.actual_arrival.tzinfo)
    late_minutes = max(
        0, int((event.actual_arrival - scheduled).total_seconds() // 60)
    )
    update_arrival(
        event.appointment_id,
        event.actual_arrival,
        late_minutes,
        path=path,
    )
    updated = get_appointment(event.appointment_id, path=path)
    if updated is None:
        raise ValueError(f"Appointment {event.appointment_id} does not exist")
    return updated


def record_consultation(
    event: ConsultationEvent, path: str = DEFAULT_DB_PATH
) -> AppointmentRecord:
    """Record consultation duration and actual wait after a saved arrival."""
    record = get_appointment(event.appointment_id, path=path)
    if record is None:
        raise ValueError(f"Appointment {event.appointment_id} does not exist")
    if record.actual_arrival is None:
        raise ValueError("An arrival must be recorded before the consultation")

    consult_duration_min = (event.end - event.start).total_seconds() / 60
    actual_wait_min = max(
        0, (event.start - record.actual_arrival).total_seconds() / 60
    )
    update_consultation(
        event.appointment_id,
        event.start,
        event.end,
        consult_duration_min,
        actual_wait_min,
        "completed",
        path=path,
    )
    updated = get_appointment(event.appointment_id, path=path)
    if updated is None:
        raise ValueError(f"Appointment {event.appointment_id} does not exist")
    return updated


def mark_no_show(
    appointment_id: int, path: str = DEFAULT_DB_PATH
) -> None:
    """Mark an appointment as a no-show."""
    if get_appointment(appointment_id, path=path) is None:
        raise ValueError(f"Appointment {appointment_id} does not exist")
    update_outcome(appointment_id, "no_show", path=path)


def cancel_appointment(
    appointment_id: int, path: str = DEFAULT_DB_PATH
) -> None:
    """Cancel an appointment, freeing its active booking capacity."""
    if get_appointment(appointment_id, path=path) is None:
        raise ValueError(f"Appointment {appointment_id} does not exist")
    update_outcome(appointment_id, "cancelled", path=path)
