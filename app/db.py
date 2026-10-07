"""SQLite persistence for clinic configuration and appointment records."""

import sqlite3
from pathlib import Path

from app.schemas import AppointmentRecord, ClinicConfig


DEFAULT_DB_PATH = "data/clinic.sqlite"
_APPOINTMENT_COLUMNS = (
    "appointment_id, booking_timestamp, appointment_date, day_of_week, "
    "scheduled_time, queue_number, patients_before, bookings_that_day, is_peak, "
    "recommended_arrival, predicted_wait_min, predicted_by, actual_arrival, "
    "late_minutes, consult_start, consult_end, consult_duration_min, "
    "actual_wait_min, outcome"
)


def get_connection(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating its parent directory when needed."""
    if path != ":memory:" and not path.startswith("file:"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(path: str = DEFAULT_DB_PATH) -> None:
    """Create the clinic configuration and appointment tables if needed."""
    connection = get_connection(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS clinic_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                config_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id INTEGER PRIMARY KEY,
                booking_timestamp TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                day_of_week TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                queue_number INTEGER NOT NULL,
                patients_before INTEGER NOT NULL,
                bookings_that_day INTEGER NOT NULL,
                is_peak INTEGER NOT NULL,
                recommended_arrival TEXT NOT NULL,
                predicted_wait_min INTEGER NOT NULL,
                predicted_by TEXT NOT NULL,
                actual_arrival TEXT,
                late_minutes INTEGER,
                consult_start TEXT,
                consult_end TEXT,
                consult_duration_min REAL,
                actual_wait_min REAL,
                outcome TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def save_clinic_config(
    config: ClinicConfig, path: str = DEFAULT_DB_PATH
) -> None:
    """Insert or replace the clinic's single configuration row."""
    init_db(path)
    connection = get_connection(path)
    try:
        connection.execute(
            """
            INSERT INTO clinic_config (id, config_json, updated_at)
            VALUES (1, ?, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            (config.model_dump_json(),),
        )
        connection.commit()
    finally:
        connection.close()


def load_clinic_config(path: str = DEFAULT_DB_PATH) -> ClinicConfig | None:
    """Load the clinic configuration, returning None when no row exists."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            "SELECT config_json FROM clinic_config WHERE id = 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return ClinicConfig.model_validate_json(row[0])


def insert_appointment(
    record: AppointmentRecord, path: str = DEFAULT_DB_PATH
) -> None:
    """Insert an appointment record, serializing dates and times as ISO strings."""
    init_db(path)
    values = record.model_dump(mode="json")
    placeholders = ", ".join("?" for _ in values)
    connection = get_connection(path)
    try:
        connection.execute(
            f"INSERT INTO appointments ({_APPOINTMENT_COLUMNS}) "
            f"VALUES ({placeholders})",
            tuple(values.values()),
        )
        connection.commit()
    finally:
        connection.close()


def get_appointment(
    appointment_id: int, path: str = DEFAULT_DB_PATH
) -> AppointmentRecord | None:
    """Load an appointment record by its ID."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            f"SELECT {_APPOINTMENT_COLUMNS} FROM appointments "
            "WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return AppointmentRecord.model_validate(dict(zip(_APPOINTMENT_COLUMNS.split(", "), row)))
