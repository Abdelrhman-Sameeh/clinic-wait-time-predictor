"""SQLite persistence for clinic configuration and appointment records."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.config_defaults import DEFAULT_CLINIC_CONFIG
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
                patient_phone TEXT,
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        appointment_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(appointments)").fetchall()
        }
        if "patient_phone" not in appointment_columns:
            connection.execute("ALTER TABLE appointments ADD COLUMN patient_phone TEXT")
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


def backfill_default_clinic_config(path: str = DEFAULT_DB_PATH) -> ClinicConfig:
    """Create or repair the single clinic config using repository defaults."""
    init_db(path)
    try:
        existing = load_clinic_config(path=path)
    except (ValidationError, ValueError):
        existing = None
    if existing is None:
        config = ClinicConfig.model_validate(DEFAULT_CLINIC_CONFIG)
        save_clinic_config(config, path=path)
        return config
    raw_existing = existing.model_dump(mode="json")
    merged = {
        **DEFAULT_CLINIC_CONFIG,
        **raw_existing,
    }
    config = ClinicConfig.model_validate(merged)
    if config != existing:
        save_clinic_config(config, path=path)
    return config


def list_knowledge_base(path: str = DEFAULT_DB_PATH) -> list[dict[str, str]]:
    """List knowledge base entries in creation order."""
    init_db(path)
    connection = get_connection(path)
    try:
        rows = connection.execute(
            "SELECT id, title, content, created_at FROM knowledge_base "
            "ORDER BY created_at, id"
        ).fetchall()
    finally:
        connection.close()
    return [
        {"id": row[0], "title": row[1], "content": row[2], "timestamp": row[3]}
        for row in rows
    ]


def save_knowledge_base_item(
    title: str,
    content: str,
    path: str = DEFAULT_DB_PATH,
    item_id: str | None = None,
) -> dict[str, str]:
    """Create or update one admin knowledge entry."""
    init_db(path)
    now = datetime.now().astimezone().isoformat()
    entry_id = item_id or str(uuid4())
    connection = get_connection(path)
    try:
        existing = connection.execute(
            "SELECT created_at FROM knowledge_base WHERE id = ?",
            (entry_id,),
        ).fetchone()
        created_at = existing[0] if existing else now
        connection.execute(
            """
            INSERT INTO knowledge_base (id, title, content, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (entry_id, title, content, created_at, now),
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "id": entry_id,
        "title": title,
        "content": content,
        "timestamp": created_at,
    }


def delete_knowledge_base_item(
    item_id: str, path: str = DEFAULT_DB_PATH
) -> bool:
    """Delete a knowledge entry by ID."""
    init_db(path)
    connection = get_connection(path)
    try:
        cursor = connection.execute(
            "DELETE FROM knowledge_base WHERE id = ?",
            (item_id,),
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def insert_appointment(
    record: AppointmentRecord,
    path: str = DEFAULT_DB_PATH,
    patient_phone: str | None = None,
) -> None:
    """Insert an appointment and optional contact phone, serializing ISO values."""
    init_db(path)
    values = record.model_dump(mode="json")
    columns = _APPOINTMENT_COLUMNS
    parameters = tuple(values.values())
    if patient_phone is not None:
        columns = f"{columns}, patient_phone"
        parameters += (patient_phone,)
    placeholders = ", ".join("?" for _ in parameters)
    connection = get_connection(path)
    try:
        connection.execute(
            f"INSERT INTO appointments ({columns}) "
            f"VALUES ({placeholders})",
            parameters,
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


def get_patient_phone(
    appointment_id: int, path: str = DEFAULT_DB_PATH
) -> str | None:
    """Read the stored contact phone for an appointment."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            "SELECT patient_phone FROM appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else row[0]


def get_appointment_data_summary(path: str = DEFAULT_DB_PATH) -> tuple[int, int]:
    """Return total appointment count and completed rows with observed waits."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN outcome = 'completed' AND actual_wait_min IS NOT NULL
                        THEN 1 ELSE 0
                    END
                )
            FROM appointments
            """
        ).fetchone()
    finally:
        connection.close()
    return int(row[0]), int(row[1] or 0)


def get_next_appointment_id(path: str = DEFAULT_DB_PATH) -> int:
    """Return the next appointment ID, starting with 1."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            "SELECT COALESCE(MAX(appointment_id), 0) + 1 FROM appointments"
        ).fetchone()
    finally:
        connection.close()
    return int(row[0])


def list_appointments_by_date(
    appointment_date: date, path: str = DEFAULT_DB_PATH
) -> list[AppointmentRecord]:
    """Load a day's appointments ordered by queue number."""
    init_db(path)
    connection = get_connection(path)
    try:
        rows = connection.execute(
            f"SELECT {_APPOINTMENT_COLUMNS} FROM appointments "
            "WHERE appointment_date = ? ORDER BY queue_number",
            (appointment_date.isoformat(),),
        ).fetchall()
    finally:
        connection.close()
    column_names = _APPOINTMENT_COLUMNS.split(", ")
    return [
        AppointmentRecord.model_validate(dict(zip(column_names, row)))
        for row in rows
    ]


def update_arrival(
    appointment_id: int,
    actual_arrival: datetime,
    late_minutes: int,
    path: str = DEFAULT_DB_PATH,
) -> None:
    """Save an arrival timestamp and the calculated lateness."""
    init_db(path)
    connection = get_connection(path)
    try:
        connection.execute(
            """
            UPDATE appointments
            SET actual_arrival = ?, late_minutes = ?
            WHERE appointment_id = ?
            """,
            (actual_arrival.isoformat(), late_minutes, appointment_id),
        )
        connection.commit()
    finally:
        connection.close()


def update_consultation(
    appointment_id: int,
    consult_start: datetime,
    consult_end: datetime,
    consult_duration_min: float,
    actual_wait_min: float,
    outcome: str,
    path: str = DEFAULT_DB_PATH,
) -> None:
    """Save consultation timing and outcome fields."""
    init_db(path)
    connection = get_connection(path)
    try:
        connection.execute(
            """
            UPDATE appointments
            SET consult_start = ?, consult_end = ?, consult_duration_min = ?,
                actual_wait_min = ?, outcome = ?
            WHERE appointment_id = ?
            """,
            (
                consult_start.isoformat(),
                consult_end.isoformat(),
                consult_duration_min,
                actual_wait_min,
                outcome,
                appointment_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def update_outcome(
    appointment_id: int, outcome: str, path: str = DEFAULT_DB_PATH
) -> None:
    """Update an appointment's outcome."""
    init_db(path)
    connection = get_connection(path)
    try:
        connection.execute(
            "UPDATE appointments SET outcome = ? WHERE appointment_id = ?",
            (outcome, appointment_id),
        )
        connection.commit()
    finally:
        connection.close()


def has_active_booking(
    patient_phone: str,
    appointment_date: date,
    path: str = DEFAULT_DB_PATH,
) -> bool:
    """Check whether a phone has a non-cancelled booking on a date."""
    init_db(path)
    connection = get_connection(path)
    try:
        row = connection.execute(
            """
            SELECT 1
            FROM appointments
            WHERE patient_phone = ? AND appointment_date = ? AND outcome != ?
            LIMIT 1
            """,
            (patient_phone, appointment_date.isoformat(), "cancelled"),
        ).fetchone()
    finally:
        connection.close()
    return row is not None
