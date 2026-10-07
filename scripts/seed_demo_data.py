"""Create an isolated demo SQLite database and sample next-day bookings."""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_connection, init_db, save_clinic_config
from app.rag import build_index
from app.schemas import BookingRequest, ClinicConfig
from app.scheduler import book_appointment, day_abbrev, generate_slots


LOGGER = logging.getLogger(__name__)
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"


def seed_demo_data(database_path: str) -> tuple[ClinicConfig, date, int]:
    """Replace appointments in a safe demo DB and create eight sample bookings."""
    resolved_path = Path(database_path).resolve()
    protected_path = (PROJECT_ROOT / "data" / "clinic.sqlite").resolve()
    if resolved_path == protected_path:
        raise ValueError("Refusing to seed data/clinic.sqlite; use data/demo.sqlite")

    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    init_db(str(resolved_path))
    connection = get_connection(str(resolved_path))
    try:
        connection.execute("DELETE FROM appointments")
        connection.commit()
    finally:
        connection.close()
    save_clinic_config(config, path=str(resolved_path))

    appointment_date = date.today() + timedelta(days=1)
    while day_abbrev(appointment_date) not in config.available_days:
        appointment_date += timedelta(days=1)
    available_slots = generate_slots(config, appointment_date)
    if len(available_slots) < 8:
        raise RuntimeError("The next open day does not have eight available slots")

    for index in range(8):
        service = config.services[index % len(config.services)] if config.services else None
        book_appointment(
            BookingRequest(
                patient_name=f"Demo Patient {index + 1}",
                patient_phone=f"+962790001{index:04d}",
                appointment_date=appointment_date,
                service=service.name if service else None,
            ),
            path=str(resolved_path),
        )
    return config, appointment_date, 8


def main() -> None:
    """Seed the demo database selected by argument, environment, or default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database_path",
        nargs="?",
        help="SQLite path (default: DB_PATH or data/demo.sqlite)",
    )
    args = parser.parse_args()
    database_path = (
        args.database_path
        or os.getenv("DB_PATH")
        or str(PROJECT_ROOT / "data" / "demo.sqlite")
    )
    config, appointment_date, booking_count = seed_demo_data(database_path)
    try:
        build_index(config, persist_dir=os.getenv("CHROMA_DIR", "chroma_db"))
        index_note = "RAG index rebuilt."
    except Exception as exc:
        LOGGER.warning("Demo data seeded, but RAG index build failed: %s", exc)
        index_note = "RAG index was not built; API can still start."

    print(
        f"Seeded {booking_count} demo bookings for {appointment_date.isoformat()} "
        f"at {config.clinic_name} in {Path(database_path).resolve()}."
    )
    print(index_note)
    print("Start the API:  .\\.venv\\Scripts\\python.exe -m uvicorn app.api:app --reload")
    print("Start the UI:   .\\.venv\\Scripts\\python.exe -m streamlit run ui\\streamlit_app.py")
    print("If using a custom DB, set $env:DB_PATH to the same path in the API shell.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
