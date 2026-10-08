"""One-time migration and rebuild of all configured clinic Chroma indexes."""

import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.api import _chroma_dir, _db_path
from app.db import backfill_default_clinic_config, init_db
from app.rag import rebuild_index_from_db


def main() -> None:
    """Backfill the current clinic, migrate old KB items, and rebuild its index."""
    database_path = _db_path()
    init_db(database_path)
    backfill_default_clinic_config(path=database_path)
    with sqlite3.connect(database_path) as connection:
        clinic_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM clinic_config ORDER BY id"
            ).fetchall()
        ]

    for clinic_id in clinic_ids:
        indexed_count = rebuild_index_from_db(
            db_path=database_path,
            persist_dir=_chroma_dir(),
            clinic_id=clinic_id,
        )
        print(
            f"Rebuilt clinic {clinic_id}: {indexed_count} documents "
            f"at {_chroma_dir()}"
        )


if __name__ == "__main__":
    main()
