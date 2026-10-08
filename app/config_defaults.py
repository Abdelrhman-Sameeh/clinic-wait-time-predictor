"""Clinic defaults sourced directly from the repository's sample config."""

import json
from pathlib import Path
from typing import Any

from app.schemas import ClinicConfig


SAMPLE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_clinic_config.json"
DEFAULT_CLINIC_CONFIG: dict[str, Any] = ClinicConfig.model_validate(
    json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
).model_dump(mode="json")
