"""Streamlit configuration-form tests with a mocked clinic API."""

import json
from datetime import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.schemas import ClinicConfig, ClinicConfigUpdate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


class FakeResponse:
    """Minimal httpx response used by Streamlit AppTest."""

    status_code = 200
    is_success = True
    text = ""

    def __init__(self, data: Any, status_code: int = 200) -> None:
        self.data = data
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self.content = json.dumps(data).encode() if status_code != 204 else b""

    def json(self) -> Any:
        return self.data


@pytest.fixture
def configuration_page(monkeypatch: pytest.MonkeyPatch):
    """Provide the form connected to an in-memory API substitute."""
    config = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    puts: list[dict[str, Any]] = []

    def request(
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **_: Any,
    ) -> FakeResponse:
        path = urlparse(url).path
        if path == "/clinic":
            public_fields = {
                "clinic_name",
                "specialty",
                "address",
                "phone",
                "available_days",
                "work_start",
                "work_end",
                "appointment_duration_min",
                "services",
                "acceptable_wait_min",
                "high_wait_threshold_min",
            }
            return FakeResponse(
                {field: config[field] for field in public_fields}
            )
        if path == "/admin/config" and method == "GET":
            return FakeResponse(config)
        if path == "/admin/config" and method == "PUT":
            assert json is not None
            validated = ClinicConfigUpdate.model_validate(json)
            config.clear()
            config.update(validated.model_dump(mode="json"))
            puts.append(dict(json))
            return FakeResponse(
                {"saved": True, "index_rebuilt": True, "chunk_count": 12}
            )
        if path == "/health":
            return FakeResponse({"status": "ok"})
        if path == "/staff/index-status":
            return FakeResponse(
                {
                    "chunk_count": 12,
                    "chroma_dir": "test-chroma",
                    "db_path": "test.sqlite",
                    "section_titles": [],
                    "vector_store_ready": True,
                    "admin_document_count": 0,
                    "llm_provider": "extractive",
                    "llm_loaded": False,
                }
            )
        if path == "/staff/knowledge":
            return FakeResponse([])
        return FakeResponse({"detail": "not found"}, status_code=404)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.radio[0].set_value(app.radio[0].options[2]).run(timeout=20)
    return app, config, puts


def _submit(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run(
        timeout=20
    )


def test_config_form_no_edit_save_and_working_hours_validation(
    configuration_page,
) -> None:
    app, original, puts = configuration_page
    expected = ClinicConfig.model_validate(original)

    assert next(item for item in app.text_input if item.label == "Clinic name").value == (
        expected.clinic_name
    )
    _submit(app, "Save Configuration")
    assert len(puts) == 1
    assert ClinicConfig.model_validate(puts[0]) == expected

    next(item for item in app.time_input if item.label == "Work starts").set_value(
        time(17, 0)
    )
    next(item for item in app.time_input if item.label == "Work ends").set_value(
        time(9, 0)
    )
    _submit(app, "Save Configuration")

    assert len(puts) == 1
    assert any("Working hours" in item.value for item in app.error)
    assert any("work_end must be after work_start" in item.value for item in app.error)

    next(item for item in app.time_input if item.label == "Work starts").set_value(
        expected.work_start
    )
    next(item for item in app.time_input if item.label == "Work ends").set_value(
        expected.work_end
    )
    next(item for item in app.time_input if item.label == "☕ Breaks 1 start").set_value(
        time(19, 0)
    )
    next(item for item in app.time_input if item.label == "☕ Breaks 1 end").set_value(
        time(18, 30)
    )
    _submit(app, "Save Configuration")
    assert len(puts) == 1
    assert any("Breaks:" in item.value for item in app.error)


def test_config_form_adds_edits_and_removes_dynamic_rows(configuration_page) -> None:
    app, saved, puts = configuration_page
    original_break_id = app.session_state["clinic_break_rows"][0]["id"]
    _submit(app, "Add a break")
    added_break = app.session_state["clinic_break_rows"][-1]
    next(item for item in app.time_input if item.label == "☕ Breaks 2 start").set_value(
        time(19, 0)
    )
    next(item for item in app.time_input if item.label == "☕ Breaks 2 end").set_value(
        time(19, 15)
    )
    _submit(app, "Save Configuration")
    assert len(puts) == 1
    assert saved["breaks"][1] == {"start": "19:00:00", "end": "19:15:00"}

    _submit(app, "Add peak hours")
    next(
        item for item in app.time_input if item.label == "📈 Peak Hours 2 start"
    ).set_value(time(16, 30))
    next(
        item for item in app.time_input if item.label == "📈 Peak Hours 2 end"
    ).set_value(time(17, 0))
    _submit(app, "Save Configuration")
    assert len(puts) == 2
    assert saved["peak_hours"][1] == {"start": "16:30:00", "end": "17:00:00"}

    _submit(app, "Add a service")
    next(item for item in app.text_input if item.label == "Service 4 name").set_value(
        "Extended consultation"
    )
    next(
        item
        for item in app.number_input
        if item.label == "Duration for service 4 (min)"
    ).set_value(30)
    next(item for item in app.number_input if item.label == "Price for service 4").set_value(
        42.5
    )
    _submit(app, "Save Configuration")
    assert len(puts) == 3
    assert saved["services"][-1] == {
        "name": "Extended consultation",
        "duration_min": 30,
        "price": 42.5,
    }

    service_id = app.session_state["clinic_service_rows"][-1]["id"]
    next(
        item for item in app.button if item.key == f"remove_service_{service_id}"
    ).click().run(timeout=20)
    assert len(app.session_state["clinic_service_rows"]) == 3
    assert len(app.session_state["clinic_break_rows"]) == 2
    assert len(app.session_state["clinic_peak_rows"]) == 2

    break_remove = next(
        item
        for item in app.button
        if item.key == f"remove_break_{original_break_id}"
    )
    break_remove.click().run(timeout=20)
    assert len(app.session_state["clinic_break_rows"]) == 1
    assert app.session_state["clinic_break_rows"][0]["id"] == added_break["id"]

    peak_id = app.session_state["clinic_peak_rows"][-1]["id"]
    next(
        item for item in app.button if item.key == f"remove_peak_{peak_id}"
    ).click().run(timeout=20)
    assert len(app.session_state["clinic_peak_rows"]) == 1
