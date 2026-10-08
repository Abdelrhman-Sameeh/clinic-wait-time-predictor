"""Offline API tests using isolated SQLite and Chroma paths."""

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.api import app, get_rag_deps
from app.db import (
    get_appointment,
    load_clinic_config,
    save_clinic_config,
)
from app.rag import build_index
from app.schemas import ClinicConfig
from app.scheduler import day_abbrev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"
TEST_STAFF_KEY = "test-only-staff-key"


class TestEmbedding:
    """Small deterministic embedding function that never downloads a model."""

    def name(self) -> str:
        """Return an identifier accepted by Chroma."""
        return "api-test-hashed-bag-of-words"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vector = [0.0] * 1024
            for word in re.findall(r"[a-z0-9]+", text.casefold()):
                index = int.from_bytes(
                    hashlib.blake2b(word.encode(), digest_size=4).digest(),
                    "big",
                ) % len(vector)
                vector[index] += 1.0
            vectors.append(vector)
        return vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed a query with the same deterministic representation."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed documents with the same deterministic representation."""
        return self(input)


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an app backed by per-test SQLite and Chroma data."""
    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    db_path = str(tmp_path / "clinic.sqlite")
    chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("CHROMA_DIR", chroma_dir)
    monkeypatch.setenv("STAFF_API_KEY", TEST_STAFF_KEY)
    monkeypatch.setenv("SEED_SAMPLE_CONFIG", "false")
    embedding = TestEmbedding()
    save_clinic_config(config, path=db_path)
    build_index(config, persist_dir=chroma_dir, embedding_function=embedding)

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        context = json.loads(user_prompt)["context"]
        return json.dumps(
            {
                "answer": context[0]["text"],
                "source_sections": [context[0]["section"]],
                "found_in_clinic_info": True,
            }
        )

    app.dependency_overrides[get_rag_deps] = lambda: (embedding, fake_llm)
    with TestClient(app) as client:
        yield client, config, db_path
    app.dependency_overrides.clear()


def next_open_day(config: ClinicConfig) -> date:
    """Find the next future configured open date."""
    candidate = date.today() + timedelta(days=1)
    while day_abbrev(candidate) not in config.available_days:
        candidate += timedelta(days=1)
    return candidate


def next_closed_day(config: ClinicConfig) -> date:
    """Find the next future date when the clinic is closed."""
    candidate = date.today() + timedelta(days=1)
    while day_abbrev(candidate) in config.available_days:
        candidate += timedelta(days=1)
    return candidate


def booking_payload(appointment_date: date, **updates: Any) -> dict[str, Any]:
    """Return a valid patient booking payload with optional changes."""
    payload: dict[str, Any] = {
        "patient_name": "API Test Patient",
        "patient_phone": "+962790000100",
        "appointment_date": appointment_date.isoformat(),
    }
    payload.update(updates)
    return payload


def staff_headers() -> dict[str, str]:
    """Return the test staff authorization header."""
    return {"X-API-Key": TEST_STAFF_KEY}


def test_health_and_public_clinic_fields(api_client) -> None:
    client, config, _ = api_client

    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/clinic")

    assert response.status_code == 200
    payload = response.json()
    assert payload["clinic_name"] == config.clinic_name
    assert payload["phone"] == config.phone
    assert payload["appointment_duration_min"] == config.appointment_duration_min
    assert payload["services"][0]["name"] == config.services[0].name
    assert "late_arrival_policy" not in payload
    assert "noshow_policy" not in payload


def test_clinic_returns_503_when_config_is_missing(api_client) -> None:
    client, _, db_path = api_client
    from app.db import get_connection

    connection = get_connection(db_path)
    try:
        connection.execute("DELETE FROM clinic_config")
        connection.commit()
    finally:
        connection.close()

    assert client.get("/clinic").status_code == 503


def test_slots_report_open_state_and_remaining_capacity(api_client) -> None:
    client, config, _ = api_client
    open_day = next_open_day(config)
    closed_day = next_closed_day(config)

    open_response = client.get("/slots", params={"date": open_day.isoformat()})
    closed_response = client.get("/slots", params={"date": closed_day.isoformat()})

    assert open_response.status_code == 200
    assert open_response.json()["open"]
    assert open_response.json()["slots"]
    assert open_response.json()["bookings_remaining"] == config.max_appointments_per_day
    assert closed_response.json()["open"] is False
    assert closed_response.json()["slots"] == []


def test_booking_returns_confirmation_and_increments_queue(api_client) -> None:
    client, config, _ = api_client
    appointment_date = next_open_day(config)
    first = client.post("/bookings", json=booking_payload(appointment_date))
    second = client.post(
        "/bookings",
        json=booking_payload(
            appointment_date, patient_phone="+962790000101"
        ),
    )

    assert first.status_code == 201
    assert first.json()["queue_number"] == 1
    assert first.json()["patients_before"] == 0
    assert first.json()["method"] == "rules"
    assert first.json()["wait_category"] in {"acceptable", "moderate", "high"}
    assert "Please arrive at" in first.json()["message"]
    assert second.json()["queue_number"] == 2


def test_booking_validation_conflicts_and_service_errors(api_client) -> None:
    client, config, _ = api_client
    open_day = next_open_day(config)
    closed_day = next_closed_day(config)

    assert client.post(
        "/bookings", json=booking_payload(closed_day)
    ).status_code == 409
    assert client.post(
        "/bookings", json=booking_payload(date.today())
    ).status_code == 422
    assert client.post(
        "/bookings", json=booking_payload(open_day, service="Unknown service")
    ).status_code == 422
    assert client.post(
        "/bookings", json=booking_payload(open_day)
    ).status_code == 201
    duplicate = client.post(
        "/bookings", json=booking_payload(open_day)
    )
    assert duplicate.status_code == 409


def test_fully_booked_day_returns_conflict(api_client) -> None:
    client, config, db_path = api_client
    limited_config = config.model_copy(update={"max_appointments_per_day": 1})
    save_clinic_config(limited_config, path=db_path)
    appointment_date = next_open_day(config)
    assert client.post(
        "/bookings", json=booking_payload(appointment_date)
    ).status_code == 201
    assert client.post(
        "/bookings",
        json=booking_payload(
            appointment_date, patient_phone="+962790000102"
        ),
    ).status_code == 409


def test_patient_cancel_requires_matching_phone_and_frees_capacity(api_client) -> None:
    client, config, _ = api_client
    appointment_date = next_open_day(config)
    booking = client.post(
        "/bookings", json=booking_payload(appointment_date)
    ).json()
    endpoint = f"/bookings/{booking['appointment_id']}/cancel"

    wrong_phone = client.post(
        endpoint, json={"patient_phone": "+962790000999"}
    )
    assert wrong_phone.status_code == 404
    record = get_appointment(booking["appointment_id"], path=api_client[2])
    assert record is not None and record.outcome == "pending"

    cancelled = client.post(
        endpoint, json={"patient_phone": "+962790000100"}
    )
    assert cancelled.status_code == 200
    assert cancelled.json() == {"status": "cancelled"}
    rebooked = client.post(
        "/bookings", json=booking_payload(appointment_date)
    )
    assert rebooked.status_code == 201


def test_ask_uses_fake_llm_and_short_question_is_rejected(api_client) -> None:
    client, _, _ = api_client

    answer = client.post("/ask", json={"question": "When is the clinic open?"})

    assert answer.status_code == 200
    assert answer.json()["found_in_clinic_info"]
    assert answer.json()["source_sections"]
    assert answer.json()["source_sections"][0] in {
        "Clinic Information",
        "Working Days and Hours",
        "Breaks",
        "Appointments and Booking Rules",
        "Peak Hours and Waiting Times",
        "Late Arrival Policy",
        "Cancellation Policy",
        "Walk-in Policy",
        "Emergency Policy",
        "No-show Policy",
        "Services and Prices",
        "Special Conditions",
    }
    assert client.post("/ask", json={"question": "Hi"}).status_code == 422


def test_staff_endpoints_require_a_configured_matching_key(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, config, _ = api_client
    params = {"date": next_open_day(config).isoformat()}

    assert client.get("/staff/appointments", params=params).status_code == 401
    assert client.get(
        "/staff/appointments",
        params=params,
        headers={"X-API-Key": "wrong"},
    ).status_code == 401
    monkeypatch.setenv("STAFF_API_KEY", "")
    assert client.get(
        "/staff/appointments",
        params=params,
        headers=staff_headers(),
    ).status_code == 503


def test_staff_arrival_consultation_and_summary_flow(api_client) -> None:
    client, config, _ = api_client
    key_headers = staff_headers()
    appointment_date = next_open_day(config)
    booking = client.post(
        "/bookings", json=booking_payload(appointment_date)
    ).json()
    appointment_id = booking["appointment_id"]
    record = get_appointment(appointment_id, path=api_client[2])
    assert record is not None
    scheduled = datetime.combine(appointment_date, record.scheduled_time)
    start = scheduled + timedelta(minutes=12)
    end = start + timedelta(minutes=20)
    event_body = {
        "appointment_id": appointment_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    early_consultation = client.post(
        "/staff/consultation", json=event_body, headers=key_headers
    )
    assert early_consultation.status_code == 409
    assert client.post(
        "/staff/arrival",
        json={
            "appointment_id": 99999,
            "actual_arrival": scheduled.isoformat(),
        },
        headers=key_headers,
    ).status_code == 404
    assert client.post(
        "/staff/consultation",
        json={
            "appointment_id": appointment_id,
            "start": end.isoformat(),
            "end": start.isoformat(),
        },
        headers=key_headers,
    ).status_code == 422

    arrival = client.post(
        "/staff/arrival",
        json={
            "appointment_id": appointment_id,
            "actual_arrival": (scheduled + timedelta(minutes=7)).isoformat(),
        },
        headers=key_headers,
    )
    assert arrival.status_code == 200
    completed = client.post(
        "/staff/consultation", json=event_body, headers=key_headers
    )
    assert completed.status_code == 200
    assert completed.json()["late_minutes"] == 7
    assert completed.json()["consult_duration_min"] == 20
    assert completed.json()["actual_wait_min"] == 5
    assert completed.json()["outcome"] == "completed"

    second_booking = client.post(
        "/bookings",
        json=booking_payload(
            appointment_date, patient_phone="+962790000104"
        ),
    ).json()
    client.post(
        f"/staff/appointments/{second_booking['appointment_id']}/no-show",
        headers=key_headers,
    )
    summary = client.get("/staff/data-summary", headers=key_headers).json()
    assert summary["total_appointments"] == 2
    assert summary["usable_rows"] == 1
    assert summary["min_rows_for_ml"] == 200
    assert summary["ready"] is False


def test_staff_no_show_and_cancel_return_updated_records(api_client) -> None:
    client, config, _ = api_client
    appointment_date = next_open_day(config)
    first = client.post(
        "/bookings", json=booking_payload(appointment_date)
    ).json()
    second = client.post(
        "/bookings",
        json=booking_payload(
            appointment_date, patient_phone="+962790000103"
        ),
    ).json()
    key_headers = staff_headers()

    no_show = client.post(
        f"/staff/appointments/{first['appointment_id']}/no-show",
        headers=key_headers,
    )
    cancelled = client.post(
        f"/staff/appointments/{second['appointment_id']}/cancel",
        headers=key_headers,
    )

    assert no_show.status_code == 200
    assert no_show.json()["outcome"] == "no_show"
    assert cancelled.status_code == 200
    assert cancelled.json()["outcome"] == "cancelled"
    assert client.get(
        "/staff/appointments/99999", headers=key_headers
    ).status_code == 404


def test_admin_config_validation_and_index_rebuild(api_client) -> None:
    client, config, db_path = api_client
    headers = staff_headers()
    invalid_config = config.model_dump(mode="json")
    invalid_config["work_end"] = "15:00:00"

    rejected = client.put(
        "/admin/config", json=invalid_config, headers=headers
    )
    assert rejected.status_code == 422
    assert load_clinic_config(path=db_path) == config

    changed_config = config.model_copy(
        update={"cancellation_policy": "Please cancel 6 hours ahead."}
    )
    saved = client.put(
        "/admin/config",
        json=changed_config.model_dump(mode="json"),
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json() == {
        "saved": True,
        "index_rebuilt": True,
        "chunk_count": client.get(
            "/staff/index-status", headers=headers
        ).json()["chunk_count"],
        "warning": None,
    }
    assert load_clinic_config(path=db_path) == changed_config


def test_admin_config_put_is_returned_by_get_and_sqlite(api_client) -> None:
    client, config, db_path = api_client
    changed_config = config.model_copy(
        update={"acceptable_wait_min": config.acceptable_wait_min + 5}
    )

    saved = client.put(
        "/admin/config",
        json=changed_config.model_dump(mode="json"),
        headers=staff_headers(),
    )
    fetched = client.get("/admin/config", headers=staff_headers())

    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert fetched.status_code == 200
    assert fetched.json() == changed_config.model_dump(mode="json")
    assert load_clinic_config(path=db_path) == changed_config


def test_admin_config_no_edit_round_trip_and_duration_update(api_client) -> None:
    client, config, db_path = api_client
    headers = staff_headers()

    unchanged = client.put(
        "/admin/config",
        json=config.model_dump(mode="json"),
        headers=headers,
    )
    assert unchanged.status_code == 200
    assert client.get("/admin/config", headers=headers).json() == (
        config.model_dump(mode="json")
    )
    assert load_clinic_config(path=db_path) == config

    changed_payload = config.model_dump(mode="json")
    changed_payload["appointment_duration_min"] += 5
    changed = client.put(
        "/admin/config",
        json=changed_payload,
        headers=headers,
    )
    public_config = client.get("/clinic")
    saved_config = client.get("/admin/config", headers=headers)

    assert changed.status_code == 200
    assert public_config.json()["appointment_duration_min"] == (
        config.appointment_duration_min + 5
    )
    assert saved_config.json()["appointment_duration_min"] == (
        config.appointment_duration_min + 5
    )


def test_admin_config_updates_services_breaks_and_peak_hours(api_client) -> None:
    client, config, db_path = api_client
    payload = config.model_dump(mode="json")
    payload["breaks"] = [{"start": "19:00:00", "end": "19:10:00"}]
    payload["peak_hours"].append({"start": "16:00:00", "end": "17:00:00"})
    payload["services"].append(
        {"name": "Extended consultation", "duration_min": 30, "price": 42.5}
    )

    response = client.put(
        "/admin/config",
        json=payload,
        headers=staff_headers(),
    )
    saved = client.get("/admin/config", headers=staff_headers())
    parsed = ClinicConfig.model_validate(saved.json())

    assert response.status_code == 200
    assert saved.status_code == 200
    assert parsed.breaks[0].start.isoformat() == "19:00:00"
    assert parsed.peak_hours[-1].end.isoformat() == "17:00:00"
    assert parsed.services[-1].name == "Extended consultation"
    assert parsed.services[-1].duration_min == 30
    assert parsed.services[-1].price == 42.5
    assert load_clinic_config(path=db_path) == parsed


def test_admin_knowledge_endpoints_ingest_retrieve_and_delete(api_client) -> None:
    client, _, _ = api_client
    headers = staff_headers()
    content = "Dr. Sarah is available every Thursday from 4 PM to 8 PM."

    added = client.post(
        "/staff/knowledge",
        json={"content": content},
        headers=headers,
    )
    assert added.status_code == 201
    entry = added.json()
    entries = client.get("/staff/knowledge", headers=headers)
    answer = client.post(
        "/ask",
        json={"question": "When is Dr. Sarah available?"},
    )
    status_response = client.get("/staff/index-status", headers=headers)

    assert entries.status_code == 200
    assert entries.json() == [entry]
    assert answer.status_code == 200
    assert "Thursday from 4 PM to 8 PM" in answer.json()["answer"]
    assert answer.json()["found_in_clinic_info"]
    assert answer.json()["source_sections"] == ["Admin Knowledge"]
    assert status_response.json()["admin_document_count"] == 1
    assert client.get("/staff/knowledge").status_code == 401

    deleted = client.delete(
        f"/staff/knowledge/{entry['id']}",
        headers=headers,
    )
    assert deleted.status_code == 204
    assert client.get("/staff/knowledge", headers=headers).json() == []


def test_admin_config_put_normalizes_omitted_default_fields(api_client) -> None:
    client, config, db_path = api_client
    payload = config.model_dump(mode="json")
    del payload["breaks"]
    del payload["buffer_min"]

    saved = client.put(
        "/admin/config",
        json=payload,
        headers=staff_headers(),
    )
    fetched = client.get("/admin/config", headers=staff_headers())
    submitted_config = ClinicConfig.model_validate(payload)
    expected = submitted_config.model_dump(mode="json")

    assert saved.status_code == 200
    assert fetched.status_code == 200
    assert fetched.json() == expected
    assert fetched.json() != payload
    assert load_clinic_config(path=db_path) == submitted_config


def test_admin_config_rejects_unknown_fields(api_client) -> None:
    client, config, db_path = api_client
    payload = config.model_dump(mode="json")
    payload["QOS"] = "Perfect"

    rejected = client.put(
        "/admin/config",
        json=payload,
        headers=staff_headers(),
    )

    assert rejected.status_code == 422
    assert any(
        error["loc"][-1] == "QOS"
        for error in rejected.json()["detail"]
    )
    assert load_clinic_config(path=db_path) == config


def test_updated_policy_is_retrievable_after_admin_put(api_client) -> None:
    client, config, db_path = api_client
    distinctive_policy = (
        "Please cancel at least 4 hours ahead. Mention the Starling marker."
    )
    payload = config.model_dump(mode="json")
    payload["cancellation_policy"] = distinctive_policy

    saved = client.put(
        "/admin/config",
        json=payload,
        headers=staff_headers(),
    )
    answer = client.post(
        "/ask",
        json={"question": "What does the Starling cancellation policy say?"},
    )

    assert saved.status_code == 200
    assert saved.json()["index_rebuilt"] is True
    assert isinstance(saved.json()["chunk_count"], int)
    assert answer.status_code == 200
    assert "Starling" in answer.json()["answer"]
    assert "Cancellation Policy" in answer.json()["source_sections"]
    assert load_clinic_config(path=db_path).cancellation_policy == distinctive_policy


def test_staff_index_status_reports_current_index_and_paths(api_client) -> None:
    client, _, db_path = api_client

    response = client.get(
        "/staff/index-status",
        headers=staff_headers(),
    )

    assert response.status_code == 200
    status = response.json()
    assert status["chunk_count"] > 0
    assert status["db_path"] == db_path
    assert status["chroma_dir"] == str(Path(db_path).parent / "chroma")
    assert "Cancellation Policy" in status["section_titles"]


def test_staff_index_status_requires_staff_key(api_client) -> None:
    client, _, _ = api_client

    response = client.get("/staff/index-status")

    assert response.status_code == 401


def test_admin_keeps_saved_config_when_index_rebuild_fails(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, config, db_path = api_client

    def failed_rebuild(**_: Any) -> int:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(api_module, "rebuild_index_from_db", failed_rebuild)
    changed_config = config.model_copy(
        update={"cancellation_policy": "Please call the clinic to cancel."}
    )
    response = client.put(
        "/admin/config",
        json=changed_config.model_dump(mode="json"),
        headers=staff_headers(),
    )

    assert response.status_code == 200
    assert response.json()["saved"] is True
    assert response.json()["index_rebuilt"] is False
    assert response.json()["chunk_count"] is None
    assert response.json()["warning"]
    assert load_clinic_config(path=db_path) == changed_config


def test_startup_seeds_sample_config_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = str(tmp_path / "seeded.sqlite")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "seed-chroma"))
    monkeypatch.setenv("SEED_SAMPLE_CONFIG", "true")
    monkeypatch.setenv("STAFF_API_KEY", TEST_STAFF_KEY)
    rebuild_calls: list[tuple[str, str]] = []

    def fake_rebuild(db_path: str, persist_dir: str) -> int:
        rebuild_calls.append((db_path, persist_dir))
        return 12

    monkeypatch.setattr(api_module, "rebuild_index_from_db", fake_rebuild)
    with TestClient(app) as client:
        response = client.get("/clinic")

    assert response.status_code == 200
    assert response.json()["clinic_name"] == "Al Noor Family Clinic"
    assert rebuild_calls == [(db_path, str(tmp_path / "seed-chroma"))]


def test_admin_get_config_requires_staff_key(api_client) -> None:
    client, config, _ = api_client
    response = client.get("/admin/config", headers=staff_headers())
    assert response.status_code == 200
    assert response.json()["clinic_name"] == config.clinic_name
