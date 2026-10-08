"""FastAPI endpoints for patient booking and clinic staff workflows."""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import date, time
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.db import (
    backfill_default_clinic_config,
    delete_knowledge_base_item,
    get_appointment,
    get_appointment_data_summary,
    get_patient_phone,
    init_db,
    list_knowledge_base,
    list_appointments_by_date,
    load_clinic_config,
    save_knowledge_base_item,
    save_clinic_config,
)
from app.chroma_store import get_chroma_path
from app.config_defaults import DEFAULT_CLINIC_CONFIG
from app.rag import (
    answer_question,
    get_index_status,
    rebuild_index_from_db,
)
from app.scheduler import (
    ClinicClosedError,
    ConfigNotFoundError,
    DuplicateBookingError,
    FullyBookedError,
    book_appointment,
    cancel_appointment,
    classify_wait,
    generate_slots,
    mark_no_show,
    record_arrival,
    record_consultation,
)
from app.schemas import (
    AppointmentRecord,
    ArrivalEvent,
    BookingRequest,
    BookingResponse,
    ClinicConfig,
    ClinicConfigUpdate,
    ConsultationEvent,
    QuestionRequest,
    RAGAnswer,
    Service,
)


load_dotenv()

LOGGER = logging.getLogger(__name__)
MIN_ROWS_FOR_ML = 200
_BOOKING_LOCK = threading.Lock()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _db_path() -> str:
    """Return the configured SQLite path."""
    return os.getenv("DB_PATH", "data/clinic.sqlite")


def _chroma_dir() -> str:
    """Return the one absolute Chroma storage path used by the API."""
    return str(get_chroma_path())


def get_rag_deps() -> tuple[Any | None, Callable[[str, str], str] | None]:
    """Return default Chroma and LLM dependencies; tests can override these."""
    return None, None


def _saved_config() -> ClinicConfig:
    config = load_clinic_config(path=_db_path())
    if config is None:
        raise ConfigNotFoundError("No clinic configuration has been saved")
    return config


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize storage and backfill the single clinic from repo defaults."""
    path = _db_path()
    init_db(path)
    backfill_default_clinic_config(path=path)
    LOGGER.info("Using Chroma storage at %s", _chroma_dir())
    rag_dependency = app.dependency_overrides.get(get_rag_deps)
    embedding_function, _ = (
        rag_dependency() if rag_dependency is not None else get_rag_deps()
    )
    if os.getenv("LLM_PROVIDER", "extractive").strip().casefold() == "qwen":
        from app.qwen import get_model_runtime_info

        runtime = get_model_runtime_info()
        LOGGER.info(
            "Qwen runtime selected at startup: device=%s dtype=%s",
            runtime["device"],
            runtime["dtype"],
        )
    try:
        document_count, _ = get_index_status(
            persist_dir=_chroma_dir(),
            embedding_function=embedding_function,
            clinic_id=1,
        )
        if document_count == 0:
            rebuild_index_from_db(
                db_path=path,
                persist_dir=_chroma_dir(),
                embedding_function=embedding_function,
                clinic_id=1,
            )
    except Exception:
        LOGGER.exception(
            "Clinic %s defaults/config loaded, but startup RAG index check/rebuild failed",
            1,
        )
    yield


app = FastAPI(title="Clinic Booking API", lifespan=lifespan)


@app.exception_handler(ClinicClosedError)
@app.exception_handler(FullyBookedError)
@app.exception_handler(DuplicateBookingError)
async def booking_conflict_handler(_, exc: Exception) -> JSONResponse:
    """Convert scheduler booking conflicts into HTTP 409 responses."""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc)},
    )


@app.exception_handler(ConfigNotFoundError)
async def config_not_found_handler(_, exc: ConfigNotFoundError) -> JSONResponse:
    """Convert missing saved configuration into HTTP 503."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


class ClinicPublicInfo(BaseModel):
    """Publicly visible clinic details for patient booking."""

    clinic_name: str
    specialty: str
    address: str
    phone: str
    available_days: list[str]
    work_start: time
    work_end: time
    appointment_duration_min: int
    services: list[Service]
    acceptable_wait_min: int
    high_wait_threshold_min: int


class SlotsResponse(BaseModel):
    """Availability for a requested clinic date."""

    appointment_date: date
    open: bool
    slots: list[time]
    bookings_remaining: int


class BookingConfirmation(BookingResponse):
    """Booking recommendation with a human-readable wait category."""

    wait_category: Literal["acceptable", "moderate", "high"]
    message: str


class PatientCancellation(BaseModel):
    """Phone verification for patient-requested cancellation."""

    patient_phone: str


class CancellationResponse(BaseModel):
    """Result of a successful patient cancellation."""

    status: Literal["cancelled"] = "cancelled"


class StaffDataSummary(BaseModel):
    """Count of collected appointment data usable for later ML."""

    total_appointments: int
    usable_rows: int
    min_rows_for_ml: int
    ready: bool


class ConfigSaveResponse(BaseModel):
    """Configuration save and index rebuild status."""

    saved: bool
    index_rebuilt: bool
    chunk_count: int | None = None
    warning: str | None = None


class IndexStatusResponse(BaseModel):
    """Current Chroma index status and configured storage paths."""

    chunk_count: int
    chroma_dir: str
    db_path: str
    section_titles: list[str]
    vector_store_ready: bool
    admin_document_count: int
    llm_provider: str
    llm_loaded: bool


class AdminKnowledgeCreate(BaseModel):
    """One free-form entry submitted by clinic staff."""

    title: str = Field(default="Admin entry", min_length=1, max_length=200)
    content: str = Field(min_length=1)


class AdminKnowledgeRecord(BaseModel):
    """A persisted free-form knowledge entry."""

    id: str
    title: str
    content: str
    timestamp: str
    index_warning: str | None = None


class AdminKnowledgeDeleteResponse(BaseModel):
    """Result of deleting a knowledge entry and rebuilding its index."""

    deleted: bool
    index_rebuilt: bool
    warning: str | None = None


def _rebuild_saved_index(
    embedding_function: Any = None,
) -> tuple[int | None, str | None]:
    """Rebuild the saved clinic's index, logging complete failure details."""
    try:
        indexed_count = rebuild_index_from_db(
            db_path=_db_path(),
            persist_dir=_chroma_dir(),
            embedding_function=embedding_function,
            clinic_id=1,
        )
    except Exception as exc:
        LOGGER.exception("RAG index rebuild failed for clinic %s", 1)
        return None, (
            "The change was saved, but the clinic RAG index could not be "
            f"rebuilt ({type(exc).__name__}: {exc}). "
            "Run scripts/rebuild_chroma_indexes.py after resolving the "
            "Chroma storage issue."
        )
    return indexed_count, None


def require_staff_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Require the configured staff key without revealing whether it is valid."""
    expected_key = os.getenv("STAFF_API_KEY", "")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="staff key not configured",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing staff key",
        )


def _format_time(value: time) -> str:
    """Format a time for short patient-facing messages."""
    return value.strftime("%I:%M %p").lstrip("0")


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report API availability and the configured generation runtime."""
    provider = os.getenv("LLM_PROVIDER", "extractive").strip().casefold()
    health_status: dict[str, str | bool] = {
        "status": "ok",
        "llm_provider": provider,
        "llm_loaded": False,
        "llm_device": "not_applicable",
        "llm_dtype": "not_applicable",
    }
    if provider == "qwen":
        from app.qwen import get_model_runtime_info, is_model_loaded

        runtime = get_model_runtime_info()
        health_status.update(
            {
                "llm_loaded": is_model_loaded(),
                "llm_device": runtime["device"],
                "llm_dtype": runtime["dtype"],
            }
        )
    return health_status


@app.get("/clinic", response_model=ClinicPublicInfo)
def clinic_info() -> ClinicPublicInfo:
    """Return only the public clinic information needed for booking."""
    config = _saved_config()
    return ClinicPublicInfo(
        clinic_name=config.clinic_name,
        specialty=config.specialty,
        address=config.address,
        phone=config.phone,
        available_days=config.available_days,
        work_start=config.work_start,
        work_end=config.work_end,
        appointment_duration_min=config.appointment_duration_min,
        services=config.services,
        acceptable_wait_min=config.acceptable_wait_min,
        high_wait_threshold_min=config.high_wait_threshold_min,
    )


@app.get("/slots", response_model=SlotsResponse)
def slots_for_date(appointment_date: date = Query(alias="date")) -> SlotsResponse:
    """Return generated appointment times and remaining daily capacity."""
    config = _saved_config()
    appointments = list_appointments_by_date(appointment_date, path=_db_path())
    active_count = sum(item.outcome != "cancelled" for item in appointments)
    slots = generate_slots(config, appointment_date)
    return SlotsResponse(
        appointment_date=appointment_date,
        open=bool(slots),
        slots=slots,
        bookings_remaining=max(0, config.max_appointments_per_day - active_count),
    )


@app.post(
    "/bookings",
    response_model=BookingConfirmation,
    status_code=status.HTTP_201_CREATED,
)
def create_booking(req: BookingRequest) -> BookingConfirmation:
    """Book via the scheduler and add the configured wait classification."""
    config = _saved_config()
    try:
        # The lock protects the scheduler's check-then-insert in this single-process demo.
        with _BOOKING_LOCK:
            response = book_appointment(req, path=_db_path())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    wait_category = classify_wait(response.estimated_wait_min, config)
    message = (
        f"Please arrive at {_format_time(response.recommended_arrival_time)}. "
        f"Expected wait: about {response.estimated_wait_min} minutes."
    )
    return BookingConfirmation(
        **response.model_dump(),
        wait_category=wait_category,
        message=message,
    )


@app.post(
    "/bookings/{appointment_id}/cancel",
    response_model=CancellationResponse,
)
def cancel_patient_booking(
    appointment_id: int, req: PatientCancellation
) -> CancellationResponse:
    """Cancel only when the submitted phone matches the stored contact."""
    record = get_appointment(appointment_id, path=_db_path())
    stored_phone = get_patient_phone(appointment_id, path=_db_path())
    if record is None or stored_phone is None or not secrets.compare_digest(
        stored_phone, req.patient_phone
    ):
        raise HTTPException(status_code=404, detail="appointment not found")
    with _BOOKING_LOCK:
        cancel_appointment(appointment_id, path=_db_path())
    return CancellationResponse()


@app.post("/ask", response_model=RAGAnswer)
def ask_clinic(
    req: QuestionRequest,
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> RAGAnswer:
    """Answer a patient question using the existing RAG pipeline."""
    embedding_function, llm = rag_deps
    return answer_question(
        req,
        db_path=_db_path(),
        persist_dir=_chroma_dir(),
        embedding_function=embedding_function,
        llm=llm,
    )


@app.get(
    "/staff/appointments",
    response_model=list[AppointmentRecord],
    dependencies=[Depends(require_staff_key)],
)
def staff_appointments(
    appointment_date: date = Query(alias="date"),
) -> list[AppointmentRecord]:
    """List a day's appointments in queue order."""
    return list_appointments_by_date(appointment_date, path=_db_path())


@app.get(
    "/staff/appointments/{appointment_id}",
    response_model=AppointmentRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_appointment(appointment_id: int) -> AppointmentRecord:
    """Return a single appointment for staff."""
    record = get_appointment(appointment_id, path=_db_path())
    if record is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    return record


@app.post(
    "/staff/arrival",
    response_model=AppointmentRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_arrival(event: ArrivalEvent) -> AppointmentRecord:
    """Record an arrival after explicitly checking that its appointment exists."""
    if get_appointment(event.appointment_id, path=_db_path()) is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    try:
        return record_arrival(event, path=_db_path())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/staff/consultation",
    response_model=AppointmentRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_consultation(event: ConsultationEvent) -> AppointmentRecord:
    """Record consultation timing after explicitly checking appointment ID."""
    if get_appointment(event.appointment_id, path=_db_path()) is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    try:
        return record_consultation(event, path=_db_path())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/staff/appointments/{appointment_id}/no-show",
    response_model=AppointmentRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_no_show(appointment_id: int) -> AppointmentRecord:
    """Mark an existing appointment as a no-show."""
    if get_appointment(appointment_id, path=_db_path()) is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    mark_no_show(appointment_id, path=_db_path())
    updated = get_appointment(appointment_id, path=_db_path())
    if updated is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    return updated


@app.post(
    "/staff/appointments/{appointment_id}/cancel",
    response_model=AppointmentRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_cancel(appointment_id: int) -> AppointmentRecord:
    """Cancel an existing appointment as staff."""
    if get_appointment(appointment_id, path=_db_path()) is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    cancel_appointment(appointment_id, path=_db_path())
    updated = get_appointment(appointment_id, path=_db_path())
    if updated is None:
        raise HTTPException(status_code=404, detail="appointment not found")
    return updated


@app.get(
    "/staff/data-summary",
    response_model=StaffDataSummary,
    dependencies=[Depends(require_staff_key)],
)
def staff_data_summary() -> StaffDataSummary:
    """Report total bookings and completed rows with observed wait times."""
    total, usable = get_appointment_data_summary(path=_db_path())
    return StaffDataSummary(
        total_appointments=total,
        usable_rows=usable,
        min_rows_for_ml=MIN_ROWS_FOR_ML,
        ready=usable >= MIN_ROWS_FOR_ML,
    )


@app.get(
    "/staff/index-status",
    response_model=IndexStatusResponse,
    dependencies=[Depends(require_staff_key)],
)
def staff_index_status(
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> IndexStatusResponse:
    """Return the current indexed content and the configured storage paths."""
    embedding_function, _ = rag_deps
    chunk_count, sections = get_index_status(
        persist_dir=_chroma_dir(),
        embedding_function=embedding_function,
        clinic_id=1,
    )
    admin_entries = list_knowledge_base(path=_db_path())
    provider = os.getenv("LLM_PROVIDER", "extractive").strip().casefold()
    model_loaded = False
    if provider == "qwen":
        try:
            from app.qwen import is_model_loaded

            model_loaded = is_model_loaded()
        except ImportError:
            model_loaded = False
    return IndexStatusResponse(
        chunk_count=chunk_count,
        chroma_dir=_chroma_dir(),
        db_path=_db_path(),
        section_titles=sections,
        vector_store_ready=chunk_count > 0,
        admin_document_count=len(admin_entries),
        llm_provider=provider,
        llm_loaded=model_loaded,
    )


@app.get(
    "/staff/knowledge",
    response_model=list[AdminKnowledgeRecord],
    dependencies=[Depends(require_staff_key)],
)
def staff_list_knowledge(
) -> list[AdminKnowledgeRecord]:
    """List admin-added knowledge persisted for the clinic."""
    return [
        AdminKnowledgeRecord(**entry)
        for entry in list_knowledge_base(path=_db_path())
    ]


@app.post(
    "/staff/knowledge",
    response_model=AdminKnowledgeRecord,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_staff_key)],
)
def staff_add_knowledge(
    entry: AdminKnowledgeCreate,
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> AdminKnowledgeRecord:
    """Persist an admin entry, then rebuild the shared clinic index."""
    if not entry.content.strip() or not entry.title.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Knowledge title and content must not be blank.",
        )
    embedding_function, _ = rag_deps
    stored = save_knowledge_base_item(
        title=entry.title.strip(),
        content=entry.content.strip(),
        path=_db_path(),
    )
    _, warning = _rebuild_saved_index(embedding_function)
    return AdminKnowledgeRecord(**stored, index_warning=warning)


@app.put(
    "/staff/knowledge/{document_id}",
    response_model=AdminKnowledgeRecord,
    dependencies=[Depends(require_staff_key)],
)
def staff_edit_knowledge(
    document_id: str,
    entry: AdminKnowledgeCreate,
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> AdminKnowledgeRecord:
    """Update an admin entry and refresh its indexed chunks."""
    if not entry.content.strip() or not entry.title.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Knowledge title and content must not be blank.",
        )
    if not any(
        item["id"] == document_id for item in list_knowledge_base(path=_db_path())
    ):
        raise HTTPException(status_code=404, detail="knowledge entry not found")
    stored = save_knowledge_base_item(
        title=entry.title.strip(),
        content=entry.content.strip(),
        path=_db_path(),
        item_id=document_id,
    )
    embedding_function, _ = rag_deps
    _, warning = _rebuild_saved_index(embedding_function)
    return AdminKnowledgeRecord(**stored, index_warning=warning)


@app.delete(
    "/staff/knowledge/{document_id}",
    response_model=AdminKnowledgeDeleteResponse,
    dependencies=[Depends(require_staff_key)],
)
def staff_delete_knowledge(
    document_id: str,
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> AdminKnowledgeDeleteResponse:
    """Remove one admin entry from persistence and refresh the index."""
    embedding_function, _ = rag_deps
    deleted = delete_knowledge_base_item(document_id, path=_db_path())
    if not deleted:
        raise HTTPException(status_code=404, detail="knowledge entry not found")
    indexed_count, warning = _rebuild_saved_index(embedding_function)
    return AdminKnowledgeDeleteResponse(
        deleted=True,
        index_rebuilt=indexed_count is not None,
        warning=warning,
    )


@app.get(
    "/admin/config",
    response_model=ClinicConfig,
    dependencies=[Depends(require_staff_key)],
)
def admin_get_config() -> ClinicConfig:
    """Return the saved configuration to authorized staff."""
    return _saved_config()


@app.put(
    "/admin/config",
    response_model=ConfigSaveResponse,
    dependencies=[Depends(require_staff_key)],
)
def admin_update_config(
    config: ClinicConfigUpdate,
    rag_deps: tuple[Any | None, Callable[[str, str], str] | None] = Depends(
        get_rag_deps
    ),
) -> ConfigSaveResponse:
    """Save a validated configuration, then report the index rebuild result."""
    embedding_function, _ = rag_deps
    existing = load_clinic_config(path=_db_path())
    allowed_fields = ClinicConfigUpdate.model_fields
    existing_data = (
        {
            key: value
            for key, value in existing.model_dump(mode="json").items()
            if key in allowed_fields
        }
        if existing
        else {}
    )
    merged = {
        **DEFAULT_CLINIC_CONFIG,
        **existing_data,
        **config.model_dump(mode="json"),
    }
    validated = ClinicConfigUpdate.model_validate(merged)
    save_clinic_config(validated, path=_db_path())
    chunk_count, warning = _rebuild_saved_index(embedding_function)
    if warning:
        return ConfigSaveResponse(
            saved=True,
            index_rebuilt=False,
            chunk_count=None,
            warning=warning,
        )
    return ConfigSaveResponse(
        saved=True,
        index_rebuilt=True,
        chunk_count=chunk_count,
    )
