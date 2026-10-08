"""Simple Streamlit patient, staff, and administrator interface."""

import os
from datetime import date, datetime, time, timedelta
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas import BreakTime, ClinicConfig, ClinicConfigUpdate, Service
from ui.styles import get_theme_css


load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HTTP_TIMEOUT = 10.0


def api_request(
    method: str,
    endpoint: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Call the API and display its errors without exposing request secrets."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.request(
            method,
            f"{API_URL}{endpoint}",
            headers=headers,
            json=payload,
            params=params,
            timeout=HTTP_TIMEOUT,
        )
    except httpx.RequestError:
        st.error(
            "The clinic API is unreachable. Start the API server and try again."
        )
        return None
    if not response.is_success:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if response.status_code == 409:
            st.error(f"Booking conflict: {detail}")
        elif response.status_code == 422:
            st.error(f"Please check the submitted information: {detail}")
        else:
            st.error(str(detail))
        return None
    if not response.content:
        return {}
    return response.json()


def _staff_headers() -> str:
    """Return the current in-memory staff key from the password widget."""
    return st.session_state.get("staff_api_key", "")


def _assistant_view() -> None:
    """Render patient booking and the conversational clinic assistant."""
    clinic = api_request("GET", "/clinic")
    if not isinstance(clinic, dict):
        return

    booking_tab, question_tab = st.tabs(["Book a visit", "Ask the clinic"])
    with booking_tab:
        st.subheader(clinic["clinic_name"])
        st.write(f"**Specialty:** {clinic['specialty']}")
        st.write(
            f"**Working days:** {', '.join(clinic['available_days'])}  \n"
            f"**Hours:** {clinic['work_start'][:5]}–{clinic['work_end'][:5]}"
        )
        st.write("**Services and prices**")
        for service in clinic["services"]:
            st.write(
                f"- {service['name']}: {service['duration_min']} min, "
                f"{service['price']:g}"
            )

        tomorrow = date.today() + timedelta(days=1)
        with st.form("patient_booking_form"):
            patient_name = st.text_input("Name")
            patient_phone = st.text_input("Phone")
            appointment_date = st.date_input(
                "Visit date", value=tomorrow, min_value=tomorrow
            )
            service_names = [""] + [
                service["name"] for service in clinic["services"]
            ]
            service = st.selectbox("Service (optional)", service_names)
            submitted = st.form_submit_button("Book a visit")
        if submitted:
            result = api_request(
                "POST",
                "/bookings",
                payload={
                    "patient_name": patient_name,
                    "patient_phone": patient_phone,
                    "appointment_date": appointment_date.isoformat(),
                    "service": service or None,
                },
            )
            if isinstance(result, dict):
                st.success("Your visit is booked.")
                st.markdown(
                    f"""
                    **Queue number:** {result['queue_number']}  
                    **Patients before you:** {result['patients_before']}  
                    **Expected consultation:** {result['expected_consultation_time']}  
                    **Recommended arrival:** {result['recommended_arrival_time']}  
                    **Estimated wait:** {result['estimated_wait_min']} minutes  
                    """
                )
                colors = {
                    "acceptable": "#198754",
                    "moderate": "#fd7e14",
                    "high": "#dc3545",
                }
                color = colors[result["wait_category"]]
                st.markdown(
                    f"<span style='background:{color};color:white;"
                    f"padding:0.25rem 0.6rem;border-radius:0.4rem'>"
                    f"{result['wait_category'].title()} wait</span>",
                    unsafe_allow_html=True,
                )
                st.info(result["message"])

    with question_tab:
        history = st.session_state.setdefault("assistant_history", [])
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("Retrieved sources"):
                        st.write(", ".join(message["sources"]))
        question = st.chat_input("Ask a question about the clinic")
        if question:
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Checking the clinic information…"):
                    result = api_request(
                        "POST", "/ask", payload={"question": question}
                    )
                if isinstance(result, dict):
                    st.markdown(result["answer"])
                    sources = result.get("source_sections", [])
                    if sources:
                        with st.expander("Retrieved sources"):
                            st.write(", ".join(sources))
                    if not result["found_in_clinic_info"]:
                        st.info(
                            "This information was not found in the clinic "
                            f"knowledge base. Please call {clinic['phone']}."
                        )
                    history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": sources,
                        }
                    )


def _staff_view() -> None:
    """Render appointment monitoring and event-entry controls."""
    api_key = _staff_headers()
    tomorrow = date.today() + timedelta(days=1)
    selected_date = st.date_input("Appointment date", value=tomorrow)
    if st.button("Refresh appointments"):
        st.rerun()

    records = api_request(
        "GET",
        "/staff/appointments",
        api_key=api_key,
        params={"date": selected_date.isoformat()},
    )
    if not isinstance(records, list):
        records = []
    if records:
        st.dataframe(
            [
                {
                    "queue_number": row["queue_number"],
                    "scheduled_time": row["scheduled_time"],
                    "recommended_arrival": row["recommended_arrival"],
                    "predicted_wait_min": row["predicted_wait_min"],
                    "outcome": row["outcome"],
                    "actual_arrival": row["actual_arrival"],
                    "actual_wait_min": row["actual_wait_min"],
                }
                for row in records
            ],
            use_container_width=True,
        )
        appointment_ids = [row["appointment_id"] for row in records]
        selected_id = st.selectbox(
            "Select appointment",
            appointment_ids,
            format_func=lambda value: (
                f"Queue {next(row['queue_number'] for row in records if row['appointment_id'] == value)} "
                f"(ID {value})"
            ),
        )
        selected = next(
            row for row in records if row["appointment_id"] == selected_id
        )
        appointment_day = date.fromisoformat(selected["appointment_date"])
        scheduled = datetime.combine(
            appointment_day,
            time.fromisoformat(selected["scheduled_time"]),
        )
        arrival_default = (
            datetime.fromisoformat(selected["actual_arrival"])
            if selected["actual_arrival"]
            else scheduled
        )
        clinic = api_request("GET", "/clinic")
        duration = (
            clinic["appointment_duration_min"]
            if isinstance(clinic, dict)
            else 15
        )
        consult_start_default = arrival_default
        consult_end_default = consult_start_default + timedelta(minutes=duration)

        st.subheader("Appointment actions")
        arrival_at = st.datetime_input(
            "Actual arrival", value=arrival_default, key="staff_arrival_time"
        )
        if st.button("Record arrival"):
            result = api_request(
                "POST",
                "/staff/arrival",
                api_key=api_key,
                payload={
                    "appointment_id": selected_id,
                    "actual_arrival": arrival_at.isoformat(timespec="seconds"),
                },
            )
            if isinstance(result, dict):
                st.success("Arrival recorded.")
                st.rerun()

        consultation_start = st.datetime_input(
            "Consultation start",
            value=consult_start_default,
            key="staff_consult_start",
        )
        consultation_end = st.datetime_input(
            "Consultation end",
            value=consult_end_default,
            key="staff_consult_end",
        )
        if st.button("Record consultation"):
            result = api_request(
                "POST",
                "/staff/consultation",
                api_key=api_key,
                payload={
                    "appointment_id": selected_id,
                    "start": consultation_start.isoformat(timespec="seconds"),
                    "end": consultation_end.isoformat(timespec="seconds"),
                },
            )
            if isinstance(result, dict):
                st.success("Consultation recorded.")
                st.rerun()

        no_show_col, cancel_col = st.columns(2)
        with no_show_col:
            if st.button("Mark no-show"):
                result = api_request(
                    "POST",
                    f"/staff/appointments/{selected_id}/no-show",
                    api_key=api_key,
                )
                if isinstance(result, dict):
                    st.success("Appointment marked as no-show.")
                    st.rerun()
        with cancel_col:
            if st.button("Cancel appointment"):
                result = api_request(
                    "POST",
                    f"/staff/appointments/{selected_id}/cancel",
                    api_key=api_key,
                )
                if isinstance(result, dict):
                    st.success("Appointment cancelled.")
                    st.rerun()
    else:
        st.info("No appointments are listed for this date.")

    st.subheader("Data collected for Phase 2")
    summary = api_request(
        "GET", "/staff/data-summary", api_key=api_key
    )
    if isinstance(summary, dict):
        st.write(
            f"Usable completed rows: {summary['usable_rows']} / "
            f"{summary['min_rows_for_ml']}"
        )
        st.progress(
            min(1.0, summary["usable_rows"] / summary["min_rows_for_ml"])
        )
        st.write("Ready for Phase 2:" + (" yes" if summary["ready"] else " no"))


_DAY_OPTIONS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_CONFIG_LABELS = {
    "clinic_name": "Clinic name",
    "specialty": "Specialty",
    "address": "Address",
    "phone": "Phone",
    "available_days": "Available days",
    "work_start": "Working hours",
    "work_end": "Working hours",
    "appointment_duration_min": "Appointment duration",
    "buffer_min": "Buffer time",
    "max_appointments_per_day": "Maximum appointments per day",
    "max_patients_per_slot": "Maximum patients per slot",
    "acceptable_wait_min": "Acceptable wait threshold",
    "high_wait_threshold_min": "High-wait threshold",
    "late_arrival_grace_min": "Late-arrival grace period",
    "late_arrival_policy": "Late-arrival policy",
    "cancellation_policy": "Cancellation policy",
    "walkin_policy": "Walk-in policy",
    "emergency_policy": "Emergency policy",
    "noshow_policy": "No-show policy",
    "services": "Services",
    "breaks": "Breaks",
    "peak_hours": "Peak hours",
    "special_conditions": "Special conditions",
}
_TEXT_FIELDS = ("clinic_name", "specialty", "address", "phone")
_POLICY_FIELDS = (
    "late_arrival_policy",
    "cancellation_policy",
    "walkin_policy",
    "emergency_policy",
    "noshow_policy",
)
_NUMBER_FIELDS = (
    ("appointment_duration_min", 1, "Length of each appointment in minutes."),
    ("buffer_min", 0, "Additional minutes reserved between appointments."),
    ("max_appointments_per_day", 1, "Maximum bookings accepted per day."),
    ("max_patients_per_slot", 1, "Maximum patients assigned to one time slot."),
    ("acceptable_wait_min", 0, "Waits up to this duration are acceptable."),
    (
        "high_wait_threshold_min",
        0,
        "Waits at or above this duration are considered high.",
    ),
    (
        "late_arrival_grace_min",
        0,
        "Grace period, in minutes, before the late-arrival policy applies.",
    ),
)


def _form_widget_key(field: str) -> str:
    return f"clinic_config_{field}"


def _initialize_config_form(api_key: str) -> bool:
    """Load the saved backend config and seed its current values into widgets."""
    saved = api_request("GET", "/admin/config", api_key=api_key)
    if not isinstance(saved, dict):
        return False
    try:
        config = ClinicConfig.model_validate(saved)
    except ValidationError as exc:
        st.error("The saved clinic configuration is invalid.")
        for error in exc.errors():
            st.error(str(error["msg"]))
        return False

    config_data = config.model_dump(mode="json")
    st.session_state["admin_config_baseline"] = config_data
    for field in (*_TEXT_FIELDS, *_POLICY_FIELDS):
        st.session_state[_form_widget_key(field)] = getattr(config, field)
    for field, _, _ in _NUMBER_FIELDS:
        st.session_state[_form_widget_key(field)] = getattr(config, field)
    st.session_state[_form_widget_key("available_days")] = list(
        config.available_days
    )
    for field in ("work_start", "work_end"):
        st.session_state[_form_widget_key(field)] = getattr(config, field)
    st.session_state[_form_widget_key("special_conditions")] = (
        config.special_conditions or ""
    )
    st.session_state["clinic_break_rows"] = [
        {"id": uuid4().hex, "start": item.start, "end": item.end}
        for item in config.breaks
    ]
    st.session_state["clinic_peak_rows"] = [
        {"id": uuid4().hex, "start": item.start, "end": item.end}
        for item in config.peak_hours
    ]
    st.session_state["clinic_service_rows"] = [
        {
            "id": uuid4().hex,
            "name": item.name,
            "duration_min": item.duration_min,
            "price": item.price,
        }
        for item in config.services
    ]
    for kind, rows in (
        ("break", st.session_state["clinic_break_rows"]),
        ("peak", st.session_state["clinic_peak_rows"]),
    ):
        for row in rows:
            st.session_state[f"clinic_{kind}_{row['id']}_start"] = row["start"]
            st.session_state[f"clinic_{kind}_{row['id']}_end"] = row["end"]
    for row in st.session_state["clinic_service_rows"]:
        st.session_state[f"clinic_service_{row['id']}_name"] = row["name"]
        st.session_state[f"clinic_service_{row['id']}_duration"] = row[
            "duration_min"
        ]
        st.session_state[f"clinic_service_{row['id']}_price"] = row["price"]
    return True


def _format_config_errors(
    exc: ValidationError, row_label: str | None = None
) -> list[str]:
    """Translate Pydantic locations to readable form labels."""
    messages = []
    for error in exc.errors():
        location = error["loc"]
        field = next(
            (str(part) for part in location if str(part) in _CONFIG_LABELS),
            None,
        )
        if field is None:
            message = str(error["msg"])
            if "work_" in message:
                field = "work_start"
            elif "high_wait_threshold_min" in message or "acceptable_wait_min" in message:
                field = "high_wait_threshold_min"
            elif "breaks" in message:
                field = "breaks"
            elif "peak_hours" in message:
                field = "peak_hours"
        label = row_label or _CONFIG_LABELS.get(field or "", "Clinic configuration")
        if not row_label and field in {"breaks", "peak_hours", "services"}:
            row_index = next(
                (part for part in location if isinstance(part, int)), None
            )
            if row_index is not None:
                label = f"{label} (row {row_index + 1})"
        messages.append(f"{label}: {error['msg']}")
    return messages


def _admin_view() -> None:
    """Render the typed clinic configuration form for authorized staff."""
    api_key = _staff_headers()
    if "admin_config_baseline" not in st.session_state:
        if not _initialize_config_form(api_key):
            return

    if st.button("Reload saved configuration"):
        if _initialize_config_form(api_key):
            st.rerun()

    baseline = ClinicConfig.model_validate(st.session_state["admin_config_baseline"])
    break_rows = st.session_state["clinic_break_rows"]
    peak_rows = st.session_state["clinic_peak_rows"]
    service_rows = st.session_state["clinic_service_rows"]
    remove_break_id = None
    remove_peak_id = None
    remove_service_id = None
    add_break = add_peak = add_service = save = False

    with st.form("clinic_configuration_form"):
        with st.container(border=True):
            st.subheader("🏥 Basic Info")
            cols = st.columns(2)
            for index, field in enumerate(_TEXT_FIELDS):
                with cols[index % 2]:
                    st.text_input(
                        _CONFIG_LABELS[field],
                        key=_form_widget_key(field),
                    )

        with st.container(border=True):
            st.subheader("📅 Appointments & Capacity")
            cols = st.columns(2)
            for index, (field, minimum, _) in enumerate(_NUMBER_FIELDS[:4]):
                with cols[index % 2]:
                    st.number_input(
                        _CONFIG_LABELS[field],
                        min_value=minimum,
                        step=1,
                        key=_form_widget_key(field),
                    )
            cols = st.columns(2)
            with cols[0]:
                st.time_input(
                    "Work starts",
                    key=_form_widget_key("work_start"),
                    format="24h",
                )
            with cols[1]:
                st.time_input(
                    "Work ends",
                    key=_form_widget_key("work_end"),
                    format="24h",
                )
            st.multiselect(
                "Available days",
                options=_DAY_OPTIONS,
                key=_form_widget_key("available_days"),
            )

        with st.container(border=True):
            st.subheader("⏱️ Wait-Time & Queue")
            cols = st.columns(3)
            for index, (field, minimum, help_text) in enumerate(_NUMBER_FIELDS[4:]):
                with cols[index % 3]:
                    st.number_input(
                        _CONFIG_LABELS[field],
                        min_value=minimum,
                        step=1,
                        help=help_text,
                        key=_form_widget_key(field),
                    )

        for label, rows, kind in (
            ("☕ Breaks", break_rows, "break"),
            ("📈 Peak Hours", peak_rows, "peak"),
        ):
            with st.container(border=True):
                st.subheader(label)
                for index, row in enumerate(rows):
                    row_cols = st.columns([2, 2, 1])
                    with row_cols[0]:
                        st.time_input(
                            f"{label} {index + 1} start",
                            key=f"clinic_{kind}_{row['id']}_start",
                            format="24h",
                        )
                    with row_cols[1]:
                        st.time_input(
                            f"{label} {index + 1} end",
                            key=f"clinic_{kind}_{row['id']}_end",
                            format="24h",
                        )
                    with row_cols[2]:
                        if st.form_submit_button(
                            "Remove",
                            key=f"remove_{kind}_{row['id']}",
                        ):
                            if kind == "break":
                                remove_break_id = row["id"]
                            else:
                                remove_peak_id = row["id"]
                if st.form_submit_button(
                    "Add a break" if kind == "break" else "Add peak hours",
                    key=f"add_{kind}_row",
                ):
                    if kind == "break":
                        add_break = True
                    else:
                        add_peak = True

        with st.container(border=True):
            st.subheader("🩺 Services")
            for index, row in enumerate(service_rows):
                row_cols = st.columns([3, 2, 2, 1])
                with row_cols[0]:
                    st.text_input(
                        f"Service {index + 1} name",
                        key=f"clinic_service_{row['id']}_name",
                    )
                with row_cols[1]:
                    st.number_input(
                        f"Duration for service {index + 1} (min)",
                        min_value=1,
                        step=1,
                        key=f"clinic_service_{row['id']}_duration",
                    )
                with row_cols[2]:
                    st.number_input(
                        f"Price for service {index + 1}",
                        min_value=0.0,
                        step=0.5,
                        key=f"clinic_service_{row['id']}_price",
                    )
                with row_cols[3]:
                    if st.form_submit_button(
                        "Remove",
                        key=f"remove_service_{row['id']}",
                    ):
                        remove_service_id = row["id"]
            if st.form_submit_button("Add a service", key="add_service_row"):
                add_service = True

        with st.container(border=True):
            st.subheader("📋 Policies")
            for field in _POLICY_FIELDS:
                st.text_area(
                    _CONFIG_LABELS[field],
                    key=_form_widget_key(field),
                )

        with st.container(border=True):
            st.subheader("📝 Special Conditions")
            st.text_area(
                "Optional additional conditions",
                key=_form_widget_key("special_conditions"),
            )

        save = st.form_submit_button("Save Configuration", type="primary")

    if add_break or add_peak:
        target_rows = break_rows if add_break else peak_rows
        target_rows.append(
            {
                "id": uuid4().hex,
                "start": baseline.work_start,
                "end": baseline.work_end,
            }
        )
        row = target_rows[-1]
        kind = "break" if add_break else "peak"
        st.session_state[f"clinic_{kind}_{row['id']}_start"] = row["start"]
        st.session_state[f"clinic_{kind}_{row['id']}_end"] = row["end"]
        st.rerun()
    if remove_break_id:
        st.session_state["clinic_break_rows"] = [
            row for row in break_rows if row["id"] != remove_break_id
        ]
        st.rerun()
    if remove_peak_id:
        st.session_state["clinic_peak_rows"] = [
            row for row in peak_rows if row["id"] != remove_peak_id
        ]
        st.rerun()
    if add_service:
        first_service_price = (
            baseline.services[0].price if baseline.services else 0.0
        )
        service_rows.append(
            {
                "id": uuid4().hex,
                "name": "",
                "duration_min": baseline.appointment_duration_min,
                "price": first_service_price,
            }
        )
        row = service_rows[-1]
        st.session_state[f"clinic_service_{row['id']}_name"] = row["name"]
        st.session_state[f"clinic_service_{row['id']}_duration"] = row[
            "duration_min"
        ]
        st.session_state[f"clinic_service_{row['id']}_price"] = row["price"]
        st.rerun()
    if remove_service_id:
        st.session_state["clinic_service_rows"] = [
            row for row in service_rows if row["id"] != remove_service_id
        ]
        st.rerun()
    if not save:
        return

    try:
        values: dict[str, Any] = {
            field: st.session_state[_form_widget_key(field)]
            for field in (*_TEXT_FIELDS, *_POLICY_FIELDS)
        }
        values.update(
            {
                field: st.session_state[_form_widget_key(field)]
                for field, _, _ in _NUMBER_FIELDS
            }
        )
        values["available_days"] = st.session_state[
            _form_widget_key("available_days")
        ]
        for field in ("work_start", "work_end"):
            values[field] = st.session_state[_form_widget_key(field)]
        values["breaks"] = [
            BreakTime(
                start=st.session_state.get(
                    f"clinic_break_{row['id']}_start", row["start"]
                ),
                end=st.session_state.get(
                    f"clinic_break_{row['id']}_end", row["end"]
                ),
            )
            for row in break_rows
        ]
        values["peak_hours"] = [
            BreakTime(
                start=st.session_state.get(
                    f"clinic_peak_{row['id']}_start", row["start"]
                ),
                end=st.session_state.get(
                    f"clinic_peak_{row['id']}_end", row["end"]
                ),
            )
            for row in peak_rows
        ]
        values["services"] = [
            Service(
                name=st.session_state.get(
                    f"clinic_service_{row['id']}_name", row["name"]
                ),
                duration_min=st.session_state.get(
                    f"clinic_service_{row['id']}_duration",
                    row["duration_min"],
                ),
                price=st.session_state.get(
                    f"clinic_service_{row['id']}_price", row["price"]
                ),
            )
            for row in service_rows
        ]
        values["special_conditions"] = (
            st.session_state[_form_widget_key("special_conditions")] or None
        )
        config = ClinicConfig(**values)
        valid_fields = set(ClinicConfigUpdate.model_fields)
        update = ClinicConfigUpdate.model_validate(
            config.model_dump(mode="python", include=valid_fields)
        )
    except ValidationError as exc:
        st.error("Please correct the following configuration fields:")
        for message in _format_config_errors(exc):
            st.error(message)
        return

    result = api_request(
        "PUT",
        "/admin/config",
        api_key=api_key,
        payload=update.model_dump(mode="json"),
    )
    if not isinstance(result, dict) or result.get("saved") is not True:
        return
    confirmed = api_request("GET", "/admin/config", api_key=api_key)
    if not isinstance(confirmed, dict):
        st.error("Configuration was submitted, but could not be confirmed.")
        return
    try:
        confirmed_config = ClinicConfig.model_validate(confirmed)
    except ValidationError as exc:
        st.error("The server returned an invalid saved configuration.")
        for message in _format_config_errors(exc):
            st.error(message)
        return
    if confirmed_config != update:
        st.error("The server's saved configuration differs from the form.")
        return
    st.session_state["admin_config_baseline"] = confirmed_config.model_dump(
        mode="json"
    )
    if result.get("index_rebuilt"):
        st.success("Configuration saved; the clinic knowledge index was updated.")
    else:
        st.warning(
            result.get("warning")
            or "Configuration saved, but the knowledge index could not be updated."
        )


def _knowledge_base_view() -> None:
    """Provide free-form staff knowledge ingestion and removal."""
    api_key = _staff_headers()
    st.header("📚 Knowledge Base")
    with st.form("admin_knowledge_form"):
        content = st.text_area(
            "Clinic knowledge",
            placeholder="Add clinic-specific information for the assistant.",
            height=160,
        )
        submitted = st.form_submit_button("Add to Knowledge Base", type="primary")
    if submitted:
        if not content.strip():
            st.error("Enter clinic information before adding it.")
        else:
            result = api_request(
                "POST",
                "/staff/knowledge",
                api_key=api_key,
                payload={"content": content.strip()},
            )
            if isinstance(result, dict):
                st.success("Knowledge added to the clinic assistant.")

    notice = st.session_state.pop("knowledge_notice", None)
    if notice:
        st.success(notice)
    entries = api_request("GET", "/staff/knowledge", api_key=api_key)
    if not isinstance(entries, list):
        return
    st.subheader("Admin-added entries")
    if not entries:
        st.info("No admin-added entries are stored yet.")
        return
    for entry in entries:
        with st.container(border=True):
            st.caption(f"Added {entry['timestamp']}")
            st.write(entry["content"])
            if st.button("Delete entry", key=f"delete_knowledge_{entry['id']}"):
                deleted = api_request(
                    "DELETE",
                    f"/staff/knowledge/{entry['id']}",
                    api_key=api_key,
                )
                if deleted is not None:
                    st.session_state["knowledge_notice"] = "Knowledge entry deleted."
                    st.rerun()


def _system_status_view() -> None:
    """Show API, vector store, document, and local model readiness."""
    api_key = _staff_headers()
    st.header("📊 System Status")
    health = api_request("GET", "/health")
    status = api_request("GET", "/staff/index-status", api_key=api_key)
    if not isinstance(status, dict):
        if health is None:
            st.error("Clinic API is unavailable.")
        return

    api_ready = isinstance(health, dict) and health.get("status") == "ok"
    cols = st.columns(4)
    cols[0].metric("Clinic API", "Connected" if api_ready else "Unavailable")
    cols[1].metric(
        "Knowledge base",
        "Ready" if status["vector_store_ready"] else "Not ready",
    )
    cols[2].metric("Indexed chunks", status["chunk_count"])
    cols[3].metric("Admin entries", status["admin_document_count"])
    st.caption(
        f"Generation provider: {status['llm_provider']} · "
        f"Qwen local model: {'loaded' if status['llm_loaded'] else 'not loaded'}"
    )
    if not status["vector_store_ready"]:
        st.warning("The clinic vector store has no indexed content.")
    with st.expander("Staff appointment operations"):
        _staff_view()


def main() -> None:
    """Run the Streamlit interface."""
    st.set_page_config(
        page_title="Clinic Management",
        page_icon="🏥",
        layout="wide",
    )
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 🏥 Clinic Management")
        mode = st.radio(
            "Navigation",
            [
                "💬 Assistant",
                "📚 Knowledge Base",
                "⚙️ Clinic Configuration",
                "📊 System Status",
            ],
        )
        st.text_input(
            "Staff API key",
            type="password",
            key="staff_api_key",
        )
    if mode == "💬 Assistant":
        _assistant_view()
    elif mode == "📚 Knowledge Base":
        _knowledge_base_view()
    elif mode == "⚙️ Clinic Configuration":
        _admin_view()
    else:
        _system_status_view()


if __name__ == "__main__":
    main()
