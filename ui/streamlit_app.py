"""Simple Streamlit patient, staff, and administrator interface."""

import json
import os
from datetime import date, datetime, time, timedelta
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv


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


def _patient_view() -> None:
    """Render the patient booking and clinic-question tabs."""
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
        question = st.text_area("What would you like to know?")
        if st.button("Ask", key="ask_clinic"):
            result = api_request(
                "POST", "/ask", payload={"question": question}
            )
            if isinstance(result, dict):
                st.write(result["answer"])
                if result["source_sections"]:
                    st.caption(
                        "Sources: " + ", ".join(result["source_sections"])
                    )
                if not result["found_in_clinic_info"]:
                    st.warning(
                        "This was not found in the clinic information. "
                        f"Please call {clinic['phone']}."
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


def _admin_view() -> None:
    """Render a JSON configuration editor for authorized administrators."""
    api_key = _staff_headers()
    if "config_json_editor" not in st.session_state:
        config = api_request("GET", "/admin/config", api_key=api_key)
        if not isinstance(config, dict):
            return
        st.session_state["config_json_editor"] = json.dumps(
            config, indent=2
        )
    if st.button("Reload configuration"):
        config = api_request("GET", "/admin/config", api_key=api_key)
        if isinstance(config, dict):
            st.session_state["config_json_editor"] = json.dumps(
                config, indent=2
            )
            st.rerun()

    config_text = st.text_area(
        "Clinic configuration JSON",
        key="config_json_editor",
        height=480,
    )
    if st.button("Save configuration"):
        try:
            config_payload = json.loads(config_text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc.msg}")
            return
        result = api_request(
            "PUT",
            "/admin/config",
            api_key=api_key,
            payload=config_payload,
        )
        if isinstance(result, dict):
            if result["index_rebuilt"]:
                st.success("Configuration saved and RAG index rebuilt.")
            else:
                st.warning(result["warning"])


def main() -> None:
    """Run the Streamlit interface."""
    st.set_page_config(page_title="Clinic Smart Booking", page_icon="🏥")
    st.title("Clinic Next-Day Smart Booking")
    mode = st.sidebar.radio("View", ["Patient", "Staff", "Admin"])
    if mode in {"Staff", "Admin"}:
        st.sidebar.text_input(
            "Staff API key",
            type="password",
            key="staff_api_key",
        )
    if mode == "Patient":
        _patient_view()
    elif mode == "Staff":
        _staff_view()
    else:
        _admin_view()


if __name__ == "__main__":
    main()
