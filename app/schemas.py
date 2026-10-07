"""Pydantic schemas for clinic configuration, bookings, and wait-time data."""

from datetime import date, datetime, time
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


DayOfWeek = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class BreakTime(BaseModel):
    """A time range used for a clinic break or peak period."""

    start: time
    end: time


class Service(BaseModel):
    """A clinic service and its duration and price."""

    name: str
    duration_min: int = Field(gt=0)
    price: float = Field(ge=0)


class ClinicConfig(BaseModel):
    """Configuration for a single clinic."""

    clinic_name: str
    specialty: str
    address: str
    phone: str
    available_days: list[DayOfWeek]
    work_start: time
    work_end: time
    breaks: list[BreakTime] = Field(default_factory=list)
    appointment_duration_min: int = Field(gt=0)
    buffer_min: int = Field(default=0, ge=0)
    max_appointments_per_day: int = Field(gt=0)
    max_patients_per_slot: int = Field(default=1, gt=0)
    peak_hours: list[BreakTime] = Field(default_factory=list)
    acceptable_wait_min: int = Field(default=15, ge=0)
    high_wait_threshold_min: int = Field(default=30, ge=0)
    late_arrival_grace_min: int = Field(default=15, ge=0)
    late_arrival_policy: str
    cancellation_policy: str
    walkin_policy: str
    emergency_policy: str
    noshow_policy: str
    services: list[Service] = Field(default_factory=list)
    special_conditions: Optional[str] = None

    @model_validator(mode="after")
    def validate_working_hours(self) -> "ClinicConfig":
        """Ensure thresholds and configured time ranges are valid."""
        if self.work_end <= self.work_start:
            raise ValueError("work_end must be after work_start")
        if self.high_wait_threshold_min < self.acceptable_wait_min:
            raise ValueError(
                "high_wait_threshold_min must be greater than or equal to "
                "acceptable_wait_min"
            )

        for field_name, ranges in (
            ("breaks", self.breaks),
            ("peak_hours", self.peak_hours),
        ):
            for time_range in ranges:
                if time_range.start >= time_range.end:
                    raise ValueError(f"Each {field_name} range must have start before end")
                if (
                    time_range.start < self.work_start
                    or time_range.end > self.work_end
                ):
                    raise ValueError(
                        f"Each {field_name} range must lie within working hours"
                    )
        return self


class QuestionRequest(BaseModel):
    """A clinic-information question."""

    question: str = Field(min_length=3, max_length=500)


class RAGAnswer(BaseModel):
    """An answer and its clinic-information sources."""

    answer: str
    source_sections: list[str]
    found_in_clinic_info: bool


class BookingRequest(BaseModel):
    """A request to book a future appointment."""

    patient_name: str
    patient_phone: str
    appointment_date: date
    service: Optional[str] = None

    @model_validator(mode="after")
    def validate_appointment_date(self) -> "BookingRequest":
        """Require the requested appointment date to be in the future."""
        if self.appointment_date <= date.today():
            raise ValueError("appointment_date must be after today")
        return self


class BookingResponse(BaseModel):
    """The appointment and arrival recommendation returned to a patient."""

    appointment_id: int
    appointment_date: date
    queue_number: int
    patients_before: int
    expected_consultation_time: time
    recommended_arrival_time: time
    estimated_wait_min: int
    method: Literal["rules", "ml"]


class ArrivalEvent(BaseModel):
    """A patient's recorded arrival."""

    appointment_id: int
    actual_arrival: datetime


class ConsultationEvent(BaseModel):
    """A consultation's start and end timestamps."""

    appointment_id: int
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_consultation_times(self) -> "ConsultationEvent":
        """Require the consultation to end after it starts."""
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class AppointmentRecord(BaseModel):
    """A stored booking and the data collected for later model training."""

    appointment_id: int
    booking_timestamp: datetime
    appointment_date: date
    day_of_week: str
    scheduled_time: time
    queue_number: int
    patients_before: int
    bookings_that_day: int
    is_peak: bool
    recommended_arrival: time
    predicted_wait_min: int
    predicted_by: Literal["rules", "ml"]
    actual_arrival: Optional[datetime] = None
    late_minutes: Optional[int] = None
    consult_start: Optional[datetime] = None
    consult_end: Optional[datetime] = None
    consult_duration_min: Optional[float] = None
    actual_wait_min: Optional[float] = None
    outcome: Literal[
        "completed", "cancelled", "no_show", "rescheduled", "pending"
    ] = "pending"


class WaitFeatures(BaseModel):
    """Input features for a wait-time prediction."""

    day_of_week: int = Field(ge=0, le=6)
    appointment_hour: int = Field(ge=0, le=23)
    queue_position: int = Field(ge=1)
    patients_before: int = Field(ge=0)
    is_peak: bool
    bookings_that_day: int = Field(ge=0)
    hist_avg_wait_min: float
    hist_avg_consult_min: float
    recent_delay_min: float = 0


class WaitPrediction(BaseModel):
    """A wait-time estimate and the method used to produce it."""

    predicted_wait_min: float = Field(ge=0)
    method: Literal["rules", "ml"]
