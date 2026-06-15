"""
app/schemas/customer.py
───────────────────────
Pydantic models for API request/response validation.

WHY PYDANTIC: FastAPI uses these schemas to:
  1. Automatically validate incoming JSON (wrong types → 422 error with clear message)
  2. Generate the OpenAPI/Swagger documentation at /docs
  3. Serialize response data to JSON

This replaces manual validation code and makes your API self-documenting.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Shared Base Input Fields ────────────────────────────────────────────────
class CustomerBase(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=80,
        description="Customer age in years (18–80).",
        examples=[28],
    )
    gender: str = Field(
        ...,
        description="Customer gender: 'Male' or 'Female'.",
        examples=["Male"],
    )
    annual_income: float = Field(
        ...,
        ge=0,
        le=200,
        description="Annual income in Lakhs of Indian Rupees (₹). Range: 0–200 Lakhs.",
        examples=[75.0],
    )
    spending_score: float = Field(
        ...,
        ge=1,
        le=100,
        description=(
            "Spending score (1–100) assigned based on purchase frequency, "
            "transaction value, and mall loyalty. "
            "1 = lowest engagement, 100 = highest engagement."
        ),
        examples=[82.0],
    )
    purchase_frequency: int = Field(
        ...,
        ge=1,
        le=52,
        description="Estimated number of shopping visits per year (1–52).",
        examples=[12],
    )

    @field_validator("gender")
    @classmethod
    def gender_must_be_valid(cls, v: str) -> str:
        if v.strip().title() not in ("Male", "Female"):
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v.strip().title()


# ─── Prediction-only Input (no name/email needed) ────────────────────────────
class CustomerInput(CustomerBase):
    """What the prediction client sends in the POST body."""
    pass


# ─── Customer CRUD Schemas ────────────────────────────────────────────────────
class CustomerCreate(CustomerBase):
    """Schema for creating a full customer profile."""
    name: str = Field(
        ...,
        min_length=1,
        description="Full name of the customer.",
        examples=["Rajesh Kumar"],
    )
    email: str = Field(
        ...,
        description="Unique email address of the customer.",
        examples=["rajesh.kumar@example.com"],
    )

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v.lower().strip()


class CustomerResponse(CustomerCreate):
    """Schema returned to represent a Customer profile."""
    id: int = Field(description="Database generated unique customer ID.")
    created_at: datetime = Field(description="Timestamp when customer was created.")

    model_config = {
        "from_attributes": True
    }


# ─── Prediction History Schema ────────────────────────────────────────────────
class PredictionRecordResponse(BaseModel):
    """Schema representing a stored prediction record."""
    id: int
    customer_id: Optional[int] = None
    age: int
    gender: str
    annual_income: float
    spending_score: float
    purchase_frequency: int
    cluster_id: int
    segment_label: str
    segment_description: str
    recommendation: str
    predicted_at: datetime

    model_config = {
        "from_attributes": True
    }


# ─── Prediction Response Schema ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """What the API sends back after a prediction."""
    cluster_id: int = Field(description="Numeric cluster ID from KMeans (0–4)")
    segment_label: str = Field(description="Business-friendly segment name")
    segment_description: str = Field(description="Human-readable segment explanation")
    recommendation: str = Field(description="Recommended business action for this segment")
    input_received: dict = Field(description="Echo of the input for traceability")
    preprocessing_warnings: list[str] = Field(
        default=[],
        description="Fields that were missing and filled with defaults"
    )
    prediction_id: Optional[int] = Field(None, description="Database record ID if persisted")


# ─── Health Check Schema ──────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool