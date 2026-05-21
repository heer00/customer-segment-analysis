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
    annual_income: float = Field(
        ...,
        ge=0,
        le=200,
        description="Annual income in k$ (thousands of dollars). Range: 0–200.",
        examples=[75],
    )
    spending_score: float = Field(
        ...,
        ge=1,
        le=100,
        description="Spending score assigned by the mall (1–100).",
        examples=[80],
    )

    @field_validator("spending_score")
    @classmethod
    def spending_score_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Spending score must be greater than 0")
        return v


# ─── Legacy Request Schema (Preserving compatibility) ───────────────────────
class CustomerInput(CustomerBase):
    """What the legacy prediction client sends in the POST body."""
    pass


# ─── Customer CRUD Schemas ────────────────────────────────────────────────────
class CustomerCreate(CustomerBase):
    """Schema for creating a customer."""
    name: str = Field(
        ...,
        min_length=1,
        description="Full name of the customer.",
        examples=["Jane Doe"],
    )
    email: str = Field(
        ...,
        description="Unique email address of the customer.",
        examples=["jane.doe@example.com"],
    )

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v.lower().strip()


class CustomerResponse(CustomerCreate):
    """Schema returned to represent a Customer profile."""
    id: int = Field(description="Database generated unique customer ID.")
    created_at: datetime = Field(description="Timestamp when customer was created.")

    model_config = {
        "from_attributes": True  # Pydantic v2 configuration to read ORM objects
    }


# ─── Prediction History Schemas ───────────────────────────────────────────────
class PredictionRecordResponse(BaseModel):
    """Schema representing a stored prediction record in history."""
    id: int
    customer_id: Optional[int] = None
    annual_income: float
    spending_score: float
    cluster_id: int
    segment_label: str
    segment_description: str
    predicted_at: datetime

    model_config = {
        "from_attributes": True
    }


# ─── Legacy Prediction Response Schema (Preserving /predict endpoint contract)
class PredictionResponse(BaseModel):
    """
    What the API sends back.
    Clients can rely on this exact shape — it's our API contract.
    """
    cluster_id: int = Field(description="Numeric cluster ID from KMeans (0–4)")
    segment_label: str = Field(description="Business-friendly segment name")
    segment_description: str = Field(description="Human-readable segment explanation")
    input_received: dict = Field(description="Echo of the input for traceability")
    prediction_id: Optional[int] = Field(None, description="Database record ID if persisted")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cluster_id": 0,
                "segment_label": "Premium Customers",
                "segment_description": "High Income, High Spending — your most valuable customers",
                "input_received": {"annual_income": 75, "spending_score": 80},
                "prediction_id": 42
            }
        }
    }


# ─── Health Check Schema ──────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
