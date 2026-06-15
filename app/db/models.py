"""
app/db/models.py
────────────────
SQLAlchemy ORM models — updated with full customer profile fields.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id                 = Column(Integer, primary_key=True, index=True)
    name               = Column(String, nullable=False)
    email              = Column(String, unique=True, index=True, nullable=False)
    age                = Column(Integer, nullable=False)
    gender             = Column(String, nullable=False)          # "Male" / "Female"
    annual_income      = Column(Float, nullable=False)           # ₹ in Lakhs
    spending_score     = Column(Float, nullable=False)           # 1–100
    purchase_frequency = Column(Integer, nullable=False)         # times/year
    created_at         = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    predictions = relationship(
        "Prediction",
        back_populates="customer",
        cascade="all, delete-orphan",
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    customer_id         = Column(
        Integer,
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Input features at time of prediction
    age                 = Column(Integer,  nullable=False)
    gender              = Column(String,   nullable=False)
    annual_income       = Column(Float,    nullable=False)
    spending_score      = Column(Float,    nullable=False)
    purchase_frequency  = Column(Integer,  nullable=False)

    # Output
    cluster_id          = Column(Integer,  nullable=False)
    segment_label       = Column(String,   nullable=False)
    segment_description = Column(String,   nullable=False)
    recommendation      = Column(String,   nullable=False)      # NEW

    predicted_at        = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="predictions")