"""
app/db/models.py
────────────────
SQLAlchemy ORM models representing our database schema.

WHY ORM MODELS:
  - Abstracts raw SQL tables into Python classes.
  - Changes in columns are tracked statically.
  - Ensures relational integrity (e.g. FK constraints, cascade deletes).
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Customer(Base):
    """
    ORM Model representing a Customer.
    Stores descriptive profiling information alongside core features.
    """
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    annual_income = Column(Float, nullable=False)  # Feature 1: Annual Income (k$)
    spending_score = Column(Float, nullable=False)  # Feature 2: Spending Score (1-100)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # One-to-Many: One customer can have multiple prediction history snapshots
    predictions = relationship(
        "Prediction",
        back_populates="customer",
        cascade="all, delete-orphan",
    )


class Prediction(Base):
    """
    ORM Model representing prediction records.
    Tracks what inputs were passed and what output the KMeans model produced at that timestamp.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(
        Integer,
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    
    # Store numerical features at prediction time (useful for tracking behavior drift)
    annual_income = Column(Float, nullable=False)
    spending_score = Column(Float, nullable=False)

    cluster_id = Column(Integer, nullable=False)
    segment_label = Column(String, nullable=False)
    segment_description = Column(String, nullable=False)

    predicted_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship to parent customer
    customer = relationship("Customer", back_populates="predictions")
