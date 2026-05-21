"""
app/repositories/customer_repo.py
──────────────────────────────────
Repository layer for database operations (CRUD).

WHY:
  - Decouples API endpoints from database query logic.
  - Allows easy unit testing by mocking the repository functions.
  - Simplifies query sharing across multiple route handlers.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from app.db import models
from app.schemas import customer as schemas


def get_customer_by_id(db: Session, customer_id: int) -> Optional[models.Customer]:
    """Fetch a customer by their ID."""
    return db.query(models.Customer).filter(models.Customer.id == customer_id).first()


def get_customer_by_email(db: Session, email: str) -> Optional[models.Customer]:
    """Fetch a customer by their email address."""
    return db.query(models.Customer).filter(models.Customer.email == email.lower().strip()).first()


def list_customers(
    db: Session,
    segment_label: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[models.Customer]:
    """
    List customers with optional pagination and segment filtering.
    Crucial for Phase 6 natural language agent query capabilities.
    """
    query = db.query(models.Customer)
    if segment_label:
        # Match segment name case-insensitively (e.g. 'premium' -> 'Premium Customers')
        query = query.join(models.Customer.predictions).filter(
            models.Prediction.segment_label.ilike(f"%{segment_label}%")
        ).distinct()
    return query.offset(skip).limit(limit).all()


def create_customer(db: Session, customer: schemas.CustomerCreate) -> models.Customer:
    """Create a new customer profile in the database."""
    db_customer = models.Customer(
        name=customer.name,
        email=customer.email.lower().strip(),
        annual_income=customer.annual_income,
        spending_score=customer.spending_score,
    )
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer


def create_prediction(
    db: Session,
    annual_income: float,
    spending_score: float,
    cluster_id: int,
    segment_label: str,
    segment_description: str,
    customer_id: Optional[int] = None,
) -> models.Prediction:
    """Log a prediction history record to the database."""
    db_prediction = models.Prediction(
        customer_id=customer_id,
        annual_income=annual_income,
        spending_score=spending_score,
        cluster_id=cluster_id,
        segment_label=segment_label,
        segment_description=segment_description,
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def list_predictions(
    db: Session,
    customer_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[models.Prediction]:
    """Retrieve prediction history, optionally filtering by a specific customer."""
    query = db.query(models.Prediction)
    if customer_id is not None:
        query = query.filter(models.Prediction.customer_id == customer_id)
    return query.order_by(models.Prediction.predicted_at.desc()).offset(skip).limit(limit).all()
