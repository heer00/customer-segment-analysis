"""
app/api/routes/customers.py
────────────────────────────
Customer administration endpoints — manages customer profiles and history.

WHY:
  - Centralizes endpoints related to client profiling.
  - Runs model inference automatically when creating customer profiles.
  - Supports query filters needed by the natural language dashboard interface.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import customer as schemas
from app.repositories import customer_repo
from app.core.model import ModelManager, get_model

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.post(
    "/",
    response_model=schemas.CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Customer Profile",
    description="Registers a new customer, automatically runs the segment prediction, and logs the initial classification.",
)
async def create_customer(
    customer: schemas.CustomerCreate,
    db: Session = Depends(get_db),
    model: ModelManager = Depends(get_model),
) -> schemas.CustomerResponse:
    # Check if customer already exists by email
    db_customer = customer_repo.get_customer_by_email(db, email=customer.email)
    if db_customer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Customer with email '{customer.email}' already exists.",
        )

    # 1. Create the customer profile
    new_customer = customer_repo.create_customer(db, customer=customer)

    # 2. Automatically perform prediction for their initial features
    try:
        prediction = model.predict(
            annual_income=customer.annual_income,
            spending_score=customer.spending_score,
        )

        # 3. Log the prediction linked to the new customer ID
        customer_repo.create_prediction(
            db=db,
            annual_income=customer.annual_income,
            spending_score=customer.spending_score,
            cluster_id=prediction["cluster_id"],
            segment_label=prediction["segment_label"],
            segment_description=prediction["segment_description"],
            customer_id=new_customer.id,
        )
    except Exception as e:
        # In a real system, you might want to rollback the customer or log as warning.
        # We will log the error but return the customer profile.
        print(f"[WARNING] Automatic prediction failed during customer registration: {e}")

    return new_customer


@router.get(
    "/",
    response_model=List[schemas.CustomerResponse],
    summary="List Customers",
    description="Retrieves a list of registered customers. Supports filtering by segment label (e.g. 'Premium Customers') and pagination.",
)
async def list_customers(
    segment_label: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> List[schemas.CustomerResponse]:
    customers = customer_repo.list_customers(
        db, segment_label=segment_label, skip=skip, limit=limit
    )
    return customers


@router.get(
    "/{customer_id}",
    response_model=schemas.CustomerResponse,
    summary="Get Customer Profile",
    description="Retrieves customer profile by database ID.",
)
async def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
) -> schemas.CustomerResponse:
    customer = customer_repo.get_customer_by_id(db, customer_id=customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found.",
        )
    return customer


@router.get(
    "/{customer_id}/predictions",
    response_model=List[schemas.PredictionRecordResponse],
    summary="Get Customer Prediction History",
    description="Retrieves all prediction logs associated with a specific customer (sorted descending by timestamp).",
)
async def get_customer_prediction_history(
    customer_id: int,
    db: Session = Depends(get_db),
) -> List[schemas.PredictionRecordResponse]:
    # Check if customer exists first
    customer = customer_repo.get_customer_by_id(db, customer_id=customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found.",
        )
    
    predictions = customer_repo.list_predictions(db, customer_id=customer_id)
    return predictions
