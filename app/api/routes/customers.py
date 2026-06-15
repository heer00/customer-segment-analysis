"""
app/api/routes/customers.py
────────────────────────────
Customer administration endpoints — manages customer profiles and history.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import customer as schemas
from app.repositories import customer_repo
from app.core.model import ModelManager, get_model
from app.core.preprocessor import Preprocessor, get_preprocessor

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.post(
    "/",
    response_model=schemas.CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Customer Profile",
    description=(
        "Registers a new customer in the database, automatically runs segment prediction, "
        "and logs the initial classification with a business recommendation."
    ),
)
async def create_customer(
    customer: schemas.CustomerCreate,
    db: Session = Depends(get_db),
    model: ModelManager = Depends(get_model),
    preprocessor: Preprocessor = Depends(get_preprocessor),
) -> schemas.CustomerResponse:

    # Check if email already exists
    if customer_repo.get_customer_by_email(db, email=customer.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Customer with email '{customer.email}' already exists.",
        )

    # 1. Create customer profile
    new_customer = customer_repo.create_customer(db, customer=customer)

    # 2. Auto-run initial prediction
    try:
        gender_encoded = preprocessor._encode_gender(customer.gender)
        prediction = model.predict(
            age                = customer.age,
            gender_encoded     = gender_encoded,
            annual_income      = customer.annual_income,
            spending_score     = customer.spending_score,
            purchase_frequency = customer.purchase_frequency,
        )

        # 3. Log prediction linked to new customer
        customer_repo.create_prediction(
            db                  = db,
            age                 = customer.age,
            gender              = customer.gender,
            annual_income       = customer.annual_income,
            spending_score      = customer.spending_score,
            purchase_frequency  = customer.purchase_frequency,
            cluster_id          = prediction["cluster_id"],
            segment_label       = prediction["segment_label"],
            segment_description = prediction["segment_description"],
            recommendation      = prediction["recommendation"],
            customer_id         = new_customer.id,
        )
    except Exception as e:
        print(f"[WARNING] Auto-prediction failed during registration: {e}")

    return new_customer


@router.get(
    "/",
    response_model=List[schemas.CustomerResponse],
    summary="List All Customers",
    description="Retrieve all registered customers. Filter by segment label (e.g. 'High Value Customers') and paginate using skip/limit.",
)
async def list_customers(
    segment_label: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> List[schemas.CustomerResponse]:
    return customer_repo.list_customers(db, segment_label=segment_label, skip=skip, limit=limit)


@router.get(
    "/{customer_id}",
    response_model=schemas.CustomerResponse,
    summary="Get Customer Profile",
    description="Retrieve a single customer profile by database ID.",
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
    description="Retrieve all prediction logs for a specific customer, sorted newest first.",
)
async def get_customer_prediction_history(
    customer_id: int,
    db: Session = Depends(get_db),
) -> List[schemas.PredictionRecordResponse]:
    customer = customer_repo.get_customer_by_id(db, customer_id=customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found.",
        )
    return customer_repo.list_predictions(db, customer_id=customer_id)
