"""
app/api/routes/predict.py
─────────────────────────
Prediction route — the core ML inference endpoint.

WHY:
  - Supports anonymous predictions (no customer profile required, input from body).
  - Supports customer-specific predictions (retrieves income/spending from DB).
  - Persists all executed predictions to the SQLite prediction history table.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import customer as schemas
from app.core.model import ModelManager, get_model
from app.repositories import customer_repo

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/",
    response_model=schemas.PredictionResponse,
    summary="Predict Customer Segment",
    description="""
    Predict segment for an input customer profile. 
    
    You can either:
    1. **Predict for an existing customer**: Provide `customer_id` as a query parameter. Features will be fetched from database.
    2. **Predict anonymously**: Omit `customer_id` and provide features (`annual_income`, `spending_score`) in request body.
    
    All predictions are logged to the database history log.
    """,
)
async def predict_segment(
    customer: Optional[schemas.CustomerInput] = None,
    customer_id: Optional[int] = None,
    db: Session = Depends(get_db),
    model: ModelManager = Depends(get_model),
) -> schemas.PredictionResponse:
    # 1. Resolve features (income and spending)
    income: float = 0.0
    spending: float = 0.0
    db_customer_id: Optional[int] = None

    if customer_id is not None:
        # Fetch customer profile from DB
        db_customer = customer_repo.get_customer_by_id(db, customer_id=customer_id)
        if not db_customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID {customer_id} not found in database.",
            )
        income = db_customer.annual_income
        spending = db_customer.spending_score
        db_customer_id = db_customer.id
    else:
        # Validate that body is provided for anonymous prediction
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either customer_id query parameter or customer details in the request body.",
            )
        income = customer.annual_income
        spending = customer.spending_score

    # 2. Run prediction
    try:
        result = model.predict(annual_income=income, spending_score=spending)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {str(e)}",
        )

    # 3. Log prediction in database
    db_pred = None
    try:
        db_pred = customer_repo.create_prediction(
            db=db,
            annual_income=income,
            spending_score=spending,
            cluster_id=result["cluster_id"],
            segment_label=result["segment_label"],
            segment_description=result["segment_description"],
            customer_id=db_customer_id,
        )
    except Exception as e:
        # We don't fail the request if logging fails, but we print a warning
        print(f"[WARNING] Failed to save prediction to history log: {e}")

    # 4. Return API contract response
    return schemas.PredictionResponse(
        cluster_id=result["cluster_id"],
        segment_label=result["segment_label"],
        segment_description=result["segment_description"],
        input_received={
            "annual_income": income,
            "spending_score": spending,
        },
        prediction_id=db_pred.id if db_pred else None,
    )
