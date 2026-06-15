"""
app/api/routes/predict.py
─────────────────────────
Prediction route — the core ML inference endpoint.

Supports:
  - Anonymous predictions (input from request body)
  - Customer-specific predictions (fetch features from DB by customer_id)
  - All predictions are persisted to the prediction history table
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import customer as schemas
from app.core.model import ModelManager, get_model
from app.core.preprocessor import Preprocessor, get_preprocessor
from app.repositories import customer_repo

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/",
    response_model=schemas.PredictionResponse,
    summary="Predict Customer Segment",
    description="""
Predict segment for a customer profile.

You can either:
1. **Anonymous prediction**: Provide all features in the request body.
2. **Customer prediction**: Provide `customer_id` as a query parameter — features are fetched from the database.

All predictions are logged to the database history table and include a business recommendation.
    """,
)
async def predict_segment(
    customer: Optional[schemas.CustomerInput] = None,
    customer_id: Optional[int] = None,
    db: Session = Depends(get_db),
    model: ModelManager = Depends(get_model),
    preprocessor: Preprocessor = Depends(get_preprocessor),
) -> schemas.PredictionResponse:

    # 1. Resolve input features
    db_customer_id: Optional[int] = None

    if customer_id is not None:
        db_customer = customer_repo.get_customer_by_id(db, customer_id=customer_id)
        if not db_customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID {customer_id} not found.",
            )
        raw_input = {
            "age":                db_customer.age,
            "gender":             db_customer.gender,
            "annual_income":      db_customer.annual_income,
            "spending_score":     db_customer.spending_score,
            "purchase_frequency": db_customer.purchase_frequency,
        }
        db_customer_id = db_customer.id
    else:
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either customer_id query param or full customer details in the request body.",
            )
        raw_input = {
            "age":                customer.age,
            "gender":             customer.gender,
            "annual_income":      customer.annual_income,
            "spending_score":     customer.spending_score,
            "purchase_frequency": customer.purchase_frequency,
        }

    # 2. Preprocess (encode gender, validate ranges)
    try:
        _, warnings = preprocessor.process(raw_input)
        gender_encoded = preprocessor._encode_gender(raw_input["gender"])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    # 3. Run prediction
    try:
        result = model.predict(
            age                = int(raw_input["age"]),
            gender_encoded     = gender_encoded,
            annual_income      = float(raw_input["annual_income"]),
            spending_score     = float(raw_input["spending_score"]),
            purchase_frequency = int(raw_input["purchase_frequency"]),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {str(e)}",
        )

    # 4. Log prediction to database
    db_pred = None
    try:
        db_pred = customer_repo.create_prediction(
            db                  = db,
            age                 = int(raw_input["age"]),
            gender              = str(raw_input["gender"]),
            annual_income       = float(raw_input["annual_income"]),
            spending_score      = float(raw_input["spending_score"]),
            purchase_frequency  = int(raw_input["purchase_frequency"]),
            cluster_id          = result["cluster_id"],
            segment_label       = result["segment_label"],
            segment_description = result["segment_description"],
            recommendation      = result["recommendation"],
            customer_id         = db_customer_id,
        )
    except Exception as e:
        print(f"[WARNING] Failed to save prediction to history: {e}")

    # 5. Return response
    return schemas.PredictionResponse(
        cluster_id              = result["cluster_id"],
        segment_label           = result["segment_label"],
        segment_description     = result["segment_description"],
        recommendation          = result["recommendation"],
        input_received          = raw_input,
        preprocessing_warnings  = warnings,
        prediction_id           = db_pred.id if db_pred else None,
    )
