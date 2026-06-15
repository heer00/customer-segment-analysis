"""
app/api/routes/analytics.py
────────────────────────────
Analytics endpoints — aggregated business metrics for the dashboard.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import get_db
from app.db import models

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/overview",
    summary="Dashboard Overview Metrics",
    description="Returns total customer count, total predictions, and segment distribution.",
)
async def get_overview(db: Session = Depends(get_db)):
    total_customers = db.query(models.Customer).count()
    total_predictions = db.query(models.Prediction).count()

    # Segment distribution from latest predictions per customer
    segment_counts = (
        db.query(
            models.Prediction.segment_label,
            func.count(models.Prediction.segment_label).label("count")
        )
        .group_by(models.Prediction.segment_label)
        .all()
    )

    segment_distribution = {row.segment_label: row.count for row in segment_counts}

    # Identify most common segment
    most_common = max(segment_distribution, key=segment_distribution.get) if segment_distribution else None

    # At-risk count
    at_risk_count = segment_distribution.get("At Risk Customers", 0)

    return {
        "total_customers":      total_customers,
        "total_predictions":    total_predictions,
        "segment_distribution": segment_distribution,
        "most_common_segment":  most_common,
        "at_risk_count":        at_risk_count,
    }