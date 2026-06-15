"""
app/core/model.py
─────────────────
ML Model Loader + Recommendation Engine using the Singleton pattern.

WHY SINGLETON: Loading .pkl files from disk takes time.
Loading once at startup and reusing for every request is standard
practice for ML serving at scale.
"""

import json
import joblib
import numpy as np
import pandas as pd
from app.core.config import settings


# ─── Recommendation Engine ────────────────────────────────────────────────────
# Maps each business segment to a concrete, actionable recommendation.
RECOMMENDATIONS: dict[str, str] = {
    "High Value Customers": (
        "Invite to an exclusive VIP loyalty program. "
        "Offer early access to new arrivals and premium membership perks."
    ),
    "Loyal Customers": (
        "Send a personalized thank-you message. "
        "Offer a 10% loyalty discount on their next visit to reinforce retention."
    ),
    "Budget Shoppers": (
        "Send weekend flash sale and bundle discount notifications. "
        "Highlight value-for-money offers and seasonal promotions."
    ),
    "At Risk Customers": (
        "Launch a win-back campaign immediately. "
        "Offer a 20% re-engagement coupon with a limited-time expiry."
    ),
    "Potential Targets": (
        "Upsell premium membership and exclusive collections. "
        "Showcase high-end products aligned with their income bracket."
    ),
}

# ─── Segment Descriptions ─────────────────────────────────────────────────────
CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "High Value Customers":  "High income, high spending, frequent buyer — your most valuable segment.",
    "Loyal Customers":       "Consistent, steady shoppers — reliable revenue base.",
    "Budget Shoppers":       "Low income but high enthusiasm — deal-driven and frequent visitors.",
    "At Risk Customers":     "Declining engagement and low spending — at risk of churn.",
    "Potential Targets":     "High income but low spending — significant upsell opportunity.",
}


# ─── Singleton Model Manager ──────────────────────────────────────────────────
class ModelManager:
    """
    Holds the loaded KMeans model, scaler, and cluster label map.
    Loads from disk once at startup; reused for every prediction request.
    """

    _instance = None
    _model    = None
    _scaler   = None
    _cluster_labels: dict[int, str] = {}

    # Feature order must match exactly what was used during training
    FEATURE_COLUMNS = [
        "Age",
        "Annual_Income",
        "Spending_Score",
        "Purchase Frequency",
        "Gender_Encoded",
    ]

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load model, scaler, and cluster label map from disk — runs only once."""
        self._model  = joblib.load(settings.model_path)
        self._scaler = joblib.load(settings.scaler_path)

        with open(settings.cluster_labels_path, "r") as f:
            raw = json.load(f)
            # JSON keys are strings; convert to int
            self._cluster_labels = {int(k): v for k, v in raw.items()}

        print(f"[OK] Model loaded from         : {settings.model_path}")
        print(f"[OK] Scaler loaded from        : {settings.scaler_path}")
        print(f"[OK] Cluster labels loaded from: {settings.cluster_labels_path}")
        print(f"[OK] Cluster labels: {self._cluster_labels}")

    def predict(
        self,
        age: int,
        gender_encoded: int,
        annual_income: float,
        spending_score: float,
        purchase_frequency: int,
    ) -> dict:
        """
        Run prediction for a single customer.

        Args:
            age:                Customer age
            gender_encoded:     0 = Female, 1 = Male
            annual_income:      Annual income in ₹ Lakhs
            spending_score:     Spending score 1–100
            purchase_frequency: Shopping visits per year

        Returns:
            dict with cluster_id, segment_label, segment_description, recommendation
        """
        # Build DataFrame with correct feature names (avoids sklearn warning)
        features_df = pd.DataFrame([{
            "Age":               age,
            "Annual_Income":     annual_income,
            "Spending_Score":    spending_score,
            "Purchase Frequency": purchase_frequency,
            "Gender_Encoded":    gender_encoded,
        }])

        # Scale using the fitted scaler
        features_scaled = self._scaler.transform(features_df)

        # Predict cluster
        cluster_id: int = int(self._model.predict(features_scaled)[0])

        # Map to business label
        segment_label = self._cluster_labels.get(cluster_id, "Unknown")

        return {
            "cluster_id":          cluster_id,
            "segment_label":       segment_label,
            "segment_description": CLUSTER_DESCRIPTIONS.get(segment_label, ""),
            "recommendation":      RECOMMENDATIONS.get(segment_label, "No recommendation available."),
        }


# ─── Convenience function ─────────────────────────────────────────────────────
def get_model() -> ModelManager:
    """FastAPI dependency — returns the singleton model manager."""
    return ModelManager.get_instance()
