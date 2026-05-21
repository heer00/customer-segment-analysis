"""
app/core/model.py
─────────────────
ML Model Loader using the Singleton pattern.

WHY SINGLETON: Loading a .pkl file from disk takes ~100–500ms.
If we loaded it on every API request, a server handling 100 req/sec
would waste 10–50 seconds per second just loading files.

The Singleton ensures the model is loaded ONCE at server startup
and reused for every prediction request. This is standard practice
at companies like Uber, Airbnb, and Netflix for ML serving.
"""

import joblib
import numpy as np
from app.core.config import settings


# ─── Business Labels ──────────────────────────────────────────────────────────
# Maps numeric KMeans cluster IDs → human-readable business segment names.
# These match what was defined in the original app.py but are now centralized.
CLUSTER_LABELS: dict[int, str] = {
    0: "Premium Customers",
    1: "Careful Spenders",
    2: "Budget Shoppers",
    3: "Low Engagement",
    4: "Average Customers",
}

CLUSTER_DESCRIPTIONS: dict[int, str] = {
    0: "High Income, High Spending — your most valuable customers",
    1: "High Income, Low Spending — potential upsell targets",
    2: "Low Income, High Spending — loyal but price-sensitive",
    3: "Low Income, Low Spending — at risk of churn",
    4: "Medium Income, Medium Spending — the stable core segment",
}


# ─── Singleton Model Manager ─────────────────────────────────────────────────
class ModelManager:
    """
    Holds the loaded KMeans model and scaler as class-level attributes.
    First call to get_instance() loads from disk. All subsequent calls
    return the already-loaded objects immediately.
    """

    _instance = None
    _model = None
    _scaler = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load model and scaler from disk — runs only once."""
        self._model = joblib.load(settings.model_path)
        self._scaler = joblib.load(settings.scaler_path)
        print(f"[OK] Model loaded from  : {settings.model_path}")
        print(f"[OK] Scaler loaded from : {settings.scaler_path}")

    def predict(self, annual_income: float, spending_score: float) -> dict:
        """
        Run prediction for a single customer.

        Args:
            annual_income:  Annual income in k$ (e.g., 75 means $75,000)
            spending_score: Spending score 1–100

        Returns:
            dict with cluster_id, segment_label, segment_description
        """
        # Reshape to 2D array as scikit-learn expects
        features = np.array([[annual_income, spending_score]])

        # Scale features using the fitted scaler (same as training)
        features_scaled = self._scaler.transform(features)

        # Predict cluster
        cluster_id: int = int(self._model.predict(features_scaled)[0])

        return {
            "cluster_id": cluster_id,
            "segment_label": CLUSTER_LABELS.get(cluster_id, "Unknown"),
            "segment_description": CLUSTER_DESCRIPTIONS.get(cluster_id, ""),
        }


# ─── Convenience function ─────────────────────────────────────────────────────
def get_model() -> ModelManager:
    """FastAPI dependency — returns the singleton model manager."""
    return ModelManager.get_instance()
