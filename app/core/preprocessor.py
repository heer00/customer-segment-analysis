"""
app/core/preprocessor.py
────────────────────────
Preprocessing layer — sits between the API and the ML model.

Responsibilities:
  1. Handle missing or null values
  2. Encode categorical variables (Gender)
  3. Validate input ranges
  4. Scale features using the saved scaler

WHY: The model expects clean, scaled, encoded data. If raw input
comes in dirty (nulls, wrong types, out-of-range), the model either
crashes or gives wrong predictions. This layer shields the model.
"""

import joblib
import numpy as np
from app.core.config import settings


# Default fallback values when a field is missing
DEFAULTS = {
    "age": 35,                  # median age
    "gender": "Female",         # most common in dataset
    "annual_income": 60.0,      # median income (₹ in Lakhs)
    "spending_score": 50.0,     # middle of range
    "purchase_frequency": 6,    # moderate frequency
}

# Valid input ranges for validation
RANGES = {
    "age": (18, 80),
    "annual_income": (0, 200),
    "spending_score": (1, 100),
    "purchase_frequency": (1, 52),
}


class Preprocessor:
    """
    Loads the saved scaler and label encoder.
    Transforms raw customer input into a model-ready numpy array.
    """

    _instance = None
    _scaler = None
    _label_encoder = None

    @classmethod
    def get_instance(cls) -> "Preprocessor":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        self._scaler = joblib.load(settings.scaler_path)
        self._label_encoder = joblib.load(settings.label_encoder_path)
        print("[OK] Preprocessor loaded scaler and label encoder")

    def _fill_missing(self, data: dict) -> dict:
        """Fill any missing fields with sensible defaults."""
        filled = {}
        for field, default in DEFAULTS.items():
            value = data.get(field)
            if value is None or value == "":
                filled[field] = default
                print(f"[PREPROCESSOR] Missing '{field}' → using default: {default}")
            else:
                filled[field] = value
        return filled

    def _validate_ranges(self, data: dict) -> list[str]:
        """Return list of validation error messages."""
        errors = []
        for field, (min_val, max_val) in RANGES.items():
            value = data.get(field)
            if value is not None and not (min_val <= float(value) <= max_val):
                errors.append(
                    f"'{field}' value {value} is out of range ({min_val}–{max_val})"
                )
        return errors

    def _encode_gender(self, gender: str) -> int:
        """Encode Male/Female → 1/0 using the saved label encoder."""
        try:
            return int(self._label_encoder.transform([gender])[0])
        except Exception:
            # Default to Female (0) if unknown value passed
            return 0

    def process(self, raw_input: dict) -> tuple[np.ndarray, list[str]]:
        """
        Main method: takes raw API input dict, returns scaled numpy array.

        Returns:
            (features_scaled, warnings)
            features_scaled → ready to pass directly into model.predict()
            warnings        → list of any fields that were defaulted
        """
        warnings = []

        # Step 1: Fill missing values
        data = self._fill_missing(raw_input)
        missing_fields = [k for k in DEFAULTS if raw_input.get(k) is None]
        if missing_fields:
            warnings.append(f"Missing fields filled with defaults: {missing_fields}")

        # Step 2: Validate ranges
        errors = self._validate_ranges(data)
        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        # Step 3: Encode gender
        gender_encoded = self._encode_gender(str(data["gender"]))

        # Step 4: Build feature array in correct order
        # Order must match training: Age, Annual_Income, Spending_Score,
        #                            Purchase Frequency, Gender_Encoded
        features = np.array([[
            float(data["age"]),
            float(data["annual_income"]),
            float(data["spending_score"]),
            float(data["purchase_frequency"]),
            float(gender_encoded),
        ]])

        # Step 5: Scale
        features_scaled = self._scaler.transform(features)

        return features_scaled, warnings


def get_preprocessor() -> Preprocessor:
    """FastAPI dependency — returns the singleton preprocessor."""
    return Preprocessor.get_instance()