from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import Settings
from app.schemas.predictions import DiabetesPredictionRequest, EcommercePredictionRequest, HousePricePredictionRequest


logger = logging.getLogger(__name__)


class ModelContractError(RuntimeError):
    """Raised when application metadata disagrees with a saved pipeline."""


class PredictionError(RuntimeError):
    """Raised when a trusted local model cannot produce a prediction."""


class ModelService:
    def __init__(self, app_settings: Settings) -> None:
        self.settings = app_settings
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        definitions = {
            "diabetes": (
                self.settings.diabetes_model_path,
                self.settings.metadata_dir / "diabetes.json",
            ),
            "house_price": (
                self.settings.house_price_model_path,
                self.settings.metadata_dir / "house_price.json",
            ),
            "ecommerce": (
                self.settings.ecommerce_model_path,
                self.settings.metadata_dir / "ecommerce.json",
            ),
        }
        for key, (model_path, metadata_path) in definitions.items():
            if not model_path.is_file():
                raise FileNotFoundError(f"Required model file not found: {model_path}")
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Required model metadata not found: {metadata_path}")
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            model = joblib.load(model_path)
            self.verify_model_contract(key, model, metadata)
            self.models[key] = model
            self.metadata[key] = metadata
            logger.info(
                "Model loaded: name=%s path=%s expected_features=%s",
                key,
                model_path,
                metadata["expected_raw_features"],
            )

    @staticmethod
    def verify_model_contract(key: str, model: Any, metadata: dict[str, Any]) -> None:
        metadata_features = metadata.get("expected_raw_features", [])
        declared_features = [field["name"] for field in metadata.get("fields", [])]
        if declared_features != metadata_features:
            raise ModelContractError(
                f"{key}: metadata fields do not match expected_raw_features order"
            )

        saved_features = getattr(model, "feature_names_in_", None)
        if saved_features is not None and list(saved_features) != metadata_features:
            raise ModelContractError(
                f"{key}: saved pipeline features {list(saved_features)!r} do not match "
                f"metadata {metadata_features!r}"
            )

        if not hasattr(model, "predict") or not hasattr(model, "named_steps"):
            raise ModelContractError(f"{key}: saved object is not a fitted prediction Pipeline")

        if key == "house_price":
            preprocessor = model.named_steps.get("preprocessor")
            numeric_columns = list(preprocessor.transformers_[0][2])
            categorical_columns = list(preprocessor.transformers_[1][2])
            metadata_numeric = [f["name"] for f in metadata["fields"] if f["type"] == "number"]
            metadata_categorical = [f["name"] for f in metadata["fields"] if f["type"] == "categorical"]
            if numeric_columns != metadata_numeric or categorical_columns != metadata_categorical:
                raise ModelContractError(
                    f"{key}: metadata numeric/categorical groups disagree with ColumnTransformer"
                )
            encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
            if getattr(encoder, "handle_unknown", None) != "ignore":
                raise ModelContractError(f"{key}: unknown category policy is not 'ignore'")
            field_by_name = {field["name"]: field for field in metadata["fields"]}
            for feature_name, fitted_categories in zip(categorical_columns, encoder.categories_):
                if field_by_name[feature_name].get("options") != fitted_categories.tolist():
                    raise ModelContractError(
                        f"{key}: metadata options for {feature_name!r} disagree with fitted encoder"
                    )

    def get_metadata(self, key: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        if key is None:
            return [self.metadata["diabetes"], self.metadata["house_price"], self.metadata["ecommerce"]]
        return self.metadata[key]

    def predict_diabetes(self, payload: DiabetesPredictionRequest) -> dict[str, Any]:
        model = self.models["diabetes"]
        metadata = self.metadata["diabetes"]
        row = payload.model_dump(by_alias=True)
        frame = pd.DataFrame([row], columns=metadata["expected_raw_features"])
        try:
            prediction = int(model.predict(frame)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                positive_index = classes.index(1)
                probability = float(model.predict_proba(frame)[0][positive_index])
        except Exception as exc:
            logger.exception("Diabetes model prediction failed")
            raise PredictionError("The diabetes model could not produce a prediction") from exc
        return {
            "prediction": prediction,
            "label": metadata["class_labels"][str(prediction)],
            "probability": probability,
            "model": metadata["model"]["name"],
            "disclaimer": metadata["disclaimer"],
        }

    def predict_house_price(self, payload: HousePricePredictionRequest) -> dict[str, Any]:
        model = self.models["house_price"]
        metadata = self.metadata["house_price"]
        row = {
            key: (np.nan if value is None else value)
            for key, value in payload.model_dump(by_alias=True).items()
        }
        frame = pd.DataFrame([row], columns=metadata["expected_raw_features"])
        try:
            prediction = float(model.predict(frame)[0])
        except Exception as exc:
            logger.exception("House-price model prediction failed")
            raise PredictionError("The house-price model could not produce a prediction") from exc
        if not np.isfinite(prediction):
            raise PredictionError("The house-price model returned a non-finite prediction")
        unit = metadata["target"]["unit"]
        return {
            "predicted_price": prediction,
            "unit": unit,
            "formatted": f"{prediction:.2f} {unit}",
            "model": metadata["model"]["name"],
            "disclaimer": metadata["disclaimer"],
        }

    def predict_ecommerce(self, payload: EcommercePredictionRequest) -> dict[str, Any]:
        model = self.models["ecommerce"]
        metadata = self.metadata["ecommerce"]
        frame = pd.DataFrame([payload.model_dump(by_alias=True)], columns=metadata["expected_raw_features"])
        try:
            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
        except Exception as exc:
            logger.exception("Ecommerce model prediction failed")
            raise PredictionError("The ecommerce model could not produce a prediction") from exc
        return {
            "prediction": prediction,
            "label": metadata["class_labels"][str(prediction)],
            "probability": probability,
            "model": metadata["model"]["name"],
            "disclaimer": metadata["disclaimer"],
        }
