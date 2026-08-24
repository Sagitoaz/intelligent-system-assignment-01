from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)


class DiabetesPredictionRequest(StrictPayload):
    pregnancies: Annotated[int, Field(alias="Pregnancies", ge=0)]
    glucose: PositiveFiniteFloat = Field(alias="Glucose")
    blood_pressure: PositiveFiniteFloat = Field(alias="BloodPressure")
    bmi: PositiveFiniteFloat = Field(alias="BMI")
    diabetes_pedigree_function: NonNegativeFiniteFloat = Field(alias="DiabetesPedigreeFunction")
    age: Annotated[int, Field(alias="Age", gt=0)]


class HousePricePredictionRequest(StrictPayload):
    province: Annotated[str, Field(alias="Province", min_length=1)]
    area: PositiveFiniteFloat = Field(alias="Area")
    frontage: PositiveFiniteFloat | None = Field(alias="Frontage")
    access_road: PositiveFiniteFloat | None = Field(alias="Access Road")
    house_direction: str | None = Field(alias="House direction")
    balcony_direction: str | None = Field(alias="Balcony direction")
    floors: PositiveFiniteFloat | None = Field(alias="Floors")
    bedrooms: PositiveFiniteFloat | None = Field(alias="Bedrooms")
    bathrooms: PositiveFiniteFloat | None = Field(alias="Bathrooms")
    legal_status: str | None = Field(alias="Legal status")
    furniture_state: str | None = Field(alias="Furniture state")

    @field_validator(
        "province",
        "house_direction",
        "balcony_direction",
        "legal_status",
        "furniture_state",
    )
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value.strip() if value is not None else None


class DiabetesPredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float | None
    model: str
    disclaimer: str


class HousePricePredictionResponse(BaseModel):
    predicted_price: float
    unit: str
    formatted: str
    model: str
    disclaimer: str
