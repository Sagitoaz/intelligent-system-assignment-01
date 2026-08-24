from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ["NEO4J_ENABLED"] = "false"

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def diabetes_demo_case() -> dict:
    return {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "BMI": 24.0,
        "DiabetesPedigreeFunction": 0.20,
        "Age": 23,
    }


@pytest.fixture
def house_demo_case() -> dict:
    return {
        "Province": "Hồ Chí Minh",
        "Area": 45.0,
        "Frontage": 4.0,
        "Access Road": 4.0,
        "House direction": "Đông - Nam",
        "Balcony direction": "Đông - Nam",
        "Floors": 3.0,
        "Bedrooms": 3.0,
        "Bathrooms": 3.0,
        "Legal status": "Have certificate",
        "Furniture state": "Full",
    }
