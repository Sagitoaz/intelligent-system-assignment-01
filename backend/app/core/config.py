from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env", override=True)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "Intelligent System Development Assignment 02 API"
    api_prefix: str = "/api/v1"
    project_root: Path = PROJECT_ROOT
    diabetes_model_path: Path = PROJECT_ROOT / "models" / "diabetes" / "diabetes_model.joblib"
    house_price_model_path: Path = PROJECT_ROOT / "models" / "house_price" / "house_price_model.joblib"
    ecommerce_model_path: Path = PROJECT_ROOT / "models" / "ecommerce" / "ecommerce_interest_model.joblib"
    metadata_dir: Path = PROJECT_ROOT / "backend" / "app" / "model_metadata"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_enabled: bool = _as_bool(os.getenv("NEO4J_ENABLED"), True)
    cors_origins_raw: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    max_request_bytes: int = int(os.getenv("MAX_REQUEST_BYTES", str(1024 * 1024)))

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


settings = Settings()
