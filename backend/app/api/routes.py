from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.graph import GraphResponse
from app.schemas.predictions import (
    DiabetesPredictionRequest,
    DiabetesPredictionResponse,
    EcommercePredictionRequest,
    EcommercePredictionResponse,
    HousePricePredictionRequest,
    HousePricePredictionResponse,
)
from app.services.neo4j_service import KnowledgeGraphUnavailable


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    model_service = request.app.state.model_service
    return {
        "status": "ok",
        "models": {
            "diabetes": "loaded" if "diabetes" in model_service.models else "unavailable",
            "house_price": "loaded" if "house_price" in model_service.models else "unavailable",
            "ecommerce": "loaded" if "ecommerce" in model_service.models else "unavailable",
        },
        "neo4j": request.app.state.neo4j_service.status,
    }


@router.get("/api/v1/models")
def list_models(request: Request) -> dict:
    return {"models": request.app.state.model_service.get_metadata()}


@router.get("/api/v1/models/diabetes")
def diabetes_metadata(request: Request) -> dict:
    return request.app.state.model_service.get_metadata("diabetes")


@router.get("/api/v1/models/house-price")
def house_price_metadata(request: Request) -> dict:
    return request.app.state.model_service.get_metadata("house_price")


@router.get("/api/v1/models/ecommerce")
def ecommerce_metadata(request: Request) -> dict:
    return request.app.state.model_service.get_metadata("ecommerce")


@router.post("/api/v1/diabetes/predict", response_model=DiabetesPredictionResponse)
def predict_diabetes(payload: DiabetesPredictionRequest, request: Request) -> dict:
    return request.app.state.model_service.predict_diabetes(payload)


@router.post("/api/v1/house-price/predict", response_model=HousePricePredictionResponse)
def predict_house_price(payload: HousePricePredictionRequest, request: Request) -> dict:
    return request.app.state.model_service.predict_house_price(payload)


@router.post("/api/v1/ecommerce/predict", response_model=EcommercePredictionResponse)
def predict_ecommerce(payload: EcommercePredictionRequest, request: Request) -> dict:
    return request.app.state.model_service.predict_ecommerce(payload)


@router.get("/api/v1/diabetes/knowledge-graph", response_model=GraphResponse)
def diabetes_knowledge_graph(request: Request) -> dict:
    try:
        return request.app.state.neo4j_service.get_diabetes_graph()
    except KnowledgeGraphUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
