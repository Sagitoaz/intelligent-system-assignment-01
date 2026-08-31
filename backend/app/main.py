from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.services.model_service import ModelService, PredictionError
from app.services.neo4j_service import Neo4jService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(settings)
    model_service.load()
    neo4j_service = Neo4jService(settings)
    neo4j_service.connect()
    app.state.model_service = model_service
    app.state.neo4j_service = neo4j_service
    yield
    neo4j_service.close()


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Local inference API for three Assignment 02 fitted scikit-learn Pipelines.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_request_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "Request body is too large"},
                )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
    return await call_next(request)


@app.exception_handler(PredictionError)
async def prediction_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Do not echo invalid values: apart from reducing data exposure, non-finite
    # JSON numbers cannot be safely serialized in an error response.
    details = [
        {
            "type": error.get("type", "validation_error"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid input"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected API failure", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Unexpected internal server error"})


app.include_router(router)
