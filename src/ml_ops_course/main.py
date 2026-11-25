import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ml_ops_course.schemas import ClientFeatures, HealthResponse, PredictionResponse

state: dict = {}


def load_settings() -> tuple[Path, str]:
    model_path = os.getenv("MODEL_PATH")
    model_version = os.getenv("MODEL_VERSION")
    if not model_path or not model_version:
        raise RuntimeError(
            "Не заданы MODEL_PATH и/или MODEL_VERSION. "
            "Скопируйте .env.example в .env и заполните значения."
        )
    if not Path(model_path).exists():
        raise RuntimeError(f"Файл модели не найден: {model_path}")
    return Path(model_path), model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    path, version = load_settings()
    state["model"] = joblib.load(path)
    state["version"] = version
    yield
    state.clear()


app = FastAPI(
    title="Bank Marketing inference",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=state.get("version", "unknown"),
        model_loaded="model" in state,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures) -> PredictionResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    X = pd.DataFrame([features.model_dump()])
    proba = float(state["model"].predict_proba(X)[0, 1])
    return PredictionResponse(
        probability=round(proba, 4),
        label="yes" if proba >= 0.5 else "no",
        model_version=state["version"],
    )
