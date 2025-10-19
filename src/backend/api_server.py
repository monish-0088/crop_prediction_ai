"""FastAPI application providing crop recommendations from simulated sensor data."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)
MODEL_ARTIFACT_PATH = Path("src/models/best_model.pkl")

app = FastAPI(title="Crop Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_label_encoder = None
_feature_columns: Tuple[str, ...] = tuple()
_scaling_params: Dict[str, Tuple[float, float]] = {}
_simulation_task: asyncio.Task | None = None


class PredictionRequest(BaseModel):
    """Input schema for crop recommendation requests."""

    N: float = Field(..., description="Nitrogen content", ge=0)
    P: float = Field(..., description="Phosphorus content", ge=0)
    K: float = Field(..., description="Potassium content", ge=0)
    temperature: float = Field(..., description="Ambient temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity percentage", ge=0)
    ph: float = Field(..., description="Soil pH level", ge=0)
    rainfall: float = Field(..., description="Rainfall in mm")


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_model_bundle(artifact_path: Path = MODEL_ARTIFACT_PATH) -> None:
    """Load the persisted model bundle into module-level state."""

    global _model, _label_encoder, _feature_columns, _scaling_params

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact missing at {artifact_path.resolve()}")

    bundle: Dict[str, Any] = joblib.load(artifact_path)

    _model = bundle["model"]
    _label_encoder = bundle["label_encoder"]
    _feature_columns = tuple(bundle["feature_columns"])
    _scaling_params = dict(bundle.get("scaling", {}))

    missing_scalers = [col for col in _feature_columns if col not in _scaling_params]
    if missing_scalers:
        raise ValueError(f"Scaling parameters missing for columns: {', '.join(missing_scalers)}")

    LOGGER.info(
        "Loaded model bundle with %d feature columns", len(_feature_columns)
    )


def _normalize_payload(payload: Dict[str, float]) -> np.ndarray:
    """Normalize incoming payload using stored scaling parameters."""

    if not _feature_columns:
        raise RuntimeError("Model features not initialised")

    normalized_values = []
    for column in _feature_columns:
        if column not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {column}")

        raw_value = float(payload[column])
        min_val, max_val = _scaling_params[column]
        if max_val == min_val:
            normalized = 0.0
        else:
            normalized = (raw_value - min_val) / (max_val - min_val)
        normalized_values.append(normalized)

    return np.asarray([normalized_values], dtype=np.float32)


async def simulated_arduino_stream(interval_seconds: int = 5) -> None:
    """Continuously emit simulated sensor readings for observability."""

    while True:
        reading = {
            "N": random.uniform(20, 140),
            "P": random.uniform(10, 90),
            "K": random.uniform(10, 90),
            "temperature": random.uniform(15, 38),
            "humidity": random.uniform(50, 95),
            "ph": random.uniform(4.5, 8.5),
            "rainfall": random.uniform(50, 300),
        }
        try:
            if _model is None or _label_encoder is None:
                LOGGER.warning("Simulation skipped; model not initialised")
                await asyncio.sleep(interval_seconds)
                continue
            features = _normalize_payload(reading)
            prediction = _model.predict(features)
            crop = _label_encoder.inverse_transform(prediction)[0]
            LOGGER.info("Simulated reading %s -> %s", reading, crop)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Simulation error: %s", exc)
        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise logging, load the model, and start simulation."""

    configure_logging()
    load_model_bundle()

    global _simulation_task
    _simulation_task = asyncio.create_task(simulated_arduino_stream())
    LOGGER.info("Simulation task started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Cancel any background simulation tasks."""

    if _simulation_task:
        _simulation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _simulation_task
        LOGGER.info("Simulation task stopped")


@app.post("/predict")
async def predict(payload: PredictionRequest) -> Dict[str, str]:
    """Predict the optimal crop for provided sensor readings."""

    if _model is None or _label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    features = _normalize_payload(payload.model_dump())
    try:
        encoded_prediction = _model.predict(features)
        crop = _label_encoder.inverse_transform(encoded_prediction)[0]
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Prediction failure: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return {"prediction": crop}


@app.get("/")
async def root() -> Dict[str, str]:
    """Simple health endpoint with pointer to the prediction route."""

    return {
        "status": "ok",
        "message": "Use POST /predict with soil and climate fields to get a crop recommendation.",
    }
