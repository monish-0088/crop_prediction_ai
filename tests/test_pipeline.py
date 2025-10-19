"""Integration-style tests covering data pipeline, model artifact, and API responses."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable when tests run without PYTHONPATH configured.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import data_loader
from src.backend.api_server import app, load_model_bundle

DATASET_PATH = Path("data/Crop_recommendation.csv")
MODEL_PATH = Path("src/models/best_model.pkl")


@pytest.fixture(scope="session", autouse=True)
def _load_model_once() -> None:
    """Ensure the model bundle is loaded before running API tests."""

    load_model_bundle()


def test_dataset_loads_and_preprocesses() -> None:
    """The dataset should load without errors and produce normalized columns."""

    raw_df = data_loader.load_dataset(DATASET_PATH)
    processed_df, scaling = data_loader.preprocess_data(raw_df, return_scaler=True)

    assert not raw_df.empty, "Raw dataset unexpectedly empty"
    assert not processed_df.empty, "Processed dataset unexpectedly empty"
    assert set(data_loader._NUMERIC_COLUMNS).issubset(processed_df.columns)
    for column, (min_val, max_val) in scaling.items():
        assert max_val >= min_val, f"Invalid scaling bounds for {column}"


def test_model_artifact_exists() -> None:
    """Persisted model bundle should exist and be readable."""

    assert MODEL_PATH.exists(), "Model artifact missing; run model trainer first"


def test_api_predict_endpoint_returns_crop() -> None:
    """Calling /predict with valid payload should return a crop label."""

    client = TestClient(app)
    payload = {
        "N": 90,
        "P": 40,
        "K": 40,
        "temperature": 26,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 200,
    }

    response = client.post("/predict", json=payload, timeout=10)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "prediction" in data, "Response missing 'prediction' key"
    assert isinstance(data["prediction"], str) and data["prediction"], "Prediction should be a non-empty string"
