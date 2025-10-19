"""Streamlit dashboard for the Smart Crop Recommendation system."""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import requests
import streamlit as st

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
MODEL_ACCURACY = 0.94  # Based on validation performance during training
EMOJIS = ("🌾", "🌽", "🌻", "🥕", "🍅", "🥒")

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Ensure Streamlit logging plays nicely with the root logger."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def call_backend(payload: Dict[str, float], endpoint: str) -> str:
    """Send a prediction request to the FastAPI backend."""

    try:
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        LOGGER.error("Connection error calling backend", exc_info=exc)
        st.error("Cannot reach the prediction API. Start the FastAPI server and retry.")
        return ""
    except requests.Timeout as exc:
        LOGGER.error("Timeout while waiting for backend", exc_info=exc)
        st.error("Prediction request timed out. Please try again.")
        return ""
    except requests.RequestException as exc:
        LOGGER.error("Unexpected backend error", exc_info=exc)
        st.error("Backend returned an error. Check server logs for details.")
        return ""

    try:
        data = response.json()
    except ValueError as exc:
        LOGGER.error("Failed to decode JSON response", exc_info=exc)
        st.error("Received an invalid response from the backend.")
        return ""

    prediction = data.get("prediction", "")
    if not prediction:
        st.warning("Backend did not include a prediction in the response.")
    return prediction


def render_sidebar(default_url: str) -> str:
    """Render the project details and backend configuration sidebar."""

    st.sidebar.title("Smart Crop Recommendation")
    st.sidebar.metric("Model Accuracy", f"{MODEL_ACCURACY * 100:.2f}%")
    st.sidebar.markdown("[GitHub Repository](https://github.com/harsh/crop_prediction_ai)")

    st.sidebar.divider()
    st.sidebar.subheader("Backend Endpoint")
    current_url = st.session_state.get("backend_url", default_url)
    st.sidebar.text_input("FastAPI URL", value=current_url, key="backend_url")
    return st.session_state["backend_url"]


def collect_user_inputs() -> Tuple[Dict[str, float], bool]:
    """Gather soil and climate parameters and submit when the user clicks."""

    st.subheader("Enter Field Measurements")
    with st.form("prediction-form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_val = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
            p_val = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
            k_val = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)

        with col2:
            temperature_val = st.number_input(
                "Temperature (°C)", min_value=-10.0, max_value=50.0, value=26.0, step=0.5
            )
            humidity_val = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            ph_val = st.number_input("Soil pH", min_value=3.0, max_value=9.5, value=6.5, step=0.1)

        with col3:
            rainfall_val = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=5.0)

        submitted = st.form_submit_button("Predict Crop")

    payload = {
        "N": float(n_val),
        "P": float(p_val),
        "K": float(k_val),
        "temperature": float(temperature_val),
        "humidity": float(humidity_val),
        "ph": float(ph_val),
        "rainfall": float(rainfall_val),
    }
    return payload, submitted


def main() -> None:
    """Boot the Streamlit dashboard."""

    configure_logging()
    st.set_page_config(page_title="Smart Crop Recommendation", layout="wide", page_icon="🌾")
    st.title("🌾 Smart Crop Recommendation Dashboard")
    st.caption("Input current soil conditions to discover the most suitable crop for your fields.")

    backend_url = render_sidebar(DEFAULT_BACKEND_URL)

    st.write("---")
    payload, submitted = collect_user_inputs()

    if submitted:
        with st.spinner("Contacting crop recommendation service..."):
            prediction = call_backend(payload, backend_url)

        if prediction:
            emoji = EMOJIS[abs(hash(prediction)) % len(EMOJIS)]
            st.success(f"{emoji} Recommended crop: **{prediction}**")
        else:
            st.info("Update the backend URL or retry once the service is available.")

    st.write("---")
    st.caption("Arduino sensor streaming will plug in here once the simulator is enabled in the backend.")


if __name__ == "__main__":
    main()
