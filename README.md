# Crop Prediction AI System (Arduino + ML + Streamlit)

## 🧠 Overview
- AI-driven crop recommendation engine that analyzes soil chemistry and climate metrics to suggest optimal crops.
- Integrates data ingestion, preprocessing, and model inference into a Streamlit dashboard for rapid insights.

## 🧩 Tech Stack
- `Python 3.13`
- `Streamlit`
- `scikit-learn`
- `XGBoost`
- `LightGBM`
- `pandas`
- `numpy`

## 📂 Folder Structure
- `data/` – Raw and processed datasets (e.g., `crop_data.csv`).
- `notebooks/` – Exploratory analysis and preprocessing notebooks.
- `src/backend/` – API services, data ingestion pipelines, and model serving utilities.
- `src/dashboard/` – Streamlit UI (`app.py`) plus shared dashboard helpers.
- `src/models/` – Training scripts, evaluation routines, and serialized artifacts.
- `src/arduino/` – Serial emulator scaffolding and future Arduino integrations.
- `docs/` – Architecture notes, diagrams, and API references.
- `scripts/` – Automation helpers for data sync, deployment, and maintenance.
- `tests/` – Unit/integration tests for preprocessing, inference, and serial IO.
- `.github/` – CI workflows and repository health files.

## ⚙️ Setup Instructions
1. Create and activate a virtual environment (Windows PowerShell):
	```powershell
	python -m venv crop_ai_env
	crop_ai_env\Scripts\activate
	```
2. Install dependencies:
	```powershell
	pip install -r requirements.txt
	```
3. Train the model (generates `src/models/best_model.pkl`):
	```powershell
	python -m src.model_trainer
	```
4. Start the FastAPI backend:
	```powershell
	uvicorn src.backend.api_server:app --reload
	```
5. In a second terminal, launch the Streamlit dashboard:
	```powershell
	streamlit run src/dashboard/app.py
	```

## 📊 Dataset
- Source: [Kaggle Crop Recommendation Data](https://www.kaggle.com/datasets/aksahaha/crop-recommendation/data)
- Place the downloaded CSV at `data/Crop_recommendation.csv`; the pipeline assumes this path.

## 🤖 ML Pipeline
- Preprocess soil nutrient and climate variables using `pandas` transformations.
- Train baseline models with `RandomForestClassifier`, `XGBoost`, and `LightGBM`.
- Evaluate performance, calibrate thresholds, and persist best models via `joblib` for downstream inference.

## 🧪 Simulated Arduino
- Real hardware integration is pending; `src/arduino/serial_emulator.py` mocks sensor payloads to unblock development and testing.

## 🧰 Contributing
- Fork the repository, then create a feature branch from `main`.
- Commit changes with meaningful messages and include tests where relevant.
- Submit a pull request summarizing changes, testing results, and context.

## 🧼 License
- Licensed under the MIT License. Review `LICENSE` for details.

## 🚀 Container Deployment
- Build and run the backend + dashboard together using Docker Compose:
	```powershell
	docker compose up --build
	```
- The FastAPI API becomes available at `http://localhost:8000`, and the Streamlit dashboard at `http://localhost:8501`.
- Pass a custom backend URL to Streamlit by setting `BACKEND_URL` (already configured as `http://backend:8000/predict` in `docker-compose.yml`).

## ✅ Future Work
- Integrate actual Arduino sensor firmware and serial ingestion.
- Enable real-time telemetry streaming and alerting.
- Deploy the container stack to a cloud host (Render, Railway, Azure App Service, etc.).

