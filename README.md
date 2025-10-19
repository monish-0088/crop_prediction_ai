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
3. Launch the Streamlit dashboard:
	```powershell
	streamlit run src/dashboard/app.py
	```

## 📊 Dataset
- Source: [Kaggle Crop Recommendation Data](https://www.kaggle.com/datasets/aksahaha/crop-recommendation/data)
- Place the downloaded CSV at `data/crop_data.csv`; the pipeline assumes this path.

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

## ✅ Future Work
- Integrate actual Arduino sensor firmware and serial ingestion.
- Enable real-time telemetry streaming and alerting.
- Package the dashboard and backend for cloud deployment.

