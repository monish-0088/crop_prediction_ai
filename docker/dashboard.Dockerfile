# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential libgomp1 \ 
    && rm -rf /var/lib/apt/lists/* \ 
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BACKEND_URL="http://localhost:8000/predict"

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
