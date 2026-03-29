# syntax=docker/dockerfile:1
# Multi-stage: backend (FastAPI) + frontend (Vite dev server). See docker-compose.yml.

# --- Backend: FastAPI + ML stack ---
FROM python:3.12-slim AS backend

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY config ./config
COPY api ./api
COPY src ./src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Frontend: Vite + React (dev server for demo) ---
FROM node:20-alpine AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

EXPOSE 5173

# In Docker Compose, override to http://backend:8000 so the Vite proxy reaches the API service.
ENV VITE_API_PROXY_TARGET=http://backend:8000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
