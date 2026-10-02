# Multi-stage build: React dashboard + FastAPI backend in one image.
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY qadapt/ qadapt/
RUN pip install --no-cache-dir ".[quantum,api]"
COPY --from=ui /ui/dist frontend/dist
ENV QADAPT_DB=/app/data/qadapt.db
EXPOSE 8000
CMD ["uvicorn", "qadapt.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
