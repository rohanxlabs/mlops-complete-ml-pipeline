# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install only runtime dependencies (no dev/test extras)
COPY requirements-prod.txt ./requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# ── Application files ─────────────────────────────────────────────────────────
COPY api/      api/
COPY models/   models/
COPY src/config.py src/config.py

# Expose API port
EXPOSE 8000

# Run with Uvicorn (use Gunicorn + UvicornWorker for multi-worker production)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
