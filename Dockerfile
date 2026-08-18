# syntax=docker/dockerfile:1

# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install system dependencies required for OpenCV, PyMuPDF, and health checks
# (Added libxcb1 as a safety net for any stray GUI dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── FIX: Force Headless OpenCV ─────────────────────────────────────────────
# Ultralytics aggressively installs standard 'opencv-python' as a dependency.
# We must explicitly uninstall it and replace it with the headless version.
RUN pip uninstall opencv-python opencv-contrib-python -y && \
    pip install --no-cache-dir opencv-python-headless

# Copy application source code and models
COPY . .

# Expose API port
EXPOSE 8000

# Health check to ensure the service is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the FastAPI application via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]