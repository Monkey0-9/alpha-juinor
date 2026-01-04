# Multi-stage Dockerfile for Institutional Quant Fund
# Phase 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Phase 2: Runtime
FROM python:3.11-slim-bookworm as runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure packages are in PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Application metadata
LABEL maintainer="Institutional Trading Team"
LABEL version="2.0.0"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output/backtests logs

# Entrypoint to allow passing arguments to main.py
ENTRYPOINT ["python", "main.py"]
