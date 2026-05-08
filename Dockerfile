# --- Base Python Layer ---
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directory for SQLite persistence
RUN mkdir -p /app/data && chmod 777 /app/data

# --- Production Layer ---
FROM base as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
EXPOSE 8001 8501

# Default command (orchestrator)
CMD ["python", "nexus_24_7.py"]
