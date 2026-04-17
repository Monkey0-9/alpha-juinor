# Institutional-grade Dockerfile for Nexus Quant Platform
FROM python:3.11-slim-bullseye

# System dependencies for numerical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY main.py /app/
COPY config/ /app/config/

# Environment setup
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Initialize local directories for data and logs
RUN mkdir -p /app/data/parquet /app/logs

# Run the engine
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "sim"]
