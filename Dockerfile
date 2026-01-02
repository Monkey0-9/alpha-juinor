# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output/backtests

# Set environment variables (override these at runtime)
ENV PYTHONUNBUFFERED=1
ENV ALPACA_API_KEY=""
ENV ALPACA_SECRET_KEY=""
ENV ALERT_EMAIL=""
ENV ALERT_PASSWORD=""
ENV ALERT_TO=""

# Run main.py
CMD ["python", "main.py"]
