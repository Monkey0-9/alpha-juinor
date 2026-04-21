#!/bin/bash
# Alpha Junior - Quick Start Script for Linux/Mac

set -e

echo "=========================================="
echo "   Alpha Junior - Starting Platform"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}[INFO]${NC} Creating .env from template..."
    cp .env.example .env
    echo -e "${YELLOW}[WARN]${NC} Please edit .env file with your API keys before running!"
    exit 1
fi

echo -e "${GREEN}[1/5]${NC} Starting PostgreSQL..."
docker-compose up -d postgres

echo -e "${GREEN}[2/5]${NC} Starting Redis..."
docker-compose up -d redis

echo -e "${GREEN}[3/5]${NC} Waiting for database..."
sleep 5

echo -e "${GREEN}[4/5]${NC} Starting Backend API..."
docker-compose up -d backend celery_worker celery_beat

echo -e "${GREEN}[5/5]${NC} Starting Frontend..."
docker-compose up -d frontend

echo ""
echo "=========================================="
echo "   Alpha Junior is RUNNING!"
echo "=========================================="
echo ""
echo -e "${GREEN}Frontend:${NC}     http://localhost:3000"
echo -e "${GREEN}API Docs:${NC}     http://localhost:8000/api/v1/docs"
echo -e "${GREEN}Flower:${NC}       http://localhost:5555"
echo -e "${GREEN}Prometheus:${NC}   http://localhost:9090"
echo -e "${GREEN}Grafana:${NC}      http://localhost:3001 (admin/admin)"
echo ""
echo "View logs: docker-compose logs -f"
echo "Stop:      docker-compose down"
