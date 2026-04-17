#!/bin/bash
# deploy.sh
# Institutional Quant Fund — Deployment & Setup Script

echo "===================================================="
echo "  MINI QUANT FUND — INSTITUTIONAL DEPLOYMENT"
echo "===================================================="

# 1. Check dependencies
echo "[1/4] Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.9+."
    exit 1
fi

# 2. Install library dependencies
echo "[2/4] Installing local dependencies..."
pip install -r requirements.txt --quiet

# 3. Docker build
echo "[3/4] Building Docker artifacts..."
if command -v docker-compose &> /dev/null; then
    docker-compose build
    echo "Docker build successful."
else
    echo "Docker compose not found. Skipping container build."
fi

# 4. Initialize working directories
echo "[4/4] Initializing data directories..."
mkdir -p output/backtests
mkdir -p data/parquet
mkdir -p logs

echo ""
echo "Deployment Ready!"
echo "----------------------------------------------------"
echo "To run a backtest:"
echo "  python main.py --mode backtest"
echo ""
echo "To launch the Registry Dashboard:"
echo "  streamlit run dashboard.py"
echo ""
echo "To launch via Docker:"
echo "  docker-compose up"
echo "----------------------------------------------------"
活跃
