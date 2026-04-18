#!/bin/bash
# ============================================================================
# NEXUS INSTITUTIONAL v0.3.0 - 24/7 LINUX SYSTEMD SETUP
# ============================================================================
# This script sets up systemd to run Nexus 24/7 on Linux/macOS

set -e

PROJECT_DIR="/home/$(whoami)/mini-quant-fund"
PYTHON_BIN=$(which python3 || which python)
SERVICE_NAME="nexus-institutional-24x7"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "========================================================================="
echo "  NEXUS INSTITUTIONAL 24/7 - LINUX SYSTEMD SETUP"
echo "========================================================================="
echo ""
echo "Configuration:"
echo "  Project Directory: $PROJECT_DIR"
echo "  Python Binary: $PYTHON_BIN"
echo "  Service Name: $SERVICE_NAME"
echo "  Service File: $SERVICE_FILE"
echo ""

# Check if running as root for system-wide installation
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Not running as root. Installing user-level service instead..."
    SERVICE_FILE="/home/$(whoami)/.config/systemd/user/${SERVICE_NAME}.service"
    mkdir -p "$(dirname $SERVICE_FILE)"
    USER_LEVEL=1
    SYSTEMCTL_CMD="systemctl --user"
else
    USER_LEVEL=0
    SYSTEMCTL_CMD="systemctl"
fi

# Create systemd service file
echo "Creating systemd service file: $SERVICE_FILE"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Nexus Institutional 24/7 Continuous Trading Platform
Documentation=file://$PROJECT_DIR/INSTITUTIONAL_DEPLOYMENT_GUIDE.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN $PROJECT_DIR/run_24_7.py --mode backtest --asset-class multi --venues 235
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/nexus_24_7_systemd.log
StandardError=append:$PROJECT_DIR/logs/nexus_24_7_systemd_error.log
Environment="PATH=$PROJECT_DIR/.venv/bin:/usr/local/bin:/usr/bin"

[Install]
WantedBy=$(if [ $USER_LEVEL -eq 1 ]; then echo "default.target"; else echo "multi-user.target"; fi)
EOF

echo "[✓] Service file created"
echo ""

# Reload systemd daemon
echo "Reloading systemd daemon..."
$SYSTEMCTL_CMD daemon-reload
echo "[✓] Daemon reloaded"
echo ""

# Enable service to start on boot
echo "Enabling service to start on boot..."
$SYSTEMCTL_CMD enable "${SERVICE_NAME}.service"
echo "[✓] Service enabled"
echo ""

# Start service
echo "Starting service..."
$SYSTEMCTL_CMD start "${SERVICE_NAME}.service"
echo "[✓] Service started"
echo ""

# Verify service status
echo "Service status:"
$SYSTEMCTL_CMD status "${SERVICE_NAME}.service" || true
echo ""

# Show logs
echo "Real-time logs (press Ctrl+C to stop):"
sleep 2
$SYSTEMCTL_CMD -e --no-pager --lines=50 STATUS "${SERVICE_NAME}.service" -u "${SERVICE_NAME}.service" || journalctl -u "${SERVICE_NAME}.service" -f --no-pager || echo "Waiting for service to generate logs..."

echo ""
echo "========================================================================="
echo "  SETUP COMPLETE"
echo "========================================================================="
echo ""
echo "✓ Nexus Institutional will run 24/7 on system startup"
echo ""
echo "Useful commands:"
echo "  View status:        $SYSTEMCTL_CMD status ${SERVICE_NAME}"
echo "  View logs:          journalctl -u ${SERVICE_NAME} -f"
echo "  Stop service:       $SYSTEMCTL_CMD stop ${SERVICE_NAME}"
echo "  Restart service:    $SYSTEMCTL_CMD restart ${SERVICE_NAME}"
echo "  Disable on boot:    $SYSTEMCTL_CMD disable ${SERVICE_NAME}"
echo "  Delete service:     rm $SERVICE_FILE && $SYSTEMCTL_CMD daemon-reload"
echo ""
echo "Log locations:"
echo "  Main logs:          $PROJECT_DIR/logs/*.log"
echo "  Systemd logs:       journalctl -u ${SERVICE_NAME}"
echo ""
