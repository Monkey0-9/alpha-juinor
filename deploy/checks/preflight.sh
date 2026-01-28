#!/bin/bash
# deploy/checks/preflight.sh
# Wrapper for python preflight check

# Ensure python is available
if ! command -v python &> /dev/null; then
    echo "Python not found!"
    exit 1
fi

echo "Running Preflight Checks..."
python deploy/checks/preflight.py

if [ $? -eq 0 ]; then
    echo "Preflight Passed."
    exit 0
else
    echo "Preflight Failed."
    exit 1
fi
