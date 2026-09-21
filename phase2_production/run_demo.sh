#!/bin/bash

echo "========================================"
echo "   VERA - DR Detection Demo Launcher"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

echo "[1/3] Checking dependencies..."
python3 -c "import streamlit" &> /dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Dependencies not installed. Installing now..."
    echo "This may take 2-3 minutes..."
    echo ""
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install dependencies"
        echo "Try manually: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "[2/3] Starting VERA demo..."
echo ""
echo "The browser will open automatically at http://localhost:8501"
echo ""
echo "To stop the demo: Press Ctrl+C"
echo ""

echo "[3/3] Launching..."
streamlit run demo_app.py
