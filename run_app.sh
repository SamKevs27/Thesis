#!/bin/bash
# Dance Grading System - Web UI Startup (Port 8080)

echo "🎯 Dance Grading System - Web UI"
echo "================================"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    exit 1
fi

source venv/bin/activate

if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q flask opencv-python mediapipe fastdtw scipy numpy
fi

echo ""
echo "✓ Starting Flask server on"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Open:  http://localhost:8080"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python app.py
