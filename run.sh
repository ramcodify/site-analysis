#!/usr/bin/env bash
# ==============================================================================
# BuildSight AI — Single Command Full Stack Launcher
# Runs MongoDB verification, Backend (FastAPI :8000), and Frontend (Vite :5173)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/buildsight-ai/backend"
FRONTEND_DIR="$SCRIPT_DIR/buildsight-ai/frontend"

# Fallback if script is run inside buildsight-ai
if [ ! -d "$BACKEND_DIR" ]; then
    BACKEND_DIR="$SCRIPT_DIR/backend"
    FRONTEND_DIR="$SCRIPT_DIR/frontend"
fi

echo "=================================================================="
echo "           🚀 Starting BuildSight AI Full Application             "
echo "=================================================================="

# 0. Check & Ensure Environment Configuration
if [ ! -f "$BACKEND_DIR/.env" ] && [ -f "$BACKEND_DIR/.env.example" ]; then
    echo "  [i] Initializing backend/.env from .env.example..."
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
fi

# 1. Check MongoDB
if systemctl is-active --quiet mongodb || systemctl is-active --quiet mongod || pgrep -x mongod > /dev/null; then
    echo "  [✓] MongoDB is active on mongodb://localhost:27017"
else
    echo "  [i] Starting MongoDB service..."
    sudo systemctl start mongodb 2>/dev/null || sudo systemctl start mongod 2>/dev/null || mongod --fork --logpath /tmp/mongod.log --dbpath /tmp 2>/dev/null || echo "  [!] Please ensure MongoDB is running"
fi

# Function to cleanly stop child processes on Ctrl+C
cleanup() {
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo "=================================================================="
    echo "  🛑 Stopping BuildSight AI services..."
    echo "=================================================================="
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# 2. Check Backend Virtual Environment & Dependencies
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
    echo "  [i] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    echo "  [i] Installing Python dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 3. Start Backend in Background
echo "  [1/2] Starting Backend API on http://localhost:8000 ..."
fuser -k 8000/tcp 2>/dev/null || true
PYTHONPATH=. python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait briefly for backend startup
sleep 2

# 4. Check Frontend Dependencies & Start
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "  [i] Installing Frontend dependencies (npm install)..."
    npm install
fi

echo "  [2/2] Starting Frontend UI on http://localhost:5173 ..."
npm run dev -- --host 0.0.0.0 --port 5173
