#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python3.12"
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    pip install --upgrade pip --quiet
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Installing agent-framework (pre-release)..."
    pip install agent-framework --pre || echo "Warning: Failed to install agent-framework. Install manually with: pip install agent-framework --pre"
else
    source venv/bin/activate
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -r requirements.txt
    fi
    # Check if agent-framework is installed
    if ! python -c "import agent_framework" 2>/dev/null; then
        echo "Installing agent-framework (pre-release)..."
        pip install agent-framework --pre || echo "Warning: Failed to install agent-framework. Install manually with: pip install agent-framework --pre"
    fi
fi

if [ ! -f ".env" ]; then
    if [ -f "env_template.txt" ]; then
        cp env_template.txt .env
        echo "Created .env file. Please update it with your Azure credentials."
    fi
fi

echo ""
echo "Setup complete!"
echo "Starting server at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
