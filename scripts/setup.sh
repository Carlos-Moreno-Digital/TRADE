#!/bin/bash
# TRADE - Setup Script
set -e

echo "=== TRADE - Autonomous Trading Agent Setup ==="

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install
echo "Installing dependencies..."
pip install -e ".[dev]"

# Create logs directory
mkdir -p logs

# Verify
echo ""
echo "Verifying installation..."
python -c "from trade.config import load_config; c = load_config(); print(f'Config loaded: mode={c.mode}')"
python -c "from trade.agents import Orchestrator; print('Agents loaded OK')"

echo ""
echo "=== Setup complete! ==="
echo "Run: python -m trade.main --symbol AAPL"
