#!/bin/bash
set -euo pipefail

cd '/Users/shaheerwamique/Projects/Full Stack AI'
source .venv/bin/activate

if ! pgrep -f "uvicorn weather_api.main:app" >/dev/null 2>&1; then
  echo "Starting weather API..."
  uvicorn weather_api.main:app --host 127.0.0.1 --port 8000 > /tmp/weather_api.log 2>&1 &
  sleep 2
fi

echo "Weather API is running."
echo "Run the simple agent with:"
echo "python LLMInteraction/weather_agent_simple.py"
