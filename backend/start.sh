#!/bin/bash

# Start VoicePrint API server

echo "Starting VoicePrint API..."

# Activate virtual environment
source venv/bin/activate

# Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
