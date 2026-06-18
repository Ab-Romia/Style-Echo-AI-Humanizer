"""Shared pytest fixtures and import path setup."""
import sys
from pathlib import Path

# Make the backend package importable from the repo root.
BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
