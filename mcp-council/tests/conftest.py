"""Shared pytest fixtures for mcp-council tests."""

import sys
from pathlib import Path

# Make package root importable (server.py, council.py, etc.).
sys.path.insert(0, str(Path(__file__).parent.parent))
