"""Shared fixtures for dialogue tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dialogue import state as dialogue_state  # noqa: E402


@pytest.fixture(autouse=True)
async def reset_dialogue_state():
    """Clear in-memory session store between tests."""
    await dialogue_state._reset_for_tests()
    yield
    await dialogue_state._reset_for_tests()
