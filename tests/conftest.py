"""Shared test isolation; runtime behavior is not modified."""
from __future__ import annotations

import sys

import pytest


@pytest.fixture
def isolated_native_ar_modules():
    """Native training and minimal inference ship different AR package trees."""
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "AR" or name.startswith("AR.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "AR" or name.startswith("AR."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
