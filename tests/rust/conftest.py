"""Legacy-free frozen regression fixtures for the Rust-backed test suite."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "assist_legacy_regression_v1.npz"
)


@pytest.fixture(scope="session")
def frozen_assist_regression() -> Mapping[str, np.ndarray]:
    """Reviewed outputs captured from the accepted legacy ASSIST parity run."""
    with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
        return {name: fixture[name].copy() for name in fixture.files}
