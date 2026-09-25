from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# AgentCF historically executes from ``agentcf/`` and uses absolute imports
# such as ``agentverse``. Keep this test faithful to that entrypoint rather
# than importing the package from the repository root.
AGENTCF_DIR = Path(__file__).resolve().parents[1] / "agentcf"
sys.path.insert(0, str(AGENTCF_DIR))

from run_jev_direct_rerank import candidate_set


def test_candidate_set_reproduces_seeded_agentcf_shuffle() -> None:
    rng = np.random.RandomState(2026)
    candidates = candidate_set(["n1", "n2", "n3", "n4"], "positive", 4, -1, rng)
    assert sorted(candidates) == ["n1", "n2", "n3", "positive"]
    assert candidates != ["n1", "n2", "n3", "positive"]


def test_candidate_set_rejects_insufficient_persisted_negatives() -> None:
    with pytest.raises(ValueError, match="shorter"):
        candidate_set(["n1"], "positive", 4, -1, np.random.RandomState(0))
