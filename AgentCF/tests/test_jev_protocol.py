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

from run_jev_direct_rerank import add_pretrained_descriptions, candidate_set, score_candidates
from run_jev_pairwise_gate_probe import build_balanced_pairwise_questions
from jev_pairwise_data import make_examples


def test_candidate_set_reproduces_seeded_agentcf_shuffle() -> None:
    rng = np.random.RandomState(2026)
    candidates = candidate_set(["n1", "n2", "n3", "n4"], "positive", 4, -1, rng)
    assert sorted(candidates) == ["n1", "n2", "n3", "positive"]
    assert candidates != ["n1", "n2", "n3", "positive"]


def test_candidate_set_rejects_insufficient_persisted_negatives() -> None:
    with pytest.raises(ValueError, match="shorter"):
        candidate_set(["n1"], "positive", 4, -1, np.random.RandomState(0))


def test_choice_permutation_bounds_fail_before_network_request() -> None:
    with pytest.raises(ValueError, match="choice_permutations"):
        score_candidates("http://unused", [], ["a", "b"], [{}, {}], 1.0, 0)


def test_pretrained_descriptions_attach_only_known_items() -> None:
    catalog = {"known": {"title": "A", "category": "B"}}
    add_pretrained_descriptions(
        catalog,
        [
            {"item_id:token": "known", "pretrained_item_description:token_seq": "rich text"},
            {"item_id:token": "absent", "pretrained_item_description:token_seq": "ignore"},
        ],
    )
    assert catalog == {"known": {"title": "A", "category": "B", "description": "rich text"}}


def test_pairwise_probe_balances_candidate_to_label_order() -> None:
    catalog = {"positive": {"title": "P"}, "negative": {"title": "N"}}
    questions = build_balanced_pairwise_questions("positive", ["negative"], catalog)
    assert list(questions["pair_0_order_0"]["criteria"]) == ["positive", "negative"]
    assert list(questions["pair_0_order_1"]["criteria"]) == ["negative", "positive"]


def test_lora_examples_exclude_history_and_balance_labels() -> None:
    catalog = {item_id: {"title": item_id} for item_id in ["h", "p", "n1", "n2", "n3"]}
    rows = [{"item_id:token": "p", "item_id_list:token_seq": "h"}]
    examples = make_examples(rows, catalog, list(catalog), 2, 1, 2026)
    assert len(examples) == 2
    assert {example["label"] for example in examples} <= {"A", "B"}
    assert all(example["negative_item"] not in {"h", "p"} for example in examples)
