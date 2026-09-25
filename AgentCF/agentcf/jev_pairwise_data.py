"""Order-balanced supervision data for a local Jev-style pairwise decision head."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np


def decision_prompt(history: List[Dict[str, str]], left: Dict[str, str], right: Dict[str, str]) -> str:
    return (
        "You are a bounded decision function. Do not explain.\n"
        f"STATE:\n{json.dumps({'recent_history': history}, ensure_ascii=False)}\n\n"
        "QUESTION:\nSelect the CD more compatible with the user's demonstrated preferences.\n\n"
        f"ALLOWED ANSWERS:\n{json.dumps({'A': left, 'B': right}, ensure_ascii=False)}\n\n"
        "Return exactly one allowed label, without spaces or punctuation.\nANSWER:"
    )


def make_examples(
    rows: Iterable[Dict[str, str]], catalog: Dict[str, Dict[str, str]], item_ids: List[str],
    negatives_per_positive: int, history_length: int, seed: int,
) -> List[Dict[str, str]]:
    """Create deterministic, order-balanced positive-vs-unseen-negative examples."""
    rng = np.random.RandomState(seed)
    examples: List[Dict[str, str]] = []
    for row in rows:
        positive = row["item_id:token"]
        history_ids = row["item_id_list:token_seq"].split()[-history_length:]
        eligible = [item_id for item_id in item_ids if item_id not in set(history_ids) | {positive}]
        if len(eligible) < negatives_per_positive:
            raise ValueError("not enough unseen items to sample negatives")
        history = [{"item_id": item_id, **catalog[item_id]} for item_id in history_ids]
        for negative in rng.choice(eligible, size=negatives_per_positive, replace=False):
            if rng.randint(2):
                left_id, right_id, label = positive, str(negative), "A"
            else:
                left_id, right_id, label = str(negative), positive, "B"
            examples.append({"prompt": decision_prompt(history, catalog[left_id], catalog[right_id]), "label": label,
                             "positive_item": positive, "negative_item": str(negative)})
    return examples
