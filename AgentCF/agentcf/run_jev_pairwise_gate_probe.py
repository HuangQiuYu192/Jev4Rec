"""Probe whether local Jev can gate pairwise AgentCF feedback decisions.

This is deliberately not a recommender metric.  It asks whether the decision
engine can identify a held-out positive over each persisted negative candidate,
with both candidate-to-label orders included to remove A/B label priors.  Only
if this probe is above chance should the decision be used to commit an AgentCF
memory update.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import requests

from run_jev_direct_rerank import (
    add_pretrained_descriptions,
    candidate_set,
    item_catalog,
    read_candidates,
    read_tsv,
)


def build_balanced_pairwise_questions(
    positive_id: str, negative_ids: Sequence[str], catalog: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, object]]:
    """Return two opposite option orders per negative, keyed deterministically."""
    questions: Dict[str, Dict[str, object]] = {}
    instructions = (
        "Select the CD more compatible with the user's demonstrated preferences. "
        "Use only the supplied history and CD descriptions, and choose exactly one."
    )
    for index, negative_id in enumerate(negative_ids):
        for direction, ordered_ids in enumerate(((positive_id, negative_id), (negative_id, positive_id))):
            questions[f"pair_{index}_order_{direction}"] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    item_id: json.dumps(catalog[item_id], ensure_ascii=False, sort_keys=True)
                    for item_id in ordered_ids
                },
            }
    return questions


def positive_probabilities(
    endpoint: str,
    history: List[Dict[str, str]],
    positive_id: str,
    negative_ids: Sequence[str],
    catalog: Dict[str, Dict[str, str]],
    timeout: float,
) -> List[float]:
    questions = build_balanced_pairwise_questions(positive_id, negative_ids, catalog)
    response = requests.post(
        endpoint,
        json={"model": "qwen3-14b-jev-style", "state": {"recent_history": history}, "questions": questions},
        timeout=timeout,
    )
    response.raise_for_status()
    answers = response.json()["answers"]
    return [
        (
            float(answers[f"pair_{index}_order_0"]["probabilities"][positive_id])
            + float(answers[f"pair_{index}_order_1"]["probabilities"][positive_id])
        )
        / 2.0
        for index in range(len(negative_ids))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/CDs-100-user-dense"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010/v1/systemone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--recall-budget", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--use-pretrained-descriptions", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tests = read_tsv(args.dataset_dir / "CDs-100-user-dense.test.inter")
    catalog = item_catalog(read_tsv(args.dataset_dir / "CDs.item"))
    if args.use_pretrained_descriptions:
        add_pretrained_descriptions(catalog, read_tsv(args.dataset_dir / "CDs.pretrained_item"))
    candidates = read_candidates(args.dataset_dir / "CDs-100-user-dense.random")
    selected = tests if args.users == 0 else tests[: args.users]
    summary = {"users": 0, "pairs": 0, "correct": 0, "errors": 0, "mean_positive_probability": 0.0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"run_metadata": vars(args), "protocol": "AgentCF persisted random candidates"}, default=str) + "\n")
        for row in selected:
            user_id, positive = row["user_id:token"], row["item_id:token"]
            try:
                pool = candidate_set(candidates[user_id], positive, args.recall_budget, -1, np.random)
                negatives = [item_id for item_id in pool if item_id != positive]
                history_ids = row["item_id_list:token_seq"].split()[-args.history_length :]
                history = [{"item_id": item_id, **catalog[item_id]} for item_id in history_ids]
                started = time.perf_counter()
                probabilities = positive_probabilities(
                    args.endpoint, history, positive, negatives, catalog, args.timeout
                )
                correct = sum(probability > 0.5 for probability in probabilities)
                record = {
                    "user_id": user_id,
                    "positive_item": positive,
                    "negative_items": negatives,
                    "positive_probabilities": probabilities,
                    "pairwise_correct": correct,
                    "pairwise_total": len(negatives),
                    "latency_seconds": time.perf_counter() - started,
                }
                summary["users"] += 1
                summary["pairs"] += len(negatives)
                summary["correct"] += correct
                summary["mean_positive_probability"] += sum(probabilities)
            except Exception as error:
                summary["errors"] += 1
                record = {"user_id": user_id, "positive_item": positive, "error": repr(error)}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    if summary["pairs"]:
        summary["pairwise_accuracy"] = summary["correct"] / summary["pairs"]
        summary["mean_positive_probability"] /= summary["pairs"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
