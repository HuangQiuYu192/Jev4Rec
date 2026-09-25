"""Frozen-state Jev-style reranking on AgentCF's persisted candidate protocol.

This runner intentionally bypasses AgentCF's mutable language memories. It is a
first, attributable test of whether local Qwen logits can rerank exactly the
candidate set assembled by ``LanguageLossTrainer``: the first ``recall_budget
- 1`` entries from ``*.random`` plus the held-out positive, followed by the
same seeded permutation when ``fix_pos == -1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import requests


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_candidates(path: Path) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if tokens:
                candidates[tokens[0]] = tokens[1:]
    return candidates


def item_catalog(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {
        row["item_id:token"]: {
            "title": row["title:token_seq"],
            "category": row.get("category:token_seq", ""),
        }
        for row in rows
    }


def candidate_set(
    negatives: Sequence[str], positive: str, recall_budget: int, fix_pos: int, rng: np.random.RandomState
) -> List[str]:
    if recall_budget < 2:
        raise ValueError("recall_budget must be at least 2")
    chosen = list(negatives[: recall_budget - 1])
    if len(chosen) != recall_budget - 1:
        raise ValueError("persisted candidate row is shorter than recall_budget - 1")
    if fix_pos in (-1, recall_budget - 1):
        chosen.append(positive)
    elif fix_pos == 0:
        chosen.insert(0, positive)
    else:
        chosen.insert(fix_pos, positive)
    if fix_pos == -1:
        rng.shuffle(chosen)
    return chosen


def score_candidates(
    endpoint: str, history: List[Dict[str, str]], candidates: Sequence[Dict[str, str]], timeout: float
) -> List[float]:
    questions = {
        f"candidate_{index}": {
            "type": "noul",
            "instructions": (
                "The following candidate CD fits the user's demonstrated preferences better than a typical "
                "unseen CD. Judge only the supplied user history and this candidate.\n"
                f"CANDIDATE: {json.dumps(candidate, ensure_ascii=False, sort_keys=True)}"
            ),
        }
        for index, candidate in enumerate(candidates)
    }
    payload = {
        "model": "qwen3-14b-jev-style",
        "state": {"recent_history": history},
        "questions": questions,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    answers = response.json()["answers"]
    return [float(answers[f"candidate_{index}"]["noul"]) for index in range(len(candidates))]


def ndcg_at_k(rank: int, k: int) -> float:
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/CDs-100-user-dense"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010/v1/systemone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=20, help="A prefix pilot only; 0 evaluates every stored test user.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--recall-budget", type=int, default=10)
    parser.add_argument("--fix-pos", type=int, default=-1)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_dir = args.dataset_dir
    tests = read_tsv(dataset_dir / "CDs-100-user-dense.test.inter")
    catalog = item_catalog(read_tsv(dataset_dir / "CDs.item"))
    candidate_rows = read_candidates(dataset_dir / "CDs-100-user-dense.random")
    selected = tests if args.users == 0 else tests[: args.users]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary = {"processed": 0, "errors": 0, "hr_at_10": 0.0, "ndcg_at_10": 0.0}
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"run_metadata": vars(args), "protocol": "AgentCF persisted random candidates"}, default=str) + "\n")
        for row in selected:
            user = row["user_id:token"]
            positive = row["item_id:token"]
            try:
                candidate_ids = candidate_set(
                    candidate_rows[user], positive, args.recall_budget, args.fix_pos, np.random
                )
                history_ids = row["item_id_list:token_seq"].split()[-args.history_length :]
                history = [{"item_id": item_id, **catalog[item_id]} for item_id in history_ids]
                started = time.perf_counter()
                probabilities = score_candidates(
                    args.endpoint, history, [catalog[item_id] for item_id in candidate_ids], args.timeout
                )
                scored = [
                    {"item_id": item_id, "score": probability}
                    for item_id, probability in zip(candidate_ids, probabilities)
                ]
                elapsed = time.perf_counter() - started
                ranked = sorted(scored, key=lambda result: result["score"], reverse=True)
                rank = next(index + 1 for index, result in enumerate(ranked) if result["item_id"] == positive)
                record = {
                    "user_id": user,
                    "positive_item": positive,
                    "candidate_ids": candidate_ids,
                    "ranked": ranked,
                    "positive_rank": rank,
                    "latency_seconds": elapsed,
                }
                summary["processed"] += 1
                summary["hr_at_10"] += float(rank <= 10)
                summary["ndcg_at_10"] += ndcg_at_k(rank, 10)
            except Exception as error:  # Preserve failures as evidence; never silently skip them.
                summary["errors"] += 1
                record = {"user_id": user, "positive_item": positive, "error": repr(error)}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    if summary["processed"]:
        summary["hr_at_10"] /= summary["processed"]
        summary["ndcg_at_10"] /= summary["processed"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
