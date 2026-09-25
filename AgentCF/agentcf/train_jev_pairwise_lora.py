"""Train only a LoRA adapter for local Jev-style A/B CD decisions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from jev_pairwise_data import make_examples
from run_jev_direct_rerank import add_pretrained_descriptions, item_catalog, read_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/CDs-100-user-dense"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("JEV_QWEN_MODEL", "Qwen/Qwen3-14B"))
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = item_catalog(read_tsv(args.dataset_dir / "CDs.item"))
    add_pretrained_descriptions(catalog, read_tsv(args.dataset_dir / "CDs.pretrained_item"))
    rows = read_tsv(args.dataset_dir / "CDs-100-user-dense.train.inter")
    examples = make_examples(rows, catalog, list(catalog), args.negatives_per_positive, args.history_length, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "pairwise_train_manifest.json"
    manifest.write_text(json.dumps({"examples": len(examples), "args": vars(args)}, default=str, indent=2), encoding="utf-8")
    print(manifest.read_text(encoding="utf-8"))
    if args.dry_run:
        return
    raise RuntimeError(
        "Dataset construction is ready. Install peft/accelerate in jev4rec before enabling LoRA optimization; "
        "this guard prevents accidentally training all Qwen3-14B parameters."
    )


if __name__ == "__main__":
    main()
