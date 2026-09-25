"""Train only a LoRA adapter for local Jev-style A/B CD decisions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
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
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-jev-train.txt in the jev4rec environment before training.") from error
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0}
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    def encode(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        label_ids = tokenizer(example["label"], add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + label_ids)[-args.max_length:]
        label_start = max(0, len(prompt_ids) - max(0, len(prompt_ids) + len(label_ids) - args.max_length))
        return {"input_ids": ids, "labels": [-100] * label_start + ids[label_start:]}

    encoded = [encode(example) for example in examples]

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [tokenizer.pad_token_id] * (width - len(row["input_ids"])) for row in batch]),
            "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["labels"])) for row in batch]),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch]),
        }

    loader = DataLoader(encoded, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = {name: value.to(model.device) for name, value in batch.items()}
        loss = model(**batch).loss / args.gradient_accumulation
        loss.backward()
        if step % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step >= args.max_steps:
            break
    if step % args.gradient_accumulation:
        optimizer.step()
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    print(json.dumps({"completed_steps": step, "adapter": str(args.output_dir / "adapter")}, indent=2))


if __name__ == "__main__":
    main()
