"""Export prepared splits as chat-style SFT examples for LoRA/QLoRA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.llm_experiments.evaluate_prompting import load_labels, read_jsonl, write_jsonl
from src.llm_experiments.prompts import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data_files/llm_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_files/llm_sft"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=("train", "dev", "test"),
    )
    return parser.parse_args()


def to_sft_row(row: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    messages = build_messages(row, labels)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"label": row["label"]}, ensure_ascii=False),
        }
    )
    return {
        "example_id": row["example_id"],
        "messages": messages,
        "label": row["label"],
        "raw_label": row["raw_label"],
        "section": row["section"],
        "encounter_id": row["encounter_id"],
    }


def main() -> None:
    args = parse_args()
    labels = load_labels(args.splits_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows = read_jsonl(args.splits_dir / f"{split}.jsonl")
        sft_rows = [to_sft_row(row, labels) for row in rows]
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, sft_rows)
        print(f"Wrote {len(sft_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
