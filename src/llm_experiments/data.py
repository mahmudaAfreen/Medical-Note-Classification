"""Prepare span-level examples from Potato/Label Studio note annotations.

The raw CSV files contain one row per SOAP section and a JSON ``label`` field
with annotated spans. This module explodes those spans into classification
examples and creates deterministic encounter-level train/dev/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_FILES = (
    "Assessment_potato.csv",
    "Objective_potato.csv",
    "Plan_potato.csv",
    "Subjective_potato.csv",
)

LABEL_ALIASES = {
    "Medications": "Medication",
    "Other Social": "Other Socials",
    "Theraputic History": "Therapeutic History",
}


def normalize_whitespace(value: Any) -> str:
    """Collapse repeated whitespace while preserving the text content."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_label(label: str, canonicalize: bool = True) -> str:
    label = normalize_whitespace(label)
    if not canonicalize:
        return label
    return LABEL_ALIASES.get(label, label)


def load_label_spans(raw_label: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_label, str) or not raw_label.strip():
        return []
    spans = json.loads(raw_label)
    if not isinstance(spans, list):
        raise ValueError(f"Expected a list of span annotations, got {type(spans)!r}")
    return spans


def infer_section(path: Path, row_section: Any) -> str:
    row_value = normalize_whitespace(row_section)
    if row_value:
        return row_value
    return path.stem.replace("_potato", "")


def iter_span_examples(
    data_dir: Path,
    data_files: tuple[str, ...] = DEFAULT_DATA_FILES,
    canonicalize_labels: bool = True,
    include_context: bool = False,
    include_note: bool = False,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped_empty_spans = 0

    for file_name in data_files:
        path = data_dir / file_name
        frame = pd.read_csv(path)
        for row_index, row in frame.iterrows():
            spans = load_label_spans(row.get("label"))
            section = infer_section(path, row.get("section"))
            for span_index, span in enumerate(spans):
                span_text = normalize_whitespace(span.get("text"))
                labels = span.get("labels") or []
                if not span_text or not labels:
                    skipped_empty_spans += 1
                    continue

                for label_index, raw_label in enumerate(labels):
                    label = canonical_label(str(raw_label), canonicalize_labels)
                    example = {
                        "example_id": (
                            f"{file_name}:{row_index}:{span_index}:{label_index}"
                        ),
                        "text": span_text,
                        "label": label,
                        "raw_label": normalize_whitespace(raw_label),
                        "section": section,
                        "encounter_id": normalize_whitespace(row.get("encounter_id")),
                        "row_id": normalize_whitespace(row.get("id")),
                        "annotation_id": normalize_whitespace(row.get("annotation_id")),
                        "source_file": file_name,
                        "span_start": span.get("start"),
                        "span_end": span.get("end"),
                    }
                    if include_context:
                        example["section_content"] = normalize_whitespace(
                            row.get("content")
                        )
                    if include_note:
                        example["note"] = normalize_whitespace(row.get("note"))
                    examples.append(example)

    if skipped_empty_spans:
        print(f"Skipped {skipped_empty_spans} empty labeled spans.")
    return examples


def split_by_group(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 13,
    group_key: str = "encounter_id",
) -> dict[str, list[dict[str, Any]]]:
    if not examples:
        raise ValueError("No examples were created from the annotation files.")
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must leave a positive test split.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        group = example.get(group_key) or example["row_id"] or example["example_id"]
        grouped[str(group)].append(example)

    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)

    n_groups = len(groups)
    train_end = max(1, round(n_groups * train_ratio))
    dev_end = train_end + max(1, round(n_groups * dev_ratio))
    if dev_end >= n_groups:
        dev_end = n_groups - 1

    split_groups = {
        "train": groups[:train_end],
        "dev": groups[train_end:dev_end],
        "test": groups[dev_end:],
    }
    return {
        split: [example for group in split_groups[split] for example in grouped[group]]
        for split in ("train", "dev", "test")
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "encounters": len({row["encounter_id"] for row in rows}),
        "sections": dict(sorted(Counter(row["section"] for row in rows).items())),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def write_metadata(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    canonicalize_labels: bool,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, Any]:
    labels = sorted({row["label"] for rows in splits.values() for row in rows})
    raw_labels = sorted({row["raw_label"] for rows in splits.values() for row in rows})
    metadata = {
        "labels": labels,
        "raw_labels": raw_labels,
        "label_aliases": LABEL_ALIASES if canonicalize_labels else {},
        "canonicalize_labels": canonicalize_labels,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "dev_ratio": dev_ratio,
        "test_ratio": round(1.0 - train_ratio - dev_ratio, 6),
        "splits": {split: summarize_split(rows) for split, rows in splits.items()},
    }
    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_files"),
        help="Directory containing *_potato.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_files/llm_splits"),
        help="Directory where JSONL splits and labels.json are written.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--keep-raw-labels",
        action="store_true",
        help="Do not map spelling/plural variants to canonical intent labels.",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include the full SOAP section text in each split row.",
    )
    parser.add_argument(
        "--include-note",
        action="store_true",
        help="Include the full note text in each split row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = iter_span_examples(
        data_dir=args.data_dir,
        canonicalize_labels=not args.keep_raw_labels,
        include_context=args.include_context,
        include_note=args.include_note,
    )
    splits = split_by_group(
        examples,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    metadata = write_metadata(
        output_dir,
        splits,
        canonicalize_labels=not args.keep_raw_labels,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    print(f"Wrote LLM splits to {output_dir}")
    for split, summary in metadata["splits"].items():
        print(
            f"{split}: {summary['examples']} examples, "
            f"{summary['encounters']} encounters"
        )


if __name__ == "__main__":
    main()
