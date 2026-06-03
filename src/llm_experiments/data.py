"""Prepare LLM examples from the encoder split files.

The encoder experiments use sentence-level CSV splits in ``final_data``. This
module converts those same splits to JSONL so LLM prompting and SFT runs evaluate
on the exact same examples and raw label spellings.

The older Potato/Label Studio annotation parser is still available with
``--source annotations`` for exploratory span-level work.
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

SPLIT_FILE_STEMS = {
    "train": "sentences_train",
    "dev": "sentences_dev",
    "test": "sentences_test",
}

LABEL_ALIASES = {
    "Medications": "Medication",
    "Other Social": "Other Socials",
    "Theraputic History": "Therapeutic History",
}


def normalize_whitespace(value: Any) -> str:
    """Collapse repeated whitespace while preserving the text content."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_label(label: str, canonicalize: bool = False) -> str:
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


def iter_encoder_split_examples(
    data_dir: Path,
    split: str,
    use_soap: bool = True,
    canonicalize_labels: bool = False,
) -> list[dict[str, Any]]:
    stem = SPLIT_FILE_STEMS[split]
    file_name = f"{stem}{'_SOAP' if use_soap else ''}.csv"
    path = data_dir / file_name
    frame = pd.read_csv(path)

    required = {"sentence", "class", "encounter_id", "text_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    examples: list[dict[str, Any]] = []
    skipped = 0
    for row_index, row in frame.iterrows():
        text = normalize_whitespace(row.get("sentence"))
        raw_label = normalize_whitespace(row.get("class"))
        if not text or not raw_label:
            skipped += 1
            continue

        text_id = normalize_whitespace(row.get("text_id")) or f"{split}_{row_index}"
        examples.append(
            {
                "example_id": f"{split}:{text_id}",
                "text": text,
                "label": canonical_label(raw_label, canonicalize_labels),
                "raw_label": raw_label,
                "section": normalize_whitespace(row.get("section")) or "UNKNOWN",
                "encounter_id": normalize_whitespace(row.get("encounter_id")),
                "text_id": text_id,
                "row_id": normalize_whitespace(row.get("id")),
                "split": split,
                "source_file": file_name,
            }
        )

    if skipped:
        print(f"Skipped {skipped} empty rows in {file_name}.")
    return examples


def load_encoder_splits(
    data_dir: Path,
    use_soap: bool = True,
    canonicalize_labels: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: iter_encoder_split_examples(
            data_dir=data_dir,
            split=split,
            use_soap=use_soap,
            canonicalize_labels=canonicalize_labels,
        )
        for split in ("train", "dev", "test")
    }


def iter_annotation_span_examples(
    data_dir: Path,
    data_files: tuple[str, ...] = DEFAULT_DATA_FILES,
    canonicalize_labels: bool = False,
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
                    raw_label_text = normalize_whitespace(raw_label)
                    example = {
                        "example_id": (
                            f"{file_name}:{row_index}:{span_index}:{label_index}"
                        ),
                        "text": span_text,
                        "label": canonical_label(raw_label_text, canonicalize_labels),
                        "raw_label": raw_label_text,
                        "section": section,
                        "encounter_id": normalize_whitespace(row.get("encounter_id")),
                        "row_id": normalize_whitespace(row.get("id")),
                        "annotation_id": normalize_whitespace(row.get("annotation_id")),
                        "source_file": file_name,
                        "span_start": span.get("start"),
                        "span_end": span.get("end"),
                    }
                    if include_context:
                        example["section_content"] = normalize_whitespace(row.get("content"))
                    if include_note:
                        example["note"] = normalize_whitespace(row.get("note"))
                    examples.append(example)

    if skipped_empty_spans:
        print(f"Skipped {skipped_empty_spans} empty labeled spans.")
    return examples


# Backward-compatible name used by older notes/scripts.
iter_span_examples = iter_annotation_span_examples


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
        "encounters": len({row["encounter_id"] for row in rows if row.get("encounter_id")}),
        "sections": dict(sorted(Counter(row.get("section", "") for row in rows).items())),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def write_metadata(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    source: str,
    canonicalize_labels: bool,
    use_soap: bool,
    seed: int | None = None,
    train_ratio: float | None = None,
    dev_ratio: float | None = None,
) -> dict[str, Any]:
    labels = sorted({row["label"] for rows in splits.values() for row in rows})
    raw_labels = sorted({row["raw_label"] for rows in splits.values() for row in rows})
    metadata = {
        "source": source,
        "labels": labels,
        "raw_labels": raw_labels,
        "label_aliases": LABEL_ALIASES if canonicalize_labels else {},
        "canonicalize_labels": canonicalize_labels,
        "use_soap": use_soap,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "dev_ratio": dev_ratio,
        "test_ratio": (
            round(1.0 - train_ratio - dev_ratio, 6)
            if train_ratio is not None and dev_ratio is not None
            else None
        ),
        "splits": {split: summarize_split(rows) for split, rows in splits.items()},
    }
    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("encoder", "annotations"),
        default="encoder",
        help="Use encoder final_data splits or explode original annotation spans.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final_data"),
        help="Directory containing final_data split CSVs or *_potato.csv files.",
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
        "--no-soap",
        action="store_true",
        help="Use final_data/sentences_{split}.csv instead of *_SOAP.csv.",
    )
    parser.add_argument(
        "--canonicalize-labels",
        action="store_true",
        help="Map raw encoder labels to canonical variants for exploratory runs.",
    )
    parser.add_argument(
        "--keep-raw-labels",
        action="store_true",
        help="Deprecated alias for the default behavior; raw labels are kept unless --canonicalize-labels is used.",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Annotation source only: include the full SOAP section text.",
    )
    parser.add_argument(
        "--include-note",
        action="store_true",
        help="Annotation source only: include the full note text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    canonicalize = bool(args.canonicalize_labels and not args.keep_raw_labels)

    if args.source == "encoder":
        splits = load_encoder_splits(
            data_dir=args.data_dir,
            use_soap=not args.no_soap,
            canonicalize_labels=canonicalize,
        )
        metadata_seed = None
        train_ratio = None
        dev_ratio = None
    else:
        examples = iter_annotation_span_examples(
            data_dir=args.data_dir,
            canonicalize_labels=canonicalize,
            include_context=args.include_context,
            include_note=args.include_note,
        )
        splits = split_by_group(
            examples,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        metadata_seed = args.seed
        train_ratio = args.train_ratio
        dev_ratio = args.dev_ratio

    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    metadata = write_metadata(
        output_dir,
        splits,
        source=args.source,
        canonicalize_labels=canonicalize,
        use_soap=not args.no_soap if args.source == "encoder" else False,
        seed=metadata_seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
    )

    print(f"Wrote LLM splits to {output_dir}")
    print(f"source: {metadata['source']}")
    for split, summary in metadata["splits"].items():
        print(
            f"{split}: {summary['examples']} examples, "
            f"{summary['encounters']} encounters"
        )


if __name__ == "__main__":
    main()
