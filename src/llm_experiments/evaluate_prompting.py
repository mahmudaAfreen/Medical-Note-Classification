"""Run zero-shot or few-shot LLM prompting for span classification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.llm_experiments.metrics import classification_report, section_reports
from src.llm_experiments.prompts import (
    build_messages,
    fallback_chat_template,
    messages_to_prompt,
    parse_label_from_response,
)
from src.llm_experiments.retrieval import FewShotRetriever


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labels(splits_dir: Path) -> list[str]:
    with (splits_dir / "labels.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return list(metadata["labels"])


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def default_candidate_splits(eval_split: str) -> list[str]:
    if eval_split == "test":
        return ["train", "dev"]
    if eval_split == "dev":
        return ["train"]
    return []


def build_eval_messages(
    eval_rows: list[dict[str, Any]],
    labels: list[str],
    setting: str,
    candidate_rows: list[dict[str, Any]],
    num_shots: int,
    section_aware: bool,
) -> list[list[dict[str, str]]]:
    retriever = None
    if setting == "few_shot":
        retriever = FewShotRetriever(candidate_rows, section_aware=section_aware)

    all_messages = []
    for row in eval_rows:
        shots = retriever.retrieve(row, num_shots) if retriever else []
        all_messages.append(build_messages(row, labels, shots))
    return all_messages


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def generate_local(
    model_name: str,
    all_messages: list[list[dict[str, str]]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    max_input_tokens: int | None,
    device_map: str,
    device: str | None,
    torch_dtype: str,
) -> tuple[list[str], list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [messages_to_prompt(tokenizer, messages) for messages in all_messages]

    model_kwargs: dict[str, Any] = {}
    if device_map.lower() != "none":
        model_kwargs["device_map"] = device_map
    if torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_map.lower() == "none" and device:
        model.to(device)
    model.eval()

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    responses: list[str] = []
    for batch_index, prompt_batch in enumerate(batched(prompts, batch_size), start=1):
        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": max_input_tokens is not None,
        }
        if max_input_tokens is not None:
            tokenize_kwargs["max_length"] = max_input_tokens
        encoded = tokenizer(prompt_batch, **tokenize_kwargs)
        input_device = next(model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            output_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
        responses.extend(text.strip() for text in decoded)
        print(f"generated batch {batch_index}/{len(batched(prompts, batch_size))}")
    return responses, prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data_files/llm_splits"),
        help="Directory containing train/dev/test JSONL and labels.json.",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument(
        "--setting",
        choices=("zero_shot", "few_shot"),
        default="zero_shot",
    )
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument(
        "--candidate-splits",
        nargs="*",
        default=None,
        help="Splits used as few-shot candidates. Defaults to train+dev for test.",
    )
    parser.add_argument(
        "--no-section-aware-few-shot",
        action="store_true",
        help="Retrieve few-shot examples globally instead of within the same section.",
    )
    parser.add_argument(
        "--backend",
        choices=("dry_run", "local"),
        default="dry_run",
        help="dry_run writes prompts without loading a model; local uses Transformers.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_prompting"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Include rendered prompts in the prediction JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.splits_dir)
    eval_rows = read_jsonl(args.splits_dir / f"{args.split}.jsonl")
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    candidate_splits = (
        args.candidate_splits
        if args.candidate_splits is not None
        else default_candidate_splits(args.split)
    )
    candidate_rows: list[dict[str, Any]] = []
    for split in candidate_splits:
        candidate_rows.extend(read_jsonl(args.splits_dir / f"{split}.jsonl"))

    if args.setting == "few_shot" and not candidate_rows:
        raise ValueError("Few-shot prompting requires candidate rows.")

    all_messages = build_eval_messages(
        eval_rows=eval_rows,
        labels=labels,
        setting=args.setting,
        candidate_rows=candidate_rows,
        num_shots=args.num_shots,
        section_aware=not args.no_section_aware_few_shot,
    )

    if args.backend == "dry_run":
        responses = ["" for _ in all_messages]
        prompts = [fallback_chat_template(messages) for messages in all_messages]
    else:
        responses, prompts = generate_local(
            model_name=args.model_name,
            all_messages=all_messages,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_input_tokens=args.max_input_tokens,
            device_map=args.device_map,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )

    results = []
    for row, response, prompt in zip(eval_rows, responses, prompts):
        predicted_label, parse_status = parse_label_from_response(response, labels)
        result = dict(row)
        result.update(
            {
                "predicted_label": predicted_label,
                "parse_status": parse_status if response else "dry_run",
                "raw_response": response,
                "correct": predicted_label == row["label"],
            }
        )
        if args.save_prompts:
            result["prompt"] = prompt
        results.append(result)

    report = classification_report(
        [row["label"] for row in results],
        [row.get("predicted_label") for row in results],
        labels,
    )
    report["by_section"] = section_reports(results, labels)
    report["setting"] = args.setting
    report["backend"] = args.backend
    report["model_name"] = args.model_name
    report["split"] = args.split
    report["candidate_splits"] = candidate_splits

    run_name = (
        f"{args.split}_{args.setting}_{args.backend}_{safe_name(args.model_name)}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{run_name}_predictions.jsonl"
    metrics_path = args.output_dir / f"{run_name}_metrics.json"
    write_jsonl(predictions_path, results)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(
        "accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} invalid={invalid}".format(
            accuracy=report["accuracy"],
            macro_f1=report["macro_f1"],
            invalid=report["invalid_predictions"],
        )
    )


if __name__ == "__main__":
    main()
