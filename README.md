# Medical-Note-Classification

Utilities for annotation, training, and evaluation on medical-note span intent
classification.

## LLM Experiments

The LLM experiment code lives in `src/llm_experiments`. It reuses the
zero-shot/few-shot prompting idea from `medical-intent-classification`, but the
label space is generated from this repository's annotation CSV files instead of
being hard-coded.

Prepare deterministic span-level splits:

```bash
python -m src.llm_experiments.data \
  --data-dir data_files \
  --output-dir data_files/llm_splits
```

This explodes the Potato/Label Studio `label` span JSON into one example per
annotated span, preserves `encounter_id` for grouped splitting, and writes
`train.jsonl`, `dev.jsonl`, `test.jsonl`, and `labels.json`. By default it maps
annotation variants to canonical labels:

- `Medications` -> `Medication`
- `Other Social` -> `Other Socials`
- `Theraputic History` -> `Therapeutic History`

Use `--keep-raw-labels` if you need exact raw annotation labels.

Preview prompts without loading a model:

```bash
python -m src.llm_experiments.evaluate_prompting \
  --splits-dir data_files/llm_splits \
  --split test \
  --setting few_shot \
  --num-shots 3 \
  --backend dry_run \
  --save-prompts \
  --limit 5
```

Run a local Hugging Face instruction model:

```bash
python -m src.llm_experiments.evaluate_prompting \
  --splits-dir data_files/llm_splits \
  --split test \
  --setting zero_shot \
  --backend local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --batch-size 2
```

Predictions are saved as JSONL with raw model responses and parsed labels.
Metrics are saved as JSON with accuracy, macro/weighted F1, per-label metrics,
and per-section reports.

Export chat-style SFT rows for LoRA/QLoRA tooling:

```bash
python -m src.llm_experiments.export_sft_data \
  --splits-dir data_files/llm_splits \
  --output-dir data_files/llm_sft
```

Keep experiments local unless the data has been approved for external API use.
