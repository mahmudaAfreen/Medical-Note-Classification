# Medical-Note-Classification

Utilities for annotation, encoder training/evaluation, and LLM experiments on
medical-note sentence intent classification.

## Encoder Experiments

The encoder task code lives in `classification_task_encoder/src`. The released sentence
splits live in `final_data`:

- `sentences_train.csv`, `sentences_dev.csv`, `sentences_test.csv`
- `sentences_train_SOAP.csv`, `sentences_dev_SOAP.csv`, `sentences_test_SOAP.csv`

The `_SOAP.csv` files include the same sentence-level labels plus a SOAP section
column. LLM experiments use these SOAP-aware splits by default so results are
aligned with the encoder train/dev/test data.

## LLM Experiments

The LLM experiment code lives in `classification_task_lms/src_llms`. It reuses the
zero-shot/few-shot prompting idea from `medical-intent-classification`, but wraps
it around this repository's encoder splits instead of making a new split from
raw annotation files.

Prepare LLM JSONL files from the encoder splits:

```bash
python -m src.llm_experiments.data \
  --source encoder \
  --data-dir final_data \
  --output-dir data_files/llm_splits
```

This writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `labels.json` using
the exact encoder examples and raw encoder labels, including `Medications`,
`Other Social`, and `Theraputic History`. Use `--canonicalize-labels` only for
exploratory runs where you intentionally want cleaned label names.

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
and per-section reports. Each run also writes encoder-style artifacts under
`outputs/llm_prompting/<run_name>/`:

- `per_class_metrics_test.csv`
- `confusion_matrix_test.csv`
- `misclassified_by_class/misclassified_<Class>.csv`

Export chat-style SFT rows for LoRA/QLoRA tooling:

```bash
python -m src.llm_experiments.export_sft_data \
  --splits-dir data_files/llm_splits \
  --output-dir data_files/llm_sft
```

Keep experiments local unless the data has been approved for external API use.

The older raw annotation span parser remains available for exploration:

```bash
python -m src.llm_experiments.data \
  --source annotations \
  --data-dir data_files \
  --output-dir data_files/llm_annotation_splits
```
