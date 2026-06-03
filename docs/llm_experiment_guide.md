# LLM Experiment Guide

This guide explains how to run the LLM experiment suite around the existing
encoder train/dev/test splits.

Run all commands from the repository root:

```bash
cd /pvc/tom_workspace/Medical-Note-Classification
```


## Docker On B200 GPUs

A CUDA 13.0 Dockerfile for B200/Blackwell runs is provided at:

```text
Dockerfile.llm-b200
```

It uses `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`, installs Python 3.12,
and installs PyTorch CUDA 13.0 wheels. It is intended for hosts where
`nvidia-smi` reports a CUDA 13.0-capable 580-series driver, for example:

```text
NVIDIA-SMI 580.159.03
Driver Version: 580.159.03
CUDA Version: 13.0
```

Build the image from the repository root:

```bash
docker build -f Dockerfile.llm-b200 -t medical-note-llm:b200-cu130 .
```

Smoke-test GPU visibility:

```bash
docker run --rm --gpus all medical-note-llm:b200-cu130 nvidia-smi
```

Start an interactive experiment shell:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD:/workspace/Medical-Note-Classification" \
  -v /pvc/huggingface_cache:/pvc/huggingface_cache \
  medical-note-llm:b200-cu130
```

Inside the container, run the normal experiment commands from:

```bash
/workspace/Medical-Note-Classification
```

For example, rebuild LLM splits:

```bash
python -m src.llm_experiments.data \
  --source encoder \
  --data-dir final_data \
  --output-dir data_files/llm_splits
```

Run a dry-run prompt check:

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

Run local LLM inference:

```bash
python -m src.llm_experiments.evaluate_prompting \
  --splits-dir data_files/llm_splits \
  --split test \
  --setting few_shot \
  --num-shots 3 \
  --backend local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --batch-size 2 \
  --max-new-tokens 64 \
  --temperature 0.0
```

If Docker cannot see the GPU, verify the host has NVIDIA Container Toolkit
installed and that containers are launched with `--gpus all`.

## 1. Confirm The Encoder Data Is Present

The LLM suite uses the same sentence-level split files as the encoder models.
The SOAP-aware files are used by default because they include the section name
in addition to the sentence and class label.

Expected files:

```text
final_data/sentences_train_SOAP.csv
final_data/sentences_dev_SOAP.csv
final_data/sentences_test_SOAP.csv
```

The raw encoder labels are preserved for comparability, including:

```text
Medications
Other Social
Theraputic History
```

## 2. Install LLM Dependencies

Use the project environment you normally use for experiments, then install the
LLM requirements:

```bash
pip install -r requirements-llm.txt
```

For local Hugging Face models, make sure your runtime has access to the model
weights and the needed GPU/CPU memory.

## 3. Build LLM JSONL Splits

Convert the encoder CSV splits into the JSONL format used by prompting and SFT:

```bash
python -m src.llm_experiments.data \
  --source encoder \
  --data-dir final_data \
  --output-dir data_files/llm_splits
```

Expected output:

```text
data_files/llm_splits/train.jsonl
data_files/llm_splits/dev.jsonl
data_files/llm_splits/test.jsonl
data_files/llm_splits/labels.json
```

The expected split sizes are:

```text
train: 5069 examples
dev: 656 examples
test: 659 examples
```

Use this command again whenever `final_data` changes.

## 4. Preview Prompts Without Loading A Model

Start with a dry run. This validates split loading, few-shot retrieval, prompt
formatting, output paths, and metric/report generation without downloading or
loading an LLM.

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

Dry-run predictions are intentionally invalid because no model response is
generated. The useful check is that prompts and report files are created.

## 5. Run Zero-Shot LLM Evaluation

Use zero-shot prompting when you want the simplest LLM baseline:

```bash
python -m src.llm_experiments.evaluate_prompting \
  --splits-dir data_files/llm_splits \
  --split test \
  --setting zero_shot \
  --backend local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --batch-size 2 \
  --max-new-tokens 64 \
  --temperature 0.0
```

Change `--model-name` to any local Hugging Face causal/instruction model you
want to evaluate.

## 6. Run Few-Shot LLM Evaluation

Few-shot prompting retrieves similar examples from train/dev and inserts them
into the prompt. Retrieval is SOAP-section-aware by default.

```bash
python -m src.llm_experiments.evaluate_prompting \
  --splits-dir data_files/llm_splits \
  --split test \
  --setting few_shot \
  --num-shots 3 \
  --backend local \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --batch-size 2 \
  --max-new-tokens 64 \
  --temperature 0.0
```

Useful variations:

```bash
# Use five retrieved examples instead of three.
--num-shots 5

# Retrieve globally instead of by SOAP section.
--no-section-aware-few-shot

# Evaluate dev using train as the candidate pool.
--split dev
```

## 7. Inspect Outputs

Each run writes JSONL predictions and JSON metrics under:

```text
outputs/llm_prompting/
```

For a run named:

```text
test_few_shot_local_Qwen_Qwen2.5-7B-Instruct
```

the main files are:

```text
outputs/llm_prompting/test_few_shot_local_Qwen_Qwen2.5-7B-Instruct_predictions.jsonl
outputs/llm_prompting/test_few_shot_local_Qwen_Qwen2.5-7B-Instruct_metrics.json
```

The prediction rows include:

```text
text
label
predicted_label
section
encounter_id
text_id
raw_response
parse_status
correct
```

## 8. Compare Against Encoder Reports

The evaluator also writes encoder-style artifacts for easier comparison with
`classification_task/expirements/...` outputs:

```text
outputs/llm_prompting/<run_name>/per_class_metrics_test.csv
outputs/llm_prompting/<run_name>/confusion_matrix_test.csv
outputs/llm_prompting/<run_name>/misclassified_by_class/
```

Use `per_class_metrics_test.csv` to compare per-class precision, recall, F1,
and support against encoder model reports.

Use `confusion_matrix_test.csv` to inspect label confusions. Invalid or
unparseable LLM outputs are counted in the `__INVALID__` prediction column.

## 9. Export SFT Data For LoRA/QLoRA

To prepare chat-format supervised fine-tuning rows:

```bash
python -m src.llm_experiments.export_sft_data \
  --splits-dir data_files/llm_splits \
  --output-dir data_files/llm_sft
```

Expected output:

```text
data_files/llm_sft/train.jsonl
data_files/llm_sft/dev.jsonl
data_files/llm_sft/test.jsonl
```

Each row contains:

```text
example_id
messages
label
raw_label
section
encounter_id
```

The `messages` field is ready for chat-style SFT pipelines such as TRL-based
LoRA/QLoRA training.

## 10. Optional: Use Annotation Span Splits

The main LLM experiments should use `--source encoder` for comparability. For
exploratory span-level experiments from the original annotation CSVs, use:

```bash
python -m src.llm_experiments.data \
  --source annotations \
  --data-dir data_files \
  --output-dir data_files/llm_annotation_splits
```

Do not compare these numbers directly with the encoder results because the
examples and split construction differ.

## 11. Recommended Experiment Order

1. Build encoder-aligned LLM splits.
2. Run a dry-run few-shot prompt preview.
3. Run zero-shot on the test split.
4. Run few-shot with `--num-shots 3`.
5. Try `--num-shots 5` if context length allows.
6. Compare `per_class_metrics_test.csv` and confusion matrices against encoder
   outputs.
7. Export SFT data and run LoRA/QLoRA only after prompt baselines are stable.

## 12. Data Handling

Keep experiments local unless the clinical-note data has been approved for
external API use. The provided evaluator defaults to local Hugging Face models
and does not call an external API.
