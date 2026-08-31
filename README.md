# Medical Note Classification

This repository contains the implementation of a **sentence-level intent classification framework for clinical notes**.

The project covers the workflow from data construction and annotation to encoder-based classification and prompt-based large language model evaluation.

> **Note:** The clinical datasets used in this project are not included in this repository.

## Repository Structure

```text
Medical-Note-Classification/
├── data_construction/
├── classification_task_encoder/
├── classification_task_llms/
├── docs/
└── README.md
```

## `data_construction/`

Contains the resources used for dataset construction and annotation.

- `Annotation guideline.pdf` — annotation guideline for the clinical intent classes
- `SOAP_splitting.ipynb` — preprocessing and SOAP-section splitting
- `Potato_formatting.ipynb` — preparation of data for the annotation interface
- `configs/` — annotation configuration files
- `surveyflow/` — survey and annotation workflow files
- `templates/` — annotation interface templates

## `classification_task_encoder/`

Contains the encoder-based classification experiments.

- `src/` — model, data module, training, and evaluation code
- `expirements/` — saved encoder experiment results
- `external_evaluation/` — external evaluation experiments
- `classification-random-split/` — experiments using a random data split
- Docker and YAML files — training and Kubernetes environment configurations

## `classification_task_llms/`

Contains the large language model experiments.

- `src_llms/` — zero-shot and few-shot prompting implementation
- `configs_llms/` — dependency, Docker, GPU, and Kubernetes configuration files
- `LLM_outputs/` — saved Qwen and Llama experiment results

## `docs/`

Contains additional documentation related to the experiments and execution environment.

## Project Workflow

```text
Clinical Notes
      │
      ▼
SOAP Section Processing
      │
      ▼
Sentence-Level Annotation
      │
      ▼
Clinical Intent Classification Dataset
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
Encoder-Based Models      Large Language Models
      │                          │
      ▼                          ▼
Supervised Fine-Tuning    Zero-Shot / Few-Shot Prompting
      │                          │
      └────────────┬─────────────┘
                   │
                   ▼
               Evaluation
```

## Experimental Approaches

### Encoder-Based Models

Transformer-based encoder models are fine-tuned for sentence-level clinical intent classification.

### Large Language Models

Instruction-tuned large language models are evaluated using:

- **Zero-shot prompting**
- **Few-shot prompting**

The experiments include models from the **Qwen** and **Llama** model families.


