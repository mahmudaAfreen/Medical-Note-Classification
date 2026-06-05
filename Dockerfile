FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime


WORKDIR /app

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install -y git tmux && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/pvc/huggingface_cache \
    TRANSFORMERS_CACHE=/pvc/huggingface_cache \
    HF_DATASETS_CACHE=/pvc/huggingface_datasets_cache

COPY requirements-llm.txt .
RUN python -m pip install -r requirements-llm.txt

COPY src ./src

CMD ["python", "-m", "src.llm_experiments.evaluate_prompting", "--help"]