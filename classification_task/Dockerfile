FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install system utilities
RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install -y git tmux && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install \
    numpy==1.23.5 \
    pandas \
    transformers \
    pytorch-lightning \
    torchmetrics \
    jsonargparse[signatures] \
    protobuf==3.20.0
RUN pip install soxr

# Set working directory
WORKDIR /src

# Copy your entire source code
COPY src/ .

# Default command for your job (LightningCLI)
CMD ["python", "/src/main.py"]
