FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p /workspace/processed/source
RUN mkdir -p /workspace/processed/target
RUN mkdir -p /workspace/checkpoints
RUN mkdir -p /workspace/data/source_voice
RUN mkdir -p /workspace/data/target_voice

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/app

# Default command
CMD ["python3", "/workspace/app/preprocess.py"]