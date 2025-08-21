# Base image
FROM nvidia/cuda:12.2.0-cudnn11-runtime-ubuntu22.04

# Set workdir
WORKDIR /workspace

# Update & install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy app code
COPY app/ ./app/

# Create data directory
RUN mkdir -p /workspace/data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "app/inference.py"]
