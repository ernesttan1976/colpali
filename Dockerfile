FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    wget \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# First ensure pip is up to date
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first for better caching
COPY requirements.txt .

# Create a modified requirements file without PyTorch packages
RUN grep -v "torch\|torchaudio\|torchvision" requirements.txt > requirements_filtered.txt

# Install PyTorch with CUDA 12.6 support
# Using version 2.7.0+cu126 which is available with CUDA 12.6
RUN pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
RUN pip install -r requirements_filtered.txt

# Copy application files
COPY . .

# Create directories for storage
RUN mkdir -p ./data/embeddings_db ./models

# Expose port
EXPOSE 7860

# Copy verification script and startup script
COPY verify_cuda.py .
COPY start-app.sh .
RUN chmod +x start-app.sh

# Don't try to install flash-attn at build time - it's too slow and often fails
# Instead, we'll let the app try to install it at runtime if needed
RUN apt-get update && \
    apt-get install -y ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Command to run the application
CMD ["./start-app.sh"]