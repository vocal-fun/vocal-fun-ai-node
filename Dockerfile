# Start with NVIDIA CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.12.8 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA 12.6 support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file
COPY .env .

# Copy the rest of the application
COPY . .

# Expose port for the server
EXPOSE 8000

# Command to run the application
CMD ["python", "launcher.py"] 