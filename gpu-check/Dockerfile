FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
LABEL maintainer="Hugging Face"

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  libsndfile1-dev \
  tesseract-ocr \
  espeak-ng \
  python3 \
  python3-pip \
  python3-venv \
  ffmpeg \
  libnvinfer-dev \
  libnvonnxparsers-dev \
  && rm -rf /var/lib/apt/lists/*

# Add CUDA libraries to the path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow with the same version as main Dockerfile
ARG TENSORFLOW='2.16.1'
RUN pip install --no-cache-dir "tensorflow==$TENSORFLOW"

# Create user and working directory
RUN useradd -m gpucheck
WORKDIR /home/gpucheck

# Copy Python scripts to working directory
COPY check_gpu.py quick_gpu_check.py ./

# Make scripts executable
RUN chmod +x *.py

# Change ownership of the files to gpucheck user
RUN chown -R gpucheck:gpucheck /home/gpucheck

# Switch to gpucheck user
USER gpucheck

# Default command to run comprehensive GPU check
CMD ["python", "check_gpu.py"] 
