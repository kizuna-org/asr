FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ARG DEBIAN_FRONTEND=noninteractive

# First stage: Update package lists
RUN apt-get update

# Second stage: Install packages
RUN apt-get install -y --no-install-recommends \
  git \
  libsndfile1-dev \
  tesseract-ocr \
  espeak-ng \
  python3 \
  python3-pip \
  python3-venv \
  ffmpeg \
  libnvinfer-dev \
  libnvonnxparsers-dev

RUN apt-get install -y --no-install-recommends \
  libcudnn8-dev \
  libcudnn8 \
  cuda-nvrtc-12-3 \
  cuda-nvrtc-dev-12-3

RUN apt-get install -y --no-install-recommends \
  cuda-cccl-12-3 \
  libcublas-12-3 \
  libcublas-dev-12-3 \
  libcufft-12-3 \
  libcufft-dev-12-3 \
  libcurand-12-3 \
  libcurand-dev-12-3 \
  libcusolver-12-3 \
  libcusolver-dev-12-3 \
  libcusparse-12-3 \
  libcusparse-dev-12-3

# Add CUDA libraries to the path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Additional environment variables for TensorFlow GPU
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

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
