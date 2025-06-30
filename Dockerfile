FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Hugging Face"
LABEL description="M-AILABS Multi-Speaker Speech Synthesis with TensorFlow"

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ARG DEBIAN_FRONTEND=noninteractive

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

# Install TensorFlow first with a compatible version
ARG TENSORFLOW='2.16.1'
RUN pip install --no-cache-dir "tensorflow==$TENSORFLOW"

# Install requests for M-AILABS dataset download
RUN pip install --no-cache-dir "requests>=2.28.0"

# Install tensorflow-probability with compatible version
RUN pip install --no-cache-dir "tensorflow-probability<0.25"

# Install additional dependencies for M-AILABS multi-speaker script
RUN pip install --no-cache-dir "librosa>=0.10.0"
RUN pip install --no-cache-dir "matplotlib>=3.5.0"
RUN pip install --no-cache-dir "numpy>=1.21.0"
RUN pip install --no-cache-dir "soundfile>=0.12.0"

# Install pydub
RUN pip install --no-cache-dir "pydub>=0.25.1"

ARG REF=main
RUN git clone https://github.com/huggingface/transformers --depth=1 -b $REF
# Install transformers without dev-tensorflow to avoid version conflicts
RUN pip install --no-cache-dir -e ./transformers[testing]

RUN pip uninstall -y torch flax
RUN pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python setup.py develop

# Create M-AILABS dataset directory
RUN mkdir -p /opt/datasets/mailabs

# Copy dataset download script
COPY ./scripts/download_dataset.py /opt/download_dataset.py

# Pre-download the M-AILABS dataset
RUN python /opt/download_dataset.py

# Verify M-AILABS dataset download
RUN ls -la /opt/datasets/ && \
    echo "M-AILABS dataset directory contents:" && \
    find /opt/datasets -type f -name "*.wav" | wc -l && \
    echo "Total dataset size:" && \
    du -sh /opt/datasets/

# Copy test script
COPY ./scripts/test_dataset.py /opt/test_dataset.py

# Copy GPU check scripts for debugging
COPY ./gpu-check/check_gpu.py /opt/check_gpu.py
COPY ./gpu-check/quick_gpu_check.py /opt/quick_gpu_check.py

# Run detailed GPU check before dataset testing
RUN echo "=== 詳細 GPU チェック実行中 ===" && \
    python /opt/check_gpu.py || echo "GPU チェック完了（警告があっても続行）"

# Test dataset loading
RUN python /opt/test_dataset.py

# Create outputs directory for visualizations
RUN mkdir -p /opt/outputs

# Copy M-AILABS multi-speaker learning script
COPY ./scripts/mailabs_demo.py /opt/mailabs_demo.py

# Create a script to run GPU check and then the main application
RUN echo '#!/bin/bash' > /opt/run_with_gpu_check.sh && \
    echo 'echo "=== 実行前 GPU チェック ==="' >> /opt/run_with_gpu_check.sh && \
    echo 'python /opt/check_gpu.py' >> /opt/run_with_gpu_check.sh && \
    echo 'echo "=== GPU チェック完了、M-AILABS マルチスピーカー学習開始 ==="' >> /opt/run_with_gpu_check.sh && \
    echo 'exec python mailabs_demo.py "$@"' >> /opt/run_with_gpu_check.sh && \
    chmod +x /opt/run_with_gpu_check.sh

# Create a non-root user for security
RUN groupadd -r ailabs && useradd -r -g ailabs -d /opt -s /bin/bash ailabs

# Change ownership of the working directory and virtual environment to the non-root user
RUN chown -R ailabs:ailabs /opt /opt/venv

# Set working directory
WORKDIR /opt

# Switch to non-root user
USER ailabs

# Default command to run M-AILABS multi-speaker training with GPU check first
CMD ["./run_with_gpu_check.sh"]
