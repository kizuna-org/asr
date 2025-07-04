FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

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

ARG REF=main
RUN git clone https://github.com/huggingface/transformers --depth=1 -b $REF
# Install transformers without dev-tensorflow to avoid version conflicts
RUN pip install --no-cache-dir -e ./transformers[testing]

COPY ./scripts/requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python setup.py develop

# Copy local dataset
COPY ./datasets /opt/datasets/

# Copy dataset download script
COPY ./scripts/download_dataset.py /opt/download_dataset.py

# Pre-download the LJSpeech dataset
RUN python /opt/download_dataset.py

# Verify dataset download
RUN ls -la /opt/datasets/ && \
    echo "Dataset directory contents:" && \
    find /opt/datasets -type f -name "*.tfrecord*" | wc -l && \
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

# Copy LJSpeech learning script
COPY ./scripts/ljspeech_demo.py /opt/ljspeech_demo.py
COPY ./scripts/fastspeech2_model.py /opt/fastspeech2_model.py
COPY ./scripts/simple_transformer_tts.py /opt/simple_transformer_tts.py
COPY ./scripts/transformer_tts_model.py /opt/transformer_tts_model.py
COPY ./scripts/fastspeech2_model.py /opt/fastspeech2_model.py

# Create a script to run GPU check and then the main application
RUN echo '#!/bin/bash' > /opt/run_with_gpu_check.sh && \
    echo 'echo "=== 実行前 GPU チェック ==="' >> /opt/run_with_gpu_check.sh && \
    echo 'python /opt/check_gpu.py' >> /opt/run_with_gpu_check.sh && \
    echo 'echo "=== GPU チェック完了、メインアプリケーション開始 ==="' >> /opt/run_with_gpu_check.sh && \
    echo 'exec python ljspeech_demo.py "$@"' >> /opt/run_with_gpu_check.sh && \
    chmod +x /opt/run_with_gpu_check.sh

# Set working directory
WORKDIR /opt

# Default command to run with GPU check first
CMD ["./run_with_gpu_check.sh", "--mode", "full", "--model", "transformer_tts"]
