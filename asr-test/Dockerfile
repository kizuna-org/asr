FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

# Accept proxy arguments
ARG HTTP_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG HTTPS_PROXY="http://http-p.srv.cc.suzuka-ct.ac.jp:8080"
ARG NO_PROXY=localhost,127.0.0.1

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

ARG DEBIAN_FRONTEND=noninteractive

# システムパッケージのインストール
ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
  && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
  && apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  gnupg \
  lsb-release \
  && rm -rf /var/lib/apt/lists/*

# Install system packages
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
  python3-dev \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
  libcudnn8-dev \
  libcudnn8 \
  cuda-nvrtc-12-3 \
  cuda-nvrtc-dev-12-3 \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
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
  libcusparse-dev-12-3 \
  && rm -rf /var/lib/apt/lists/*

# Add CUDA libraries to the path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Additional environment variables for TensorFlow GPU
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# メモリ管理の環境変数
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV MALLOC_MMAP_MAX_=65536
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install TensorFlow with GPU support
RUN pip install --no-cache-dir tensorflow==2.13.0

# Install other Python dependencies
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt && \
  pip install pandas==2.1.4

# ALSAエラー抑制の環境変数
ENV ALSA_PCM_CARD=0
ENV ALSA_PCM_DEVICE=0
ENV PYTHONWARNINGS=ignore
ENV ALSA_CONFIG_PATH=/dev/null
ENV ALSA_PCM_NAME=null
ENV PULSE_SERVER=unix:/tmp/pulse-socket
ENV PULSE_COOKIE=/tmp/pulse-cookie
ENV AUDIODEV=null
ENV AUDIODRIVER=null

# ALSA設定ファイルを作成（エラーを抑制）
RUN mkdir -p /etc/alsa/conf.d && \
  echo 'pcm.!default { type null }' > /etc/alsa/conf.d/99-null.conf && \
  echo 'ctl.!default { type null }' >> /etc/alsa/conf.d/99-null.conf && \
  echo 'pcm.!default { type null }' > /etc/asound.conf && \
  echo 'ctl.!default { type null }' >> /etc/asound.conf

# オーディオデバイスを無効化
RUN mkdir -p /dev/snd && \
  ln -sf /dev/null /dev/snd/controlC0 && \
  ln -sf /dev/null /dev/snd/pcmC0D0p && \
  ln -sf /dev/null /dev/snd/pcmC0D0c

# Set working directory
WORKDIR /app

# Copy GPU check script
COPY gpu_check.py ./gpu_check.py

# ディレクトリの作成
RUN mkdir -p data/raw data/processed models

# アプリケーションコードのコピー
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
COPY start_services.sh ./
COPY start_streamlit_safe.sh ./
COPY start_streamlit_minimal.sh ./
COPY .streamlit/ ./.streamlit/

# appディレクトリをPythonパッケージとして設定
RUN touch app/__init__.py

# 起動スクリプトに実行権限を付与
RUN chmod +x start_services.sh
RUN chmod +x start_streamlit_safe.sh
RUN chmod +x start_streamlit_minimal.sh

# Run GPU check during build (continue even if it fails)
RUN python gpu_check.py || echo "GPU check completed (warnings are OK)" || true

# Add NVIDIA Container Runtime check
RUN echo "=== NVIDIA Container Runtime Check ===" && \
  echo "Checking if we're running in NVIDIA container..." && \
  if [ -f /proc/driver/nvidia/version ]; then \
  echo "✅ NVIDIA driver detected in container"; \
  cat /proc/driver/nvidia/version; \
  else \
  echo "⚠️  NVIDIA driver not detected in container (this is normal during build)"; \
  fi && \
  echo "Container GPU check completed"

# ポートの公開
EXPOSE 8501 8000

# アプリケーションの起動
CMD ["./start_services.sh"]

