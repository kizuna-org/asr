# backend/Dockerfile
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# apt-getを非対話的に実行するための設定
ENV DEBIAN_FRONTEND=noninteractive

# Accept proxy arguments (empty by default for local builds)
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1"

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# Configure apt proxy only when proxy is provided
RUN if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then \
    echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/01proxy; \
    echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/01proxy; \
    fi

# Pythonとpipをインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    file \
    bsdutils \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python, pip3 -> pip のシンボリックリンクを作成
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# 依存関係をインストール
# run.shでrsyncしてからbuildするため、コンテキストはプロジェクトルートになる
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# アプリケーションのソースコードをコピー
COPY backend/app /app/app
COPY backend/config.yaml /app/config.yaml
COPY backend/download_ljspeech.py /app/download_ljspeech.py
COPY backend/test_training_fix.py /app/test_training_fix.py
COPY gpu_check.py /app/gpu_check.py
RUN chmod +x /app/download_ljspeech.py
RUN chmod +x /app/gpu_check.py

# ポートを開放 (ドキュメント用、実際にはcomposeでマッピング)
EXPOSE 8000

