# backend/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# apt-getを非対話的に実行するための設定
ENV DEBIAN_FRONTEND=noninteractive

# Pythonとpipをインストール
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python, pip3 -> pip のシンボリックリンクを作成
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# 依存関係をインストール
# run.shでrsyncしてからbuildするため、コンテキストはプロジェクトルートになる
COPY ./backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY ./backend/app /app/app
COPY ./backend/config.yaml /app/config.yaml

# ポートを開放 (ドキュメント用、実際にはcomposeでマッピング)
EXPOSE 8000

