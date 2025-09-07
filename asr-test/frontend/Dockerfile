# frontend/Dockerfile
FROM python:3.9-slim

# Accept proxy arguments (empty by default for local builds)
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=localhost,127.0.0.1,asr-api

# Set proxy environment variables for build
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# Configure apt proxy only when proxy is provided
RUN if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then \
      echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/01proxy; \
      echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/01proxy; \
    fi


WORKDIR /app

# 依存関係をインストール
# run.shでrsyncしてからbuildするため、コンテキストはプロジェクトルートになる
COPY ./frontend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY ./frontend/app.py /app/

# ポートを開放 (ドキュメント用、実際にはcomposeでマッピング)
EXPOSE 8501
