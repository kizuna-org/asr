FROM python:3.11-slim

ARG HTTP_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080
ARG HTTPS_PROXY=http://http-p.srv.cc.suzuka-ct.ac.jp:8080
ARG NO_PROXY=localhost,127.0.0.1

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# メモリ管理の環境変数
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV MALLOC_MMAP_MAX_=65536

# システムパッケージのインストール
ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
  && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
  && apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  portaudio19-dev \
  python3-dev \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# Python依存関係のコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ディレクトリの作成
RUN mkdir -p data/raw data/processed models

# アプリケーションコードのコピー
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/
COPY start_services.sh ./

# appディレクトリをPythonパッケージとして設定
RUN touch app/__init__.py

# 起動スクリプトに実行権限を付与
RUN chmod +x start_services.sh

# ポートの公開
EXPOSE 8501 8000

# アプリケーションの起動
CMD ["./start_services.sh"]

