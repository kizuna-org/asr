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
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# システムパッケージのインストール
ARG PROXY_URL=http://http-p.srv.cc.suzuka-ct.ac.jp:8080/

RUN echo "Acquire::http::Proxy \"${PROXY_URL}\";" > /etc/apt/apt.conf.d/01proxy \
  && echo "Acquire::https::Proxy \"${PROXY_URL}\";" >> /etc/apt/apt.conf.d/01proxy \
  && apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  python3-dev \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# Python依存関係のコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
  pip install --no-cache-dir -r requirements.txt && \
  pip install pandas==2.1.4

# キャッシュが消えない様になるべくこれ以降を編集すること


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

# ポートの公開
EXPOSE 8501 8000

# アプリケーションの起動
CMD ["./start_services.sh"]

