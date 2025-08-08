FROM python:3.11-slim

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
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

# ポートの公開
EXPOSE 8501 8000

# アプリケーションの起動
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

