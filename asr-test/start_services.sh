#!/bin/bash

echo "🚀 リアルタイム音声認識システムを起動中..."

# 環境変数の設定
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# メモリ設定
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536

# プロセス管理の改善
trap 'kill $(jobs -p)' EXIT

# Streamlitサーバーを起動
echo "📊 Streamlitサーバーを起動します..."
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false &

# 少し待機
sleep 3

# FastAPIサーバーを起動
echo "🌐 FastAPIサーバーを起動します..."
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1 &

# 両方のプロセスを待機
wait
