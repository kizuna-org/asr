#!/bin/bash

echo "🚀 安全なStreamlitサーバーを起動中..."

# 環境変数の設定
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# メモリクリア
echo "🧹 メモリクリア中..."
python -c "import gc; gc.collect()"

# プロセス管理
trap 'echo "🛑 プロセス終了中..."; kill $(jobs -p) 2>/dev/null; exit' EXIT

# Streamlitサーバーを起動（メモリ制限付き）
echo "📊 Streamlitサーバーを起動します..."
ulimit -v 2097152  # 2GBメモリ制限
streamlit run app/main.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 200 \
    --server.maxMessageSize 200 \
    --browser.gatherUsageStats false

echo "✅ Streamlitサーバーが終了しました"
