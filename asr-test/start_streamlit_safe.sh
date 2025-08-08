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
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export STREAMLIT_RUNNER_MAGIC_ENABLED=false
export PYARROW_IGNORE_IMPORT_ERROR=1

# メモリクリア
echo "🧹 メモリクリア中..."
python -c "import gc; gc.collect()"

# プロセス管理
trap 'echo "🛑 プロセス終了中..."; kill $(jobs -p) 2>/dev/null; exit' EXIT

# Streamlitサーバーを起動（メモリ制限付き）
echo "📊 Streamlitサーバーを起動します..."
ulimit -v 2097152  # 2GBメモリ制限
STREAMLIT_SERVER_HEADLESS=true \
STREAMLIT_SERVER_ENABLE_CORS=false \
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200 \
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200 \
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
streamlit run app/main.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 200 \
    --server.maxMessageSize 200 \
    --browser.gatherUsageStats false \
    --global.developmentMode false

echo "✅ Streamlitサーバーが終了しました"
