#!/bin/bash

echo "🚀 最小限設定でStreamlitサーバーを起動中..."

# 環境変数の設定（最小限）
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
export STREAMLIT_RUNNER_MAGIC_ENABLED=false

# メモリ制限を緩和
ulimit -v 4194304  # 4GBメモリ制限

# プロセス管理
trap 'echo "🛑 プロセス終了中..."; kill $(jobs -p) 2>/dev/null; exit' EXIT

# Streamlitサーバーを起動（最小限設定）
echo "📊 Streamlitサーバーを起動します..."
python -m streamlit.web.cli run app/main.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    --global.developmentMode false

echo "✅ Streamlitサーバーが終了しました"
