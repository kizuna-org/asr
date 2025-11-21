#!/bin/bash

# 機能ポリシー警告を抑制するための環境変数を設定
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false
export STREAMLIT_RUNNER_MAGIC_ENABLED=false
export STREAMLIT_RUNNER_INSTALL_TRACER=false

# ブラウザの機能ポリシーを設定
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false

# 機能ポリシー関連の警告を抑制
export PYTHONWARNINGS="ignore::UserWarning"

echo "🚀 Streamlitを機能ポリシー警告抑制モードで起動中..."

# Streamlitを起動
streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false --client.showErrorDetails=false --runner.magicEnabled=false --runner.installTracer=false





