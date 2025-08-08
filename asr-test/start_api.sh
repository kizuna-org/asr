#!/bin/bash

echo "🚀 リアルタイム音声認識APIサーバーを起動中..."

# アプリケーションディレクトリに移動
cd /app

# FastAPIサーバーを起動
echo "🌐 FastAPIサーバーを起動します..."
echo "📡 API: http://localhost:8000"
echo "🎤 WebUI: http://localhost:8000/static/index.html"
echo "📚 API Docs: http://localhost:8000/docs"

# uvicornでFastAPIサーバーを起動
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
