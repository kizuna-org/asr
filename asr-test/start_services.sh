#!/bin/bash

echo "🚀 リアルタイム音声認識システムを起動中..."

# Streamlitサーバーを起動
echo "📊 Streamlitサーバーを起動します..."
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 &

# FastAPIサーバーを起動
echo "🌐 FastAPIサーバーを起動します..."
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# 両方のプロセスを待機
wait
