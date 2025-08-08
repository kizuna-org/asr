#!/bin/bash

echo "🚀 リアルタイム音声認識モデル学習システムを起動中..."

# Docker Composeでアプリケーションを起動
sudo docker compose up --build

echo "✅ アプリケーションが起動しました！"
echo "🌐 ブラウザで http://localhost:8501 にアクセスしてください"
