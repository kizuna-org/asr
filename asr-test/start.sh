#!/bin/bash

echo "🚀 リアルタイム音声認識モデル学習システムを起動中..."

# Docker Composeでアプリケーションを起動
sudo docker compose up --build

echo "✅ アプリケーションが起動しました！"
echo "🌐 リアルタイム音声認識: http://localhost:58080/static/index.html"
echo "🌐 モデル学習: http://localhost:58081"
