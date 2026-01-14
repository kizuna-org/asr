#!/bin/bash
# ローカルでフロントエンドを起動し、バックエンドだけSSHトンネル経由で接続

set -e

echo "🚀 ローカルフロントエンド起動スクリプト"
echo "================================"

# バックエンドへのSSHトンネルを確立
echo "📡 バックエンドへのSSHトンネルを確立中..."
ssh -f -N -L 58081:172.16.98.181:58081 edu-gpu
echo "✅ SSHトンネル確立完了 (localhost:58081 -> server:58081)"

# フロントエンドの依存関係をインストール
cd "$(dirname "$0")/frontend"
echo "📦 フロントエンド依存関係をインストール中..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 環境変数を設定
export BACKEND_HOST="localhost"
export BACKEND_PORT="58081"

echo "🎨 フロントエンドを起動中..."
echo "📍 アクセスURL: http://localhost:8501"
echo "🔗 バックエンド接続先: http://localhost:58081 (SSH tunnel)"
echo ""
echo "✨ WebRTCはローカルで動作するため、マイクアクセスが正常に機能します"
echo "🛑 終了するには Ctrl+C を押してください"
echo ""

# フロントエンドを起動
streamlit run app.py --server.port 8501

# クリーンアップ
echo "🧹 SSHトンネルをクリーンアップ中..."
pkill -f "ssh -f -N -L 58081"
