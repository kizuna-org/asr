#!/bin/bash

# このスクリプトはWebRTC機能が動作する直接アクセス用のURLを表示します

SSH_HOST="edu-gpu"

echo "🔍 サーバーのIPアドレスを取得しています..."
SERVER_IP=$(ssh ${SSH_HOST} "hostname -I | awk '{print \$1}'")

if [ -z "${SERVER_IP}" ]; then
    echo "❌ エラー: サーバーのIPアドレスを取得できませんでした。"
    exit 1
fi

echo ""
echo "✅ サーバーIP: ${SERVER_IP}"
echo ""
echo "📱 WebRTC機能を使用するには、以下のURLに**直接**アクセスしてください:"
echo ""
echo "   🌐 フロントエンド (Streamlit): http://${SERVER_IP}:58080"
echo "   🔧 バックエンド API:           http://${SERVER_IP}:58081"
echo ""
echo "⚠️  重要: localhost経由ではWebRTCが動作しません！"
echo "          上記のIPアドレスを使用してブラウザでアクセスしてください。"
echo ""
echo "💡 ファイアウォールで以下のポートが開放されている必要があります:"
echo "   - TCP 58080 (Streamlit frontend)"
echo "   - TCP 58081 (FastAPI backend)"
echo "   - UDP 範囲 (WebRTC media traffic)"
echo ""
