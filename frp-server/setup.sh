#!/bin/bash
# setup.sh - Cloudflare Tunnel + FRP Server セットアップスクリプト

set -e

echo "=== Cloudflare Tunnel + FRP Server セットアップ ==="

# 1. 環境変数の確認
if [ ! -f .env ]; then
    echo "❌ .envファイルが見つかりません"
    echo "env.templateをコピーして.envを作成し、適切な値を設定してください"
    exit 1
fi

source .env

# 2. 必要な環境変数のチェック
required_vars=("CLOUDFLARE_TUNNEL_TOKEN" "FRP_TOKEN")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ 環境変数 $var が設定されていません"
        exit 1
    fi
done

# 3. Cloudflare認証情報の確認
if [ ! -f cloudflared/credentials.json ]; then
    echo "❌ cloudflared/credentials.json が見つかりません"
    echo "cloudflared/credentials.json.template をコピーして適切な値を設定してください"
    exit 1
fi

# 4. ログディレクトリの作成
mkdir -p logs

# 5. Cloudflare Tunnelの設定確認
echo "✅ Cloudflare Tunnel設定を確認中..."
if ! command -v cloudflared &> /dev/null; then
    echo "⚠️  cloudflaredがインストールされていません（Docker版を使用）"
else
    echo "✅ cloudflaredが利用可能です"
fi

# 6. Docker Composeでサービス起動
echo "🚀 Docker Composeでサービスを起動中..."
docker compose up -d

# 7. 起動確認
echo "⏳ サービスの起動を待機中..."
sleep 10

echo "🔍 サービスの状態を確認中..."
docker compose ps

# 8. アクセス情報の表示
echo ""
echo "=== アクセス情報 ==="
echo "📊 FRP Dashboard (ローカル): http://localhost:8000"
echo "🌐 Jenkins (Cloudflare): https://jenkins.yourdomain.com"
echo "🌐 Gitea (Cloudflare): https://gitea.yourdomain.com"
echo "🔧 FRP Admin (Cloudflare): https://frp-admin.yourdomain.com"
echo ""
echo "⚠️  yourdomain.com を実際のドメインに置き換えてください"
echo ""

# 9. 次のステップの表示
echo "=== 次のステップ ==="
echo "1. Cloudflareダッシュボードでドメインの設定を確認"
echo "2. frpクライアント（jenkins/gitea側）の設定を更新"
echo "3. DNS設定が反映されるまで待機"
echo ""
echo "✅ セットアップ完了！" 
