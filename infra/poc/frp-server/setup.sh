#!/bin/bash
# setup.sh - FRP Server セットアップスクリプト

set -eu

echo "=== FRP Server セットアップ ==="

# 1. 環境変数の確認
if [ ! -f .env ]; then
    echo "❌ .envファイルが見つかりません"
    echo "env.templateをコピーして.envを作成し、適切な値を設定してください"
    exit 1
fi

source .env

# 2. 必要な環境変数のチェック
required_vars=("FRP_TOKEN")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ 環境変数 $var が設定されていません"
        exit 1
    fi
done

# 3. ログディレクトリの作成
mkdir -p logs

# 4. Docker Composeでサービス起動
echo "🚀 Docker Composeでサービスを起動中..."
docker compose up -d

# 5. 起動確認
echo "⏳ サービスの起動を待機中..."
sleep 10

echo "🔍 サービスの状態を確認中..."
docker compose ps

# 6. アクセス情報の表示
echo ""
echo "=== アクセス情報 ==="
echo "📊 FRP Dashboard: http://localhost:8000"
echo "🔧 FRP Management: http://localhost:7000"
echo ""
echo "🌐 Jenkins: https://jenkins.shiron.dev"
echo "🌐 Gitea: https://gitea.shiron.dev"
echo "🔧 FRP Admin: https://frp-admin.shiron.dev"
echo ""

# 7. 次のステップの表示
echo "=== 次のステップ ==="
echo "1. Cloudflareダッシュボードでトンネルを設定"
echo "2. 以下のドメインを設定:"
echo "   - jenkins.shiron.dev → http://frps.shiron.dev:80"
echo "   - gitea.shiron.dev → http://frps.shiron.dev:80"
echo "   - frp-admin.shiron.dev → http://frps.shiron.dev:8000"
echo "3. frpクライアント（jenkins/gitea側）の起動"
echo ""
echo "✅ セットアップ完了！" 
