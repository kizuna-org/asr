# Rovo Dev CI/CD System

GCP Pub/SubとGitHub Actionsを活用したイベント駆動型CI/CDシステム

## 🏗️ アーキテクチャ概要

このシステムは、GitHubリポジトリへのプッシュをトリガーとして、機械学習モデルの学習やデータ処理などのアプリケーションを自動でビルド・実行し、成果物を公開するCI/CDシステムです。

### 主要コンポーネント

- **GitHub Actions**: パイプラインのトリガーとオーケストレーション
- **GCP Pub/Sub**: システムコンポーネント間の非同期メッセージング
- **GPUサーバー**: アプリケーションのビルドと実行
- **GitHub Container Registry (GHCR)**: コンテナイメージの保存
- **Cloudflare R2**: ログとステータス情報の集約
- **Cloudflare Pages**: ダッシュボードによる可視化
- **Hugging Face Hub**: 成果物の公開

## 🚀 セットアップ

### 1. GCP Pub/Sub の設定

```bash
cd infrastructure
chmod +x setup-pubsub.sh
./setup-pubsub.sh
```

### 2. GitHub Secrets の設定

以下のシークレットをGitHubリポジトリに追加してください：

- `GCP_PROJECT_ID`: GCPプロジェクトID
- `GCP_SA_KEY`: サービスアカウントキー（Base64エンコード済み）

### 3. GPUサーバーの設定

```bash
cd infrastructure
chmod +x gpu-server-setup.sh
./gpu-server-setup.sh
```

### 4. 環境変数の設定

GPUサーバーで以下の環境変数を設定してください：

```bash
# GCP設定
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Cloudflare R2設定
export R2_ENDPOINT_URL="https://your-account-id.r2.cloudflarestorage.com"
export R2_ACCESS_KEY_ID="your-r2-access-key"
export R2_SECRET_ACCESS_KEY="your-r2-secret-key"
export R2_BUCKET_NAME="your-bucket-name"

# Hugging Face設定
export HF_TOKEN="your-huggingface-token"
```

### 5. Docker Composeでの起動（オプション）

```bash
# 環境変数を設定
cp .env.example .env
# .envファイルを編集

# サービス起動
docker-compose up -d

# ログ確認
docker-compose logs -f
```

### 6. Systemdサービスでの起動

```bash
# サービス有効化
sudo systemctl enable whaled-build whaled-app

# サービス開始
sudo systemctl start whaled-build whaled-app

# ステータス確認
sudo systemctl status whaled-build whaled-app
```

## 📊 ダッシュボード

ダッシュボードをCloudflare Pagesにデプロイ：

1. `dashboard/` ディレクトリをCloudflare Pagesにデプロイ
2. `dashboard.js` 内の `r2BaseUrl` を設定
3. R2バケットのCORS設定を行う

## 🔄 実行フロー

1. **コードプッシュ**: `main`ブランチに`git push`
2. **GHAトリガー**: GitHub Actionsが`build-triggers`トピックにメッセージ発行
3. **ビルドプロセス**: Build SubscriberがコンテナイメージをビルドしてGHCRにプッシュ
4. **実行トリガー**: ビルド成功時に`app-triggers`トピックにメッセージ発行
5. **実行プロセス**: App Subscriberがイメージをプルして実行
6. **タスク実行**: アプリケーションコンテナが主処理を実行
7. **成果物公開**: 完了後、成果物をHugging Face Hubにアップロード

## 📁 ディレクトリ構造

```
.
├── .github/workflows/
│   └── ci-cd.yml                 # GitHub Actions ワークフロー
├── whaled/
│   ├── build/
│   │   ├── subscriber.py         # ビルドサブスクライバー
│   │   └── Dockerfile
│   └── app/
│       ├── subscriber.py         # アプリサブスクライバー
│       └── Dockerfile
├── app/
│   ├── main.py                   # アプリケーションテンプレート
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── index.html                # ダッシュボードUI
│   └── dashboard.js
├── infrastructure/
│   ├── setup-pubsub.sh          # Pub/Sub設定スクリプト
│   └── gpu-server-setup.sh      # GPUサーバー設定スクリプト
├── docker-compose.yml
└── README.md
```

## 🛠️ カスタマイズ

### アプリケーションのカスタマイズ

`app/main.py` を編集して、独自の機械学習タスクやデータ処理を実装してください。

### ダッシュボードのカスタマイズ

`dashboard/dashboard.js` を編集して、R2からの実際のデータ取得ロジックを実装してください。

## 📝 ログとモニタリング

- **ビルドログ**: `/{jobId}/build.log`
- **実行ログ**: `/{jobId}/app.log`
- **ステータス**: `/{jobId}/status.json`

すべてのログとステータスはCloudflare R2に保存され、ダッシュボードで確認できます。

## 🔧 トラブルシューティング

### サービスログの確認

```bash
# Build subscriber
sudo journalctl -u whaled-build -f

# App subscriber
sudo journalctl -u whaled-app -f
```

### Docker Composeログの確認

```bash
docker-compose logs build-subscriber
docker-compose logs app-subscriber
```

### 権限エラー

Docker実行権限を確認：
```bash
sudo usermod -aG docker $USER
# ログアウト・ログインが必要
```

## 📚 参考資料

- [アーキテクチャドキュメント](docs/architecture.md)
- [実装履歴](docs/dev/)

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。