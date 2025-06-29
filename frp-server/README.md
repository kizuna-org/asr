# FRP Server + Cloudflare Tunnel セットアップ

このディレクトリには、frpサーバーとCloudflare Tunnelを組み合わせたホスティング環境の設定が含まれています。

## 概要

- **frps**: 逆プロキシサーバー（Jenkins/Gitea等のサービスを公開）
- **Cloudflare Tunnel**: セキュアな外部公開（直接ポート開放不要）

## アーキテクチャ

```
[Jenkins/Gitea] → [frpc] → インターネット → [Cloudflare] → [cloudflared] → [frps] → [Dashboard]
```

## セットアップ手順

### 1. 前提条件

- Cloudflareアカウント
- ドメインがCloudflareで管理されている
- Docker & Docker Compose

### 2. Cloudflare Tunnel作成

```bash
# cloudflaredをインストール（ローカル設定用）
# macOS
brew install cloudflared

# Ubuntu/Debian
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

```bash
# Cloudflareにログイン
cloudflared tunnel login

# Tunnelを作成
cloudflared tunnel create jenkins-frp-tunnel

# Tunnel情報を確認
cloudflared tunnel list
```

### 3. 認証情報設定

```bash
# 認証情報をコピー
cp cloudflared/credentials.json.template cloudflared/credentials.json

# credentials.jsonを編集
# - AccountTag: CloudflareアカウントID
# - TunnelSecret: Tunnel作成時に生成された秘密鍵
# - TunnelID: Tunnel作成時に生成されたID
# - TunnelName: jenkins-frp-tunnel
```

### 4. 環境変数設定

```bash
# 環境変数ファイルをコピー
cp env.template .env

# .envを編集
nano .env
```

必要な設定項目：
- `CLOUDFLARE_TUNNEL_TOKEN`: Cloudflareダッシュボードから取得
- `FRP_TOKEN`: frpサーバーとクライアント間の認証トークン
- その他認証情報

### 5. DNS設定

Cloudflareダッシュボードで以下のCNAMEレコードを追加：

```
jenkins.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
gitea.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
frp-admin.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
```

### 6. サービス起動

```bash
# 自動セットアップ実行
./setup.sh

# または手動実行
docker compose up -d
```

## アクセス方法

### 外部公開URL（Cloudflare Tunnel経由）
- **Jenkins**: https://jenkins.yourdomain.com
- **Gitea**: https://gitea.yourdomain.com
- **FRP Dashboard**: https://frp-admin.yourdomain.com

### ローカルアクセス（サーバー内部）
- **FRP Dashboard**: http://localhost:8000
- **FRP管理**: http://localhost:7000

## 設定ファイル

| ファイル | 説明 |
|---------|------|
| `compose.yaml` | Docker Compose設定 |
| `frps.toml` | frpサーバー設定 |
| `cloudflared/config.yml` | Cloudflare Tunnel設定 |
| `cloudflared/credentials.json` | Cloudflare認証情報 |
| `.env` | 環境変数 |

## セキュリティ設定

### frp認証
- トークンベース認証
- 管理ダッシュボードの認証保護

### Cloudflare保護
- DDoS保護
- WAF (Web Application Firewall)
- SSL/TLS暗号化

### ファイアウォール設定
```bash
# 必要なポートのみ開放
sudo ufw allow 50000/tcp  # Jenkins エージェント
sudo ufw allow 2222/tcp   # Gitea SSH
# HTTP/HTTPSは直接開放不要（Cloudflare Tunnel経由）
```

## 監視とログ

### ログファイル
- **frps**: `logs/frps.log`
- **cloudflared**: `logs/cloudflared.log`

### ヘルスチェック
```bash
# サービス状態確認
docker compose ps

# ログ確認
docker compose logs frps
docker compose logs cloudflared

# Cloudflare Tunnel状態確認
curl -H "Host: jenkins.yourdomain.com" http://localhost:80
```

## トラブルシューティング

### よくある問題

1. **Tunnel接続エラー**
   ```bash
   # 認証情報確認
   docker compose logs cloudflared
   
   # Tunnel状態確認
   cloudflared tunnel info jenkins-frp-tunnel
   ```

2. **DNS解決エラー**
   ```bash
   # DNS設定確認
   dig jenkins.yourdomain.com
   nslookup jenkins.yourdomain.com
   ```

3. **frp接続エラー**
   ```bash
   # frpサーバーログ確認
   docker compose logs frps
   
   # ポート確認
   netstat -tulpn | grep 7000
   ```

### デバッグコマンド
```bash
# 全サービス再起動
docker compose restart

# 設定テスト
cloudflared tunnel --config cloudflared/config.yml ingress validate

# 手動Tunnel実行（デバッグ）
docker run --rm -v $(pwd)/cloudflared:/etc/cloudflared \
  cloudflare/cloudflared:latest tunnel --config /etc/cloudflared/config.yml run
```

## 本番環境設定

### SSL/TLS設定
Cloudflare設定で「Full (strict)」に設定

### セキュリティ強化
- Access Policies設定
- Rate Limiting設定
- Bot Fight Mode有効化

### バックアップ
```bash
# 設定ファイルのバックアップ
tar -czf frp-server-backup-$(date +%Y%m%d).tar.gz \
  *.toml *.yml .env cloudflared/
```

## 更新手順

```bash
# イメージ更新
docker compose pull

# サービス再起動
docker compose up -d

# 動作確認
./setup.sh
``` 
