# FRP Server セットアップ (Cloudflare Dashboard管理)

このディレクトリには、frpサーバーの設定が含まれています。Cloudflare Tunnelの設定はCloudflareダッシュボードで行います。

## 概要

- **frps**: 逆プロキシサーバー（Jenkins/Gitea等のサービスを公開）
- **Cloudflare Tunnel**: Cloudflareダッシュボードで管理

## アーキテクチャ

```
[Jenkins/Gitea Container] → [frpc] → インターネット → [frps.shiron.dev] → [Cloudflare Tunnel] → [外部公開]
```

## セットアップ手順

### 1. 前提条件

- Cloudflareアカウント
- shiron.devドメインがCloudflareで管理されている
- Docker & Docker Compose

### 2. frpサーバー起動

```bash
cd frp-server

# 環境変数設定
cp env.template .env
# .envを編集（FRP_TOKEN等）

# サービス起動
./setup.sh
```

### 3. Cloudflare Dashboard設定

#### 3.1 Tunnelの作成
1. Cloudflareダッシュボードにログイン
2. Zero Trust → Access → Tunnels
3. "Create a tunnel" をクリック
4. "Cloudflared" を選択
5. Tunnel名を入力（例: jenkins-frp-tunnel）

#### 3.2 ルートの設定
以下のPublic Hostnameを追加：

| Subdomain | Domain | Service |
|-----------|--------|---------|
| jenkins | shiron.dev | HTTP, frps.shiron.dev:80 |
| gitea | shiron.dev | HTTP, frps.shiron.dev:80 |
| frp-admin | shiron.dev | HTTP, frps.shiron.dev:8000 |

#### 3.3 cloudflaredの起動
Tunnelを保存すると、cloudflaredの起動コマンドが表示されます：

```bash
# 例：
sudo cloudflared service install your-tunnel-token
```

## 公開URL

### 外部公開URL（Cloudflare Tunnel経由）
- **Jenkins**: https://jenkins.shiron.dev
- **Gitea**: https://gitea.shiron.dev
- **FRP Dashboard**: https://frp-admin.shiron.dev

### ローカルアクセス（サーバー内部）
- **FRP Dashboard**: http://localhost:8000
- **FRP管理**: http://localhost:7000

## 設定ファイル

| ファイル | 説明 |
|---------|------|
| `compose.yaml` | Docker Compose設定 |
| `frps.toml` | frpサーバー設定 |
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
sudo ufw allow 80/tcp     # HTTP（Cloudflare Tunnel経由）
sudo ufw allow 443/tcp    # HTTPS（Cloudflare Tunnel経由）
sudo ufw allow 7000/tcp   # frp管理
sudo ufw allow 8000/tcp   # frpダッシュボード
sudo ufw allow 50000/tcp  # Jenkins エージェント
sudo ufw allow 2222/tcp   # Gitea SSH
```

## 監視とログ

### ログファイル
- **frps**: `logs/frps.log`

### ヘルスチェック
```bash
# サービス状態確認
docker compose ps

# ログ確認
docker compose logs frps

# ポート確認
curl http://localhost:8000
```

## トラブルシューティング

### よくある問題

1. **frpc接続エラー**
   ```bash
   # frpサーバーログ確認
   docker compose logs frps
   
   # ポート確認
   netstat -tulpn | grep 7000
   ```

2. **Cloudflare Tunnel接続エラー**
   ```bash
   # Cloudflareダッシュボードでトンネル状態確認
   # cloudflaredサービス状態確認
   sudo systemctl status cloudflared
   ```

3. **DNS解決エラー**
   ```bash
   # DNS設定確認
   dig jenkins.shiron.dev
   nslookup jenkins.shiron.dev
   ```

### デバッグコマンド
```bash
# 全サービス再起動
docker compose restart

# 設定テスト
curl -H "Host: jenkins.shiron.dev" http://localhost:80
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
  *.toml .env logs/
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

## Cloudflare Tunnel管理コマンド

```bash
# トンネル一覧表示
cloudflared tunnel list

# トンネル情報表示
cloudflared tunnel info jenkins-frp-tunnel

# トンネル削除
cloudflared tunnel delete jenkins-frp-tunnel

# サービス停止
sudo systemctl stop cloudflared

# サービス開始
sudo systemctl start cloudflared
``` 
