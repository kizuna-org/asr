# 2025-06-29 FRP Jenkins Hosting Setup

## 概要
frp（Fast Reverse Proxy）を使用してJenkinsとGiteaをHTTP/HTTPSでホスティングする環境を構築しました。
**更新**: frpサーバーを別の場所に分離し、Cloudflare Tunnel経由で外部公開するように構成を変更しました。

## アーキテクチャ

### 更新後のアーキテクチャ
```
[Jenkins/Gitea Container] → [frpc] → インターネット → [外部frpsサーバー] → [Cloudflare Tunnel] → [外部公開]
```

### frpサーバー分離の利点
- **セキュリティ向上**: 本番環境のJenkins/Giteaサーバーで不要なポートを開放しない
- **可用性向上**: 専用のfrpsサーバーで安定した外部接続
- **管理の分離**: ホスティング環境とCI/CD環境の分離

## frpとは
frpは高速な逆プロキシサーバーで、NAT/ファイアウォール経由でのサービス公開を可能にします。主な特徴：

- TCP/UDP/HTTP/HTTPS プロトコル対応
- ロードバランシング機能
- 暗号化通信対応
- ヘルスチェック機能
- Web管理ダッシュボード

## 実装内容

### 1. frpサーバー分離（`/frp-server/`）

frpサーバーを専用のcomposeファイルに分離しました：

#### frpサーバー構成
```yaml
# frp-server/compose.yaml
services:
  frps:
    image: snowdreamtech/frps:latest
    ports:
      - "127.0.0.1:7000:7000"   # 内部のみ
      - "127.0.0.1:8000:8000"   # 内部のみ
      - "127.0.0.1:80:80"       # Cloudflare Tunnel経由
      - "127.0.0.1:443:443"     # Cloudflare Tunnel経由
      - "50000:50000"           # Jenkins エージェント（直接）
      - "2222:2222"             # Gitea SSH（直接）

  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel --no-autoupdate run
    environment:
      - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
```

#### Cloudflare Tunnel設定
```yaml
# frp-server/cloudflared/config.yml
ingress:
  - hostname: jenkins.yourdomain.com
    service: http://frps:80
    originRequest:
      httpHostHeader: jenkins.yourdomain.com
  
  - hostname: gitea.yourdomain.com
    service: http://frps:80
    originRequest:
      httpHostHeader: gitea.yourdomain.com
  
  - hostname: frp-admin.yourdomain.com
    service: http://frps:8000
```

### 2. pocディレクトリの構成変更

#### frpsサービス削除
- pocディレクトリからfrpsサービスを削除
- frpcのみ残し、外部frpsサーバーに接続

#### frpc設定更新
```toml
# poc/frp/frpc.toml
serverAddr = "${FRP_SERVER_HOST:-your-frp-server.com}"
serverPort = 7000

[[proxies]]
name = "jenkins-http"
customDomains = ["jenkins.yourdomain.com"]

[[proxies]]
name = "gitea-http"
customDomains = ["gitea.yourdomain.com"]
```

## セットアップ手順

### 1. frpサーバーの起動（ホスティングサーバー）

```bash
cd frp-server

# 環境変数設定
cp env.template .env
# .envを編集（Cloudflare設定等）

# Cloudflare認証情報設定
cp cloudflared/credentials.json.template cloudflared/credentials.json
# credentials.jsonを編集

# サービス起動
./setup.sh
```

### 2. Cloudflare設定

#### DNS設定
```
jenkins.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
gitea.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
frp-admin.yourdomain.com → jenkins-frp-tunnel.cfargotunnel.com
```

### 3. Jenkins/Gitea環境の起動（CI/CDサーバー）

```bash
cd poc

# 環境変数設定
cp env.template .env
# FRP_SERVER_HOST=your-frp-server.com を設定

# サービス起動（frpサーバーが起動済みであること）
docker compose --progress plain up -d
```

## アクセス方法

### 外部公開URL（Cloudflare Tunnel経由）
- **Jenkins**: https://jenkins.yourdomain.com
- **Gitea**: https://gitea.yourdomain.com
- **FRP Dashboard**: https://frp-admin.yourdomain.com

### 直接アクセス（開発時）
- **Jenkins**: http://localhost:8080
- **Gitea**: http://localhost:3000

### frpサーバー管理（ホスティングサーバー内部）
- **FRP Dashboard**: http://localhost:8000

## セキュリティ設定

### ネットワーク分離
- **frpsサーバー**: HTTP/HTTPSポートは内部のみ（127.0.0.1）
- **CI/CDサーバー**: frpcのみで外部接続
- **Cloudflare**: DDoS保護、WAF、SSL/TLS

### ファイアウォール設定

#### frpsサーバー
```bash
# 最小限のポート開放
sudo ufw allow 50000/tcp  # Jenkins エージェント
sudo ufw allow 2222/tcp   # Gitea SSH
# HTTP/HTTPSは開放不要（Cloudflare Tunnel経由）
```

#### CI/CDサーバー
```bash
# 開発用ポートのみ（必要に応じて）
sudo ufw allow 8080/tcp   # Jenkins（開発時のみ）
sudo ufw allow 3000/tcp   # Gitea（開発時のみ）
```

## 監視とトラブルシューティング

### ログ確認

#### frpsサーバー
```bash
cd frp-server
docker compose logs frps
docker compose logs cloudflared
```

#### CI/CDサーバー
```bash
cd poc
docker compose logs frpc
```

### 接続テスト

#### frpc → frps接続確認
```bash
# CI/CDサーバーから
docker compose exec frpc cat /tmp/frpc.log
```

#### Cloudflare Tunnel確認
```bash
# frpsサーバーから
curl -H "Host: jenkins.yourdomain.com" http://localhost:80
```

### よくある問題

1. **frpc接続エラー**
   - FRP_SERVER_HOSTの設定確認
   - ネットワーク接続確認
   - 認証トークンの一致確認

2. **Cloudflare Tunnel接続エラー**
   - TUNNEL_TOKENの確認
   - DNS設定の確認
   - credentials.jsonの確認

3. **ドメイン解決エラー**
   - Cloudflare DNS設定確認
   - キャッシュクリア

## 運用のベストプラクティス

### バックアップ
```bash
# frpsサーバー設定
cd frp-server
tar -czf frps-backup-$(date +%Y%m%d).tar.gz *.toml *.yml .env cloudflared/

# CI/CD設定
cd poc
tar -czf cicd-backup-$(date +%Y%m%d).tar.gz frp/ jenkins/ gitea/ *.yaml
```

### 更新手順
```bash
# frpsサーバー更新
cd frp-server
docker compose pull && docker compose up -d

# CI/CDサーバー更新
cd poc
docker compose pull && docker compose up -d
```

### 監視項目
- frps/frpc間の接続状態
- Cloudflare Tunnel状態
- SSL証明書の有効期限
- サービスのヘルスチェック

## 本番環境への展開

### 1. frpsサーバーの設置
- 高可用性のクラウドインスタンス
- 固定IPアドレス
- 適切なセキュリティグループ設定

### 2. Cloudflare設定
- SSL/TLS: Full (strict)
- Security Level: Medium以上
- Bot Fight Mode: ON

### 3. CI/CDサーバー設定
- 本番用FRP_SERVER_HOSTの設定
- セキュリティポリシーの適用
- バックアップスケジュールの設定

## 次のステップ

### Webhook設定
Cloudflare Tunnel経由でのGitea-Jenkins Webhook連携

### 負荷分散
複数frpsサーバーでの負荷分散設定

### 監視強化
Prometheus/Grafanaによるメトリクス監視

## 参考資料
- [frp GitHub Repository](https://github.com/fatedier/frp)
- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Docker Hub - snowdreamtech/frps](https://hub.docker.com/r/snowdreamtech/frps)
- [Cloudflare Tunnel Examples](https://github.com/cloudflare/cloudflared) 
