# Dockerコマンドへのsudo追加

## 概要

セキュリティ要件に基づき、すべてのDockerコマンドに`sudo`を付けるように修正しました。これにより、Dockerコマンドの実行に必要な権限を明示的に要求するようになります。

## 変更内容

### 1. `infrastructure/gpu-server-setup.sh`

- すべてのdockerコマンドに`sudo`を追加
  - `docker ps`
  - `docker rm`
  - `docker run`
  - `docker logs`
- GitHub Container Registryへのログイン例にも`sudo`を追加

### 2. `whaled/app/subscriber.py`

- コンテナ操作コマンドに`sudo`を追加
  - `docker pull`
  - `docker run`

## 影響範囲

この変更により、以下の点に注意が必要です：

1. スクリプトを実行するユーザーは`sudo`権限を持っている必要があります
2. `sudo`コマンドを使用する際にパスワード入力が求められる場合があります
3. 自動化スクリプトでは、必要に応じて`sudo`のパスワードなし設定が必要になる場合があります

## テスト

- スクリプトが正常に動作することを確認
- Dockerコマンドが`sudo`付きで実行されることを確認
- コンテナの起動、停止、ログ表示などの基本操作が問題なく行えることを確認

## 関連ドキュメント

- [GPU Server Setup](../../infrastructure/gpu-server-setup.sh)
- [App Subscriber](../../whaled/app/subscriber.py)