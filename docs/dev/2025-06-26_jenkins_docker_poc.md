# 2025-06-26 Jenkins Docker PoC 環境構築

## 概要
JenkinsでDocker in Docker (DinD) および Docker out of Docker (DooD) の概念実証 (PoC) 環境を構築しました。

## 構築内容

### 1. 初期環境構築 (DinD)
- `poc` ディレクトリを作成し、`compose.yaml` と `Dockerfile` を配置。
- `compose.yaml` で `jenkins` サービスと `dind` サービスを定義し、Jenkinsコンテナ内でDockerコマンドを実行できるように設定。
- `Dockerfile` で `jenkins/jenkins:lts` イメージをベースに `docker-ce-cli` をインストール。

### 2. `docker-ce-cli` インストール問題の解決
- 初期の `Dockerfile` で `docker-ce-cli` のインストールに失敗。
- 原因: `Dockerfile` 内でのDockerリポジトリ追加時に、CPUアーキテクチャの指定が `amd64` に固定されていたため、`arm64` 環境でパッケージが見つからなかった。また、`apt-key` が非推奨であった。
- 修正:
    - `Dockerfile` を更新し、`$(dpkg --print-architecture)` を使用してビルド時にコンテナのCPUアーキテクチャを自動判別するように変更。
    - 非推奨の `apt-key` の代わりに、新しいGPGキーの管理方法 (`/etc/apt/keyrings/docker.gpg`) を使用するように修正。
    - `compose.yaml` から `version: '3'` の記述を削除。

### 3. Docker out of Docker (DooD) への変更
- JenkinsコンテナがホストマシンのDockerデーモンと直接通信するように設定を変更。
- `poc/compose.yaml` から `dind` サービスを削除。
- `jenkins` サービスにホストのDockerソケット (`/var/run/docker.sock`) をボリュームとしてマウント。

## 今後の作業
- JenkinsのWeb UIにアクセスし、初期設定を完了する。
- Jenkinsパイプラインジョブを作成し、`docker` コマンドが正常に実行できることを確認する。
