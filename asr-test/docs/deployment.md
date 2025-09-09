# デプロイ & インフラ構成設計書

このドキュメントは、`docker-compose.yml` および `docker-compose.gpu.yml`、並びにローカル実行用の `run-local.sh` に基づき、本アプリケーションのデプロイ/実行手順とインフラ構成について記述します。

## 1. 概要

開発用PCでは基本的にコンテナで実行します。ローカル実行時は `run-local.sh` を使用します（直接 `python` を実行しないこと）。GPUサーバーでの実行時は `docker-compose.gpu.yml` を併用して GPU を割り当てます。

## 2. インフラストラクチャ

-   **リモートサーバー（任意）**: GPUを搭載したサーバー。NVIDIA Container Runtime が設定済みであること。
-   **プロジェクトパス**: 任意の作業パスに配置。
-   **コンテナ環境**: Docker および Docker Compose。NVIDIA Container Runtime が有効。

## 3. 実行手順

1.  **ローカル（CPU/GPU問わず）**:
    -   `asr-test/` 直下で `run-local.sh` を実行します。全ての実行はコンテナ内で行われます。
2.  **GPU サーバー上での実行**:
    -   `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
    -   事前に `docker compose build` を行う場合は、`docker compose build` を利用します。

2.  **コンテナの再起動**:
    -   `docker compose down`: リモートサーバー上で現在実行中のコンテナをすべて停止・削除する。
    -   `docker build . -t asr-app`: リモートサーバー上でプロジェクトのルートから `Dockerfile` を探し、`asr-app` というタグでDockerイメージをビルドする。
    -   `docker compose up -d`: `docker-compose.yml` に基づいてコンテナをバックグラウンドで起動する。
    -   **[注記]** 本来 `docker compose up --build` でビルドと起動を同時に実行できますが、デプロイ対象のリモートサーバーが特殊なネットワーク環境下にある等の理由で、ビルドと起動のコマンドを分離して実行する必要があることを前提としています。

3.  **動作確認**:
    -   `check_nvidia_runtime.sh`: サーバーのNVIDIA Container Runtime設定が正しいことを確認する。
    -   `docker compose exec asr-api python gpu_check.py`: `asr-api` コンテナ内からPythonスクリプトを実行し、GPUが正常に認識されていることを確認する。

4.  **ポートフォワーディング**:
    -   SSHのControlMaster機能を利用して、リモートサーバーとの間に永続的なマスターセッションを確立する。
    -   このセッションを介して、リモートサーバーのポート `58080` と `58081` をローカルPCの同名ポートに転送する。
    -   これにより、ローカルPCのブラウザから `http://localhost:58080` のようにアクセスできる。
    -   `run.sh` を `Ctrl+C` で終了すると、ポートフォワーディングも自動的に停止する。

## 4. コンテナ構成（実体）

バックエンド/フロントはそれぞれ `docker-compose.yml` でビルド・起動されます。GPU 割り当ては `docker-compose.gpu.yml` で上書きします。

- `docker-compose.yml` の主なポイント
  - `asr-api`: `backend/Dockerfile` をビルド。`58081:8000` を公開。`./data` と `./checkpoints` を `/app/data`, `/app/checkpoints` にマウント。
  - `frontend`: `frontend/Dockerfile` をビルド。`58080:8501` を公開。`BACKEND_HOST=asr-api`, `BACKEND_PORT=8000` を環境変数で指定。

- `docker-compose.gpu.yml` の主なポイント
  - `asr-api`: NVIDIA GPU を 1 枚予約（`deploy.resources.reservations.devices`）。イメージ名での起動を想定。

### 4.1. サービス詳細

-   **`asr-api` (バックエンド)**
    -   **役割**: FastAPIを用いて、学習・推論のAPIエンドポイントとWebSocketサーバーを提供する。
    -   **ビルド**: `backend/Dockerfile` からビルドされます（ローカル）。GPU環境ではビルド済みイメージを使用可。
    -   **ポート**: コンテナのポート `8000` をホストの `58081` にマッピング。
    -   **GPU**: NVIDIA GPUを1つ割り当てるように設定。
    -   **ボリューム**:
        -   `./data` -> `/app/data`: データセットを読み込むためにマウント。
        -   `./checkpoints` -> `/app/checkpoints`: 学習のチェックポイントを永続化するためにマウント。

-   **`frontend` (フロントエンド)**
    -   **役割**: Streamlitを用いて、学習の制御や結果を可視化するWeb GUIを提供する。
    -   **ビルド**: `frontend/Dockerfile` からビルドされます。
    -   **ポート**: コンテナのポート `8501` をホストの `58080` にマッピング。

## 5. 他の設計への影響

-   **パス設定 (`config.yaml`)**:
    -   `datasets.ljspeech.path` はコンテナ内のパス `/app/data/ljspeech` を指定します。
-   **ディレクトリ構成 (`main.md`)**:
    -   `checkpoints` と `data` はホスト側ディレクトリをマウントし、永続化します。

## 6. 運用ガイド

- ローカルでの操作は必ず `run-local.sh` を使用してください（直接 `python` は実行しない）。
- GPU 環境では `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` を推奨します。
