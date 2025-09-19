# デプロイ & インフラ構成設計書

このドキュメントは、`docker-compose.yml` および `docker-compose.gpu.yml`、並びにローカル実行用の `run-local.sh` に基づき、本アプリケーションのデプロイ/実行手順とインフラ構成について記述します。

## 1. 概要

開発用PCでは基本的にコンテナで実行します。ローカル実行時は `run-local.sh` を使用します（直接 `python` を実行しないこと）。GPUサーバーでの実行時は `docker-compose.gpu.yml` を併用して GPU を割り当てます。`./asr-test` は別ホストのGPUサーバーで動作するため、ローカルでは `run-local.sh` を必ず使用します。

## 2. インフラストラクチャ

- **リモートサーバー（任意）**: GPUを搭載したサーバー。NVIDIA Container Runtime が設定済みであること。
- **プロジェクトパス**: 任意の作業パスに配置。
- **コンテナ環境**: Docker および Docker Compose。NVIDIA Container Runtime が有効。

## 3. 実行手順

1. **ローカル（CPU/GPU問わず）**
   - `asr-test/` 直下で `./run-local.sh` を実行します。全ての実行はコンテナ内で行われます。
2. **GPU サーバー上での実行**
   - `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
   - 事前にビルドする場合は `docker compose build` を利用します。

3. **コンテナの再起動**
   - `docker compose down`: 現在実行中のコンテナを停止・削除します。
   - `docker build . -t asr-app`: ルートから `backend/Dockerfile` を参照してビルドします（GPU用は別イメージも可）。
   - `docker compose up -d`: `docker-compose.yml` に基づいて起動します。
   - 注記: `docker compose up --build` で同時実行も可能ですが、環境によってはビルドと起動を分離してください。

4. **動作確認**
   - `check_nvidia_runtime.sh`: NVIDIA Container Runtime設定の確認。
   - `docker compose exec asr-api python gpu_check.py`: コンテナ内でGPUが認識されていることを確認（ローカルで直接 `python` は叩かない）。

5. **ポートフォワーディング**
   - SSHのControlMaster機能でマスターセッションを確立。
   - リモートの `58080` と `58081` をローカルへ転送。
   - これにより `http://localhost:58080` でフロント、`http://localhost:58081` でAPIにアクセス可能。
   - `run.sh` 終了でフォワーディングも停止します。

## 4. コンテナ構成（実体）

バックエンド/フロントはそれぞれ `docker-compose.yml` でビルド・起動されます。GPU 割り当ては `docker-compose.gpu.yml` で上書きします。

- `docker-compose.yml` の主なポイント
  - `asr-api`: `backend/Dockerfile` をビルド。`58081:8000` を公開。`./data` と `./checkpoints` を `/app/data`, `/app/checkpoints` にマウント。
  - `frontend`: `frontend/Dockerfile` をビルド。`58080:8501` を公開。`BACKEND_HOST=asr-api`, `BACKEND_PORT=8000` を環境変数で指定。

- `docker-compose.gpu.yml` の主なポイント
  - `asr-api`: NVIDIA GPU を 1 枚予約（`deploy.resources.reservations.devices`）。イメージ名での起動を想定。

### 4.1. サービス詳細

-   **`asr-api` (バックエンド)**
    -   **役割**: FastAPIを用いて、学習・推論のAPIエンドポイント（プレフィックス `/api`）とWebSocketサーバー（`/ws`）を提供する。
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

- **パス設定 (`config.yaml`)**
  - `datasets.ljspeech.path` はコンテナ内のパス `/app/data/ljspeech` を指定します。
- **ディレクトリ構成 (`main.md`)**
  - `checkpoints` と `data` はホスト側ディレクトリをマウントし、永続化します。

## 6. 運用ガイド

- ローカルでの操作は必ず `run-local.sh` を使用してください（直接 `python` は実行しない）。
- GPU 環境では `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` を推奨します。
