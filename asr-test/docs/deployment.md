# デプロイ & インフラ構成設計書

このドキュメントは、`docker-compose.yml` および `docker-compose.gpu.yml`、並びにローカル実行用の `run-local.sh` に基づき、本アプリケーションのデプロイ/実行手順とインフラ構成について記述します。

## 1. 概要

開発用PCでは基本的にコンテナで実行します。ローカル実行時は `run-local.sh` を使用します（直接 `python` を実行しないこと）。GPUサーバーでの実行時は `docker-compose.gpu.yml` を併用して GPU を割り当てます。`./asr-test` は別ホストのGPUサーバーで動作するため、ローカルでは `run-local.sh` を必ず使用します。

**主要機能:**
- 学習制御: 新規学習、チェックポイントからの再開、学習停止（conformer/realtimeモデル対応）
- 推論機能: ファイルアップロードによる推論、リアルタイム音声推論（WebRTC統合）
- モデル管理: 学習済みモデルの一覧表示、削除、フィルタリング機能
- チェックポイント管理: チェックポイントの一覧表示、学習再開、詳細情報表示
- データセット管理: データセットのダウンロード、自動展開
- リアルタイム通信: WebSocketによる学習進捗のリアルタイム更新、音声ストリーミング

## 2. インフラストラクチャ

- **リモートサーバー（任意）**: GPUを搭載したサーバー。NVIDIA Container Runtime が設定済みであること。
- **プロジェクトパス**: 任意の作業パスに配置。
- **コンテナ環境**: Docker および Docker Compose。NVIDIA Container Runtime が有効。

## 3. 実行手順

### 3.1. ローカル実行（推奨）

**ローカル（CPU/GPU問わず）**
- `asr-test/` 直下で `./run-local.sh` を実行します。全ての実行はコンテナ内で行われます。
- ローカル環境では直接 `python` コマンドを実行しないでください。

### 3.2. GPU サーバー上での実行

**GPU サーバー上での実行**
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
- 事前にビルドする場合は `docker compose build` を利用します。

### 3.3. コンテナの再起動

**コンテナの再起動**
- `docker compose down`: 現在実行中のコンテナを停止・削除します。
- `docker build . -t asr-app`: ルートから `backend/Dockerfile` を参照してビルドします（GPU用は別イメージも可）。
- `docker compose up -d`: `docker-compose.yml` に基づいて起動します。
- 注記: `docker compose up --build` で同時実行も可能ですが、環境によってはビルドと起動を分離してください。

### 3.4. 動作確認

**動作確認**
- `check_nvidia_runtime.sh`: NVIDIA Container Runtime設定の確認。
- `docker compose exec asr-api python gpu_check.py`: コンテナ内でGPUが認識されていることを確認（ローカルで直接 `python` は叩かない）。

### 3.5. ポートフォワーディング

**ポートフォワーディング**
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
  - `frontend`: イメージ名での起動を想定。

### 4.1. サービス詳細

-   **`asr-api` (バックエンド)**
    -   **役割**: FastAPIを用いて、学習・推論のAPIエンドポイント（プレフィックス `/api`）とWebSocketサーバー（`/ws`）を提供する。
    -   **ビルド**: `backend/Dockerfile` からビルドされます（ローカル）。GPU環境ではビルド済みイメージを使用可。
    -   **ポート**: コンテナのポート `8000` をホストの `58081` にマッピング。
    -   **GPU**: NVIDIA GPUを1つ割り当てるように設定。
    -   **ボリューム**:
        -   `./data` -> `/app/data`: データセットを読み込むためにマウント。
        -   `./checkpoints` -> `/app/checkpoints`: 学習のチェックポイントを永続化するためにマウント。
    -   **機能**:
        - 学習制御: 新規学習、チェックポイントからの再開、学習停止（conformer/realtimeモデル対応）
        - 推論機能: ファイルアップロードによる推論、リアルタイム音声推論（WebRTC統合）
        - モデル管理: 学習済みモデルの一覧表示、削除、フィルタリング機能
        - チェックポイント管理: チェックポイントの一覧表示、学習再開、詳細情報表示
        - データセット管理: データセットのダウンロード、自動展開
        - リアルタイム通信: WebSocketによる学習進捗のリアルタイム更新、音声ストリーミング

-   **`frontend` (フロントエンド)**
    -   **役割**: Streamlitを用いて、学習の制御や結果を可視化するWeb GUIを提供する。
    -   **ビルド**: `frontend/Dockerfile` からビルドされます。
    -   **ポート**: コンテナのポート `8501` をホストの `58080` にマッピング。
    -   **機能**:
        - メインダッシュボード: 学習制御、推論テスト、リアルタイム推論、進捗表示
        - モデル管理: 学習済みモデルの一覧表示と削除、フィルタリング機能
        - チェックポイント管理: チェックポイントの一覧表示と学習再開、詳細情報表示
        - リアルタイム推論: マイク入力によるリアルタイム音声認識（WebRTC統合）

## 5. 他の設計への影響

- **パス設定 (`config.yaml`)**
  - `datasets.ljspeech.path` はコンテナ内のパス `/app/data/ljspeech` を指定します。
- **ディレクトリ構成 (`main.md`)**
  - `checkpoints` と `data` はホスト側ディレクトリをマウントし、永続化します。

## 6. 運用ガイド

### 6.1. 基本運用

- ローカルでの操作は必ず `run-local.sh` を使用してください（直接 `python` は実行しない）。
- GPU 環境では `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` を推奨します。

### 6.2. トラブルシューティング

**よくある問題と解決方法**

1. **コンテナが起動しない場合**
   - Docker と Docker Compose が正しくインストールされているか確認
   - ポート 58080, 58081 が他のプロセスで使用されていないか確認

2. **GPU が認識されない場合**
   - NVIDIA Container Runtime が正しくインストールされているか確認
   - `check_nvidia_runtime.sh` を実行して設定を確認

3. **フロントエンドにアクセスできない場合**
   - バックエンドサービスが正常に起動しているか確認
   - プロキシ設定（HTTP_PROXY, HTTPS_PROXY, NO_PROXY）を確認

### 6.3. ログ確認

**ログの確認方法**
- バックエンドログ: `docker compose logs asr-api`
- フロントエンドログ: `docker compose logs frontend`
- リアルタイムログ: `docker compose logs -f asr-api`
