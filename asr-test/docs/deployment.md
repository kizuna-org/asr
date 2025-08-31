# デプロイ & インフラ構成設計書

このドキュメントは、`run.sh` スクリプトの分析に基づき、本アプリケーションのデプロイ手順とインフラ構成について記述します。

**注意:** `docker-compose.yml` の完全な内容をツールで読み取れなかったため、一部推測に基づいています。

## 1. 概要

開発用PCから `run.sh` を実行することで、リモートのGPUサーバー (`edu-gpu`) にアプリケーションをデプロイします。デプロイプロセスには、ファイル同期、Dockerコンテナのビルドと起動、ローカルPCへのポートフォワーディングが含まれます。

## 2. インフラストラクチャ

-   **リモートサーバー**: `edu-gpu` というホスト名を持つSSHアクセス可能なサーバー。GPUを搭載している。
-   **プロジェクトパス**: サーバー上の `/home/students/r03i/r03i18/asr-test/asr/asr-test` にプロジェクトファイルが配置される。
-   **コンテナ環境**: DockerおよびDocker Composeが利用可能。NVIDIA Container Runtimeがセットアップされており、コンテナ内からGPUを利用できる。

## 3. デプロイ手順 (`run.sh` の処理フロー)

1.  **ファイル同期**:
    -   `rsync` を使用して、ローカルのプロジェクトファイルをリモートサーバーにコピーする。
    -   `__pycache__/` と `models/` ディレクトリは同期対象から除外される。
    -   **[考察]** `models/` が除外されているのは、大規模な学習済みモデルを毎回転送しないようにするためと考えられます。チェックポイントは `checkpoints/` に保存されるため、このディレクトリはサーバー上で永続化されるべきです（後述のボリューム設定を参照）。

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

## 4. コンテナ構成 (docker-compose.yml の推測)

```yaml
# 推定される docker-compose.yml の内容
services:
  # バックエンドAPIサービス
  asr-api:
    # run.sh の `docker build` コマンドから、イメージ名が直接指定されている可能性
    image: asr-app
    # もしくは、コンテキストを直接指定
    # build:
    #   context: ./backend
    #   dockerfile: Dockerfile
    ports:
      - "58081:8000" # ホスト:コンテナ (FastAPIのデフォルトは8000)
    volumes:
      - ./data:/app/data # データセット用
      - ./checkpoints:/app/checkpoints # 学習済みモデル/チェックポイント用
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  # フロントエンドGUIサービス
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "58080:8501" # ホスト:コンテナ (Streamlitのデフォルトは8501)
    depends_on:
      - asr-api
    command: streamlit run app.py --server.port 8501

volumes:
  data:
  checkpoints:
```

### 4.1. サービス詳細

-   **`asr-api` (バックエンド)**
    -   **役割**: FastAPIを用いて、学習・推論のAPIエンドポイントとWebSocketサーバーを提供する。
    -   **ビルド**: `backend/Dockerfile` からビルドされる（と推測）。
    -   **ポート**: コンテナのポート `8000` をホストの `58081` にマッピング。
    -   **GPU**: NVIDIA GPUを1つ割り当てるように設定。
    -   **ボリューム**:
        -   `./data` -> `/app/data`: データセットを読み込むためにマウント。
        -   `./checkpoints` -> `/app/checkpoints`: 学習のチェックポイントを永続化するためにマウント。

-   **`frontend` (フロントエンド)**
    -   **役割**: Streamlitを用いて、学習の制御や結果を可視化するWeb GUIを提供する。
    -   **ビルド**: `frontend/Dockerfile` からビルドされる（と推測）。
    -   **ポート**: コンテナのポート `8501` をホストの `58080` にマッピング。

## 5. 他の設計への影響

-   **パス設定 (`config.yaml`)**:
    -   `config_spec.md` に記載されているデータセットのパス (`datasets.ljspeech.path`) は、コンテナ内のパス (`/app/data/ljspeech` など) を指すように記述する必要があります。
-   **ディレクトリ構成 (`main.md`)**:
    -   `run.sh` の挙動から、`main.md` に記載のディレクトリ構成案は妥当であると判断できます。
    -   ただし、`checkpoints` と `data` ディレクトリは `rsync` の対象外とし、サーバー側で永続的に管理するのが望ましいです。
