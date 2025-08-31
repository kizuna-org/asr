# API仕様書

このドキュメントは、ASR学習POCアプリケーションのバックエンドAPIとWebSocket通信の仕様を定義します。

## 1. REST API

ベースURL: `http://<backend-host>:<port>`

### 1.1. 学習制御

#### `POST /train/start`

学習プロセスを開始します。

**リクエストボディ (application/json)**

```json
{
  "model_name": "conformer",
  "dataset_name": "ljspeech"
}
```

-   `model_name` (string, required): 使用するモデル名。`config.yaml`の`available_models`に定義されている必要があります。
-   `dataset_name` (string, required): 使用するデータセット名。`config.yaml`の`available_datasets`に定義されている必要があります。

**レスポンス**

-   **202 Accepted**: 学習プロセスが正常にバックグラウンドで開始された場合。
    ```json
    {
      "status": "Training started in background."
    }
    ```
-   **400 Bad Request**: リクエストボディが不正な場合や、指定されたモデル/データセットが存在しない場合。
    ```json
    {
      "detail": "Model 'invalid_model' not found in config."
    }
    ```
-   **409 Conflict**: 既に学習プロセスが実行中の場合。
    ```json
    {
      "detail": "Training is already in progress."
    }
    ```

#### `POST /train/stop`

現在実行中の学習プロセスを停止します。

**リクエストボディ**

なし

**レスポンス**

-   **200 OK**: 学習プロセスに停止シグナルを送信した場合。
    ```json
    {
      "status": "Stop signal sent to training process."
    }
    ```
-   **404 Not Found**: 実行中の学習プロセスがない場合。
    ```json
    {
      "detail": "No training process is running."
    }
    ```

### 1.2. 推論

#### `POST /inference`

アップロードされた音声ファイルから文字起こしを行います。

**リクエストボディ (multipart/form-data)**

-   `file` (file, required): 推論対象の音声ファイル (WAV, FLACなど)。

**レスポンス**

-   **200 OK**: 推論が成功した場合。
    ```json
    {
      "transcription": "hello world"
    }
    ```
-   **400 Bad Request**: ファイルが提供されなかった場合。
-   **500 Internal Server Error**: 推論中にエラーが発生した場合。

### 1.3. 設定情報

#### `GET /config`

現在のバックエンド設定（利用可能なモデルやデータセットなど）を取得します。

**レスポンス**

-   **200 OK**:
    ```json
    {
      "available_models": ["conformer", "rnn-t"],
      "available_datasets": ["ljspeech"],
      "training_config": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam"
      }
    }
    ```

## 2. WebSocket API

エンドポイント: `ws://<backend-host>:<port>/ws`

フロントエンドが接続し、学習の進捗状況をリアルタイムで受信するために使用します。

### 2.1. サーバーからクライアントへのメッセージ

サーバーは学習の進捗に応じて、以下のJSONオブジェクトをブロードキャストします。

#### 学習進捗 (`progress`)

```json
{
  "type": "progress",
  "payload": {
    "epoch": 1,
    "total_epochs": 100,
    "step": 50,
    "total_steps": 1000,
    "loss": 0.1234,
    "learning_rate": 0.0009
  }
}
```

#### システムログ (`log`)

```json
{
  "type": "log",
  "payload": {
    "level": "INFO",
    "message": "Epoch 1 finished. Starting validation..."
  }
}
```

#### 検証結果 (`validation_result`)

各エポックの検証完了時に送信されます。

```json
{
  "type": "validation_result",
  "payload": {
    "epoch": 1,
    "val_loss": 0.0876
  }
}
```

#### 学習完了 (`status`)

```json
{
  "type": "status",
  "payload": {
    "status": "completed",
    "message": "Training finished successfully."
  }
}
```

#### エラー (`error`)

```json
{
  "type": "error",
  "payload": {
    "message": "CUDA out of memory."
  }
}
```

### 2.2. クライアントからサーバーへのメッセージ

現在の設計では、クライアントからサーバーへのメッセージ送信は想定していません。接続確立後の通信はサーバーからの一方的なプッシュのみです。
