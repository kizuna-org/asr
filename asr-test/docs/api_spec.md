# API仕様書

このドキュメントは、ASR学習POCアプリケーションのバックエンドAPIとWebSocket通信の仕様を定義します。実装は FastAPI で提供され、HTTP API は全て `/api` プレフィックス配下で公開されます（WebSocketはプレフィックスなし）。

## 1. REST API

ベースURL: `http://<backend-host>:<port>/api`

### 1.1. 学習制御

#### `POST /train/start`

学習プロセスを開始します。

**リクエストボディ (application/json)**

```json
{
  "model_name": "conformer",
  "dataset_name": "ljspeech",
  "epochs": 10,
  "batch_size": 32,
  "resume_from_checkpoint": true,
  "specific_checkpoint": "conformer-ljspeech-epoch-5.pt",
  "lightweight": false,
  "limit_samples": null
}
```

-   `model_name` (string, required): 使用するモデル名。`config.yaml`の`available_models`に定義されている必要があります。
-   `dataset_name` (string, required): 使用するデータセット名。`config.yaml`の`available_datasets`に定義されている必要があります。
-   `epochs` (integer, optional): 学習エポック数。デフォルトは設定ファイルの値。
-   `batch_size` (integer, optional): バッチサイズ。デフォルトは設定ファイルの値。
-   `resume_from_checkpoint` (boolean, optional): チェックポイントから学習を再開するかどうか。デフォルトは`true`。
-   `specific_checkpoint` (string, optional): 特定のチェックポイントから再開する場合のチェックポイント名。
-   `lightweight` (boolean, optional): 軽量実行モード（サンプル数を制限）。デフォルトは`false`。
-   `limit_samples` (integer, optional): 学習に使用するサンプル数の上限。`lightweight`が`true`の場合は10に設定される。

**レスポンス**

-   **202 Accepted**: 学習プロセスが正常にバックグラウンドで開始された場合。
    ```json
    {
      "message": "Training started in background."
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

#### `POST /train/resume`

学習をチェックポイントから再開します。

**リクエストボディ (application/json)**

```json
{
  "model_name": "conformer",
  "dataset_name": "ljspeech",
  "specific_checkpoint": "conformer-ljspeech-epoch-5.pt",
  "epochs": 10,
  "batch_size": 32,
  "lightweight": false,
  "limit_samples": null
}
```

-   `model_name` (string, required): 使用するモデル名。
-   `dataset_name` (string, required): 使用するデータセット名。
-   `specific_checkpoint` (string, optional): 特定のチェックポイントから再開する場合のチェックポイント名。省略時は最新のチェックポイントを使用。
-   `epochs` (integer, optional): 学習エポック数。デフォルトは10。
-   `batch_size` (integer, optional): バッチサイズ。デフォルトは32。
-   `lightweight` (boolean, optional): 軽量実行モード。デフォルトは`false`。
-   `limit_samples` (integer, optional): 学習に使用するサンプル数の上限。

**レスポンス**

-   **200 OK**: 学習再開が正常に開始された場合。
    ```json
    {
      "message": "Training resumed from checkpoint in background."
    }
    ```
-   **400 Bad Request**: リクエストボディが不正な場合。
-   **404 Not Found**: 指定されたチェックポイントが存在しない場合。
-   **409 Conflict**: 既に学習プロセスが実行中の場合。

#### `POST /train/stop`

現在実行中の学習プロセスを停止します。

**リクエストボディ**

なし

**レスポンス**

-   **200 OK**: 学習プロセスに停止シグナルを送信した場合。
    ```json
    {
      "message": "Stop signal sent to training process."
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
-   `model_name` (query, optional): 既定値 `conformer`

**レスポンス**

-   **200 OK**: 推論が成功した場合。
    ```json
    {
      "transcription": "hello world",
      "first_token_time_ms": 45.2,
      "inference_time_ms": 123.4,
      "total_time_ms": 168.6
    }
    ```
    - `transcription` (string): 文字起こし結果
    - `first_token_time_ms` (number): 音声入力から最初の出力が来るまでの時間（ミリ秒）
    - `inference_time_ms` (number): 推論処理時間（ミリ秒）
    - `total_time_ms` (number): 音声入力から最後の出力が来るまでの総時間（ミリ秒）
-   **400 Bad Request**: ファイルが提供されなかった場合。
-   **500 Internal Server Error**: 推論中にエラーが発生した場合。

### 1.3. 設定情報

#### `GET /config`

現在のバックエンド設定（利用可能なモデルやデータセットなど）を取得します。

**レスポンス**

-   **200 OK**:
    ```json
    {
      "available_models": ["conformer"],
      "available_datasets": ["ljspeech"],
      "training_config": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "AdamW"
      }
    }
    ```

### 1.4. ステータス・進捗

#### `GET /status`

学習の実行有無を返します。

**レスポンス**

-   **200 OK**
    ```json
    { "is_training": true }
    ```

#### `GET /progress`

最新の学習進捗スナップショットを返します。

例:

```json
{
  "is_training": true,
  "current_epoch": 1,
  "current_step": 50,
  "current_loss": 0.1234,
  "current_learning_rate": 0.0009,
  "progress": 0.05,
  "total_epochs": 10,
  "total_steps": 1000,
  "server_time": 1719555555.12
}
```

### 1.5. データセットダウンロード

#### `POST /dataset/download`

指定データセットをコンテナ内 `/app/data` にダウンロードして展開します（現状 `ljspeech` のみサポート）。

**リクエストボディ (application/json)**

```json
{ "dataset_name": "ljspeech" }
```

**レスポンス**

-   成功時 200: `message`, `path`, `num_wavs`
-   既存時 200: `message`, `path`, `num_wavs`
-   エラー時 4xx/5xx: `detail` にメッセージ

### 1.6. チェックポイント管理

#### `GET /checkpoints`

利用可能なチェックポイントの一覧を取得します。

**クエリパラメータ**

-   `model_name` (string, optional): モデル名でフィルタリング
-   `dataset_name` (string, optional): データセット名でフィルタリング

**レスポンス**

-   **200 OK**: チェックポイント一覧の取得成功
    ```json
    {
      "checkpoints": [
        {
          "name": "conformer-ljspeech-epoch-10.pt",
          "model_name": "conformer",
          "dataset_name": "ljspeech",
          "epoch": 10,
          "path": "/app/checkpoints/conformer-ljspeech-epoch-10.pt",
          "size_mb": 245.67,
          "file_count": 7,
          "created_at": 1719555555.12,
          "files": ["config.json", "model.safetensors", "optimizer.pt", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]
        }
      ]
    }
    ```
    - `name` (string): チェックポイント名
    - `model_name` (string): モデル名
    - `dataset_name` (string): データセット名
    - `epoch` (integer): エポック数
    - `path` (string): チェックポイントファイルのパス
    - `size_mb` (number): チェックポイントサイズ（MB）
    - `file_count` (number): 含まれるファイル数
    - `created_at` (number): 作成日時（Unix timestamp）
    - `files` (array): 含まれるファイル名のリスト

### 1.7. 学習済みモデル管理

#### `GET /models`

学習済みモデルの一覧を取得します。

**レスポンス**

-   **200 OK**: モデル一覧の取得成功
    ```json
    {
      "models": [
        {
          "name": "conformer-ljspeech-epoch-10.pt",
          "path": "/app/checkpoints/conformer-ljspeech-epoch-10.pt",
          "epoch": "10",
          "size_mb": 245.67,
          "file_count": 7,
          "created_at": 1719555555.12,
          "files": ["config.json", "model.safetensors", "optimizer.pt", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]
        }
      ]
    }
    ```
    - `name` (string): モデル名
    - `path` (string): モデルファイルのパス
    - `epoch` (string|null): エポック数（抽出可能な場合）
    - `size_mb` (number): モデルサイズ（MB）
    - `file_count` (number): 含まれるファイル数
    - `created_at` (number): 作成日時（Unix timestamp）
    - `files` (array): 含まれるファイル名のリスト

#### `DELETE /models/{model_name}`

指定された学習済みモデルを削除します。

**パラメータ**

-   `model_name` (string, required): 削除するモデル名

**レスポンス**

-   **200 OK**: モデル削除の成功
    ```json
    {
      "message": "Model 'conformer-ljspeech-epoch-10.pt' deleted successfully"
    }
    ```
-   **400 Bad Request**: 無効なモデル名の場合
    ```json
    {
      "detail": "Invalid model name"
    }
    ```
-   **404 Not Found**: 指定されたモデルが存在しない場合
    ```json
    {
      "detail": "Model 'invalid-model' not found"
    }
    ```
-   **500 Internal Server Error**: 削除処理中にエラーが発生した場合

### 1.8. テスト

#### `GET /api/test`

疎通確認用エンドポイント。利用可能なAPIのリストを返します。
```json
{ "message": "Test endpoint is working", "endpoints": ["/config", "/status", "/progress", "/train/start", "/train/resume", "/train/stop", "/dataset/download", "/models", "/checkpoints"] }
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

### 2.2. クライアントからサーバーへのメッセージ（拡張: リアルタイム推論）

エンドポイント: `ws://<backend-host>:<port>/ws`

クライアントは以下のプロトコルでメッセージを送信し、音声をストリーミングして部分/最終の文字起こしを受信できます。

- Text(JSON) `start`:
  ```json
  {"type": "start", "model_name": "conformer", "sample_rate": 48000, "format": "i16"}
  ```
  - `model_name`: 使用するモデル名（省略時 `conformer`）
  - `sample_rate`: クライアント送信のサンプルレート（例: 48000）
  - `format`: `i16` (PCM16LE) または `f32` (float32, little endian)

- Binary 音声フレーム:
  - `start` 後、一定長の音声フレームを連続で送信します。
  - モノラル想定。サーバー側で 16kHz mono へリサンプリング。

- Text(JSON) `stop`:
  ```json
  {"type": "stop"}
  ```

サーバーからの応答（追加）:

- 部分結果 `partial`:
  ```json
  {"type": "partial", "payload": {"text": "hello wor"}}
  ```

- 最終結果 `final`:
  ```json
  {"type": "final", "payload": {"text": "hello world"}}
  ```

- ステータス `status`（準備完了/停止）:
  ```json
  {"type": "status", "payload": {"status": "ready"}}
  ```

- エラー `error`:
  ```json
  {"type": "error", "payload": {"message": "Invalid JSON"}}
  ```
