# 学習プロセス 詳細フロー

このドキュメントは、`POST /train/start` が呼び出されてから学習が完了するまでの、バックエンド内部の処理フローを詳細に記述します。

主要コンポーネント:
-   `api.py`: HTTPリクエストの受付
-   `trainer.py`: 学習プロセスの本体
-   `config_loader.py`: 設定ファイルの読み込み
-   `websocket.py`: フロントエンドへの進捗通知
-   `app/models/{model_name}.py`: モデル定義
-   `app/datasets/{dataset_name}.py`: データセット定義

## 1. 学習開始リクエスト (POST /train/start)

1.  **[api.py]** `/train/start` エンドポイントがリクエスト（モデル名、データセット名を含む）を受信します。
2.  **[api.py]** 既に学習が実行中でないか、グローバルな状態フラグをチェックします。実行中であれば `409 Conflict` を返します。
3.  **[api.py]** FastAPIの `BackgroundTasks` を使用して、`trainer.start_training` 関数を非同期タスクとして登録します。
4.  **[api.py]** `202 Accepted` レスポンスをクライアントに即座に返します。

## 2. 学習プロセスの初期化 (trainer.start_training)

バックグラウンドで実行される処理です。このプロセス全体は `try...except` ブロックで囲まれ、設定不備やファイルの欠損など、学習ループ開始前に発生したエラーはすべて捕捉され、フロントエンドに通知されます。

1.  **[trainer.py]** グローバルな状態フラグを「学習中」に設定します。
2.  **[trainer.py -> websocket.py]** WebSocketマネージャーを通じて「学習準備中...」のログメッセージをブロードキャストします。
3.  **[trainer.py]** **設定検証と読み込み:**
    -   `config.yaml` を読み込み、リクエストで指定されたモデルとデータセットの設定が存在するか検証します。
    -   データセットのパスが実際に存在するか検証します。
    -   設定やパスに不備がある場合は、`error` 型のWebSocketメッセージを送信し、処理を中断します。
4.  **[trainer.py]** **動的モジュール読み込み:**
    -   `importlib` を使用して、`app.models.{model_name}` と `app.datasets.{dataset_name}` から対応するクラスを動的にインポートします。
    -   モジュールやクラスが見つからない場合は、`error` 型のWebSocketメッセージを送信し、処理を中断します。
5.  **[trainer.py]** **コンポーネント初期化:**
    -   データセットクラスをインスタンス化します（`train` スプリットと `validation` スプリット）。
    -   `torch.utils.data.DataLoader` を作成します。
    -   モデルクラスをインスタンス化し、`.to(device)` でGPUに転送します。
    -   オプティマイザと学習率スケジューラを初期化します。
6.  **[trainer.py]** **チェックポイントの読み込み:**
    -   最新のチェックポイントが存在すれば、`model.load_checkpoint()` を呼び出して学習を再開します。開始エポック番号も更新します。
7.  **[trainer.py -> websocket.py]** WebSocketマネージャーを通じて「学習開始」のログメッセージをブロードキャストします。

## 3. 学習ループ

```python
# trainer.py (擬似コード)
for epoch in range(start_epoch, num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # 3.1. フォワード・バックワード
        optimizer.zero_grad()
        batch = send_to_device(batch, device)
        loss = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # 3.2. 進捗の記録と通知
        if i % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress_data = {
                "type": "progress",
                "payload": { "epoch": epoch, "loss": loss.item(), ... }
            }
            websocket_manager.broadcast(progress_data)

    # 3.3. バリデーション
    run_validation(model, valid_loader)

    # 3.4. チェックポイント保存
    if epoch % checkpoint_interval == 0:
        checkpoint_path = f"{checkpoints_dir}/{model_name}-{dataset_name}-epoch-{epoch}.pt"
        model.save_checkpoint(checkpoint_path, optimizer, epoch)
```

### 3.1. フォワード・バックワード
-   データローダーからバッチを取得し、GPUに転送します。
-   モデルのフォワードパスを実行して損失を計算します。
-   逆伝播を行い、勾配を計算します。
-   勾配クリッピングを適用します。
-   オプティマイザでモデルの重みを更新します。
-   学習率スケジューラを更新します。

### 3.2. 進捗の記録と通知
-   `log_interval` ごとに現在の損失、学習率などを `progress` 型のメッセージとしてWebSocketでブロードキャストします。

### 3.3. バリデーション
-   各エポックの終了時に、検証データセットでモデルの性能を評価します（`torch.no_grad()` 環境下で実行）。
-   検証結果（例: Validation Loss, WER）を `log` 型のメッセージとしてWebSocketでブロードキャストします。

### 3.4. チェックポイント保存
-   `checkpoint_interval` ごとに、モデルの `state_dict`、オプティマイザの `state_dict`、エポック番号をファイルに保存します。
-   **命名規則:** チェックポイントファイルは `{checkpoints_dir}/{model_name}-{dataset_name}-epoch-{epoch}.pt` という形式で保存されます。

## 4. プロセスの終了

1.  **正常終了**: 全てのエポックが完了した場合。
    -   **[trainer.py]** グローバルな状態フラグを「待機中」に戻します。
    -   **[trainer.py -> websocket.py]** `status` 型のメッセージ（`"status": "completed"`）をブロードキャストします。
2.  **手動停止**: `POST /train/stop` が呼び出された場合。
    -   **[api.py]** 停止フラグを立てます。
    -   **[trainer.py]** 学習ループが各ステップの開始時に停止フラグをチェックし、フラグが立っていればループを中断します。
    -   **[trainer.py]** 最後のチェックポイントを保存します。
    -   **[trainer.py]** グローバルな状態フラグを「待機中」に戻します。
    -   **[trainer.py -> websocket.py]** `status` 型のメッセージ（`"status": "stopped"`）をブロードキャストします。
3.  **エラー発生**: `try...except` ブロックで学習中の例外（例: `RuntimeError: CUDA out of memory`）を捕捉した場合。
    -   **[trainer.py]** エラー内容をログに記録します。
    -   **[trainer.py]** グローバルな状態フラグを「待機中」に戻します。
    -   **[trainer.py -> websocket.py]** `error` 型のメッセージをブロードキャストします。
