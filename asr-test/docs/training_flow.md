# 学習プロセス 詳細フロー

このドキュメントは、`POST /train/start` が呼び出されてから学習が完了するまでの、バックエンド内部の処理フローを詳細に記述します。

(主要コンポーネントのリストは変更なし)

## 1. 学習開始リクエスト (POST /train/start)

(変更なし)

## 2. 学習プロセスの初期化 (trainer.start_training)

(前段の説明は変更なし)

1.  (変更なし)
2.  (変更なし)
3.  (変更なし)
4.  (変更なし)
5.  **[trainer.py]** **コンポーネント初期化:**
    -   データセットクラスをインスタンス化します。`ljspeech`のように検証セットが分離していないデータセットの場合、このクラス内で訓練データから一定割合をランダムに分割して検証セットを作成します（例: 95% train, 5% validation）。
    -   訓練用と検証用の `torch.utils.data.DataLoader` をそれぞれ作成します。
    -   モデルクラスをインスタンス化し、`.to(device)` でGPUに転送します。
    -   オプティマイザと学習率スケジューラを初期化します。
6.  **[trainer.py]** **チェックポイントの読み込み:**
    -   `checkpoints/` ディレクトリ内をスキャンし、現在のモデル名とデータセット名に一致するチェックポイントファイル (`{model_name}-{dataset_name}-epoch-*.pt`) を探します。
    -   最もエポック番号の大きいファイルを最新のチェックポイントとして特定します。
    -   最新のチェックポイントが存在する場合、`model.load_checkpoint()` を呼び出してモデルの重みとオプティマイザの状態を復元します。学習を開始するエポック番号 (`start_epoch`) も復元した値に設定します。
    -   存在しない場合は、エポック0から学習を開始します。
7.  (変更なし)

## 3. 学習サイクル (1エポック)

```python
# trainer.py (擬似コード)
for epoch in range(start_epoch, num_epochs):
    # 3.1. 訓練ループ
    model.train()
    for i, batch in enumerate(train_loader):
        # ... (フォワード・バックワード処理)
        # ... (進捗の記録と通知)

    # 3.2. 検証ループ
    val_loss = run_validation(model, valid_loader, device)
    websocket_manager.broadcast_sync({
        "type": "validation_result",
        "payload": { "epoch": epoch + 1, "val_loss": val_loss }
    })

    # 3.3. チェックポイント保存
    if (epoch + 1) % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, epoch + 1)
```

### 3.1. 訓練ループ (Training Loop)
-   (旧3.1と3.2の内容を統合、変更なし)

### 3.2. 検証ループ (Validation Loop)
-   各エポックの訓練が終了した後、検証データセットでモデルの性能を評価します。
-   `torch.no_grad()` 環境下で実行し、勾配計算を無効化します。
-   検証データセット全体を反復処理し、平均の損失（Validation Loss）を計算します。
-   計算した検証ロスを `validation_result` 型のメッセージとしてWebSocketでブロードキャストします。

### 3.3. チェックポイント保存 (Checkpointing)
-   `checkpoint_interval` で指定されたエポックごとに実行されます。
-   **命名規則:** `{checkpoints_dir}/{model_name}-{dataset_name}-epoch-{epoch}.pt` という形式でファイルを保存します。
-   **最新チェックポイントの管理:** 保存後、同じディレクトリに `{model_name}-{dataset_name}-latest.pt` という名前で、今保存したファイルへのシンボリックリンクを作成（または上書き）します。これにより、最新のチェックポイントを容易に特定できます。

## 4. プロセスの終了

(変更なし。ただし、「手動停止」の場合、停止前に最後のチェックポイントを保存するステップが含まれることを強調)
