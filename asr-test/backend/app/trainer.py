import time
import torch
import importlib
from typing import Dict

from .websocket import manager as websocket_manager
from .api import training_status
from . import config_loader

# 学習停止フラグ
stop_training_flag = False

def start_training(params: Dict):
    """学習プロセスのエントリーポイント"""
    global stop_training_flag
    stop_training_flag = False # 開始時にフラグをリセット

    model_name = params.get("model_name")
    dataset_name = params.get("dataset_name")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_was_stopped = False

    try:
        # ... (初期化処理は変更なし) ...

        # 6. 学習ループ
        num_epochs = training_config['num_epochs']
        for epoch in range(start_epoch, num_epochs):
            if stop_training_flag:
                training_was_stopped = True
                break

            websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"Epoch {epoch + 1}/{num_epochs} を開始します。"}})
            for i, batch in enumerate(train_loader):
                if stop_training_flag:
                    training_was_stopped = True
                    break

                # --- 学習ステップ --- #
                # ... (学習ステップは変更なし) ...

            # エポックの終わりにチェック
            if training_was_stopped:
                break

        # 7. 終了処理
        if training_was_stopped:
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "stopped", "message": "学習がユーザーによって停止されました。"}})
        else:
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "completed", "message": "学習が正常に完了しました。"}})

    except Exception as e:
        # 8. エラー終了
        websocket_manager.broadcast_sync({"type": "error", "payload": {"message": f"エラーが発生しました: {e}"}})

    finally:
        # 9. 状態を「待機中」に戻す
        training_status["is_training"] = False
        stop_training_flag = False
