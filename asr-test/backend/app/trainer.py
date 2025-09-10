import time
import torch
import importlib
import os
import re
import glob
import logging
from typing import Dict

from .websocket import manager as websocket_manager
from .state import training_status
from . import config_loader

# 学習停止フラグ
stop_training_flag = False

# ロガー設定（uvicorn/fastapiの親ロガー配下にぶら下げる）
logger = logging.getLogger("asr-api")

# --- ヘルパー関数 ---

def get_latest_checkpoint(model_name: str, dataset_name: str, checkpoints_dir: str = "./checkpoints") -> str:
    # ディレクトリ形式のチェックポイントを探す（新しい形式）
    dir_pattern = f"{checkpoints_dir}/{model_name}-{dataset_name}-epoch-*.pt"
    dir_checkpoints = glob.glob(dir_pattern)
    
    # ファイル形式のチェックポイントを探す（旧形式）
    file_pattern = f"{checkpoints_dir}/{model_name}-{dataset_name}-epoch-*.pt"
    file_checkpoints = glob.glob(file_pattern)
    
    all_checkpoints = []
    
    # ディレクトリ形式のチェックポイントを追加
    for checkpoint in dir_checkpoints:
        if os.path.isdir(checkpoint):
            all_checkpoints.append(checkpoint)
    
    # ファイル形式のチェックポイントを追加（ディレクトリでないもの）
    for checkpoint in file_checkpoints:
        if not os.path.isdir(checkpoint):
            all_checkpoints.append(checkpoint)
    
    if not all_checkpoints:
        return None
    
    # エポック番号でソートして最新を返す
    return max(all_checkpoints, key=lambda p: int(re.search(r"epoch-(\d+)", p).group(1)))

def save_checkpoint(model, optimizer, epoch, model_name, dataset_name, checkpoints_dir="./checkpoints"):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-epoch-{epoch}.pt")
    model.save_checkpoint(checkpoint_path, optimizer, epoch)
    latest_path = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-latest.pt")
    if os.path.lexists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(checkpoint_path), latest_path)
    message = f"チェックポイントを保存しました: {checkpoint_path}"
    logger.info(message)
    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": message}})

@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss = model(batch)
        total_loss += loss.item()
    return total_loss / len(loader)

# --- 学習プロセス本体 ---

def start_training(params: Dict):
    global stop_training_flag
    stop_training_flag = False
    model_name = params.get("model_name")
    dataset_name = params.get("dataset_name")
    epochs = params.get("epochs", 10)  # フロントエンドから送信されるエポック数
    batch_size = params.get("batch_size", 32)  # フロントエンドから送信されるバッチサイズ
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CPU 実行時はメモリ保護のためバッチサイズを自動的に制限
    if device == "cpu" and batch_size > 4:
        original_bs = batch_size
        batch_size = 4
        logger.warning(f"CPU 実行のためバッチサイズを {original_bs} -> {batch_size} に調整しました")
    training_was_stopped = False

    try:
        training_status["is_training"] = True
        # 進捗情報をリセット
        training_status["current_epoch"] = 0
        training_status["current_step"] = 0
        training_status["current_loss"] = 0.0
        training_status["current_learning_rate"] = 0.0
        training_status["progress"] = 0.0
        # 暫定のトータル値を先に埋めて 0/0 表示を避ける
        training_status["total_epochs"] = epochs or 0
        training_status["total_steps"] = max(1, (epochs or 1))
        training_status.setdefault("latest_logs", [])
        training_status.pop("latest_error", None)

        start_msg = f"学習を開始します: model={model_name}, dataset={dataset_name}, epochs={epochs}, batch_size={batch_size}, device={device}"
        logger.info(start_msg)
        training_status["latest_logs"].append(start_msg)
        websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": start_msg}})

        model_config = config_loader.get_model_config(model_name)
        dataset_config = config_loader.get_dataset_config(dataset_name)
        training_config = config_loader.get_training_config()
        
        # フロントエンドから送信されたパラメータで設定を上書き
        training_config['num_epochs'] = epochs
        training_config['batch_size'] = batch_size

        # データセット名に応じてクラス名を適切に設定
        if dataset_name == "ljspeech":
            class_name = "LJSpeechDataset"
        else:
            class_name = f"{dataset_name.capitalize()}Dataset"
        DatasetClass = getattr(importlib.import_module(f".datasets.{dataset_name}", "app"), class_name)
        # リアルタイムモデルの場合は特別なクラス名を使用
        if model_name == "realtime":
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), "RealtimeASRModel")
        else:
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        collate_fn = getattr(importlib.import_module(f".datasets.{dataset_name}", "app"), "collate_fn")

        # 軽量実行フラグ/サンプル制限の反映
        # params: { lightweight: bool, limit_samples: int }
        dataset_overrides = {}
        if params.get("lightweight"):
            dataset_overrides["max_samples"] = 10
        if isinstance(params.get("limit_samples"), int) and params.get("limit_samples") > 0:
            dataset_overrides["max_samples"] = params.get("limit_samples")

        # 設定をコピーしてオーバーライド（検証は常に同数で十分軽いので同じ制限を適用）
        train_ds_conf = {**dataset_config, **dataset_overrides}
        valid_ds_conf = {**dataset_config, **dataset_overrides}

        train_dataset = DatasetClass(train_ds_conf, split='train')
        valid_dataset = DatasetClass(valid_ds_conf, split='validation')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=training_config['batch_size'], shuffle=False, collate_fn=collate_fn)

        model = ModelClass(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

        start_epoch = 0
        latest_checkpoint_path = get_latest_checkpoint(model_name, dataset_name)
        if latest_checkpoint_path:
            websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"チェックポイントを読み込んでいます: {latest_checkpoint_path}"}})
            start_epoch = model.load_checkpoint(latest_checkpoint_path, optimizer)

        num_epochs = training_config['num_epochs']
        training_status['total_epochs'] = num_epochs
        training_status['total_steps'] = max(1, len(train_loader) * num_epochs)
        # 正確な合計がわかったら一度通知
        websocket_manager.broadcast_sync({
            "type": "progress",
            "payload": {
                "epoch": 0,
                "total_epochs": training_status['total_epochs'],
                "step": 0,
                "total_steps": training_status['total_steps'],
                "loss": None,
                "learning_rate": None,
            }
        })
        # 1秒ごとの同期用のタイムスタンプ
        last_broadcast_time = 0.0
        
        for epoch in range(start_epoch, num_epochs):
            if stop_training_flag:
                training_was_stopped = True
                break
            
            model.train()
            for i, batch in enumerate(train_loader):
                if stop_training_flag:
                    training_was_stopped = True
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                
                # 進捗情報を更新
                current_step = epoch * len(train_loader) + i
                training_status['current_epoch'] = epoch + 1
                training_status['current_step'] = current_step
                training_status['current_loss'] = loss.item()
                training_status['current_learning_rate'] = optimizer.param_groups[0]['lr']
                training_status['progress'] = current_step / training_status['total_steps']
                # 1秒ごとに現在のエポック/ステップをブロードキャスト
                now = time.time()
                if now - last_broadcast_time >= 1.0:
                    websocket_manager.broadcast_sync({
                        "type": "progress",
                        "payload": {
                            "epoch": epoch + 1,
                            "total_epochs": num_epochs,
                            "step": current_step,
                            "total_steps": training_status['total_steps'],
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr']
                        }
                    })
                    last_broadcast_time = now
                
                if i % training_config['log_interval'] == 0:
                    log_message = f"Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    logger.info(log_message)
                    training_status['latest_logs'].append(log_message)
                    if len(training_status['latest_logs']) > 50:
                        training_status['latest_logs'] = training_status['latest_logs'][-50:]

                    websocket_manager.broadcast_sync({"type": "progress", "payload": {"epoch": epoch + 1, "total_epochs": num_epochs, "step": i, "total_steps": len(train_loader), "loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']}})
                    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": log_message}})
            
            if training_was_stopped:
                break

            val_loss = run_validation(model, valid_loader, device)
            val_message = f"Validation: epoch={epoch + 1}, val_loss={val_loss:.4f}"
            logger.info(val_message)
            websocket_manager.broadcast_sync({"type": "validation_result", "payload": { "epoch": epoch + 1, "val_loss": val_loss }})
            websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": val_message}})

            if (epoch + 1) % training_config.get("checkpoint_interval", 1) == 0:
                save_checkpoint(model, optimizer, epoch + 1, model_name, dataset_name)

        if training_was_stopped:
            save_checkpoint(model, optimizer, epoch, model_name, dataset_name) # 停止時も保存
            logger.info("学習がユーザーによって停止されました。")
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "stopped", "message": "学習がユーザーによって停止されました。"}})
        else:
            logger.info("学習が正常に完了しました。")
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "completed", "message": "学習が正常に完了しました。"}})

    except Exception as e:
        import traceback
        error_msg = f"学習中にエラーが発生しました: {e}"
        error_traceback = traceback.format_exc()
        logger.error(error_msg)
        logger.error(error_traceback)
        # API経由でも確認できるように保持
        training_status["latest_error"] = {"message": str(e), "traceback": error_traceback}
        training_status.setdefault("latest_logs", []).append(error_msg)
        websocket_manager.broadcast_sync({"type": "error", "payload": {"message": error_msg, "traceback": error_traceback}})
    finally:
        training_status["is_training"] = False
        # 学習終了時に数値系の進捗情報はリセット（ログ/エラーは保持）
        training_status["current_epoch"] = 0
        training_status["current_step"] = 0
        training_status["current_loss"] = 0.0
        training_status["current_learning_rate"] = 0.0
        training_status["progress"] = 0.0
        stop_training_flag = False

