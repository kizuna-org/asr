import time
import torch
import importlib
import os
import re
import glob
from typing import Dict

from .websocket import manager as websocket_manager
from .state import training_status
from . import config_loader

# 学習停止フラグ
stop_training_flag = False

# --- ヘルパー関数 ---

def get_latest_checkpoint(model_name: str, dataset_name: str, checkpoints_dir: str = "./checkpoints") -> str:
    pattern = f"{checkpoints_dir}/{model_name}-{dataset_name}-epoch-*.pt"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(re.search(r"epoch-(\d+)\.pt", p).group(1)))

def save_checkpoint(model, optimizer, epoch, model_name, dataset_name, checkpoints_dir="./checkpoints"):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-epoch-{epoch}.pt")
    model.save_checkpoint(checkpoint_path, optimizer, epoch)
    latest_path = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-latest.pt")
    if os.path.lexists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(checkpoint_path), latest_path)
    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"チェックポイントを保存しました: {checkpoint_path}"}})

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
    training_was_stopped = False

    try:
        training_status["is_training"] = True
        # 進捗情報をリセット
        training_status["current_epoch"] = 0
        training_status["current_step"] = 0
        training_status["current_loss"] = 0.0
        training_status["current_learning_rate"] = 0.0
        training_status["progress"] = 0.0
        training_status["latest_logs"] = []
        
        websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"学習を開始します: model={model_name}, dataset={dataset_name}, epochs={epochs}, batch_size={batch_size}, device={device}"}})

        model_config = config_loader.get_model_config(model_name)
        dataset_config = config_loader.get_dataset_config(dataset_name)
        training_config = config_loader.get_training_config()
        
        # フロントエンドから送信されたパラメータで設定を上書き
        training_config['num_epochs'] = epochs
        training_config['batch_size'] = batch_size

        DatasetClass = getattr(importlib.import_module(f".datasets.{dataset_name}", "app"), f"{dataset_name.capitalize()}Dataset")
        ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        collate_fn = getattr(importlib.import_module(f".datasets.{dataset_name}", "app"), "collate_fn")

        train_dataset = DatasetClass(dataset_config, split='train')
        valid_dataset = DatasetClass(dataset_config, split='validation')
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
        training_status['total_steps'] = len(train_loader) * num_epochs
        
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
                
                if i % training_config['log_interval'] == 0:
                    log_message = f"Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    training_status['latest_logs'].append(log_message)
                    # 最新の10件のみ保持
                    if len(training_status['latest_logs']) > 10:
                        training_status['latest_logs'] = training_status['latest_logs'][-10:]
                    
                    websocket_manager.broadcast_sync({"type": "progress", "payload": {"epoch": epoch + 1, "total_epochs": num_epochs, "step": i, "total_steps": len(train_loader), "loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']}})
            
            if training_was_stopped:
                break

            val_loss = run_validation(model, valid_loader, device)
            websocket_manager.broadcast_sync({"type": "validation_result", "payload": { "epoch": epoch + 1, "val_loss": val_loss }})

            if (epoch + 1) % training_config.get("checkpoint_interval", 1) == 0:
                save_checkpoint(model, optimizer, epoch + 1, model_name, dataset_name)

        if training_was_stopped:
            save_checkpoint(model, optimizer, epoch, model_name, dataset_name) # 停止時も保存
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "stopped", "message": "学習がユーザーによって停止されました。"}})
        else:
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "completed", "message": "学習が正常に完了しました。"}})

    except Exception as e:
        websocket_manager.broadcast_sync({"type": "error", "payload": {"message": f"学習中にエラーが発生しました: {e}"}})
    finally:
        training_status["is_training"] = False
        # 学習終了時に進捗情報をクリア
        training_status["current_epoch"] = 0
        training_status["current_step"] = 0
        training_status["current_loss"] = 0.0
        training_status["current_learning_rate"] = 0.0
        training_status["progress"] = 0.0
        training_status["latest_logs"] = []
        stop_training_flag = False

