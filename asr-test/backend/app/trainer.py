import time
import torch
import importlib
import os
import re
import glob
import logging
import math
import json
from typing import Dict
from datetime import datetime

from .websocket import manager as websocket_manager
from .state import training_status
from . import config_loader

# 学習停止フラグ
stop_training_flag = False

# ロガー設定（uvicorn/fastapiの親ロガー配下にぶら下げる）
logger = logging.getLogger("asr-api")

# --- ヘルパー関数 ---

class WarmupLR:
    """WarmupLRスケジューラーの実装"""
    
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        """学習率を更新"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # ウォームアップ期間中は線形に学習率を増加
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # ウォームアップ後は基本学習率を維持
            lr = self.base_lr
        
        # オプティマイザの学習率を更新
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """現在の学習率を取得"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

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

def save_checkpoint(model, optimizer, epoch, model_name, dataset_name, scheduler=None, checkpoints_dir="./checkpoints", training_metadata=None):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    # モデル固有の保存形式を使用
    checkpoint_dir = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-epoch-{epoch}")
    
    # モデルのsave_checkpointメソッドを使用（ディレクトリ形式で保存）
    model.save_checkpoint(checkpoint_dir, optimizer, epoch)
    
    # スケジューラーの状態も保存（オプション）
    if scheduler is not None:
        scheduler_data = {
            'step_count': scheduler.step_count,
            'warmup_steps': scheduler.warmup_steps,
            'base_lr': scheduler.base_lr
        }
        torch.save(scheduler_data, os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # 学習メタデータを保存
    if training_metadata is not None:
        metadata_file = os.path.join(checkpoint_dir, "training_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
    
    # 最新チェックポイントへのシンボリックリンクを作成
    latest_dir = os.path.join(checkpoints_dir, f"{model_name}-{dataset_name}-latest")
    if os.path.lexists(latest_dir):
        os.remove(latest_dir)
    os.symlink(os.path.basename(checkpoint_dir), latest_dir)
    
    # 古いチェックポイントの自動削除
    cleanup_old_checkpoints(model_name, dataset_name, checkpoints_dir)
    
    message = f"チェックポイントを保存しました: {checkpoint_dir}"
    logger.info(message)
    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": message}})

def cleanup_old_checkpoints(model_name, dataset_name, checkpoints_dir="./checkpoints"):
    """古いチェックポイントを自動削除する"""
    try:
        # 設定から保持数を取得
        training_config = config_loader.get_training_config()
        retention_count = training_config.get('checkpoint_retention', 5)
        
        # 該当するチェックポイントを検索
        pattern = f"{checkpoints_dir}/{model_name}-{dataset_name}-epoch-*.pt"
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) <= retention_count:
            return  # 保持数以下の場合は何もしない
        
        # エポック番号でソート
        def extract_epoch(path):
            match = re.search(r"epoch-(\d+)", path)
            return int(match.group(1)) if match else 0
        
        checkpoints.sort(key=extract_epoch, reverse=True)
        
        # 古いチェックポイントを削除
        checkpoints_to_delete = checkpoints[retention_count:]
        for checkpoint_path in checkpoints_to_delete:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logger.info(f"古いチェックポイントを削除しました: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"チェックポイントの削除に失敗しました: {checkpoint_path}, エラー: {e}")
                
    except Exception as e:
        logger.warning(f"チェックポイントの自動削除中にエラーが発生しました: {e}")

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
        
        # 学習メタデータを作成
        training_start_time = datetime.now().isoformat()
        training_metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "training_start_time": training_start_time,
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": training_config.get('learning_rate'),
                "optimizer": training_config.get('optimizer'),
                "scheduler": training_config.get('scheduler'),
                "warmup_steps": training_config.get('warmup_steps'),
                "device": device
            },
            "model_config": model_config,
            "dataset_config": dataset_config,
            "training_config": training_config
        }
        
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

        # 学習率スケジューラーの初期化
        scheduler = None
        if training_config.get('scheduler') == 'WarmupLR':
            warmup_steps = training_config.get('warmup_steps', 4000)
            scheduler = WarmupLR(optimizer, warmup_steps, training_config['learning_rate'])
            logger.info(f"WarmupLRスケジューラーを初期化しました: warmup_steps={warmup_steps}, base_lr={training_config['learning_rate']}")

        start_epoch = 0
        resume_from_checkpoint = params.get("resume_from_checkpoint", True)
        specific_checkpoint = params.get("specific_checkpoint")
        
        if resume_from_checkpoint:
            if specific_checkpoint:
                # 特定のチェックポイントから再開
                checkpoint_path = f"./checkpoints/{specific_checkpoint}"
                if os.path.exists(checkpoint_path):
                    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"指定されたチェックポイントを読み込んでいます: {checkpoint_path}"}})
                    start_epoch = model.load_checkpoint(checkpoint_path, optimizer)
                    
                    # スケジューラーの状態も復元
                    if scheduler is not None:
                        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                        if 'scheduler_state_dict' in checkpoint:
                            scheduler_state = checkpoint['scheduler_state_dict']
                            scheduler.step_count = scheduler_state['step_count']
                            logger.info(f"スケジューラーの状態を復元しました: step_count={scheduler.step_count}")
                else:
                    logger.warning(f"指定されたチェックポイントが見つかりません: {checkpoint_path}")
            else:
                # 最新のチェックポイントから再開
                latest_checkpoint_path = get_latest_checkpoint(model_name, dataset_name)
                if latest_checkpoint_path:
                    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": f"最新のチェックポイントを読み込んでいます: {latest_checkpoint_path}"}})
                    start_epoch = model.load_checkpoint(latest_checkpoint_path, optimizer)
                    
                    # スケジューラーの状態も復元
                    if scheduler is not None:
                        checkpoint = torch.load(latest_checkpoint_path, map_location=lambda storage, loc: storage)
                        if 'scheduler_state_dict' in checkpoint:
                            scheduler_state = checkpoint['scheduler_state_dict']
                            scheduler.step_count = scheduler_state['step_count']
                            logger.info(f"スケジューラーの状態を復元しました: step_count={scheduler.step_count}")
                else:
                    logger.info("チェックポイントが見つからないため、最初から学習を開始します")
        else:
            logger.info("チェックポイントからの再開が無効化されているため、最初から学習を開始します")

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
                
                # 学習率スケジューラーの更新
                if scheduler is not None:
                    scheduler.step()
                
                # 進捗情報を更新
                current_step = epoch * len(train_loader) + i
                training_status['current_epoch'] = epoch + 1
                training_status['current_step'] = current_step
                training_status['current_loss'] = loss.item()
                # スケジューラーがある場合はスケジューラーから学習率を取得、なければオプティマイザから取得
                if scheduler is not None:
                    training_status['current_learning_rate'] = scheduler.get_last_lr()[0]
                else:
                    training_status['current_learning_rate'] = optimizer.param_groups[0]['lr']
                training_status['progress'] = current_step / training_status['total_steps']
                # 1秒ごとに現在のエポック/ステップをブロードキャスト
                now = time.time()
                if now - last_broadcast_time >= 1.0:
                    # スケジューラーがある場合はスケジューラーから学習率を取得、なければオプティマイザから取得
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                    websocket_manager.broadcast_sync({
                        "type": "progress",
                        "payload": {
                            "epoch": epoch + 1,
                            "total_epochs": num_epochs,
                            "step": current_step,
                            "total_steps": training_status['total_steps'],
                            "loss": loss.item(),
                            "learning_rate": current_lr
                        }
                    })
                    last_broadcast_time = now
                
                if i % training_config['log_interval'] == 0:
                    # スケジューラーがある場合はスケジューラーから学習率を取得、なければオプティマイザから取得
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                    log_message = f"Epoch {epoch + 1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                    logger.info(log_message)
                    training_status['latest_logs'].append(log_message)
                    if len(training_status['latest_logs']) > 50:
                        training_status['latest_logs'] = training_status['latest_logs'][-50:]

                    websocket_manager.broadcast_sync({"type": "progress", "payload": {"epoch": epoch + 1, "total_epochs": num_epochs, "step": i, "total_steps": len(train_loader), "loss": loss.item(), "learning_rate": current_lr}})
                    websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": log_message}})
            
            if training_was_stopped:
                break

            val_loss = run_validation(model, valid_loader, device)
            val_message = f"Validation: epoch={epoch + 1}, val_loss={val_loss:.4f}"
            logger.info(val_message)
            websocket_manager.broadcast_sync({"type": "validation_result", "payload": { "epoch": epoch + 1, "val_loss": val_loss }})
            websocket_manager.broadcast_sync({"type": "log", "payload": {"level": "INFO", "message": val_message}})

            if (epoch + 1) % training_config.get("checkpoint_interval", 1) == 0:
                # メタデータを更新
                training_metadata["current_epoch"] = epoch + 1
                training_metadata["last_checkpoint_time"] = datetime.now().isoformat()
                save_checkpoint(model, optimizer, epoch + 1, model_name, dataset_name, scheduler, training_metadata=training_metadata)

        if training_was_stopped:
            # メタデータを更新
            training_metadata["current_epoch"] = epoch
            training_metadata["training_end_time"] = datetime.now().isoformat()
            training_metadata["training_status"] = "stopped"
            save_checkpoint(model, optimizer, epoch, model_name, dataset_name, scheduler, training_metadata=training_metadata) # 停止時も保存
            logger.info("学習がユーザーによって停止されました。")
            websocket_manager.broadcast_sync({"type": "status", "payload": {"status": "stopped", "message": "学習がユーザーによって停止されました。"}})
        else:
            # メタデータを更新
            training_metadata["current_epoch"] = num_epochs
            training_metadata["training_end_time"] = datetime.now().isoformat()
            training_metadata["training_status"] = "completed"
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

