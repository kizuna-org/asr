import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import wer

from app.model import LightweightASRModel, FastASRModel, ID_TO_CHAR
from app.dataset import AudioPreprocessor, TextPreprocessor, ASRDataset, create_dataloader
from app.ljspeech_dataset import create_ljspeech_dataloader


class ControlledASRTrainer:
    """制御可能な音声認識モデルのトレーナー"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 max_epochs: int = 100,
                 patience: int = 10,
                 model_save_dir: str = 'models',
                 gradient_clip: float = 1.0,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_save_dir = model_save_dir
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        
        # 学習制御変数
        self.is_training = False
        self.is_paused = False
        self.current_epoch = 0
        self.current_batch = 0
        self.should_stop = False
        
        # 最適化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学習率スケジューラー
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 履歴
        self.train_losses = []
        self.val_losses = []
        self.train_wers = []
        self.val_wers = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # コールバック関数
        self.progress_callback = None
        self.epoch_callback = None
        
        # ディレクトリ作成
        os.makedirs(model_save_dir, exist_ok=True)
    
    def set_progress_callback(self, callback: Callable):
        """進捗コールバックを設定"""
        self.progress_callback = callback
    
    def set_epoch_callback(self, callback: Callable):
        """エポックコールバックを設定"""
        self.epoch_callback = callback
    
    def start_training(self):
        """学習開始"""
        if self.is_training:
            return {"status": "error", "message": "学習は既に実行中です"}
        
        self.is_training = True
        self.is_paused = False
        self.should_stop = False
        
        # 学習スレッドを開始
        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        return {"status": "success", "message": "学習を開始しました"}
    
    def pause_training(self):
        """学習一時停止"""
        if not self.is_training:
            return {"status": "error", "message": "学習が実行されていません"}
        
        self.is_paused = True
        return {"status": "success", "message": "学習を一時停止しました"}
    
    def resume_training(self):
        """学習再開"""
        if not self.is_training:
            return {"status": "error", "message": "学習が実行されていません"}
        
        self.is_paused = False
        return {"status": "success", "message": "学習を再開しました"}
    
    def stop_training(self):
        """学習停止"""
        if not self.is_training:
            return {"status": "error", "message": "学習が実行されていません"}
        
        self.should_stop = True
        self.is_training = False
        self.is_paused = False
        return {"status": "success", "message": "学習を停止しました"}
    
    def save_checkpoint(self, epoch: int = None):
        """チェックポイントを保存"""
        if epoch is None:
            epoch = self.current_epoch
        
        checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_{epoch}.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'current_batch': self.current_batch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_wers': self.train_wers,
            'val_wers': self.val_wers
        }, checkpoint_path)
        
        return {"status": "success", "message": f"チェックポイントを保存しました: {checkpoint_path}"}
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントを読み込み"""
        if not os.path.exists(checkpoint_path):
            return {"status": "error", "message": "チェックポイントファイルが見つかりません"}
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.current_batch = checkpoint.get('current_batch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_wers = checkpoint.get('train_wers', [])
            self.val_wers = checkpoint.get('val_wers', [])
            
            return {"status": "success", "message": f"チェックポイントを読み込みました: {checkpoint_path}"}
        
        except Exception as e:
            return {"status": "error", "message": f"チェックポイントの読み込みに失敗しました: {str(e)}"}
    
    def get_training_status(self) -> dict:
        """学習状態を取得"""
        return {
            "is_training": self.is_training,
            "is_paused": self.is_paused,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "total_batches": len(self.train_loader),
            "max_epochs": self.max_epochs,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_wers": self.train_wers,
            "val_wers": self.val_wers,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "gradient_clip": self.gradient_clip,
            "early_stopping_patience": self.early_stopping_patience,
            "validation_split": self.validation_split
        }
    
    def _training_loop(self):
        """学習ループ"""
        print(f"Training started on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            if self.should_stop:
                break
            
            self.current_epoch = epoch
            
            # 一時停止チェック
            while self.is_paused and not self.should_stop:
                time.sleep(0.1)
            
            if self.should_stop:
                break
            
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 50)
            
            # 学習
            train_loss, train_wer = self._train_epoch()
            
            # 検証
            val_loss, val_wer = self._validate()
            
            # 学習率の更新
            self.scheduler.step(val_loss)
            
            # 履歴の保存
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_wers.append(train_wer)
            self.val_wers.append(val_wer)
            
            # 結果の表示
            print(f"Train Loss: {train_loss:.4f}, Train WER: {train_wer:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # ベストモデルの保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch + 1)
                print(f"New best model saved! (Epoch {epoch + 1})")
            
            # 定期的なチェックポイント保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # エポックコールバック
            if self.epoch_callback:
                self.epoch_callback(epoch, train_loss, val_loss, train_wer, val_wer)
            
            # 早期停止のチェック
            if epoch - self.best_epoch >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.is_training = False
        print("Training completed.")
    
    def _train_epoch(self) -> Tuple[float, float]:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        
        for batch_idx, (audio_features, text_ids, audio_lengths, text_lengths) in enumerate(self.train_loader):
            if self.should_stop:
                break
            
            # 一時停止チェック
            while self.is_paused and not self.should_stop:
                time.sleep(0.1)
            
            if self.should_stop:
                break
            
            self.current_batch = batch_idx
            
            # データをデバイスに移動
            audio_features = audio_features.to(self.device)
            text_ids = text_ids.to(self.device)
            audio_lengths = audio_lengths.to(self.device)
            text_lengths = text_lengths.to(self.device)
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 順伝播
            logits = self.model(audio_features, audio_lengths)
            
            # 損失計算
            loss = self.model.compute_loss(logits, text_ids, audio_lengths, text_lengths)
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            
            # パラメータ更新
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item()
            
            # WER計算（サンプル数制限）
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    decoded_sequences = self.model.decode(logits, audio_lengths)
                    wer_score = self._calculate_wer(decoded_sequences, text_ids, text_lengths)
                    total_wer += wer_score
            
            num_batches += 1
            
            # 進捗コールバック
            if self.progress_callback:
                progress = {
                    "epoch": self.current_epoch + 1,
                    "batch": batch_idx + 1,
                    "total_batches": len(self.train_loader),
                    "loss": loss.item(),
                    "wer": wer_score if batch_idx % 10 == 0 else None
                }
                self.progress_callback(progress)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = total_wer / (num_batches // 10) if num_batches > 0 else 0.0
        
        return avg_loss, avg_wer
    
    def _validate(self) -> Tuple[float, float]:
        """検証"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio_features, text_ids, audio_lengths, text_lengths in self.val_loader:
                # データをデバイスに移動
                audio_features = audio_features.to(self.device)
                text_ids = text_ids.to(self.device)
                audio_lengths = audio_lengths.to(self.device)
                text_lengths = text_lengths.to(self.device)
                
                # 順伝播
                logits = self.model(audio_features, audio_lengths)
                
                # 損失計算
                loss = self.model.compute_loss(logits, text_ids, audio_lengths, text_lengths)
                
                # WER計算
                decoded_sequences = self.model.decode(logits, audio_lengths)
                wer_score = self._calculate_wer(decoded_sequences, text_ids, text_lengths)
                
                total_loss += loss.item()
                total_wer += wer_score
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = total_wer / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_wer
    
    def _calculate_wer(self, decoded_sequences: List[List[int]], 
                      text_ids: torch.Tensor, 
                      text_lengths: torch.Tensor) -> float:
        """WER（Word Error Rate）を計算"""
        total_wer = 0.0
        num_samples = 0
        
        for i, decoded in enumerate(decoded_sequences):
            # デコードされたID列をテキストに変換
            decoded_text = self._ids_to_text(decoded)
            
            # 正解のID列をテキストに変換
            target_ids = text_ids[i][:text_lengths[i]].cpu().numpy().tolist()
            target_text = self._ids_to_text(target_ids)
            
            # WER計算
            if target_text.strip():
                wer_score = wer(target_text, decoded_text)
                total_wer += wer_score
                num_samples += 1
        
        return total_wer / num_samples if num_samples > 0 else 0.0
    
    def _ids_to_text(self, ids: List[int]) -> str:
        """ID列をテキストに変換"""
        text = ""
        for id_val in ids:
            if id_val in ID_TO_CHAR and ID_TO_CHAR[id_val] != '<blank>':
                text += ID_TO_CHAR[id_val]
        return text
    
    def get_available_checkpoints(self) -> List[str]:
        """利用可能なチェックポイントを取得"""
        checkpoints = []
        for file in os.listdir(self.model_save_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
                checkpoints.append(file)
        return sorted(checkpoints)
    
    def plot_training_curves(self):
        """学習曲線のプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 損失曲線
        ax1.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # WER曲線
        ax2.plot(self.train_wers, label='Train WER')
        if self.val_wers:
            ax2.plot(self.val_wers, label='Val WER')
        ax2.set_title('Training and Validation WER')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('WER')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_curves.png'))
        plt.close()
