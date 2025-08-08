import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import wer

from app.model import LightweightASRModel, FastASRModel, ID_TO_CHAR
from app.dataset import AudioPreprocessor, TextPreprocessor, ASRDataset, create_dataloader


class ASRTrainer:
    """音声認識モデルのトレーナー"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 max_epochs: int = 100,
                 patience: int = 10,
                 model_save_dir: str = 'models'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_save_dir = model_save_dir
        
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
        
        # ディレクトリ作成
        os.makedirs(model_save_dir, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (audio_features, text_ids, audio_lengths, text_lengths) in enumerate(progress_bar):
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item()
            
            # WER計算（サンプル数制限）
            if batch_idx % 10 == 0:  # 10バッチごとに計算
                with torch.no_grad():
                    decoded_sequences = self.model.decode(logits, audio_lengths)
                    wer_score = self._calculate_wer(decoded_sequences, text_ids, text_lengths)
                    total_wer += wer_score
            
            num_batches += 1
            
            # プログレスバーの更新
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'WER': f'{wer_score if batch_idx % 10 == 0 else "N/A"}'
            })
        
        avg_loss = total_loss / num_batches
        avg_wer = total_wer / (num_batches // 10) if num_batches > 0 else 0.0
        
        return avg_loss, avg_wer
    
    def validate(self) -> Tuple[float, float]:
        """検証"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio_features, text_ids, audio_lengths, text_lengths in tqdm(self.val_loader, desc="Validation"):
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
    
    def train(self) -> Dict[str, List[float]]:
        """学習の実行"""
        print(f"Training started on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 50)
            
            # 学習
            train_loss, train_wer = self.train_epoch()
            
            # 検証
            val_loss, val_wer = self.validate()
            
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
                self.save_model(f"best_model.pth")
                print(f"New best model saved! (Epoch {epoch + 1})")
            
            # 定期的なモデル保存
            if (epoch + 1) % 10 == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")
            
            # 早期停止のチェック
            if epoch - self.best_epoch >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 学習曲線の保存
        self.plot_training_curves()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_wers': self.train_wers,
            'val_wers': self.val_wers
        }
    
    def save_model(self, filename: str):
        """モデルの保存"""
        save_path = os.path.join(self.model_save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_wers': self.train_wers,
            'val_wers': self.val_wers
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """モデルの読み込み"""
        load_path = os.path.join(self.model_save_dir, filename)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_wers = checkpoint.get('train_wers', [])
            self.val_wers = checkpoint.get('val_wers', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Model loaded from {load_path}")
        else:
            print(f"Model file not found: {load_path}")
    
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


class FastTrainer(ASRTrainer):
    """高速学習用のトレーナー（軽量モデル用）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # より高速な学習設定
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=kwargs.get('learning_rate', 2e-3),  # より高い学習率
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
    
    def train_epoch(self) -> Tuple[float, float]:
        """高速化された1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        total_wer = 0.0
        num_batches = 0
        
        for audio_features, text_ids, audio_lengths, text_lengths in self.train_loader:
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
            
            # パラメータ更新
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, 0.0  # WER計算をスキップして高速化
