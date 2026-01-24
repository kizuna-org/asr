# backend/app/models/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class BaseASRModel(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        学習・検証時のフォワードパス。
        Args:
            batch: データローダーからの出力 (例: {'waveforms': ..., 'tokens': ...})
        Returns:
            loss: 計算された損失 (スカラーテンソル)
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def inference(self, waveform: torch.Tensor) -> str:
        """
        推論処理。音声波形からテキストを生成する。
        Args:
            waveform: 単一の音声波形テンソル
        Returns:
            transcription: 文字起こし結果の文字列
        """
        pass

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int):
        """チェックポイントを保存する"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """チェックポイントを読み込む"""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)

    @torch.no_grad()
    def inference_stream(self, waveform: torch.Tensor, context: Optional[torch.Tensor] = None) -> str:
        """
        ストリーミング推論処理（リアルタイム用）
        
        Args:
            waveform: 単一の音声波形テンソル
            context: 前のコンテキスト（オプション）
            
        Returns:
            str: 文字起こし結果の文字列
        """
        # デフォルト実装では通常のinferenceを呼び出す
        # サブクラスでオーバーライドしてストリーミング最適化を実装可能
        return self.inference(waveform)
