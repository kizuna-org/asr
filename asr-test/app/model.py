import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
import math


class LightweightASRModel(nn.Module):
    """
    軽量で高速な音声認識モデル
    CNN + LSTM + CTC アーキテクチャ
    """
    
    def __init__(self, 
                 input_dim=80, 
                 hidden_dim=128, 
                 num_layers=2, 
                 num_classes=29,  # 英数字 + 特殊文字
                 dropout=0.1):
        super(LightweightASRModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # CNN特徴抽出器（軽量）
        self.conv_layers = nn.Sequential(
            # 1層目: 80 -> 64
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 2層目: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 3層目: 128 -> hidden_dim
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 出力層
        self.output_layer = nn.Linear(hidden_dim * 2, num_classes)
        
        # CTC損失関数
        self.ctc_loss = CTCLoss(blank=0, zero_infinity=True)
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, time_steps, input_dim)
            lengths: 各シーケンスの長さ
        Returns:
            logits: (batch_size, time_steps, num_classes)
        """
        batch_size, time_steps, _ = x.size()
        
        # CNN特徴抽出
        # (batch_size, time_steps, input_dim) -> (batch_size, input_dim, time_steps)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        # (batch_size, hidden_dim, time_steps) -> (batch_size, time_steps, hidden_dim)
        x = x.transpose(1, 2)
        
        # LSTM処理
        if lengths is not None:
            # PackedSequenceを使用して効率化
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=time_steps
            )
        else:
            x, _ = self.lstm(x)
        
        # 出力層
        logits = self.output_layer(x)
        
        return logits
    
    def compute_loss(self, logits, targets, logit_lengths, target_lengths):
        """
        CTC損失を計算
        """
        # logits: (batch_size, time_steps, num_classes) -> (time_steps, batch_size, num_classes)
        logits = logits.transpose(0, 1)
        logits = F.log_softmax(logits, dim=-1)
        
        return self.ctc_loss(logits, targets, logit_lengths, target_lengths)
    
    def decode(self, logits, lengths=None):
        """
        推論時のデコード（貪欲法）
        """
        # logits: (batch_size, time_steps, num_classes)
        if lengths is not None:
            # 各シーケンスの実際の長さに基づいてマスク
            mask = torch.arange(logits.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.to(logits.device)
        else:
            mask = None
        
        # 最も確率の高いクラスを選択
        predictions = torch.argmax(logits, dim=-1)
        
        # CTCデコード（重複除去とブランク除去）
        decoded_sequences = []
        for i, pred in enumerate(predictions):
            if mask is not None:
                pred = pred[mask[i]]
            
            # CTCデコード
            decoded = self._ctc_decode(pred)
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def _ctc_decode(self, pred):
        """
        CTCデコード（重複除去とブランク除去）
        """
        decoded = []
        prev = None
        
        for p in pred:
            if p != prev and p != 0:  # 0はブランク
                decoded.append(p.item())
            prev = p
        
        return decoded


class FastASRModel(nn.Module):
    """
    超軽量で高速な音声認識モデル
    リアルタイム推論に最適化
    """
    
    def __init__(self, 
                 input_dim=80, 
                 hidden_dim=64,  # より小さく
                 num_classes=29,
                 dropout=0.1):
        super(FastASRModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 軽量CNN
        self.conv_layers = nn.Sequential(
            # 1層目
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 2層目
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # 単層LSTM（高速化）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # 単方向で高速化
        )
        
        # 出力層
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # CTC損失
        self.ctc_loss = CTCLoss(blank=0, zero_infinity=True)
    
    def forward(self, x, lengths=None):
        batch_size, time_steps, _ = x.size()
        
        # CNN特徴抽出
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        
        # LSTM処理
        x, _ = self.lstm(x)
        
        # 出力層
        logits = self.output_layer(x)
        
        return logits
    
    def compute_loss(self, logits, targets, logit_lengths, target_lengths):
        logits = logits.transpose(0, 1)
        logits = F.log_softmax(logits, dim=-1)
        return self.ctc_loss(logits, targets, logit_lengths, target_lengths)
    
    def decode(self, logits):
        predictions = torch.argmax(logits, dim=-1)
        return [self._ctc_decode(pred) for pred in predictions]
    
    def _ctc_decode(self, pred):
        decoded = []
        prev = None
        
        for p in pred:
            if p != prev and p != 0:
                decoded.append(p.item())
            prev = p
        
        return decoded


# 文字マッピング
CHAR_TO_ID = {
    '<blank>': 0,
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
    'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
    'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
    ' ': 27, "'": 28
}

ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHAR_TO_ID)
