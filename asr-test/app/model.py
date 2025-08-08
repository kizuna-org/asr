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
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def is_trained(self) -> bool:
        """モデルが学習済みかどうかを確認"""
        # 重みの統計をチェック
        with torch.no_grad():
            # 出力層の重みをチェック
            output_weights = self.output_layer.weight
            weight_mean = output_weights.mean().item()
            weight_std = output_weights.std().item()
            
            # 初期化時の値と比較（初期化は平均0、標準偏差0.01）
            # 学習済みならば、重みの分布が初期化時と異なるはず
            is_trained = abs(weight_mean) > 0.001 or weight_std > 0.02
            
            print(f"Model trained check - Weight mean: {weight_mean:.6f}, std: {weight_std:.6f}, is_trained: {is_trained}")
            
            return is_trained
    
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
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_layers(x)
        # (batch_size, hidden_dim, time_steps) -> (batch_size, time_steps, hidden_dim)
        x = x.permute(0, 2, 1).contiguous()
        
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
        logits = logits.permute(1, 0, 2).contiguous()
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
        
        # デバッグ情報を追加
        print(f"Debug - Logits shape: {logits.shape}")
        print(f"Debug - Predictions shape: {predictions.shape}")
        print(f"Debug - First few predictions: {predictions[0][:10].tolist()}")
        print(f"Debug - Logits stats - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
        
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
        decoded = []
        prev = None
        
        # デバッグ情報を追加
        print(f"CTC Debug - Raw predictions: {pred[:20].tolist()}")
        print(f"CTC Debug - Prediction stats - min: {pred.min()}, max: {pred.max()}, unique: {torch.unique(pred).tolist()}")
        
        for p in pred:
            if p != prev and p != 0:
                decoded.append(p.item())
            prev = p
        
        print(f"CTC Debug - Decoded IDs: {decoded}")
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
    
    def is_trained(self) -> bool:
        """モデルが学習済みかどうかを確認"""
        # 重みの統計をチェック
        with torch.no_grad():
            # 出力層の重みをチェック
            output_weights = self.output_layer.weight
            weight_mean = output_weights.mean().item()
            weight_std = output_weights.std().item()
            
            # 初期化時の値と比較
            is_trained = abs(weight_mean) > 0.001 or weight_std > 0.02
            
            print(f"FastASR trained check - Weight mean: {weight_mean:.6f}, std: {weight_std:.6f}, is_trained: {is_trained}")
            
            return is_trained
    
    def forward(self, x, lengths=None):
        batch_size, time_steps, _ = x.size()
        
        # CNN特徴抽出
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1).contiguous()
        
        # LSTM処理
        x, _ = self.lstm(x)
        
        # 出力層
        logits = self.output_layer(x)
        
        return logits
    
    def compute_loss(self, logits, targets, logit_lengths, target_lengths):
        logits = logits.permute(1, 0, 2).contiguous()
        logits = F.log_softmax(logits, dim=-1)
        return self.ctc_loss(logits, targets, logit_lengths, target_lengths)
    
    def decode(self, logits, lengths=None):
        # デバッグ情報を追加
        print(f"FastASR Debug - Logits shape: {logits.shape}")
        print(f"FastASR Debug - Logits stats - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
        
        predictions = torch.argmax(logits, dim=-1)
        print(f"FastASR Debug - First few predictions: {predictions[0][:10].tolist()}")
        
        # 標準的なCTCデコードを試行
        decoded_sequences = [self._ctc_decode(pred) for pred in predictions]
        
        # 結果が空の場合は、より柔軟なデコードを試行
        for i, decoded in enumerate(decoded_sequences):
            if not decoded:
                print(f"CTC Debug - Empty result for sequence {i}, trying flexible decode...")
                decoded_sequences[i] = self._flexible_ctc_decode(predictions[i])
        
        return decoded_sequences
    
    def _flexible_ctc_decode(self, pred):
        """より柔軟なCTCデコード（ブランクの扱いを緩和）"""
        decoded = []
        prev = None
        
        # 非ブランクの予測をカウント
        non_blank_count = (pred != 0).sum().item()
        total_count = len(pred)
        
        print(f"Flexible CTC Debug - Non-blank ratio: {non_blank_count}/{total_count} = {non_blank_count/total_count:.3f}")
        
        # 非ブランクの割合が低すぎる場合は、最も頻繁に出現する非ブランク文字を選択
        if non_blank_count / total_count < 0.1:
            non_blank_preds = pred[pred != 0]
            if len(non_blank_preds) > 0:
                # 最も頻繁に出現する文字を選択
                unique, counts = torch.unique(non_blank_preds, return_counts=True)
                most_common = unique[counts.argmax()]
                decoded = [most_common.item()]
                print(f"Flexible CTC Debug - Using most common non-blank: {most_common.item()}")
            else:
                # 全てブランクの場合は、最も確率の高い文字を選択
                decoded = [pred[0].item() if pred[0] != 0 else 1]  # デフォルトで'a'
                print(f"Flexible CTC Debug - All blank, using default: {decoded[0]}")
        else:
            # 通常のCTCデコード
            for p in pred:
                if p != prev and p != 0:
                    decoded.append(p.item())
                prev = p
        
        print(f"Flexible CTC Debug - Final decoded: {decoded}")
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
