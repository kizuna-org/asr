# backend/app/models/realtime.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import traceback

from .interface import BaseASRModel

# モデル専用のロガー
logger = logging.getLogger("model")

class RealtimeEncoder(nn.Module):
    """リアルタイム音声認識用のエンコーダ"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 音響特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RNN層（GRU使用）
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 出力投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, chunk_features: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            chunk_features: [batch_size, seq_len, input_dim]
            hidden_state: 前のチャンクの隠れ状態
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            new_hidden_state: 次のチャンク用の隠れ状態
        """
        # 特徴抽出
        features = self.feature_extractor(chunk_features)
        
        # RNN処理
        output, new_hidden_state = self.rnn(features, hidden_state)
        
        # 出力投影
        output = self.output_projection(output)
        
        return output, new_hidden_state


class RealtimeCTCDecoder(nn.Module):
    """リアルタイム音声認識用のCTCデコーダ"""
    
    def __init__(self, input_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.vocab_size = vocab_size
        
        # CTC出力層
        self.ctc_head = nn.Linear(input_dim, vocab_size + 1)  # +1 for blank token
        
        # 文字確率の正規化
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, seq_len, input_dim]
        
        Returns:
            log_probs: [batch_size, seq_len, vocab_size + 1]
        """
        # CTC出力
        logits = self.ctc_head(encoder_output)
        
        # 確率正規化
        log_probs = self.softmax(logits)
        
        return log_probs
    
    def decode_realtime(self, log_probs: torch.Tensor, threshold: float = -10.0) -> List[int]:
        """
        リアルタイムデコード（簡易版）
        
        Args:
            log_probs: [seq_len, vocab_size + 1]
            threshold: 文字検出の閾値（対数確率）
        
        Returns:
            detected_chars: 検出された文字のリスト
        """
        detected_chars = []
        blank_id = len(self.vocab)  # 最後のインデックスがblank
        
        logger.info(f"Decoding realtime with threshold: {threshold}, log_probs shape: {log_probs.shape}")
        
        for t in range(log_probs.size(0)):
            # 最大確率の文字を取得
            max_prob, max_char = torch.max(log_probs[t], dim=-1)
            
            # デバッグ情報
            if t < 5:  # 最初の5ステップのみログ出力
                logger.info(f"Step {t}: max_prob={max_prob.item():.3f}, max_char={max_char.item()}, blank_id={blank_id}")
            
            # 閾値を超え、かつblankでない場合（対数確率なので閾値は負の値）
            if max_prob > threshold and max_char != blank_id:
                detected_chars.append(max_char.item())
                logger.info(f"Detected char: {max_char.item()} at step {t}")
        
        logger.info(f"Total detected chars: {len(detected_chars)}")
        return detected_chars


class RealtimeASRModel(BaseASRModel):
    """リアルタイム性特化型音声認識モデル"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 設定の取得
        encoder_config = config.get('encoder', {})
        decoder_config = config.get('decoder', {})
        processing_config = config.get('processing', {})
        
        # エンコーダとデコーダの初期化
        self.encoder = RealtimeEncoder(
            input_dim=encoder_config.get('input_dim', 80),
            hidden_dim=encoder_config.get('hidden_dim', 256),
            num_layers=encoder_config.get('num_layers', 3),
            dropout=encoder_config.get('dropout', 0.1)
        )
        
        self.decoder = RealtimeCTCDecoder(
            input_dim=decoder_config.get('input_dim', 256),
            vocab_size=decoder_config.get('vocab_size', 1000)
        )
        
        # 状態管理
        self.hidden_state = None
        
        # 音響特徴抽出の設定
        self.sample_rate = processing_config.get('sample_rate', 16000)
        self.n_mels = processing_config.get('n_mels', 80)
        self.n_fft = processing_config.get('n_fft', 1024)
        self.hop_length = processing_config.get('hop_length', 160)
        self.chunk_size_ms = processing_config.get('chunk_size_ms', 100)
        
        # メルスペクトログラム変換器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=self.sample_rate // 2
        )
        
        # 語彙マッピング（簡易版）
        self.vocab = self._create_simple_vocab(decoder_config.get('vocab_size', 1000))
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}
        
        # 特殊トークンの設定
        self.blank_token = decoder_config.get('blank_token', '_')
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.vocab_size = len(self.vocab)
        
        logger.info("RealtimeASRModel initialized successfully", 
                   extra={"extra_fields": {"component": "model", "action": "init_complete", 
                                         "encoder_hidden_dim": encoder_config.get('hidden_dim', 256),
                                         "vocab_size": decoder_config.get('vocab_size', 1000)}})
    
    def _create_simple_vocab(self, vocab_size: int) -> List[str]:
        """簡易的な語彙を作成"""
        # 基本的な文字セット
        basic_chars = list("abcdefghijklmnopqrstuvwxyz ")
        # 数字
        numbers = list("0123456789")
        # 句読点
        punctuation = list(".,!?;:")
        
        vocab = basic_chars + numbers + punctuation
        
        # 語彙サイズに合わせて調整
        if len(vocab) > vocab_size - 1:  # -1 for blank token
            vocab = vocab[:vocab_size - 1]
        else:
            # 不足分を特殊文字で埋める
            while len(vocab) < vocab_size - 1:
                vocab.append(f"<unk_{len(vocab)}>")
        
        return vocab
    
    def extract_features(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """音声チャンクからメルスペクトログラム特徴を抽出"""
        # 音声を正規化
        audio_chunk = audio_chunk.float()
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)  # [1, time]
        
        # メルスペクトログラムに変換
        mel_spec = self.mel_transform(audio_chunk)  # [1, n_mels, time]
        
        # 対数変換
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # 転置して [1, time, n_mels] に
        mel_spec = mel_spec.transpose(1, 2)
        
        return mel_spec
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        学習時のフォワードパス（CTC損失計算）
        """
        waveforms = batch["waveforms"]
        texts = batch["texts"]
        
        # バッチ内の各音声を処理
        batch_outputs = []
        batch_lengths = []
        
        for waveform in waveforms:
            # 特徴抽出
            features = self.extract_features(waveform)
            batch_outputs.append(features)
            batch_lengths.append(features.size(1))
        
        # パディング
        max_length = max(batch_lengths)
        padded_outputs = []
        
        for output in batch_outputs:
            if output.size(1) < max_length:
                padding = torch.zeros(output.size(0), max_length - output.size(1), output.size(2))
                output = torch.cat([output, padding], dim=1)
            padded_outputs.append(output)
        
        # バッチ化
        batch_features = torch.cat(padded_outputs, dim=0)  # [batch_size, max_time, n_mels]
        
        # エンコーダ処理
        encoder_output, _ = self.encoder(batch_features)
        
        # CTCデコード
        log_probs = self.decoder(encoder_output)
        
        # CTC損失計算
        # テキストをトークン化
        target_lengths = []
        targets = []
        
        for text in texts:
            # 簡易的な文字レベルトークン化
            text_tokens = []
            for char in text.lower():
                if char in self.vocab:
                    text_tokens.append(self.vocab.index(char))
                else:
                    # 未知文字は語彙の最後のインデックスを使用
                    text_tokens.append(len(self.vocab) - 1)
            
            targets.extend(text_tokens)
            target_lengths.append(len(text_tokens))
        
        if not targets:
            # 空のターゲットの場合は小さな損失を返す
            return torch.tensor(0.1, requires_grad=True, device=log_probs.device)
        
        # ターゲットをテンソルに変換
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=log_probs.device)
        target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long, device=log_probs.device)
        
        # 入力長（バッチ内の各サンプルの長さ）
        input_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=log_probs.device)
        
        # CTC損失計算
        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # [time, batch, vocab] に変換
            targets_tensor,
            input_lengths,
            target_lengths_tensor,
            blank=self.vocab_size,  # blank tokenのインデックス
            reduction='mean'
        )
        
        return ctc_loss
    
    @torch.no_grad()
    def inference(self, waveform: torch.Tensor) -> str:
        """
        推論処理（リアルタイム用）
        """
        logger.info("Starting realtime model inference", 
                   extra={"extra_fields": {"component": "model", "action": "inference_start", 
                                         "waveform_shape": waveform.shape, "dtype": str(waveform.dtype)}})
        
        try:
            # 音声が短すぎる場合は空文字を返す
            if waveform.numel() < 1600:  # 0.1秒未満
                logger.debug("Audio too short for inference", 
                            extra={"extra_fields": {"component": "model", "action": "audio_too_short", 
                                                  "samples": waveform.numel(), "duration_sec": waveform.numel() / self.sample_rate}})
                return ""
            
            # 特徴抽出
            features = self.extract_features(waveform)
            logger.info(f"Extracted features shape: {features.shape}")
            
            # エンコーダ処理
            encoder_output, self.hidden_state = self.encoder(features, self.hidden_state)
            logger.info(f"Encoder output shape: {encoder_output.shape}")
            
            # CTCデコード
            log_probs = self.decoder(encoder_output)
            logger.info(f"CTC log_probs shape: {log_probs.shape}")
            logger.info(f"CTC log_probs min/max: {log_probs.min().item():.3f}/{log_probs.max().item():.3f}")
            
            # リアルタイムデコード
            detected_chars = self.decoder.decode_realtime(log_probs[0])
            logger.info(f"Detected chars: {detected_chars}")
            
            # 後処理
            recognized_text = self._post_process_ctc_output(detected_chars)
            logger.info(f"Post-processed text: '{recognized_text}'")
            
            logger.info("Realtime inference completed successfully", 
                       extra={"extra_fields": {"component": "model", "action": "inference_complete", 
                                             "transcription": recognized_text, "chars_count": len(detected_chars)}})
            
            return recognized_text
            
        except Exception as e:
            logger.error("Error during realtime inference", 
                        extra={"extra_fields": {"component": "model", "action": "inference_error", 
                                              "error": str(e), "traceback": traceback.format_exc()}})
            return ""
    
    def _post_process_ctc_output(self, ctc_sequence: List[int]) -> str:
        """
        CTC出力を最終的な文字列に変換
        """
        # 1. 重複除去
        deduplicated = []
        prev_char = None
        
        for char_id in ctc_sequence:
            if char_id != prev_char:
                deduplicated.append(char_id)
            prev_char = char_id
        
        # 2. ブランク削除と文字変換
        final_text = ''.join([self.id_to_char.get(char_id, '') for char_id in deduplicated 
                             if char_id < len(self.vocab)])  # blank tokenは除外
        
        return final_text
    
    def reset_state(self):
        """セッション開始時に状態をリセット"""
        self.hidden_state = None
        logger.info("Model state reset", 
                   extra={"extra_fields": {"component": "model", "action": "state_reset"}})
    
    def process_audio_chunk(self, audio_chunk: torch.Tensor) -> str:
        """
        単一の音声チャンクを処理
        
        Args:
            audio_chunk: 音声データ（1,600サンプル）
        
        Returns:
            recognized_text: 認識された文字列
        """
        return self.inference(audio_chunk)
    
    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int):
        """realtimeモデルのチェックポイントをディレクトリ形式で保存"""
        import os
        import json
        
        # ディレクトリを作成
        os.makedirs(path, exist_ok=True)
        
        # モデルの状態辞書を保存
        try:
            # safetensorsファイルとして保存
            from safetensors.torch import save_file
            save_file(self.state_dict(), os.path.join(path, "model.safetensors"))
            logger.info(f"Successfully saved safetensors checkpoint to {path}")
        except ImportError:
            # safetensorsが利用できない場合は、通常のtorch.saveを使用
            logger.warning("safetensors not available, falling back to torch.save")
            torch.save(self.state_dict(), os.path.join(path, "model.safetensors"))
            logger.info(f"Successfully saved torch checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}")
            raise
        
        # 設定を保存
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        # 語彙を保存
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.vocab, f, indent=2)
        
        # 特殊トークンマップを保存
        special_tokens_map = {
            "blank_token": self.blank_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token
        }
        with open(os.path.join(path, "special_tokens_map.json"), "w") as f:
            json.dump(special_tokens_map, f, indent=2)
        
        # 前処理設定を保存
        preprocessor_config = {
            "chunk_size_ms": self.chunk_size_ms,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length
        }
        with open(os.path.join(path, "preprocessor_config.json"), "w") as f:
            json.dump(preprocessor_config, f, indent=2)
        
        # トークナイザー設定を保存
        tokenizer_config = {
            "vocab_size": len(self.vocab),
            "blank_token": self.blank_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # オプティマイザーを保存
        super().save_checkpoint(os.path.join(path, "optimizer.pt"), optimizer, epoch)
    
    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """realtimeモデルのチェックポイントをディレクトリ形式から読み込み"""
        import os
        import json
        
        # モデルの状態辞書を読み込み
        model_path = os.path.join(path, "model.safetensors")
        if os.path.exists(model_path):
            try:
                # safetensorsファイルの読み込み
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                self.load_state_dict(state_dict)
                logger.info(f"Successfully loaded safetensors checkpoint from {model_path}")
            except ImportError:
                # safetensorsが利用できない場合は、通常のtorch.loadを試す
                logger.warning("safetensors not available, falling back to torch.load")
                try:
                    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
                    self.load_state_dict(state_dict)
                    logger.info(f"Successfully loaded torch checkpoint from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint from {model_path}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Failed to load safetensors checkpoint from {model_path}: {e}")
                raise
        else:
            # 旧形式の単一ファイルの場合のフォールバック
            return super().load_checkpoint(path, optimizer)
        
        # オプティマイザーを読み込み
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path) and optimizer:
            return super().load_checkpoint(optimizer_path, optimizer)
        
        return 0


class RealtimeASRPipeline:
    """リアルタイム音声認識パイプライン"""
    
    def __init__(self, model: RealtimeASRModel):
        self.model = model
        self.accumulated_text = ""
        
    def process_audio_chunk(self, audio_chunk: torch.Tensor) -> str:
        """音声チャンクを処理して認識結果を返す"""
        chunk_text = self.model.process_audio_chunk(audio_chunk)
        
        if chunk_text:
            self.accumulated_text += chunk_text
            return chunk_text
        
        return ""
    
    def get_accumulated_text(self) -> str:
        """蓄積されたテキストを取得"""
        return self.accumulated_text
    
    def reset(self):
        """パイプラインをリセット"""
        self.model.reset_state()
        self.accumulated_text = ""


def create_audio_chunks(audio_stream, chunk_size_ms: int = 100, sample_rate: int = 16000):
    """
    音声ストリームを固定サイズのチャンクに分割
    
    Args:
        audio_stream: 連続音声ストリーム
        chunk_size_ms: チャンクサイズ（ミリ秒）
        sample_rate: サンプリングレート
    
    Returns:
        Generator yielding audio chunks
    """
    chunk_samples = int(sample_rate * chunk_size_ms / 1000)
    
    while True:
        chunk = audio_stream.read(chunk_samples)
        if len(chunk) < chunk_samples:
            break
        yield torch.tensor(chunk, dtype=torch.float32)


def post_process_ctc_output(ctc_sequence: List[int], id_to_char: Dict[int, str], blank_token: str = '_') -> str:
    """
    CTC出力を最終的な文字列に変換
    
    Args:
        ctc_sequence: CTCデコーダの出力文字列
        id_to_char: IDから文字へのマッピング
        blank_token: 空白トークン
    
    Returns:
        final_text: 処理済みの文字列
    """
    # 1. 重複除去
    deduplicated = []
    prev_char = None
    
    for char_id in ctc_sequence:
        if char_id != prev_char:
            deduplicated.append(char_id)
        prev_char = char_id
    
    # 2. ブランク削除と文字変換
    final_text = ''.join([id_to_char.get(char_id, '') for char_id in deduplicated 
                         if char_id < len(id_to_char)])  # blank tokenは除外
    
    return final_text
