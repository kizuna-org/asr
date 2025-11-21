# backend/app/realtime_processor.py
import torch
import torchaudio
import numpy as np
import logging
from typing import List, Optional
from collections import deque

logger = logging.getLogger("realtime_processor")

class RealtimeAudioProcessor:
    """リアルタイム音声処理クラス"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size_seconds: float = 1.0,
                 buffer_size_seconds: float = 3.0,
                 min_chunk_length: float = 0.5,
                 max_silence_duration: float = 2.0):
        """
        Args:
            sample_rate: サンプルレート
            chunk_size_seconds: チャンクサイズ（秒）
            buffer_size_seconds: バッファサイズ（秒）
            min_chunk_length: 最小チャンク長（秒）
            max_silence_duration: 最大無音時間（秒）
        """
        self.sample_rate = sample_rate
        self.chunk_size_samples = int(chunk_size_seconds * sample_rate)
        self.buffer_size_samples = int(buffer_size_seconds * sample_rate)
        self.min_chunk_samples = int(min_chunk_length * sample_rate)
        self.max_silence_samples = int(max_silence_duration * sample_rate)
        
        # 音声バッファ（スライディングウィンドウ）
        self.audio_buffer = deque(maxlen=self.buffer_size_samples)
        
        # 無音検出用
        self.silence_threshold = 0.01  # 無音の閾値
        self.silence_count = 0
        
        # リサンプラー（必要に応じて）
        self.resampler = None
        
        logger.info(f"RealtimeAudioProcessor initialized: "
                   f"sample_rate={sample_rate}, chunk_size={chunk_size_seconds}s, "
                   f"buffer_size={buffer_size_seconds}s")
    
    def add_audio_chunk(self, audio_tensor: torch.Tensor, input_sample_rate: Optional[int] = None) -> bool:
        """
        音声チャンクをバッファに追加
        
        Args:
            audio_tensor: 音声テンソル
            input_sample_rate: 入力サンプルレート（Noneの場合はself.sample_rateと仮定）
            
        Returns:
            bool: バッファが十分な長さになったかどうか
        """
        try:
            # サンプルレート変換が必要な場合
            if input_sample_rate and input_sample_rate != self.sample_rate:
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=input_sample_rate,
                        new_freq=self.sample_rate
                    )
                audio_tensor = self.resampler(audio_tensor)
            
            # バッファに追加
            self.audio_buffer.extend(audio_tensor.tolist())
            
            # 無音検出
            is_silent = self._detect_silence(audio_tensor)
            if is_silent:
                self.silence_count += len(audio_tensor)
            else:
                self.silence_count = 0
            
            # バッファが十分な長さになったかチェック
            return len(self.audio_buffer) >= self.min_chunk_samples
            
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
            return False
    
    def get_audio_buffer(self) -> Optional[torch.Tensor]:
        """
        現在の音声バッファを取得
        
        Returns:
            torch.Tensor: 音声バッファ（十分な長さがない場合はNone）
        """
        if len(self.audio_buffer) < self.min_chunk_samples:
            return None
        
        # バッファをテンソルに変換
        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_array)
        
        return audio_tensor
    
    def clear_buffer(self):
        """音声バッファをクリア"""
        self.audio_buffer.clear()
        self.silence_count = 0
        logger.debug("Audio buffer cleared")
    
    def _detect_silence(self, audio_tensor: torch.Tensor) -> bool:
        """
        無音を検出
        
        Args:
            audio_tensor: 音声テンソル
            
        Returns:
            bool: 無音かどうか
        """
        # RMS（Root Mean Square）を計算
        rms = torch.sqrt(torch.mean(audio_tensor ** 2))
        return rms < self.silence_threshold
    
    def is_silent_too_long(self) -> bool:
        """
        無音が長すぎるかチェック
        
        Returns:
            bool: 無音が長すぎるかどうか
        """
        return self.silence_count > self.max_silence_samples
    
    def get_buffer_info(self) -> dict:
        """
        バッファの情報を取得
        
        Returns:
            dict: バッファ情報
        """
        return {
            "buffer_length": len(self.audio_buffer),
            "buffer_duration_seconds": len(self.audio_buffer) / self.sample_rate,
            "silence_count": self.silence_count,
            "silence_duration_seconds": self.silence_count / self.sample_rate,
            "is_silent": self.silence_count > 0,
            "is_silent_too_long": self.is_silent_too_long()
        }
    
    def preprocess_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        音声の前処理
        
        Args:
            audio_tensor: 入力音声テンソル
            
        Returns:
            torch.Tensor: 前処理済み音声テンソル
        """
        # 正規化
        if audio_tensor.abs().max() > 0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        # パディング（必要に応じて）
        if len(audio_tensor) < self.min_chunk_samples:
            padding = self.min_chunk_samples - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        
        return audio_tensor
    
    def should_process(self) -> bool:
        """
        処理すべきかどうかを判定
        
        Returns:
            bool: 処理すべきかどうか
        """
        # バッファが十分な長さがある
        has_enough_data = len(self.audio_buffer) >= self.min_chunk_samples
        
        # 無音が長すぎない
        not_silent_too_long = not self.is_silent_too_long()
        
        return has_enough_data and not_silent_too_long





