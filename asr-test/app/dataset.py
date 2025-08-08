import torch
import torchaudio
import librosa
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import soundfile as sf
from .model import CHAR_TO_ID, ID_TO_CHAR


class AudioPreprocessor:
    """音声データの前処理クラス"""
    
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=80,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 f_min=0,
                 f_max=8000):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
        # MelSpectrogram変換器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        # 対数変換
        self.log_transform = torchaudio.transforms.AmplitudeToDB()
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        音声ファイルを前処理してメルスペクトログラムに変換
        """
        # 音声ファイルの読み込み
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # モノラルに変換
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # サンプリングレートの統一
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # メルスペクトログラムに変換
        mel_spec = self.mel_transform(waveform)
        
        # 対数変換
        log_mel_spec = self.log_transform(mel_spec)
        
        # 正規化
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        return log_mel_spec.squeeze(0).transpose(0, 1)  # (time, features)
    
    def preprocess_audio_from_array(self, audio_array: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        numpy配列から音声を前処理
        """
        # テンソルに変換
        waveform = torch.from_numpy(audio_array).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # サンプリングレートの統一
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # メルスペクトログラムに変換
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.log_transform(mel_spec)
        
        # 正規化
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        return log_mel_spec.squeeze(0).transpose(0, 1)


class TextPreprocessor:
    """テキストデータの前処理クラス"""
    
    def __init__(self, char_to_id: dict = CHAR_TO_ID):
        self.char_to_id = char_to_id
        self.id_to_char = {v: k for k, v in char_to_id.items()}
    
    def text_to_ids(self, text: str) -> List[int]:
        """テキストをID列に変換"""
        text = text.lower().strip()
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                # 未知文字はスキップ
                continue
        return ids
    
    def ids_to_text(self, ids: List[int]) -> str:
        """ID列をテキストに変換"""
        text = ""
        for id_val in ids:
            if id_val in self.id_to_char and self.id_to_char[id_val] != '<blank>':
                text += self.id_to_char[id_val]
        return text


class ASRDataset(Dataset):
    """音声認識データセット"""
    
    def __init__(self, 
                 data_dir: str,
                 audio_preprocessor: AudioPreprocessor,
                 text_preprocessor: TextPreprocessor,
                 max_length: int = 1000):
        self.data_dir = data_dir
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.max_length = max_length
        
        # データリストの読み込み
        self.data_list = self._load_data_list()
    
    def _load_data_list(self) -> List[Tuple[str, str]]:
        """データリストを読み込み"""
        data_list = []
        
        # メタデータファイルを探す
        metadata_files = ['metadata.json', 'transcript.txt', 'labels.txt']
        
        for metadata_file in metadata_files:
            metadata_path = os.path.join(self.data_dir, metadata_file)
            if os.path.exists(metadata_path):
                if metadata_file.endswith('.json'):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        for item in metadata:
                            audio_path = os.path.join(self.data_dir, item['audio'])
                            text = item['text']
                            if os.path.exists(audio_path):
                                data_list.append((audio_path, text))
                else:
                    # テキストファイル形式
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    audio_path = os.path.join(self.data_dir, parts[0])
                                    text = parts[1]
                                    if os.path.exists(audio_path):
                                        data_list.append((audio_path, text))
                break
        
        # メタデータファイルがない場合は、音声ファイルを直接探索
        if not data_list:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_path = os.path.join(root, file)
                        # ファイル名からテキストを推測（簡易版）
                        text = os.path.splitext(file)[0].replace('_', ' ').lower()
                        data_list.append((audio_path, text))
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        audio_path, text = self.data_list[idx]
        
        # 音声の前処理
        audio_features = self.audio_preprocessor.preprocess_audio(audio_path)
        
        # テキストの前処理
        text_ids = self.text_preprocessor.text_to_ids(text)
        
        # 長さの制限
        if audio_features.shape[0] > self.max_length:
            audio_features = audio_features[:self.max_length]
        
        if len(text_ids) > self.max_length // 4:  # テキストは音声より短い
            text_ids = text_ids[:self.max_length // 4]
        
        return (
            audio_features,
            torch.tensor(text_ids, dtype=torch.long),
            audio_features.shape[0],
            len(text_ids)
        )


def collate_fn(batch):
    """バッチ処理用のコラテーション関数"""
    audio_features, text_ids, audio_lengths, text_lengths = zip(*batch)
    
    # 音声特徴量のパディング
    max_audio_length = max(audio_lengths)
    padded_audio = []
    for audio in audio_features:
        if audio.shape[0] < max_audio_length:
            padding = torch.zeros(max_audio_length - audio.shape[0], audio.shape[1])
            audio = torch.cat([audio, padding], dim=0)
        padded_audio.append(audio)
    
    # テキストIDのパディング
    max_text_length = max(text_lengths)
    padded_text = []
    for text in text_ids:
        if text.shape[0] < max_text_length:
            padding = torch.zeros(max_text_length - text.shape[0], dtype=torch.long)
            text = torch.cat([text, padding], dim=0)
        padded_text.append(text)
    
    return (
        torch.stack(padded_audio),
        torch.stack(padded_text),
        torch.tensor(audio_lengths),
        torch.tensor(text_lengths)
    )


def create_dataloader(dataset: ASRDataset, 
                     batch_size: int = 8, 
                     shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """データローダーを作成"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )


class SyntheticDataset:
    """合成データセット（テスト用）"""
    
    def __init__(self, 
                 audio_preprocessor: AudioPreprocessor,
                 text_preprocessor: TextPreprocessor,
                 num_samples: int = 100):
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.num_samples = num_samples
        
        # サンプルテキスト
        self.sample_texts = [
            "hello world",
            "good morning",
            "how are you",
            "thank you",
            "please help",
            "nice to meet you",
            "have a good day",
            "see you later",
            "goodbye",
            "excuse me"
        ]
    
    def generate_synthetic_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        """合成データを生成"""
        data = []
        
        for i in range(self.num_samples):
            # ランダムな音声特徴量を生成
            time_steps = np.random.randint(100, 500)
            audio_features = torch.randn(time_steps, 80)  # 80次元のメル特徴量
            
            # ランダムなテキストを選択
            text = np.random.choice(self.sample_texts)
            text_ids = self.text_preprocessor.text_to_ids(text)
            
            data.append((
                audio_features,
                torch.tensor(text_ids, dtype=torch.long),
                time_steps,
                len(text_ids)
            ))
        
        return data
