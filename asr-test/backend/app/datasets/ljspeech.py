# backend/app/datasets/ljspeech.py
import torch
import torchaudio
from datasets import load_dataset
from transformers import T5Tokenizer
from typing import Dict, Any

from .interface import BaseASRDataset

class LJSpeechDataset(BaseASRDataset):
    """LJSpeechデータセットを扱うクラス"""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        super().__init__(config, split)

    def _load_data(self):
        """Hugging Faceのdatasetsライブラリを使ってデータをロードする"""
        # LJSpeechはデフォルトでtrainスプリットしかないので、split引数は無視する
        dataset = load_dataset("ljspeech", split='train')
        # デバッグ用に小さなサブセットを使う場合は以下を有効化
        # return dataset.select(range(100))
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        sample_rate = item["audio"]["sampling_rate"]

        # 1. 音声の前処理
        waveform = self._preprocess_audio(waveform, sample_rate)

        # 2. テキストの前処理
        text = item["text"]
        # TODO: text_cleanersを適用する
        token_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {
            "waveform": waveform,
            "sample_rate": self.config["sample_rate"],
            "text": text,
            "token_ids": token_ids
        }

    def _preprocess_audio(self, waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
        """音声データをリサンプリングし、メルスペクトログラムに変換する"""
        target_sample_rate = self.config["sample_rate"]

        # リサンプリング
        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        # メルスペクトログラム計算
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=self.config["n_fft"],
            win_length=self.config["win_length"],
            hop_length=self.config["hop_length"],
            n_mels=self.config["n_mels"],
            f_min=self.config["f_min"],
            f_max=self.config["f_max"]
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)

        return mel_spectrogram

def collate_fn(batch):
    """バッチ内のデータをパディングしてテンソルにまとめる"""
    waveforms = [item['waveform'].squeeze(0).T for item in batch] # (Time, Freq)
    token_ids = [item['token_ids'] for item in batch]

    # 波形をパディング
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).transpose(1, 2) # (Batch, Freq, Time)

    # トークンIDをパディング
    padded_token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True)

    # テキストはそのままリストで返す
    texts = [item['text'] for item in batch]

    return {
        "waveforms": padded_waveforms,
        "waveform_lengths": torch.tensor([wf.shape[0] for wf in waveforms]),
        "token_ids": padded_token_ids,
        "token_lengths": torch.tensor([len(t) for t in token_ids]),
        "texts": texts
    }
