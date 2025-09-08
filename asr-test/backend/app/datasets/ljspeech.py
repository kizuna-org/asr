# backend/app/datasets/ljspeech.py
import os
import csv
from typing import Dict, Any, List, Tuple

import torch
import torchaudio

from .interface import BaseASRDataset


class LJSpeechDataset(BaseASRDataset):
    """ローカル展開済みの LJSpeech-1.1 を読み込むデータセット。

    期待構成:
      {root}/LJSpeech-1.1/
        ├── metadata.csv  (id|text|normalized_text)
        └── wavs/
             └── {id}.wav
    """

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        super().__init__(config, split)

    def _load_data(self) -> List[Tuple[str, str]]:
        """metadata.csv を読み込み、(wav_path, text) のリストを返す。

        訓練/検証の分割は config の validation_size と random_seed に従う。
        """
        root_dir = self.config.get("path", "/app/data/ljspeech")
        ljs_dir = os.path.join(root_dir, "LJSpeech-1.1")
        metadata_path = os.path.join(ljs_dir, "metadata.csv")
        wavs_dir = os.path.join(ljs_dir, "wavs")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
        if not os.path.isdir(wavs_dir):
            raise FileNotFoundError(f"wavs directory not found: {wavs_dir}")

        entries: List[Tuple[str, str]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if not row:
                    continue
                utt_id = row[0].strip()
                # metadata.csv は id|text|normalized_text
                text = row[2].strip() if len(row) > 2 and row[2].strip() else row[1].strip()
                wav_path = os.path.join(wavs_dir, f"{utt_id}.wav")
                if os.path.exists(wav_path):
                    entries.append((wav_path, text))

        if len(entries) == 0:
            raise RuntimeError(f"No wav entries found under {wavs_dir}")

        # 分割設定
        val_size = float(self.config.get("validation_size", 0.05))
        random_seed = int(self.config.get("random_seed", 42))

        # 決定論的にシャッフル
        g = torch.Generator()
        g.manual_seed(random_seed)
        indices = torch.randperm(len(entries), generator=g).tolist()

        split_idx = int(len(entries) * (1.0 - val_size))
        if self.split == 'train':
            chosen = [entries[i] for i in indices[:split_idx]]
        else:
            chosen = [entries[i] for i in indices[split_idx:]]

        return chosen

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        wav_path, text = self.data[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = self._preprocess_audio(waveform, sample_rate)
        return {
            "waveform": waveform.squeeze(0),
            "sample_rate": 16000,
            "text": text,
        }

    def _preprocess_audio(self, waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
        target_sample_rate = 16000

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        return waveform


def collate_fn(batch):
    """バッチ内の波形をパディングし、テキストはリストで返す"""
    waveforms = [item['waveform'] for item in batch]
    texts = [item['text'] for item in batch]

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    waveform_lengths = torch.tensor([wf.shape[0] for wf in waveforms], dtype=torch.long)

    return {
        "waveforms": padded_waveforms,
        "waveform_lengths": waveform_lengths,
        "texts": texts,
    }
