# backend/app/datasets/jsut.py
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
import torchaudio

from .interface import BaseASRDataset


class JSUTDataset(BaseASRDataset):
    """ローカル展開済みの JSUT (Japanese Speech Utterance Corpus) を読み込むデータセット。

    期待構成:
      {root}/jsut_ver1.1/
        ├── basic5000/
        │   ├── wav/
        │   │   └── *.wav
        │   └── transcript_utf8.txt  (ファイルIDとテキストの対応)
        ├── onomatopee300/
        │   ├── wav/
        │   └── transcript_utf8.txt
        └── ...
    """

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        super().__init__(config, split)

    def _load_data(self) -> List[Tuple[str, str]]:
        """JSUTのwavファイルとlabファイルを読み込み、(wav_path, text) のリストを返す。

        訓練/検証の分割は config の validation_size と random_seed に従う。
        """
        root_dir = self.config.get("path", "/app/data/jsut")
        jsut_dir = os.path.join(root_dir, "jsut_ver1.1")

        # ディレクトリが存在しない場合、代替パスを試す
        if not os.path.isdir(jsut_dir):
            # jsut-label-master などの別名で展開されている可能性
            parent_dir = Path(root_dir)
            if parent_dir.exists():
                # 親ディレクトリ内のjsut関連ディレクトリを探す
                for item in parent_dir.iterdir():
                    if item.is_dir() and 'jsut' in item.name.lower():
                        jsut_dir = str(item)
                        print(f"Found JSUT directory with alternative name: {jsut_dir}")
                        break
                else:
                    # ディレクトリ構造を確認してデバッグ情報を出力
                    print(f"Debug: Root directory contents: {list(parent_dir.iterdir())}")
                    raise FileNotFoundError(
                        f"JSUT directory not found: {os.path.join(root_dir, 'jsut_ver1.1')}\n"
                        f"Available directories in {root_dir}: {[d.name for d in parent_dir.iterdir() if d.is_dir()]}"
                    )
            else:
                raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # JSUTのサブディレクトリを取得（basic5000, onomatopee300など）
        subdirs = self.config.get("subdirs", ["basic5000"])
        if isinstance(subdirs, str):
            subdirs = [subdirs]

        entries: List[Tuple[str, str]] = []

        # デバッグ: jsut_dirの内容を確認
        jsut_path = Path(jsut_dir)
        if jsut_path.exists():
            available_subdirs = [d.name for d in jsut_path.iterdir() if d.is_dir()]
            print(f"Debug: Available subdirectories in {jsut_dir}: {available_subdirs}")

        for subdir_name in subdirs:
            subdir_path = os.path.join(jsut_dir, subdir_name)
            if not os.path.isdir(subdir_path):
                print(f"Warning: Subdirectory not found: {subdir_path}, skipping...")
                # 利用可能なサブディレクトリを提案
                if jsut_path.exists():
                    available = [d.name for d in jsut_path.iterdir() if d.is_dir()]
                    print(f"  Available subdirectories: {available}")
                continue

            wav_dir = os.path.join(subdir_path, "wav")
            transcript_path = os.path.join(subdir_path, "transcript_utf8.txt")

            # wavディレクトリの確認
            if not os.path.isdir(wav_dir):
                print(f"Warning: wav directory not found: {wav_dir}, skipping...")
                print(f"  Contents of {subdir_path}: {list(Path(subdir_path).iterdir())}")
                continue

            # transcript_utf8.txtの確認
            if not os.path.exists(transcript_path):
                print(f"Warning: transcript_utf8.txt not found: {transcript_path}, skipping...")
                continue

            # transcript_utf8.txtからテキストを読み込む
            # 形式: 各行に "ファイルID: テキスト" または "ファイルID\tテキスト" など
            transcript_dict = {}
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # コロンまたはタブで分割を試す
                        if ':' in line:
                            parts = line.split(':', 1)
                        elif '\t' in line:
                            parts = line.split('\t', 1)
                        else:
                            # スペースで分割（最初の部分がファイルID、残りがテキスト）
                            parts = line.split(None, 1)
                        
                        if len(parts) >= 2:
                            file_id = parts[0].strip()
                            text = parts[1].strip()
                            transcript_dict[file_id] = text
                        else:
                            # 形式が異なる場合、行全体をテキストとして扱う
                            print(f"Warning: Unexpected transcript format: {line}")
            except Exception as e:
                print(f"Warning: Failed to read transcript file {transcript_path}: {e}")
                continue

            # wavファイルを取得
            wav_files = sorted(Path(wav_dir).glob("*.wav"))
            print(f"Debug: Found {len(wav_files)} wav files in {wav_dir}")
            print(f"Debug: Found {len(transcript_dict)} entries in transcript_utf8.txt")
            
            if len(wav_files) == 0:
                print(f"Warning: No wav files found in {wav_dir}")
                continue

            # wavファイルとtranscriptを対応付ける
            for wav_path in wav_files:
                # ファイル名（拡張子なし）をキーとして使用
                file_id = wav_path.stem
                
                # transcriptからテキストを取得
                if file_id in transcript_dict:
                    text = transcript_dict[file_id]
                    if text:
                        entries.append((str(wav_path), text))
                else:
                    # ファイルIDが見つからない場合、警告を出すがスキップ
                    print(f"Warning: No transcript found for {file_id} in {transcript_path}")
                    continue

        if len(entries) == 0:
            # より詳細なエラーメッセージを提供
            error_msg = f"No wav/lab entries found under {jsut_dir}\n"
            error_msg += f"  Root directory: {root_dir}\n"
            error_msg += f"  JSUT directory: {jsut_dir}\n"
            if os.path.exists(jsut_dir):
                error_msg += f"  Contents: {[d.name for d in Path(jsut_dir).iterdir()]}\n"
            error_msg += f"  Searched subdirs: {subdirs}\n"
            raise RuntimeError(error_msg)

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

        # 軽量テスト用: 先頭から最大 max_samples 件に制限
        max_samples = self.config.get("max_samples") or self.config.get("limit_samples")
        if isinstance(max_samples, int) and max_samples > 0:
            chosen = chosen[:max_samples]

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

