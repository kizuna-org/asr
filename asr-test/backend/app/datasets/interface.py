# backend/app/datasets/interface.py
from abc import ABC, abstractmethod
from typing import Dict
from torch.utils.data import Dataset, DataLoader

class BaseASRDataset(Dataset, ABC):
    def __init__(self, config: Dict, split: str = 'train'):
        self.config = config
        self.split = split
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self):
        """データセットのメタデータを読み込む処理"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """
        音声波形とテキストのペアを辞書形式で返す
        (例: {'waveform': tensor, 'text': "hello world"})
        """
        pass

def create_dataloader(dataset: BaseASRDataset, batch_size: int, shuffle: bool, num_workers: int = 0, collate_fn=None) -> DataLoader:
    """データローダーを作成する共通関数"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
