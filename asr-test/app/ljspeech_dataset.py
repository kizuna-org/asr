import tensorflow as tf
import numpy as np
import os
import json
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from app.dataset import AudioPreprocessor, TextPreprocessor


class LJSpeechDataset(Dataset):
    """LJSpeechデータセット用のクラス"""
    
    def __init__(self, 
                 data_dir: str = "/app/datasets/ljspeech/1.1.1",
                 audio_preprocessor: AudioPreprocessor = None,
                 text_preprocessor: TextPreprocessor = None,
                 max_length: int = 1000,
                 split: str = "train"):
        
        self.data_dir = data_dir
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.max_length = max_length
        self.split = split
        
        # メタデータの読み込み
        self.metadata = self._load_metadata()
        
        # TFRecordファイルのリスト
        self.tfrecord_files = self._get_tfrecord_files()
        
        # データセットの初期化
        self.dataset = self._create_tf_dataset()
        
        # データの長さを取得
        self._length = self._get_dataset_length()
    
    def _load_metadata(self) -> dict:
        """メタデータを読み込み"""
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _get_tfrecord_files(self) -> List[str]:
        """TFRecordファイルのリストを取得"""
        tfrecord_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith(f"ljspeech-{self.split}.tfrecord-") and file.endswith(".tfrecord"):
                tfrecord_files.append(os.path.join(self.data_dir, file))
        return sorted(tfrecord_files)
    
    def _create_tf_dataset(self):
        """TensorFlowデータセットを作成"""
        if not self.tfrecord_files:
            return None
        
        # TFRecordデータセットを作成
        dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        
        # パース関数を定義
        def parse_tfrecord(example_proto):
            feature_description = {
                'audio': tf.io.FixedLenFeature([], tf.string),
                'text': tf.io.FixedLenFeature([], tf.string),
                'audio_length': tf.io.FixedLenFeature([], tf.int64),
                'text_length': tf.io.FixedLenFeature([], tf.int64)
            }
            
            parsed_features = tf.io.parse_single_example(example_proto, feature_description)
            
            # 音声データをデコード
            audio = tf.io.parse_tensor(parsed_features['audio'], out_type=tf.float32)
            text = parsed_features['text']
            
            return audio, text
        
        # データセットにパース関数を適用
        dataset = dataset.map(parse_tfrecord)
        
        return dataset
    
    def _get_dataset_length(self) -> int:
        """データセットの長さを取得"""
        if not self.dataset:
            return 0
        
        # サンプル数を数える（時間がかかるので注意）
        count = 0
        for _ in self.dataset:
            count += 1
            if count > 1000:  # 最大1000サンプルまで数える
                break
        
        return count
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, int, int]:
        """データセットからアイテムを取得"""
        if not self.dataset:
            raise IndexError("データセットが空です")
        
        # データセットをイテレートして指定されたインデックスのデータを取得
        for i, (audio, text) in enumerate(self.dataset):
            if i == idx:
                # 音声データをnumpy配列に変換
                audio_np = audio.numpy()
                text_str = text.numpy().decode('utf-8')
                
                # 音声の前処理
                if self.audio_preprocessor:
                    audio_features = self.audio_preprocessor.preprocess_audio_from_array(
                        audio_np, 22050  # LJSpeechは22.05kHz
                    )
                else:
                    # 簡易的な前処理
                    audio_features = torch.from_numpy(audio_np).float()
                
                # テキストの前処理
                if self.text_preprocessor:
                    text_ids = self.text_preprocessor.text_to_ids(text_str)
                else:
                    text_ids = [ord(c) % 29 for c in text_str.lower() if c.isalpha() or c == ' ']
                
                # 長さの制限
                if audio_features.shape[0] > self.max_length:
                    audio_features = audio_features[:self.max_length]
                
                if len(text_ids) > self.max_length // 4:
                    text_ids = text_ids[:self.max_length // 4]
                
                return (
                    audio_features,
                    torch.tensor(text_ids, dtype=torch.long),
                    audio_features.shape[0],
                    len(text_ids)
                )
        
        raise IndexError(f"インデックス {idx} が見つかりません")
    
    def get_sample_texts(self, num_samples: int = 10) -> List[str]:
        """サンプルテキストを取得"""
        texts = []
        for i, (_, text) in enumerate(self.dataset):
            if i >= num_samples:
                break
            texts.append(text.numpy().decode('utf-8'))
        return texts


class LJSpeechDataLoader:
    """LJSpeechデータローダー"""
    
    def __init__(self, 
                 data_dir: str = "/app/datasets/ljspeech/1.1.1",
                 audio_preprocessor: AudioPreprocessor = None,
                 text_preprocessor: TextPreprocessor = None,
                 batch_size: int = 8,
                 max_length: int = 1000):
        
        self.data_dir = data_dir
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.batch_size = batch_size
        self.max_length = max_length
        
        # データセットを作成
        self.dataset = LJSpeechDataset(
            data_dir=data_dir,
            audio_preprocessor=audio_preprocessor,
            text_preprocessor=text_preprocessor,
            max_length=max_length
        )
    
    def get_dataloader(self):
        """PyTorch DataLoaderを作成"""
        from torch.utils.data import DataLoader
        from app.dataset import collate_fn
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Docker環境では0に設定
            pin_memory=True
        )
    
    def get_dataset_info(self) -> dict:
        """データセット情報を取得"""
        return {
            "total_samples": len(self.dataset),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "data_dir": self.data_dir,
            "tfrecord_files": len(self.dataset.tfrecord_files),
            "sample_texts": self.dataset.get_sample_texts(5)
        }


def create_ljspeech_dataloader(data_dir: str = "/app/datasets/ljspeech/1.1.1",
                              audio_preprocessor: AudioPreprocessor = None,
                              text_preprocessor: TextPreprocessor = None,
                              batch_size: int = 8,
                              max_length: int = 1000):
    """LJSpeechデータローダーを作成"""
    loader = LJSpeechDataLoader(
        data_dir=data_dir,
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        batch_size=batch_size,
        max_length=max_length
    )
    return loader.get_dataloader(), loader.get_dataset_info()
