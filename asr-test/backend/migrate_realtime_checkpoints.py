#!/usr/bin/env python3
"""
realtimeモデルのチェックポイントを旧形式（ファイル）から新形式（ディレクトリ）に移行するスクリプト
"""

import os
import json
import torch
import shutil
from pathlib import Path

def migrate_realtime_checkpoints(checkpoints_dir="./checkpoints"):
    """realtimeモデルのチェックポイントを移行"""
    
    # realtimeモデルの旧形式チェックポイントを探す
    old_pattern = f"{checkpoints_dir}/realtime-ljspeech-epoch-*.pt"
    old_checkpoints = []
    
    for file_path in Path(checkpoints_dir).glob("realtime-ljspeech-epoch-*.pt"):
        if file_path.is_file():
            old_checkpoints.append(file_path)
    
    if not old_checkpoints:
        print("移行対象のrealtimeチェックポイントが見つかりませんでした。")
        return
    
    print(f"移行対象のチェックポイント: {len(old_checkpoints)}個")
    
    for old_checkpoint in old_checkpoints:
        print(f"移行中: {old_checkpoint}")
        
        # 新しいディレクトリ名を作成
        new_dir = old_checkpoint.with_suffix('')  # .ptを除去
        
        # ディレクトリが既に存在する場合はスキップ
        if new_dir.exists():
            print(f"  スキップ: {new_dir} は既に存在します")
            continue
        
        try:
            # 新しいディレクトリを作成
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # 旧チェックポイントを読み込み
            checkpoint = torch.load(old_checkpoint, map_location='cpu')
            
            # モデルの状態辞書を保存
            torch.save(checkpoint['model_state_dict'], new_dir / "model.safetensors")
            
            # オプティマイザーを保存
            if 'optimizer_state_dict' in checkpoint:
                torch.save({
                    'epoch': checkpoint.get('epoch', 0),
                    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
                }, new_dir / "optimizer.pt")
            
            # デフォルト設定を作成（実際の設定は学習時に保存される）
            default_config = {
                "encoder": {
                    "input_dim": 80,
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "rnn_type": "GRU",
                    "dropout": 0.1
                },
                "decoder": {
                    "input_dim": 256,
                    "vocab_size": 1000,
                    "blank_token": "_"
                },
                "processing": {
                    "chunk_size_ms": 100,
                    "sample_rate": 16000,
                    "feature_type": "mel_spectrogram",
                    "n_mels": 80,
                    "n_fft": 1024,
                    "hop_length": 160
                }
            }
            
            with open(new_dir / "config.json", "w") as f:
                json.dump(default_config, f, indent=2)
            
            # デフォルト語彙を作成
            default_vocab = {str(i): chr(ord('a') + i) for i in range(26)}  # a-z
            default_vocab.update({
                "26": "_",  # blank token
                "27": "<pad>",
                "28": "<unk>"
            })
            
            with open(new_dir / "vocab.json", "w") as f:
                json.dump(default_vocab, f, indent=2)
            
            # 特殊トークンマップ
            special_tokens_map = {
                "blank_token": "_",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            }
            with open(new_dir / "special_tokens_map.json", "w") as f:
                json.dump(special_tokens_map, f, indent=2)
            
            # 前処理設定
            preprocessor_config = {
                "chunk_size_ms": 100,
                "sample_rate": 16000,
                "n_mels": 80,
                "n_fft": 1024,
                "hop_length": 160
            }
            with open(new_dir / "preprocessor_config.json", "w") as f:
                json.dump(preprocessor_config, f, indent=2)
            
            # トークナイザー設定
            tokenizer_config = {
                "vocab_size": 1000,
                "blank_token": "_",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            }
            with open(new_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            
            print(f"  完了: {new_dir}")
            
        except Exception as e:
            print(f"  エラー: {e}")
            # エラーが発生した場合は作成したディレクトリを削除
            if new_dir.exists():
                shutil.rmtree(new_dir)
    
    print("移行完了！")

if __name__ == "__main__":
    migrate_realtime_checkpoints()
