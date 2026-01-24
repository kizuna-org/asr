#!/usr/bin/env python3
"""
既存のモデルファイルを新しいディレクトリ形式に移行するスクリプト
"""
import os
import shutil
import torch
from pathlib import Path

def migrate_old_models():
    """古い形式のモデルファイルを新しいディレクトリ形式に移行"""
    
    # 古いモデルファイルの場所
    old_models_dir = Path("./old.asr-test/asr/asr-test/models")
    new_checkpoints_dir = Path("./checkpoints")
    
    # 新しいチェックポイントディレクトリを作成
    new_checkpoints_dir.mkdir(exist_ok=True)
    
    if not old_models_dir.exists():
        print(f"古いモデルディレクトリが見つかりません: {old_models_dir}")
        return
    
    # 古いモデルファイルを検索
    old_model_files = list(old_models_dir.glob("*.pth"))
    print(f"見つかった古いモデルファイル: {len(old_model_files)}個")
    
    for old_model_file in old_model_files:
        print(f"移行中: {old_model_file.name}")
        
        # ファイル名からエポック番号を抽出
        filename = old_model_file.stem  # checkpoint_epoch_10
        if "checkpoint_epoch_" in filename:
            epoch = filename.replace("checkpoint_epoch_", "")
        else:
            epoch = "unknown"
        
        # 新しいディレクトリ名を作成
        new_model_dir = new_checkpoints_dir / f"conformer-ljspeech-epoch-{epoch}"
        
        # ディレクトリが既に存在する場合はスキップ
        if new_model_dir.exists():
            print(f"  スキップ: {new_model_dir} は既に存在します")
            continue
        
        try:
            # 新しいディレクトリを作成
            new_model_dir.mkdir(parents=True, exist_ok=True)
            
            # 古いモデルファイルを読み込み
            checkpoint = torch.load(old_model_file, map_location='cpu')
            
            # モデルの状態辞書を新しい形式で保存
            if 'model_state_dict' in checkpoint:
                torch.save(checkpoint['model_state_dict'], new_model_dir / "model.safetensors")
            else:
                # 古い形式の場合はそのまま保存
                torch.save(checkpoint, new_model_dir / "model.safetensors")
            
            # オプティマイザーの状態も保存
            if 'optimizer_state_dict' in checkpoint:
                torch.save({
                    'epoch': checkpoint.get('epoch', int(epoch)),
                    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
                }, new_model_dir / "optimizer.pt")
            
            # デフォルト設定を作成
            default_config = {
                "input_dim": 80,
                "encoder_dim": 256,
                "num_encoder_layers": 4,
                "num_heads": 4,
                "kernel_size": 31,
                "dropout": 0.1,
                "huggingface_model_name": "facebook/wav2vec2-base-960h",
                "tokenizer": {
                    "type": "SentencePiece",
                    "vocab_size": 5000
                }
            }
            
            import json
            with open(new_model_dir / "config.json", "w") as f:
                json.dump(default_config, f, indent=2)
            
            # 語彙ファイルを作成
            vocab = {"<pad>": 0, "<unk>": 1, "<blank>": 2}
            with open(new_model_dir / "vocab.json", "w") as f:
                json.dump(vocab, f, indent=2)
            
            print(f"  完了: {new_model_dir}")
            
        except Exception as e:
            print(f"  エラー: {new_model_dir} の移行に失敗しました: {e}")
            # 失敗したディレクトリを削除
            if new_model_dir.exists():
                shutil.rmtree(new_model_dir)
    
    print("移行完了!")

if __name__ == "__main__":
    migrate_old_models()





