#!/usr/bin/env python3
"""
リアルタイムモデルのデモスクリプト
"""
import sys
import os
import torch
import torchaudio
import numpy as np
import time
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, '/app')

from app.models.realtime import RealtimeASRModel, RealtimeASRPipeline, create_audio_chunks
from app import config_loader

def generate_test_audio(duration_sec=5.0, sample_rate=16000, frequency=440):
    """テスト用の音声を生成"""
    t = torch.linspace(0, duration_sec, int(sample_rate * duration_sec))
    # 複数の周波数を混ぜてより自然な音声に
    audio = (torch.sin(2 * np.pi * frequency * t) + 
             0.5 * torch.sin(2 * np.pi * frequency * 2 * t) +
             0.3 * torch.sin(2 * np.pi * frequency * 3 * t))
    return audio

def simulate_realtime_processing():
    """リアルタイム処理のシミュレーション"""
    print("🎵 Simulating realtime audio processing...")
    
    # 設定を読み込み
    config = config_loader.load_config()
    realtime_config = config.get('models', {}).get('realtime', {})
    
    if not realtime_config:
        print("❌ Realtime model config not found")
        return
    
    # モデルとパイプラインを初期化
    model = RealtimeASRModel(realtime_config)
    pipeline = RealtimeASRPipeline(model)
    
    # テスト音声を生成（10秒間）
    test_audio = generate_test_audio(duration_sec=10.0)
    sample_rate = 16000
    chunk_size_ms = 100
    chunk_samples = int(sample_rate * chunk_size_ms / 1000)
    
    print(f"📊 Audio: {test_audio.shape[0]} samples ({test_audio.shape[0]/sample_rate:.1f}s)")
    print(f"📊 Chunk size: {chunk_samples} samples ({chunk_size_ms}ms)")
    print(f"📊 Total chunks: {test_audio.shape[0] // chunk_samples}")
    print()
    
    # チャンクごとに処理をシミュレート
    accumulated_text = ""
    chunk_count = 0
    
    for i in range(0, test_audio.shape[0], chunk_samples):
        chunk = test_audio[i:i+chunk_samples]
        
        if chunk.shape[0] < chunk_samples:
            # 最後のチャンクが短い場合はパディング
            padding = torch.zeros(chunk_samples - chunk.shape[0])
            chunk = torch.cat([chunk, padding])
        
        # リアルタイム処理のタイミングをシミュレート
        start_time = time.time()
        
        # チャンクを処理
        chunk_text = pipeline.process_audio_chunk(chunk)
        
        processing_time = time.time() - start_time
        
        if chunk_text:
            accumulated_text += chunk_text
        
        chunk_count += 1
        
        # 進捗表示
        progress = (i + chunk_samples) / test_audio.shape[0] * 100
        print(f"\r🔄 Chunk {chunk_count:3d} | "
              f"Progress: {progress:5.1f}% | "
              f"Processing: {processing_time*1000:4.1f}ms | "
              f"Text: '{accumulated_text}'", end="", flush=True)
        
        # リアルタイム感を出すために少し待機
        time.sleep(0.05)  # 50ms待機
    
    print(f"\n\n✅ Processing completed!")
    print(f"📝 Final accumulated text: '{accumulated_text}'")
    print(f"📊 Total chunks processed: {chunk_count}")
    
    # パイプラインをリセット
    pipeline.reset()
    print("🔄 Pipeline reset completed")

def demo_model_components():
    """モデルコンポーネントのデモ"""
    print("🔧 Demonstrating model components...")
    
    config = config_loader.load_config()
    realtime_config = config.get('models', {}).get('realtime', {})
    
    if not realtime_config:
        print("❌ Realtime model config not found")
        return
    
    model = RealtimeASRModel(realtime_config)
    
    # テスト音声を生成
    test_audio = generate_test_audio(duration_sec=1.0)
    
    print(f"📊 Input audio shape: {test_audio.shape}")
    
    # 特徴抽出のデモ
    features = model.extract_features(test_audio)
    print(f"📊 Extracted features shape: {features.shape}")
    
    # エンコーダのデモ
    encoder_output, hidden_state = model.encoder(features)
    print(f"📊 Encoder output shape: {encoder_output.shape}")
    print(f"📊 Hidden state shape: {hidden_state.shape}")
    
    # デコーダのデモ
    log_probs = model.decoder(encoder_output)
    print(f"📊 Decoder output shape: {log_probs.shape}")
    
    # リアルタイムデコードのデモ
    detected_chars = model.decoder.decode_realtime(log_probs[0])
    print(f"📊 Detected characters: {detected_chars}")
    
    # 後処理のデモ
    final_text = model._post_process_ctc_output(detected_chars)
    print(f"📊 Final text: '{final_text}'")

def main():
    """メインデモ関数"""
    print("🚀 RealtimeASRModel Demo")
    print("=" * 50)
    
    try:
        # コンポーネントデモ
        demo_model_components()
        print("\n" + "=" * 50)
        
        # リアルタイム処理シミュレーション
        simulate_realtime_processing()
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
