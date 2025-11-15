#!/usr/bin/env python3
"""
リアルタイムモデルのテストスクリプト
"""
import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, '/app')

from app.models.realtime import RealtimeASRModel, RealtimeASRPipeline, create_audio_chunks
from app import config_loader

def test_realtime_model():
    """リアルタイムモデルの基本動作をテスト"""
    print("🧪 Testing RealtimeASRModel...")
    
    # 設定を読み込み
    config = config_loader.load_config()
    realtime_config = config.get('models', {}).get('realtime', {})
    
    if not realtime_config:
        print("❌ Realtime model config not found")
        return False
    
    print(f"✅ Config loaded: {realtime_config}")
    
    try:
        # モデルを初期化
        model = RealtimeASRModel(realtime_config)
        print("✅ Model initialized successfully")
        
        # テスト用の音声データを生成（1秒間のサイン波）
        sample_rate = 16000
        duration = 1.0  # 1秒
        frequency = 440  # A音
        t = torch.linspace(0, duration, int(sample_rate * duration))
        test_audio = torch.sin(2 * np.pi * frequency * t)
        
        print(f"✅ Test audio generated: {test_audio.shape}")
        
        # 推論テスト
        model.eval()
        with torch.no_grad():
            result = model.inference(test_audio)
            print(f"✅ Inference result: '{result}'")
        
        # パイプラインのテスト
        pipeline = RealtimeASRPipeline(model)
        chunk_result = pipeline.process_audio_chunk(test_audio)
        print(f"✅ Pipeline result: '{chunk_result}'")
        
        # 状態リセットのテスト
        model.reset_state()
        print("✅ State reset successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_chunking():
    """音声チャンキングのテスト"""
    print("\n🧪 Testing audio chunking...")
    
    try:
        # テスト用の音声ストリーム（模擬）
        class MockAudioStream:
            def __init__(self, audio_data, chunk_size):
                self.audio_data = audio_data
                self.chunk_size = chunk_size
                self.position = 0
            
            def read(self, size):
                if self.position >= len(self.audio_data):
                    return []
                end = min(self.position + size, len(self.audio_data))
                chunk = self.audio_data[self.position:end]
                self.position = end
                return chunk
        
        # 5秒間のテスト音声
        sample_rate = 16000
        duration = 5.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        test_audio = torch.sin(2 * np.pi * 440 * t)
        
        # チャンクサイズ（100ms）
        chunk_size_ms = 100
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        
        mock_stream = MockAudioStream(test_audio, chunk_samples)
        
        chunks = list(create_audio_chunks(mock_stream, chunk_size_ms, sample_rate))
        print(f"✅ Generated {len(chunks)} chunks")
        print(f"✅ First chunk shape: {chunks[0].shape if chunks else 'No chunks'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """モデルコンポーネントの個別テスト"""
    print("\n🧪 Testing model components...")
    
    try:
        from app.models.realtime import RealtimeEncoder, RealtimeCTCDecoder
        
        # エンコーダのテスト
        encoder = RealtimeEncoder(input_dim=80, hidden_dim=256, num_layers=3)
        test_input = torch.randn(1, 10, 80)  # [batch, seq_len, input_dim]
        output, hidden = encoder(test_input)
        print(f"✅ Encoder output shape: {output.shape}")
        print(f"✅ Hidden state shape: {hidden.shape}")
        
        # デコーダのテスト
        decoder = RealtimeCTCDecoder(input_dim=256, vocab_size=1000)
        log_probs = decoder(output)
        print(f"✅ Decoder output shape: {log_probs.shape}")
        
        # リアルタイムデコードのテスト
        detected_chars = decoder.decode_realtime(log_probs[0])
        print(f"✅ Detected chars: {detected_chars}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("🚀 Starting RealtimeASRModel tests...\n")
    
    tests = [
        ("Model Components", test_model_components),
        ("Audio Chunking", test_audio_chunking),
        ("Realtime Model", test_realtime_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
