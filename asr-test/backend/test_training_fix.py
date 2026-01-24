#!/usr/bin/env python3
"""
学習修正のテストスクリプト
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, '/app')

from app.models.realtime import RealtimeASRModel
from app import config_loader

def test_realtime_model_forward():
    """リアルタイムモデルのforwardメソッドをテスト"""
    print("🧪 Testing RealtimeASRModel forward method...")
    
    # 設定を読み込み
    config = config_loader.load_config()
    realtime_config = config.get('models', {}).get('realtime', {})
    
    if not realtime_config:
        print("❌ Realtime model config not found")
        return False
    
    try:
        # モデルを初期化
        model = RealtimeASRModel(realtime_config)
        model.train()  # 学習モードに設定
        
        print("✅ Model initialized successfully")
        
        # テスト用のバッチデータを作成
        batch_size = 2
        sample_rate = 16000
        duration = 1.0  # 1秒
        
        # 音声データを生成
        waveforms = []
        texts = []
        
        for i in range(batch_size):
            # テスト音声を生成
            t = torch.linspace(0, duration, int(sample_rate * duration))
            freq = 440 + i * 100  # 異なる周波数
            waveform = torch.sin(2 * np.pi * freq * t)
            waveforms.append(waveform)
            
            # テストテキスト
            texts.append(f"test text {i}")
        
        # バッチを作成
        batch = {
            "waveforms": waveforms,
            "texts": texts
        }
        
        print(f"✅ Test batch created: {len(waveforms)} waveforms, {len(texts)} texts")
        
        # forwardメソッドを実行
        loss = model(batch)
        
        print(f"✅ Forward pass completed")
        print(f"📊 Loss value: {loss.item():.6f}")
        print(f"📊 Loss requires_grad: {loss.requires_grad}")
        print(f"📊 Loss device: {loss.device}")
        
        # 損失が0でないことを確認
        if loss.item() == 0.0:
            print("❌ Loss is still 0.0 - forward method may not be working correctly")
            return False
        
        # 勾配計算のテスト
        loss.backward()
        
        # パラメータに勾配が設定されているかチェック
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                print(f"✅ Parameter '{name}' has gradients: {param.grad.abs().sum().item():.6f}")
                break
        
        if not has_gradients:
            print("❌ No gradients found - backward pass may not be working correctly")
            return False
        
        print("✅ Gradients computed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctc_loss_calculation():
    """CTC損失計算の詳細テスト"""
    print("\n🧪 Testing CTC loss calculation...")
    
    try:
        # 設定を読み込み
        config = config_loader.load_config()
        realtime_config = config.get('models', {}).get('realtime', {})
        
        model = RealtimeASRModel(realtime_config)
        model.train()
        
        # 簡単なテストケース
        batch_size = 1
        seq_len = 10
        vocab_size = 1000
        
        # ダミーのlog_probsを作成
        log_probs = torch.randn(seq_len, batch_size, vocab_size + 1)
        log_probs = F.log_softmax(log_probs, dim=-1)
        
        # ダミーのターゲット
        targets = torch.tensor([1, 2, 3], dtype=torch.long)  # "abc"に対応
        target_lengths = torch.tensor([3], dtype=torch.long)
        input_lengths = torch.tensor([seq_len], dtype=torch.long)
        
        # CTC損失を計算
        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # [time, batch, vocab]
            targets,
            input_lengths,
            target_lengths,
            blank=vocab_size,
            reduction='mean'
        )
        
        print(f"✅ CTC loss calculated: {ctc_loss.item():.6f}")
        print(f"✅ CTC loss requires_grad: {ctc_loss.requires_grad}")
        
        if ctc_loss.item() == 0.0:
            print("❌ CTC loss is 0.0 - this may indicate an issue")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CTC loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading_in_trainer():
    """trainerでのモデル読み込みテスト"""
    print("\n🧪 Testing model loading in trainer...")
    
    try:
        import importlib
        
        # trainerと同じ方法でモデルクラスを取得
        model_name = "realtime"
        if model_name == "realtime":
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), "RealtimeASRModel")
        else:
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        
        print(f"✅ Model class loaded: {ModelClass.__name__}")
        
        # 設定を読み込み
        config = config_loader.load_config()
        realtime_config = config.get('models', {}).get('realtime', {})
        
        # モデルをインスタンス化
        model = ModelClass(realtime_config)
        print(f"✅ Model instantiated: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("🚀 Starting training fix tests...\n")
    
    tests = [
        ("Model Loading in Trainer", test_model_loading_in_trainer),
        ("CTC Loss Calculation", test_ctc_loss_calculation),
        ("Realtime Model Forward", test_realtime_model_forward),
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
        print("🎉 All tests passed! Training should now work correctly.")
        return 0
    else:
        print("💥 Some tests failed! Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
