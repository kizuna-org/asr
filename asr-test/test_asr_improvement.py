#!/usr/bin/env python3
"""
音声認識改善のテストスクリプト
"""

import torch
import numpy as np
import librosa
import time
import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.model import FastASRModel, LightweightASRModel, CHAR_TO_ID, ID_TO_CHAR
from app.dataset import AudioPreprocessor, TextPreprocessor
from app.utils import AudioProcessor, PerformanceMonitor


def test_model_initialization():
    """モデル初期化のテスト"""
    print("=== モデル初期化テスト ===")
    
    # FastASRModelのテスト
    fast_model = FastASRModel(hidden_dim=64)
    print(f"FastASRModel parameters: {sum(p.numel() for p in fast_model.parameters()):,}")
    
    # LightweightASRModelのテスト
    light_model = LightweightASRModel(hidden_dim=128)
    print(f"LightweightASRModel parameters: {sum(p.numel() for p in light_model.parameters()):,}")
    
    print("✅ モデル初期化テスト完了\n")


def test_audio_preprocessing():
    """音声前処理のテスト"""
    print("=== 音声前処理テスト ===")
    
    # ダミー音声データを作成
    sample_rate = 16000
    duration = 2.0  # 2秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 複数の周波数の音声を合成
    audio = (np.sin(2 * np.pi * 440 * t) +  # A音
             0.5 * np.sin(2 * np.pi * 880 * t) +  # A音の倍音
             0.3 * np.sin(2 * np.pi * 1320 * t))  # さらに高い倍音
    
    # ノイズを追加
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    print(f"元の音声データ: shape={audio.shape}, range=[{audio.min():.4f}, {audio.max():.4f}]")
    
    # 音声前処理
    preprocessor = AudioPreprocessor()
    features = preprocessor.preprocess_audio_from_array(audio, sample_rate)
    
    print(f"前処理後の特徴量: shape={features.shape}, range=[{features.min():.4f}, {features.max():.4f}]")
    print("✅ 音声前処理テスト完了\n")
    
    return features


def test_model_inference(features):
    """モデル推論のテスト"""
    print("=== モデル推論テスト ===")
    
    # モデルを初期化
    model = FastASRModel(hidden_dim=64)
    model.eval()
    
    # 入力データを準備
    input_tensor = features.unsqueeze(0)  # バッチ次元を追加
    print(f"入力テンソル: shape={input_tensor.shape}")
    
    # 推論実行
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_tensor)
        inference_time = time.time() - start_time
    
    print(f"ロジット出力: shape={logits.shape}")
    print(f"推論時間: {inference_time:.4f}秒")
    
    # デコードテスト
    decoded_sequences = model.decode(logits, beam_size=3)
    print(f"デコード結果: {decoded_sequences}")
    
    # テキスト変換
    text_preprocessor = TextPreprocessor()
    if decoded_sequences and decoded_sequences[0]:
        text = text_preprocessor.ids_to_text(decoded_sequences[0])
        print(f"認識テキスト: '{text}'")
    else:
        print("認識テキスト: (空)")
    
    print("✅ モデル推論テスト完了\n")


def test_audio_enhancement():
    """音声品質向上のテスト"""
    print("=== 音声品質向上テスト ===")
    
    # ダミー音声データ
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # ノイズを追加
    noise = np.random.normal(0, 0.2, len(audio))
    noisy_audio = audio + noise
    
    print(f"ノイズ付き音声: SNR={20*np.log10(np.std(audio)/np.std(noise)):.2f}dB")
    
    # 音声品質向上を適用
    enhanced_audio = AudioProcessor.normalize_audio(noisy_audio)
    enhanced_audio = AudioProcessor.apply_preemphasis(enhanced_audio)
    enhanced_audio = AudioProcessor.remove_silence(enhanced_audio)
    
    print(f"品質向上後: shape={enhanced_audio.shape}")
    print("✅ 音声品質向上テスト完了\n")


def test_performance_monitor():
    """パフォーマンス監視のテスト"""
    print("=== パフォーマンス監視テスト ===")
    
    monitor = PerformanceMonitor()
    
    # ダミーデータを記録
    for i in range(5):
        inference_time = 0.1 + np.random.normal(0, 0.02)  # 0.1秒前後
        audio_duration = 2.0 + np.random.normal(0, 0.1)   # 2秒前後
        monitor.record_inference(inference_time, audio_duration)
    
    # 統計情報を取得
    stats = monitor.get_stats()
    print(f"統計情報: {stats}")
    print("✅ パフォーマンス監視テスト完了\n")


def test_beam_search():
    """ビームサーチのテスト"""
    print("=== ビームサーチテスト ===")
    
    # ダミーのロジットを作成
    batch_size, time_steps, num_classes = 1, 10, 29
    logits = torch.randn(batch_size, time_steps, num_classes)
    
    # モデルを初期化
    model = FastASRModel(hidden_dim=64)
    
    # 異なるビームサイズでテスト
    for beam_size in [1, 3, 5]:
        start_time = time.time()
        decoded_sequences = model.decode(logits, beam_size=beam_size)
        decode_time = time.time() - start_time
        
        print(f"ビームサイズ {beam_size}: 結果={decoded_sequences}, 時間={decode_time:.4f}秒")
    
    print("✅ ビームサーチテスト完了\n")


def main():
    """メイン関数"""
    print("🚀 音声認識改善テスト開始\n")
    
    try:
        # 各テストを実行
        test_model_initialization()
        features = test_audio_preprocessing()
        test_model_inference(features)
        test_audio_enhancement()
        test_performance_monitor()
        test_beam_search()
        
        print("🎉 全てのテストが完了しました！")
        print("\n改善点:")
        print("- ビームサーチによる高精度デコード")
        print("- 音声品質向上（ノイズ除去、プリエンファシス）")
        print("- 音声活動検出（VAD）")
        print("- 信頼度フィルタリング")
        print("- 認識結果の平滑化")
        print("- パフォーマンス監視の改善")
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
