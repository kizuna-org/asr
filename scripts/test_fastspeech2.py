#!/usr/bin/env python3
"""
FastSpeech 2 Model Test Script
This script tests the basic functionality of the FastSpeech 2 model implementation.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add current directory to path to import fastspeech2_model
sys.path.append(os.path.dirname(__file__))

def test_fastspeech2_model():
    """Test FastSpeech 2 model basic functionality."""
    print("=== FastSpeech 2 モデルテスト ===")
    
    try:
        # Import FastSpeech 2 components
        from fastspeech2_model import (
            FastSpeech2, 
            create_fastspeech2_config,
            MultiHeadAttention,
            PositionwiseFeedForward,
            TransformerBlock,
            FastSpeechEncoder,
            VariancePredictor,
            VarianceAdaptor,
            FastSpeechDecoder,
            PostNet
        )
        print("✅ FastSpeech 2 モジュールのインポート成功")
        
        # Test configuration creation
        config = create_fastspeech2_config()
        print("✅ 設定ファイルの作成成功")
        print(f"   設定内容: {config}")
        
        # Test model creation
        model = FastSpeech2(config)
        print("✅ FastSpeech 2 モデルの作成成功")
        
        # Test model with dummy input
        batch_size = 2
        seq_length = 10
        dummy_input = tf.random.uniform((batch_size, seq_length), 0, config['vocab_size'], dtype=tf.int32)
        
        print(f"🔍 ダミー入力でモデルをテスト中...")
        print(f"   入力形状: {dummy_input.shape}")
        
        # Forward pass
        outputs = model(dummy_input, training=False)
        
        print("✅ モデルの前向き計算成功")
        print(f"   出力キー: {list(outputs.keys())}")
        
        # Check output shapes
        expected_outputs = ['mel_output', 'mel_output_refined', 'duration_pred', 'pitch_pred', 'energy_pred']
        for key in expected_outputs:
            if key in outputs:
                print(f"   {key}: {outputs[key].shape}")
            else:
                print(f"❌ 期待される出力 '{key}' が見つかりません")
                return False
        
        # Test individual components
        print("\n🔍 個別コンポーネントのテスト...")
        
        # Test MultiHeadAttention
        mha = MultiHeadAttention(config['attention_dim'], config['encoder_attention_heads'])
        dummy_seq = tf.random.normal((batch_size, seq_length, config['attention_dim']))
        mha_output = mha(dummy_seq)
        print(f"✅ MultiHeadAttention: {mha_output.shape}")
        
        # Test PositionwiseFeedForward
        ffn = PositionwiseFeedForward(config['attention_dim'], config['encoder_ffn_filter_size'], 
                                    config['encoder_ffn_kernel_size'])
        ffn_output = ffn(dummy_seq)
        print(f"✅ PositionwiseFeedForward: {ffn_output.shape}")
        
        # Test TransformerBlock
        transformer_block = TransformerBlock(config['attention_dim'], config['encoder_attention_heads'],
                                           config['encoder_ffn_filter_size'], config['encoder_ffn_kernel_size'])
        tb_output = transformer_block(dummy_seq)
        print(f"✅ TransformerBlock: {tb_output.shape}")
        
        # Test Encoder
        encoder = FastSpeechEncoder(config['vocab_size'], config['attention_dim'], 
                                  config['encoder_layers'], config['encoder_attention_heads'],
                                  config['encoder_ffn_filter_size'], config['encoder_ffn_kernel_size'])
        encoder_output = encoder(dummy_input)
        print(f"✅ FastSpeechEncoder: {encoder_output.shape}")
        
        # Test VariancePredictor
        variance_pred = VariancePredictor(config['attention_dim'], config['predictor_layers'],
                                        config['predictor_kernel_size'], config['predictor_filter_size'])
        var_output = variance_pred(encoder_output)
        print(f"✅ VariancePredictor: {var_output.shape}")
        
        # Test VarianceAdaptor
        variance_adaptor = VarianceAdaptor(config['attention_dim'], config['predictor_layers'],
                                         config['predictor_kernel_size'], config['predictor_filter_size'])
        va_output, duration, pitch, energy = variance_adaptor(encoder_output)
        print(f"✅ VarianceAdaptor: {va_output.shape}, {duration.shape}, {pitch.shape}, {energy.shape}")
        
        # Test Decoder
        decoder = FastSpeechDecoder(config['attention_dim'], config['decoder_layers'],
                                  config['decoder_attention_heads'], config['decoder_ffn_filter_size'],
                                  config['decoder_ffn_kernel_size'])
        decoder_output = decoder(va_output)
        print(f"✅ FastSpeechDecoder: {decoder_output.shape}")
        
        # Test PostNet
        postnet = PostNet(config['num_mels'], config['postnet_layers'],
                         config['postnet_kernel_size'], config['postnet_filters'])
        mel_dummy = tf.random.normal((batch_size, seq_length, config['num_mels']))
        postnet_output = postnet(mel_dummy)
        print(f"✅ PostNet: {postnet_output.shape}")
        
        print("\n🎉 すべてのテストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_integration():
    """Test FastSpeech2Trainer integration."""
    print("\n=== FastSpeech2Trainer 統合テスト ===")
    
    try:
        # Import modules
        sys.path.append(os.path.dirname(__file__))
        from ljspeech_demo import FastSpeech2Trainer, FastSpeech2Loss
        from fastspeech2_model import FastSpeech2, create_fastspeech2_config
        
        print("✅ トレーナーモジュールのインポート成功")
        
        # Create model
        config = create_fastspeech2_config()
        fastspeech2_model = FastSpeech2(config)
        
        # Create trainer
        trainer = FastSpeech2Trainer(fastspeech2_model)
        trainer.compile(optimizer='adam', metrics=['mae'])
        
        print("✅ FastSpeech2Trainer の作成成功")
        
        # Test with dummy data
        batch_size = 2
        seq_length = 10
        dummy_x = tf.random.uniform((batch_size, seq_length), 0, config['vocab_size'], dtype=tf.int32)
        dummy_y = tf.random.normal((batch_size, seq_length, config['num_mels']))
        
        # Test forward pass
        outputs = trainer(dummy_x, training=False)
        print(f"✅ トレーナーの前向き計算成功: 出力キー = {list(outputs.keys())}")
        
        # Test loss function
        loss_fn = FastSpeech2Loss()
        loss_value = loss_fn(dummy_y, outputs)
        print(f"✅ 損失関数の計算成功: loss = {loss_value:.4f}")
        
        print("\n🎉 統合テストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("FastSpeech 2 モデル包括テスト開始")
    print("=" * 50)
    
    # Set memory growth
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"🔧 GPU設定完了: {len(physical_devices)} GPU(s)")
        else:
            print("🔧 CPUモードで実行")
    except Exception as e:
        print(f"⚠️  GPU設定警告: {e}")
    
    # Run tests
    test1_passed = test_fastspeech2_model()
    test2_passed = test_trainer_integration()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 すべてのテストが成功しました！")
        print("✅ FastSpeech 2 モデルは正常に動作しています")
        return 0
    else:
        print("❌ 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
