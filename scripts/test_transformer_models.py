#!/usr/bin/env python3
"""
Test script for Transformer-based TTS models.
This script tests the newly implemented Transformer TTS and VITS models.
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Ensure the script can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_transformer_tts import build_simple_transformer_tts_model, create_simple_transformer_config, SimpleTransformerTTSLoss
    from simple_vits import build_simple_vits_model, create_simple_vits_config, SimpleVITSLoss
    print("✅ Successfully imported transformer models")
except ImportError as e:
    print(f"❌ Failed to import transformer models: {e}")
    sys.exit(1)


def test_transformer_tts():
    """Test the Transformer TTS model."""
    print("\n" + "="*50)
    print("🧪 Testing Transformer TTS Model")
    print("="*50)
    
    try:
        # Create model configuration
        config = create_simple_transformer_config()
        print(f"📋 Configuration: {config}")
        
        # Build model
        model = build_simple_transformer_tts_model(config=config)
        print(f"✅ Model built successfully with {model.count_params():,} parameters")
        
        # Test with dummy data
        batch_size = 2
        seq_len = 20
        dummy_text = tf.random.uniform((batch_size, seq_len), 0, config['vocab_size'], dtype=tf.int32)
        
        print(f"🔄 Testing with input shape: {dummy_text.shape}")
        
        # Forward pass
        output = model(dummy_text, training=False)
        print(f"✅ Forward pass successful")
        
        # Check output shapes
        print("📊 Output shapes:")
        for key, value in output.items():
            print(f"  {key}: {value.shape}")
        
        # Test loss function
        target_mel = tf.random.normal((batch_size, seq_len, config['n_mels']))
        loss_fn = SimpleTransformerTTSLoss()
        loss = loss_fn(target_mel, output)
        print(f"✅ Loss computation successful: {loss.numpy():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer TTS test failed: {e}")
        return False


def test_vits():
    """Test the VITS model."""
    print("\n" + "="*50)
    print("🧪 Testing VITS Model")
    print("="*50)
    
    try:
        # Create model configuration
        config = create_simple_vits_config()
        print(f"📋 Configuration: {config}")
        
        # Build model
        model = build_simple_vits_model(config=config)
        print(f"✅ Model built successfully with {model.count_params():,} parameters")
        
        # Test with dummy data
        batch_size = 2
        text_seq_len = 20
        mel_seq_len = 100
        
        dummy_text = tf.random.uniform((batch_size, text_seq_len), 0, config['vocab_size'], dtype=tf.int32)
        dummy_mel = tf.random.normal((batch_size, mel_seq_len, config['n_mels']))
        
        print(f"🔄 Testing with input shapes:")
        print(f"  Text: {dummy_text.shape}")
        print(f"  Mel: {dummy_mel.shape}")
        
        # Forward pass (training mode)
        output_train = model(dummy_text, dummy_mel, training=True)
        print(f"✅ Training forward pass successful")
        
        # Forward pass (inference mode)
        output_inf = model(dummy_text, training=False)
        print(f"✅ Inference forward pass successful")
        
        # Check output shapes
        print("📊 Training output shapes:")
        for key, value in output_train.items():
            if isinstance(value, tf.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("📊 Inference output shapes:")
        for key, value in output_inf.items():
            if isinstance(value, tf.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Test loss function
        loss_fn = SimpleVITSLoss()
        loss = loss_fn(dummy_mel, output_train)
        print(f"✅ Loss computation successful: {loss.numpy():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ VITS test failed: {e}")
        return False


def test_integration():
    """Test integration with ljspeech_demo.py"""
    print("\n" + "="*50)
    print("🧪 Testing Integration with LJSpeech Demo")
    print("="*50)
    
    try:
        from ljspeech_demo import TTSModel, build_text_to_spectrogram_model, get_model_config
        
        # Test Transformer TTS integration
        print("🔄 Testing Transformer TTS integration...")
        config = get_model_config(TTSModel.TRANSFORMER_TTS)
        model = build_text_to_spectrogram_model(
            vocab_size=1000, 
            mel_bins=80, 
            max_sequence_length=430, 
            model_type=TTSModel.TRANSFORMER_TTS
        )
        print("✅ Transformer TTS integration successful")
        
        # Test VITS integration
        print("🔄 Testing VITS integration...")
        config = get_model_config(TTSModel.VITS)
        model = build_text_to_spectrogram_model(
            vocab_size=1000, 
            mel_bins=80, 
            max_sequence_length=430, 
            model_type=TTSModel.VITS
        )
        print("✅ VITS integration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Transformer-based TTS Models Test Suite")
    print("=" * 60)
    
    # Set memory growth to avoid GPU memory issues
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"💻 Found {len(physical_devices)} GPU(s)")
        else:
            print("💻 No GPU devices found, using CPU")
    except Exception as e:
        print(f"⚠️  Warning: Could not configure GPU memory growth: {e}")
    
    # Run tests
    tests = [
        ("Transformer TTS", test_transformer_tts),
        ("VITS", test_vits),
        ("Integration", test_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Transformer-based TTS models are ready to use.")
        print("\n💡 Usage examples:")
        print("  # Train with Transformer TTS")
        print("  python ljspeech_demo.py --mode mini --epochs 100 --model transformer_tts")
        print("\n  # Train with VITS")
        print("  python ljspeech_demo.py --mode mini --epochs 100 --model vits")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
