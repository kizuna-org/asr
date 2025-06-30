#!/usr/bin/env python3
"""
Simplified Transformer TTS Model Implementation
This script implements a working transformer-based TTS model with basic functionality.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class PositionalEncoding(tf.keras.layers.Layer):
    """位置エンコーディング層 - Transformerに系列の順序情報を提供"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 位置エンコーディングを事前計算
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.positional_encoding = tf.constant(pe, dtype=tf.float32)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.positional_encoding[:seq_len]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length
        })
        return config


class NoamLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Noam学習率スケジューラ - Attention Is All You Needで提案された手法"""
    
    def __init__(self, d_model: int, warmup_steps: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            'd_model': int(self.d_model),
            'warmup_steps': int(self.warmup_steps)
        }


class SimpleTransformerTTS(tf.keras.Model):
    """Simplified Transformer TTS model."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.n_mels = config['n_mels']
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Embedding layers
        self.text_embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        
        # 位置エンコーディング層を追加
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate
            )
            self.transformer_layers.append(layer)
        
        # Layer normalization
        self.layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(self.num_layers)]
        
        # Feed forward networks
        self.ffns = []
        for i in range(self.num_layers):
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
                tf.keras.layers.Dense(self.d_model),
                tf.keras.layers.Dropout(self.dropout_rate)
            ])
            self.ffns.append(ffn)
        
        # Output projection
        self.mel_linear = tf.keras.layers.Dense(self.n_mels)
        
        # 強化されたPostnet（Tacotron 2スタイルの5層構造）
        self.postnet = tf.keras.Sequential([
            # 1層目
            tf.keras.layers.Conv1D(512, 5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # 2層目
            tf.keras.layers.Conv1D(512, 5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # 3層目
            tf.keras.layers.Conv1D(512, 5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # 4層目
            tf.keras.layers.Conv1D(512, 5, padding='same', activation='tanh'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            
            # 5層目（出力層、活性化関数なし）
            tf.keras.layers.Conv1D(self.n_mels, 5, padding='same')
        ])
    
    def get_config(self):
        """Return the config of the model for serialization."""
        base_config = super().get_config()
        return {**base_config, 'config': self.config}
    
    @classmethod
    def from_config(cls, config):
        """Create a model from its config."""
        # Extract the model-specific config
        model_config = config.pop('config', {})
        return cls(model_config, **config)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, **kwargs) -> Dict[str, tf.Tensor]:
        # Text embedding
        x = self.text_embedding(inputs)
        
        # 位置エンコーディングを追加
        x = self.positional_encoding(x)
        
        # Transformer layers
        for i in range(self.num_layers):
            # Multi-head attention
            attn_output = self.transformer_layers[i](x, x, training=training)
            x = self.layer_norms[i](x + attn_output)
            
            # Feed forward
            ffn_output = self.ffns[i](x, training=training)
            x = self.layer_norms[i](x + ffn_output)
        
        # Mel-spectrogram prediction
        mel_output = self.mel_linear(x)
        
        # 強化されたPostnetで補正
        mel_postnet = self.postnet(mel_output, training=training)
        mel_output_refined = mel_output + mel_postnet
        
        return {
            'mel_output': mel_output,
            'mel_output_refined': mel_output_refined,
        }


def create_simple_transformer_config() -> Dict[str, Any]:
    """Create default configuration for Simple Transformer TTS model."""
    return {
        'vocab_size': 10000,
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'n_mels': 80,
        'dropout_rate': 0.15,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,  # Noamスケジューラ用のウォームアップステップ数
    }


def build_simple_transformer_tts_model(vocab_size: int = 10000, 
                                      n_mels: int = 80,
                                      config: Optional[Dict[str, Any]] = None) -> 'TTSModelTrainer':
    """Build and return a Simple Transformer TTS model wrapped in TTSModelTrainer."""
    # Import here to avoid circular imports
    from ljspeech_demo import TTSModelTrainer, TTSModel
    
    if config is None:
        config = create_simple_transformer_config()
    
    config['vocab_size'] = vocab_size
    config['n_mels'] = n_mels
    
    # Create the base model
    simple_model = SimpleTransformerTTS(config)
    
    # Build the model by calling it with dummy data
    dummy_text = tf.zeros((1, 10), dtype=tf.int32)
    _ = simple_model(dummy_text, training=False)
    
    # Wrap with trainer class
    model = TTSModelTrainer(simple_model, TTSModel.TRANSFORMER_TTS)
    
    # Build the wrapper model too
    _ = model(dummy_text, training=False)
    
    # Noam学習率スケジューラを使用したOptimizerを作成
    learning_rate_schedule = NoamLearningRateSchedule(
        d_model=config['d_model'],
        warmup_steps=config.get('warmup_steps', 4000)
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        metrics=['mae']
    )
    
    print(f"✅ Simple Transformer TTS model created successfully")
    print(f"📍 位置エンコーディング: 有効")
    print(f"📈 Noam学習率スケジューラ: 有効 (ウォームアップステップ: {config.get('warmup_steps', 4000)})")
    print(f"🔧 強化されたPost-net: 5層構造")
    
    # Try to get parameter count with error handling
    try:
        param_count = model.count_params()
        print(f"Model parameters: {param_count:,}")
    except ValueError as e:
        print(f"Model parameters: Could not count parameters - {e}")
    
    return model


class SimpleTransformerTTSLoss(tf.keras.losses.Loss):
    """Simple loss function for Transformer TTS model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        """
        Compute loss for Simple Transformer TTS model.
        
        y_true: mel-spectrogram (batch_size, time_steps, n_mels)
        y_pred: dict containing model outputs
        """
        # Mel-spectrogram loss (both before and after postnet)
        mel_loss = self.mse(y_true, y_pred['mel_output'])
        mel_postnet_loss = self.mse(y_true, y_pred['mel_output_refined'])
        total_loss = mel_loss + mel_postnet_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the model
    config = create_simple_transformer_config()
    model = build_simple_transformer_tts_model(config=config)
    
    # Test with dummy input
    dummy_text = tf.random.uniform((2, 20), 0, config['vocab_size'], dtype=tf.int32)
    output = model(dummy_text, training=False)
    
    print("Model output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}") 
