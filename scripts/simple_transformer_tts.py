#!/usr/bin/env python3
"""
Encoder-Decoder Transformer TTS Model Implementation
This script implements a Transformer-based TTS model with encoder-decoder architecture.
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


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformerエンコーダ層"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Multi-Head Self-Attention
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=None, mask=None):
        # Self-Attention
        attn_output = self.self_attention(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-Forward Network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Transformerデコーダ層"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Masked Multi-Head Self-Attention
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Multi-Head Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, encoder_output, training=None, look_ahead_mask=None, padding_mask=None):
        # Masked Self-Attention
        attn1 = self.self_attention(x, x, attention_mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Cross-Attention
        attn2 = self.cross_attention(out1, encoder_output, attention_mask=padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        # Feed-Forward Network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config


class Prenet(tf.keras.layers.Layer):
    """デコーダへの入力品質を高めるための小さな全結合ネットワーク"""
    
    def __init__(self, units: int = 256, dropout_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=None):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config


class SimpleTransformerTTS(tf.keras.Model):
    """Encoder-Decoder Transformer TTS model."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.num_heads = config['num_heads']
        self.d_ff = config['d_ff']
        self.n_mels = config['n_mels']
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # テキストエンコーダ (Text Encoder)
        self.text_embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.encoder_pos_encoding = PositionalEncoding(self.d_model)
        
        self.encoder_layers_list = [
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.encoder_layers)
        ]
        
        # メルデコーダ (Mel Decoder)
        self.prenet = Prenet(units=256, dropout_rate=0.5)
        self.decoder_projection = tf.keras.layers.Dense(self.d_model)
        self.decoder_pos_encoding = PositionalEncoding(self.d_model)
        
        self.decoder_layers_list = [
            TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.decoder_layers)
        ]
        
        # 最終出力部 (Final Output Stage)
        # メルスペクトログラム生成
        self.mel_linear = tf.keras.layers.Dense(self.n_mels)
        
        # Stop Token予測
        self.stop_linear = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Post-net（5層の畳み込みネットワーク）
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
    
    def create_look_ahead_mask(self, size):
        """未来のフレームをカンニングしないようにマスクを作成"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def encode(self, text_inputs, training=None):
        """テキストエンコーダ"""
        # テキスト埋め込み + 位置エンコーディング
        x = self.text_embedding(text_inputs)
        x = self.encoder_pos_encoding(x)
        
        # エンコーダ層を通す
        for encoder_layer in self.encoder_layers_list:
            x = encoder_layer(x, training=training)
        
        return x
    
    def decode_step(self, mel_inputs, encoder_output, training=None):
        """メルデコーダの1ステップ"""
        # Prenet
        x = self.prenet(mel_inputs, training=training)
        x = self.decoder_projection(x)
        x = self.decoder_pos_encoding(x)
        
        # Look-ahead mask作成
        seq_len = tf.shape(x)[1]
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        
        # デコーダ層を通す
        for decoder_layer in self.decoder_layers_list:
            x = decoder_layer(
                x, encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask
            )
        
        return x
    
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
        """
        Forward pass for training mode (teacher forcing)
        inputs: text inputs (batch_size, text_seq_len)
        """
        # For training, we need mel targets for teacher forcing
        # This is a simplified implementation that assumes mel targets are provided
        # In practice, you would need to modify this based on your training setup
        
        # テキストエンコーダ
        encoder_output = self.encode(inputs, training=training)
        
        # 簡略化のため、固定長のメル出力を生成
        # 実際の実装では、自己回帰的に生成するか、teacher forcingを使用
        batch_size = tf.shape(inputs)[0]
        max_mel_length = 100  # 適切な長さに調整
        
        # ダミーのメル入力（実際はGTメルスペクトログラムまたは前のステップの出力）
        dummy_mel = tf.zeros((batch_size, max_mel_length, self.n_mels))
        
        # デコーダ
        decoder_output = self.decode_step(dummy_mel, encoder_output, training=training)
        
        # 最終出力
        mel_output = self.mel_linear(decoder_output)
        stop_tokens = self.stop_linear(decoder_output)
        
        # Post-netで補正
        mel_postnet = self.postnet(mel_output, training=training)
        mel_output_refined = mel_output + mel_postnet
        
        return {
            'encoder_output': encoder_output,
            'decoder_output': decoder_output,
            'mel_output': mel_output,
            'mel_output_refined': mel_output_refined,
            'stop_tokens': stop_tokens,
        }


def create_simple_transformer_config() -> Dict[str, Any]:
    """Create default configuration for Encoder-Decoder Transformer TTS model."""
    return {
        'vocab_size': 10000,
        'd_model': 512,
        'encoder_layers': 6,
        'decoder_layers': 6,
        'num_heads': 8,
        'd_ff': 2048,  # d_modelの4倍
        'n_mels': 80,
        'dropout_rate': 0.1,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
    }


def build_simple_transformer_tts_model(vocab_size: int = 10000, 
                                      n_mels: int = 80,
                                      config: Optional[Dict[str, Any]] = None) -> 'TTSModelTrainer':
    """Build and return a Encoder-Decoder Transformer TTS model wrapped in TTSModelTrainer."""
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
    
    print(f"✅ Encoder-Decoder Transformer TTS model created successfully")
    print(f"📍 テキストエンコーダ: {config['encoder_layers']}層")
    print(f"🎯 メルデコーダ: {config['decoder_layers']}層")
    print(f"🔧 Multi-Head Attention: {config['num_heads']}ヘッド")
    print(f"📈 FFN内部次元: {config['d_ff']} (d_modelの4倍)")
    print(f"🎵 Prenet: 256ユニット (ドロップアウト0.5)")
    print(f"🏁 Stop Token予測: 有効")
    print(f"📈 Noam学習率スケジューラ: 有効 (ウォームアップステップ: {config.get('warmup_steps', 4000)})")
    print(f"🔧 Post-net: 5層畳み込み構造")
    
    # Try to get parameter count with error handling
    try:
        param_count = model.count_params()
        print(f"Model parameters: {param_count:,}")
    except ValueError as e:
        print(f"Model parameters: Could not count parameters - {e}")
    
    return model


class SimpleTransformerTTSLoss(tf.keras.losses.Loss):
    """Loss function for Encoder-Decoder Transformer TTS model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
    
    def call(self, y_true, y_pred):
        """
        Compute loss for Encoder-Decoder Transformer TTS model.
        
        y_true: dict containing mel-spectrogram and stop tokens
        y_pred: dict containing model outputs
        """
        # Mel-spectrogram loss (both before and after postnet)
        mel_loss = self.mse(y_true['mel'], y_pred['mel_output'])
        mel_postnet_loss = self.mse(y_true['mel'], y_pred['mel_output_refined'])
        
        # Stop token loss
        stop_loss = self.bce(y_true['stop_tokens'], y_pred['stop_tokens'])
        
        total_loss = mel_loss + mel_postnet_loss + stop_loss
        
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
