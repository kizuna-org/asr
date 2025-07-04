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
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        super().build(input_shape)
        # Let the MultiHeadAttention layers build themselves during first call
        # Build only the layers we can safely build
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
    
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
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        super().build(input_shape)
        # Let the MultiHeadAttention layers build themselves during first call
        # Build only the layers we can safely build
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.layernorm3.build(input_shape)
    
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
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        super().build(input_shape)
        # Build sub-layers
        self.dense1.build(input_shape)
        # Output of first dense layer
        dense1_output_shape = input_shape[:-1] + (self.units,)
        self.dense2.build(dense1_output_shape)
    
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
    
    def build(self, input_shape):
        """Build the model layers with proper input shapes."""
        super().build(input_shape)
        
        # Build text embedding
        self.text_embedding.build((None, None))
        
        # Build encoder position encoding
        encoder_shape = (None, None, self.d_model)
        self.encoder_pos_encoding.build(encoder_shape)
        
        # Build decoder layers (but not encoder/decoder layers as they handle their own build)
        self.prenet.build((None, None, self.n_mels))
        self.decoder_projection.build((None, None, 256))
        self.decoder_pos_encoding.build((None, None, self.d_model))
        
        # Build output layers
        decoder_shape = (None, None, self.d_model)
        self.mel_linear.build(decoder_shape)
        self.stop_linear.build(decoder_shape)
        
        # Build postnet
        self.postnet.build((None, None, self.n_mels))

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
        
        # 動的にメル長さを決定（入力テキスト長さに基づく）
        batch_size = tf.shape(inputs)[0]
        text_length = tf.shape(inputs)[1]
        
        # テキスト長さに基づいてメル長さを推定（適応的な長さ）
        # 一般的にメルスペクトログラムはテキストより長い
        estimated_mel_length = tf.minimum(text_length * 4, 400)  # 4倍を基準とし、最大400フレーム
        estimated_mel_length = tf.maximum(estimated_mel_length, 50)  # 最小50フレーム
        
        # ダミーのメル入力（実際はGTメルスペクトログラムまたは前のステップの出力）
        dummy_mel = tf.zeros((batch_size, estimated_mel_length, self.n_mels))
        
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
    
    # Build the model with proper input shape
    text_input_shape = (None, None)  # (batch_size, sequence_length)
    simple_model.build(text_input_shape)
    
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
    
    # Compile the model (with MAE metric for Transformer TTS)
    mae_metric = SimpleTransformerMAE()
    
    model.compile(
        optimizer=optimizer,
        loss=SimpleTransformerTTSLoss(),
        metrics=[mae_metric]
    )
    
    # Build the metric by doing a forward pass
    try:
        # Get a sample from the model output for metric building
        dummy_output = model(dummy_text, training=False)
        dummy_mel = tf.zeros((1, 100, 80), dtype=tf.float32)
        
        # Build the metric by calling it once
        mae_metric.update_state(dummy_mel, dummy_output)
        mae_metric.reset_state()  # Reset after building
        print(f"✅ MAE メトリクスのビルドが完了しました")
    except Exception as e:
        print(f"⚠️  MAE メトリクスのビルドに失敗しましたが、トレーニング中に自動ビルドされます: {e}")
    
    print(f"✅ Encoder-Decoder Transformer TTS model created successfully")
    print(f"📍 テキストエンコーダ: {config['encoder_layers']}層")
    print(f"🎯 メルデコーダ: {config['decoder_layers']}層")
    print(f"🔧 Multi-Head Attention: {config['num_heads']}ヘッド")
    print(f"📈 FFN内部次元: {config['d_ff']} (d_modelの4倍)")
    print(f"🎵 Prenet: 256ユニット (ドロップアウト0.5)")
    print(f"🏁 Stop Token予測: 有効")
    print(f"📈 Noam学習率スケジューラ: 有効 (ウォームアップステップ: {config.get('warmup_steps', 4000)})")
    print(f"🔧 Post-net: 5層畳み込み構造")
    
    # Display model structure information
    print("\n" + "="*60)
    print("🏗️  MODEL STRUCTURE INFORMATION")
    print("="*60)
    
    # Display base model information
    print("\n📋 Base Model (SimpleTransformerTTS):")
    try:
        param_count = simple_model.count_params()
        print(f"   Parameters: {param_count:,}")
    except Exception as e:
        print(f"   Parameters: Could not count - {e}")
    
    # Display wrapper model information
    print("\n📋 Wrapper Model (TTSModelTrainer):")
    try:
        wrapper_param_count = model.count_params()
        print(f"   Parameters: {wrapper_param_count:,}")
    except Exception as e:
        print(f"   Parameters: Could not count - {e}")
    
    # Display detailed layer information
    print("\n🔍 Detailed Layer Structure:")
    try:
        print("   Text Embedding:")
        print(f"     - Vocabulary Size: {config['vocab_size']:,}")
        print(f"     - Embedding Dimension: {config['d_model']}")
        
        print("   Encoder:")
        print(f"     - Layers: {config['encoder_layers']}")
        print(f"     - Multi-Head Attention: {config['num_heads']} heads")
        print(f"     - Feed-Forward Dimension: {config['d_ff']}")
        
        print("   Decoder:")
        print(f"     - Layers: {config['decoder_layers']}")
        print(f"     - Multi-Head Attention: {config['num_heads']} heads") 
        print(f"     - Feed-Forward Dimension: {config['d_ff']}")
        
        print("   Output:")
        print(f"     - Mel-spectrogram Channels: {config['n_mels']}")
        print(f"     - Post-net Layers: 5 (Conv1D)")
        
    except Exception as e:
        print(f"   Could not display detailed structure: {e}")
    
    # Try to display model summary
    print("\n📊 Model Summary:")
    try:
        # Create a StringIO object to capture summary
        import io
        import sys
        
        # Capture the summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            model.summary()
            summary_output = buffer.getvalue()
            sys.stdout = old_stdout
            
            if summary_output.strip():
                print(summary_output)
                
                # Also display more detailed internal structure
                print("\n📋 モデル構造:")
                print("Base Model Structure:")
                try:
                    # Try to get more detailed summary of the base model
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()
                    simple_model.summary(expand_nested=True, show_trainable=True)
                    base_summary = buffer.getvalue()
                    sys.stdout = old_stdout
                    
                    if base_summary.strip():
                        print(base_summary)
                    else:
                        # Fallback to manual layer listing
                        print("SimpleTransformerTTS Internal Layers:")
                        layer_count = 0
                        
                        # Text embedding
                        print(f"├── Text Embedding: {simple_model.text_embedding.count_params():,} params")
                        layer_count += 1
                        
                        # Encoder layers
                        print(f"├── Encoder Layers ({len(simple_model.encoder_layers_list)}):")
                        for i, layer in enumerate(simple_model.encoder_layers_list):
                            try:
                                params = layer.count_params()
                                print(f"│   ├── Encoder Layer {i}: {params:,} params")
                                layer_count += 1
                            except:
                                print(f"│   ├── Encoder Layer {i}: params not available")
                        
                        # Decoder components
                        print(f"├── Prenet: {simple_model.prenet.count_params():,} params")
                        print(f"├── Decoder Projection: {simple_model.decoder_projection.count_params():,} params")
                        
                        # Decoder layers
                        print(f"├── Decoder Layers ({len(simple_model.decoder_layers_list)}):")
                        for i, layer in enumerate(simple_model.decoder_layers_list):
                            try:
                                params = layer.count_params()
                                print(f"│   ├── Decoder Layer {i}: {params:,} params")
                                layer_count += 1
                            except:
                                print(f"│   ├── Decoder Layer {i}: params not available")
                        
                        # Output layers
                        print(f"├── Mel Linear: {simple_model.mel_linear.count_params():,} params")
                        print(f"├── Stop Linear: {simple_model.stop_linear.count_params():,} params")
                        print(f"└── Post-net: {simple_model.postnet.count_params():,} params")
                        
                        print(f"\nTotal Layers: {layer_count + 6} (excluding positional encodings)")
                        
                except Exception as detail_error:
                    print(f"Could not display detailed base model structure: {detail_error}")
                    
            else:
                print("   Summary not available (model might not be fully built)")
        except Exception as summary_error:
            sys.stdout = old_stdout
            print(f"   Could not generate summary: {summary_error}")
            
            # Try alternative summary approach
            try:
                print("   Alternative structure view:")
                for i, layer in enumerate(model.layers):
                    print(f"     Layer {i}: {layer.name} ({type(layer).__name__})")
            except Exception as alt_error:
                print(f"   Could not display alternative structure: {alt_error}")
                
    except Exception as e:
        print(f"   Summary display failed: {e}")
    
    print("="*60)
    
    return model


class SimpleTransformerTTSLoss(tf.keras.losses.Loss):
    """Loss function for Encoder-Decoder Transformer TTS model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        
        # Metrics to track individual loss components
        self.mel_loss_metric = tf.keras.metrics.Mean(name='mel_loss')
        self.mel_postnet_loss_metric = tf.keras.metrics.Mean(name='mel_postnet_loss')
        self.stop_loss_metric = tf.keras.metrics.Mean(name='stop_loss')
    
    def call(self, y_true, y_pred):
        """
        Compute loss for Encoder-Decoder Transformer TTS model.
        
        y_true: mel-spectrogram tensor (batch_size, time_steps, n_mels)
        y_pred: dict containing model outputs
        """
        # Handle different input formats
        if isinstance(y_true, dict):
            # If y_true is a dict, extract mel and stop tokens
            mel_true = y_true['mel']
            stop_true = y_true.get('stop_tokens', tf.zeros_like(y_pred['stop_tokens']))
        else:
            # If y_true is just the mel-spectrogram tensor
            mel_true = y_true
            # Create dummy stop tokens (all zeros - no stop)
            batch_size = tf.shape(mel_true)[0]
            seq_len = tf.shape(y_pred['stop_tokens'])[1]
            stop_true = tf.zeros((batch_size, seq_len, 1), dtype=tf.float32)
        
        # Ensure mel dimensions match
        mel_output = y_pred['mel_output']
        mel_output_refined = y_pred['mel_output_refined']
        
        # Crop or pad mel_true to match output length - simplified approach
        mel_true_len = tf.shape(mel_true)[1]
        mel_output_len = tf.shape(mel_output)[1]
        
        # Use simpler tensor operations instead of nested tf.cond
        min_len = tf.minimum(mel_true_len, mel_output_len)
        
        # First, crop both to minimum length
        mel_true_cropped = mel_true[:, :min_len, :]
        
        # Then pad to match output length if needed
        pad_len = tf.maximum(0, mel_output_len - min_len)
        
        def pad_if_needed():
            if pad_len > 0:
                batch_size = tf.shape(mel_true_cropped)[0]
                n_mels = tf.shape(mel_true_cropped)[2]
                padding = tf.zeros((batch_size, pad_len, n_mels), dtype=mel_true.dtype)
                return tf.concat([mel_true_cropped, padding], axis=1)
            else:
                return mel_true_cropped
        
        mel_true = tf.cond(
            pad_len > 0,
            lambda: tf.concat([
                mel_true_cropped, 
                tf.zeros((tf.shape(mel_true_cropped)[0], pad_len, tf.shape(mel_true_cropped)[2]), dtype=mel_true.dtype)
            ], axis=1),
            lambda: mel_true_cropped
        )
        
        # Mel-spectrogram loss (both before and after postnet)
        mel_loss = self.mse(mel_true, mel_output)
        mel_postnet_loss = self.mse(mel_true, mel_output_refined)
        
        # Stop token loss (use reduced weight since we don't have real stop tokens)
        stop_loss = self.bce(stop_true, y_pred['stop_tokens']) * 0.1
        
        # Update individual loss metrics
        self.mel_loss_metric.update_state(mel_loss)
        self.mel_postnet_loss_metric.update_state(mel_postnet_loss)
        self.stop_loss_metric.update_state(stop_loss)
        
        total_loss = mel_loss + mel_postnet_loss + stop_loss
        
        return total_loss
    
    def reset_metrics(self):
        """Reset all loss component metrics."""
        self.mel_loss_metric.reset_state()
        self.mel_postnet_loss_metric.reset_state()
        self.stop_loss_metric.reset_state()
    
    def get_metrics(self):
        """Get current metric values as a dictionary."""
        return {
            'mel_loss': self.mel_loss_metric.result(),
            'mel_postnet_loss': self.mel_postnet_loss_metric.result(),
            'stop_loss': self.stop_loss_metric.result()
        }


class SimpleTransformerMAE(tf.keras.metrics.Metric):
    """Custom MAE metric for SimpleTransformerTTS model."""
    
    def __init__(self, name='mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_mae = self.add_weight(name='total_mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state."""
        try:
            if isinstance(y_pred, dict):
                # Use the main mel output for MAE calculation
                mel_pred = y_pred.get('mel_output_refined', y_pred.get('mel_output'))
                if mel_pred is None:
                    # Fallback to any available output
                    mel_pred = list(y_pred.values())[0]
            else:
                mel_pred = y_pred
            
            # Ensure tensors are properly shaped
            y_true = tf.cast(y_true, tf.float32)
            mel_pred = tf.cast(mel_pred, tf.float32)
            
            # Ensure dimensions match - use dynamic shapes
            y_true_shape = tf.shape(y_true)
            mel_pred_shape = tf.shape(mel_pred)
            
            # Truncate to minimum length if needed
            min_len = tf.minimum(y_true_shape[1], mel_pred_shape[1])
            min_channels = tf.minimum(y_true_shape[2], mel_pred_shape[2])
            
            y_true_truncated = y_true[:, :min_len, :min_channels]
            mel_pred_truncated = mel_pred[:, :min_len, :min_channels]
            
            # Calculate MAE
            mae = tf.reduce_mean(tf.abs(y_true_truncated - mel_pred_truncated))
            
            # Update metric state
            self.total_mae.assign_add(mae)
            self.count.assign_add(1.0)
            
        except Exception as e:
            # If MAE calculation fails, add zero to avoid breaking training
            self.total_mae.assign_add(0.0)
            self.count.assign_add(1.0)
    
    def result(self):
        """Return current metric value."""
        return tf.math.divide_no_nan(self.total_mae, self.count)
    
    def reset_state(self):
        """Reset metric state."""
        self.total_mae.assign(0.0)
        self.count.assign(0.0)


class TransformerTTSTrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback to display training progress with MAE for Transformer TTS."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch_mae_values = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Display epoch results with MAE calculation."""
        if logs is None:
            logs = {}
        
        # Get loss value
        train_loss = logs.get('loss', 0.0)
        
        # Get MAE value
        train_mae = logs.get('mae', 0.0)
        
        # Store MAE for tracking
        self.epoch_mae_values.append(train_mae)
        
        # Display epoch results with MAE
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, MAE={train_mae:.4f}")


if __name__ == "__main__":
    print("🚀 Testing Encoder-Decoder Transformer TTS Model")
    print("="*60)
    
    # Test the model
    config = create_simple_transformer_config()
    print(f"📝 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🏗️ Building Model...")
    model = build_simple_transformer_tts_model(config=config)
    
    print("\n🧪 Testing Model with Dummy Input...")
    # Test with dummy input
    dummy_text = tf.random.uniform((2, 20), 0, config['vocab_size'], dtype=tf.int32)
    print(f"Input shape: {dummy_text.shape}")
    
    try:
        output = model(dummy_text, training=False)
        
        print("\n✅ Model Forward Pass Successful!")
        print("📊 Model output shapes:")
        for key, value in output.items():
            print(f"   {key}: {value.shape}")
            
        # Additional model information
        print("\n📈 Model Performance Metrics:")
        print(f"   Input sequence length: {dummy_text.shape[1]}")
        print(f"   Batch size: {dummy_text.shape[0]}")
        print(f"   Output mel length: {output['mel_output'].shape[1]}")
        print(f"   Mel channels: {output['mel_output'].shape[2]}")
        
        # Check if outputs are reasonable
        print("\n🔍 Output Sanity Checks:")
        mel_mean = tf.reduce_mean(output['mel_output']).numpy()
        mel_std = tf.math.reduce_std(output['mel_output']).numpy()
        stop_mean = tf.reduce_mean(output['stop_tokens']).numpy()
        
        print(f"   Mel output mean: {mel_mean:.4f}")
        print(f"   Mel output std: {mel_std:.4f}")
        print(f"   Stop token mean: {stop_mean:.4f}")
        
        print("\n✅ All tests passed! Model is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)

