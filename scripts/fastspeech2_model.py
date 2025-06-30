#!/usr/bin/env python3
"""
FastSpeech 2 Model Implementation
This script implements the FastSpeech 2 model components with the specified configuration.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention layer for FastSpeech 2."""
    
    def __init__(self, attention_dim: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert attention_dim % num_heads == 0
        self.depth = attention_dim // num_heads
        
        self.wq = tf.keras.layers.Dense(attention_dim)
        self.wk = tf.keras.layers.Dense(attention_dim)
        self.wv = tf.keras.layers.Dense(attention_dim)
        self.dense = tf.keras.layers.Dense(attention_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, 
                                   mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, 
             training: Optional[bool] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.attention_dim))
        
        output = self.dense(concat_attention)
        return output


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """Position-wise Feed Forward Network."""
    
    def __init__(self, attention_dim: int, ffn_filter_size: int, 
                 ffn_kernel_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.ffn_filter_size = ffn_filter_size
        self.ffn_kernel_size = ffn_kernel_size
        self.dropout_rate = dropout_rate
        
        self.conv1 = tf.keras.layers.Conv1D(
            filters=ffn_filter_size,
            kernel_size=ffn_kernel_size,
            activation='relu',
            padding='same'
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(
            filters=attention_dim,
            kernel_size=ffn_kernel_size,
            padding='same'
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer Block for FastSpeech 2."""
    
    def __init__(self, attention_dim: int, num_heads: int, ffn_filter_size: int,
                 ffn_kernel_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.ffn_filter_size = ffn_filter_size
        self.ffn_kernel_size = ffn_kernel_size
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PositionwiseFeedForward(attention_dim, ffn_filter_size, 
                                         ffn_kernel_size, dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        # Multi-head attention
        attn_output = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class FastSpeechEncoder(tf.keras.layers.Layer):
    """FastSpeech 2 Encoder."""
    
    def __init__(self, vocab_size: int, attention_dim: int, num_layers: int,
                 num_heads: int, ffn_filter_size: int, ffn_kernel_size: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_filter_size = ffn_filter_size
        self.ffn_kernel_size = ffn_kernel_size
        self.dropout_rate = dropout_rate
        
        # Embedding layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, attention_dim)
        self.pos_encoding = self.positional_encoding(5000, attention_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(attention_dim, num_heads, ffn_filter_size,
                           ffn_kernel_size, dropout_rate)
            for _ in range(num_layers)
        ]
    
    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """Generate positional encoding."""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """Calculate angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        
        # Embedding + positional encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask, training=training)
        
        return x


class VariancePredictor(tf.keras.layers.Layer):
    """Variance Predictor for duration, pitch, and energy prediction."""
    
    def __init__(self, attention_dim: int, predictor_layers: int,
                 predictor_kernel_size: int, predictor_filter_size: int,
                 predictor_dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.predictor_layers = predictor_layers
        self.predictor_kernel_size = predictor_kernel_size
        self.predictor_filter_size = predictor_filter_size
        self.predictor_dropout_rate = predictor_dropout_rate
        
        self.conv_layers = []
        self.layer_norms = []
        self.dropouts = []
        
        for i in range(predictor_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    filters=predictor_filter_size,
                    kernel_size=predictor_kernel_size,
                    activation='relu',
                    padding='same'
                )
            )
            self.layer_norms.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.dropouts.append(tf.keras.layers.Dropout(predictor_dropout_rate))
        
        self.projection = tf.keras.layers.Dense(1)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = inputs
        
        for conv, norm, dropout in zip(self.conv_layers, self.layer_norms, self.dropouts):
            x = conv(x)
            x = norm(x)
            x = dropout(x, training=training)
        
        x = self.projection(x)
        return x


class VarianceAdaptor(tf.keras.layers.Layer):
    """Variance Adaptor containing duration, pitch, and energy predictors."""
    
    def __init__(self, attention_dim: int, predictor_layers: int,
                 predictor_kernel_size: int, predictor_filter_size: int,
                 predictor_dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        
        # Duration predictor
        self.duration_predictor = VariancePredictor(
            attention_dim, predictor_layers, predictor_kernel_size,
            predictor_filter_size, predictor_dropout_rate
        )
        
        # Pitch predictor
        self.pitch_predictor = VariancePredictor(
            attention_dim, predictor_layers, predictor_kernel_size,
            predictor_filter_size, predictor_dropout_rate
        )
        
        # Energy predictor
        self.energy_predictor = VariancePredictor(
            attention_dim, predictor_layers, predictor_kernel_size,
            predictor_filter_size, predictor_dropout_rate
        )
        
        # Pitch and energy embedding
        self.pitch_embedding = tf.keras.layers.Dense(attention_dim)
        self.energy_embedding = tf.keras.layers.Dense(attention_dim)
    
    def length_regulator(self, encoder_output: tf.Tensor, 
                        duration: tf.Tensor) -> tf.Tensor:
        """Length regulation using duration."""
        # Simplified length regulation - expand based on duration
        # In a real implementation, this would use actual duration values
        # to expand the sequence length properly
        return encoder_output
    
    def call(self, encoder_output: tf.Tensor, duration: Optional[tf.Tensor] = None,
             pitch: Optional[tf.Tensor] = None, energy: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        
        # Predict duration, pitch, and energy
        duration_pred = self.duration_predictor(encoder_output, training=training)
        pitch_pred = self.pitch_predictor(encoder_output, training=training)
        energy_pred = self.energy_predictor(encoder_output, training=training)
        
        # Use predicted values if ground truth is not provided
        if duration is None:
            duration = duration_pred
        if pitch is None:
            pitch = pitch_pred
        if energy is None:
            energy = energy_pred
        
        # Length regulation
        expanded_output = self.length_regulator(encoder_output, duration)
        
        # Add pitch and energy embeddings
        pitch_emb = self.pitch_embedding(pitch)
        energy_emb = self.energy_embedding(energy)
        
        variance_output = expanded_output + pitch_emb + energy_emb
        
        return variance_output, duration_pred, pitch_pred, energy_pred


class FastSpeechDecoder(tf.keras.layers.Layer):
    """FastSpeech 2 Decoder."""
    
    def __init__(self, attention_dim: int, num_layers: int, num_heads: int,
                 ffn_filter_size: int, ffn_kernel_size: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_filter_size = ffn_filter_size
        self.ffn_kernel_size = ffn_kernel_size
        self.dropout_rate = dropout_rate
        
        # Positional encoding
        self.pos_encoding = self.positional_encoding(5000, attention_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(attention_dim, num_heads, ffn_filter_size,
                           ffn_kernel_size, dropout_rate)
            for _ in range(num_layers)
        ]
    
    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """Generate positional encoding."""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """Calculate angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        
        # Add positional encoding
        x = inputs + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask, training=training)
        
        return x


class PostNet(tf.keras.layers.Layer):
    """Post-net for improving mel-spectrogram quality."""
    
    def __init__(self, num_mels: int, postnet_layers: int, postnet_kernel_size: int,
                 postnet_filters: int, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_mels = num_mels
        self.postnet_layers = postnet_layers
        self.postnet_kernel_size = postnet_kernel_size
        self.postnet_filters = postnet_filters
        self.dropout_rate = dropout_rate
        
        self.conv_layers = []
        self.batch_norms = []
        self.dropouts = []
        
        for i in range(postnet_layers):
            if i == 0:
                # First layer
                self.conv_layers.append(
                    tf.keras.layers.Conv1D(
                        filters=postnet_filters,
                        kernel_size=postnet_kernel_size,
                        activation='tanh',
                        padding='same'
                    )
                )
            elif i == postnet_layers - 1:
                # Last layer
                self.conv_layers.append(
                    tf.keras.layers.Conv1D(
                        filters=num_mels,
                        kernel_size=postnet_kernel_size,
                        activation=None,
                        padding='same'
                    )
                )
            else:
                # Middle layers
                self.conv_layers.append(
                    tf.keras.layers.Conv1D(
                        filters=postnet_filters,
                        kernel_size=postnet_kernel_size,
                        activation='tanh',
                        padding='same'
                    )
                )
            
            self.batch_norms.append(tf.keras.layers.BatchNormalization())
            self.dropouts.append(tf.keras.layers.Dropout(dropout_rate))
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = inputs
        
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            x = conv(x)
            if i < len(self.conv_layers) - 1:  # No batch norm on the last layer
                x = bn(x, training=training)
            x = dropout(x, training=training)
        
        return x


class FastSpeech2(tf.keras.Model):
    """FastSpeech 2 Model."""
    
    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)
        
        # Extract configuration parameters
        self.vocab_size = config['vocab_size']
        self.num_mels = config['num_mels']
        self.attention_dim = config['attention_dim']
        self.dropout_rate = config['dropout_rate']
        
        # Encoder parameters
        self.encoder_layers = config['encoder_layers']
        self.encoder_attention_heads = config['encoder_attention_heads']
        self.encoder_ffn_filter_size = config['encoder_ffn_filter_size']
        self.encoder_ffn_kernel_size = config['encoder_ffn_kernel_size']
        
        # Decoder parameters
        self.decoder_layers = config['decoder_layers']
        self.decoder_attention_heads = config['decoder_attention_heads']
        self.decoder_ffn_filter_size = config['decoder_ffn_filter_size']
        self.decoder_ffn_kernel_size = config['decoder_ffn_kernel_size']
        
        # Variance adaptor parameters
        self.predictor_layers = config['predictor_layers']
        self.predictor_kernel_size = config['predictor_kernel_size']
        self.predictor_filter_size = config['predictor_filter_size']
        self.predictor_dropout_rate = config['predictor_dropout_rate']
        
        # Post-net parameters
        self.postnet_layers = config['postnet_layers']
        self.postnet_kernel_size = config['postnet_kernel_size']
        self.postnet_filters = config['postnet_filters']
        
        # Build model components
        self.encoder = FastSpeechEncoder(
            self.vocab_size, self.attention_dim, self.encoder_layers,
            self.encoder_attention_heads, self.encoder_ffn_filter_size,
            self.encoder_ffn_kernel_size, self.dropout_rate
        )
        
        self.variance_adaptor = VarianceAdaptor(
            self.attention_dim, self.predictor_layers,
            self.predictor_kernel_size, self.predictor_filter_size,
            self.predictor_dropout_rate
        )
        
        self.decoder = FastSpeechDecoder(
            self.attention_dim, self.decoder_layers,
            self.decoder_attention_heads, self.decoder_ffn_filter_size,
            self.decoder_ffn_kernel_size, self.dropout_rate
        )
        
        self.mel_linear = tf.keras.layers.Dense(self.num_mels)
        
        self.postnet = PostNet(
            self.num_mels, self.postnet_layers,
            self.postnet_kernel_size, self.postnet_filters,
            self.dropout_rate
        )
    
    def call(self, inputs: tf.Tensor, duration: Optional[tf.Tensor] = None,
             pitch: Optional[tf.Tensor] = None, energy: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> dict:
        
        # Encoder
        encoder_output = self.encoder(inputs, training=training)
        
        # Variance Adaptor
        variance_output, duration_pred, pitch_pred, energy_pred = self.variance_adaptor(
            encoder_output, duration, pitch, energy, training=training
        )
        
        # Decoder
        decoder_output = self.decoder(variance_output, training=training)
        
        # Mel-spectrogram prediction
        mel_output = self.mel_linear(decoder_output)
        
        # Post-net
        mel_postnet = self.postnet(mel_output, training=training)
        mel_output_refined = mel_output + mel_postnet
        
        return {
            'mel_output': mel_output,
            'mel_output_refined': mel_output_refined,
            'duration_pred': duration_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred
        }


def create_fastspeech2_config() -> dict:
    """Create FastSpeech 2 configuration with specified parameters."""
    return {
        # Global configuration
        'vocab_size': 80,
        'num_mels': 80,
        'attention_dim': 256,
        'dropout_rate': 0.2,
        
        # Encoder
        'encoder_layers': 4,
        'encoder_attention_heads': 2,
        'encoder_ffn_filter_size': 1024,
        'encoder_ffn_kernel_size': 9,
        
        # Decoder
        'decoder_layers': 4,
        'decoder_attention_heads': 2,
        'decoder_ffn_filter_size': 1024,
        'decoder_ffn_kernel_size': 9,
        
        # Variance Adaptor
        'predictor_layers': 2,
        'predictor_kernel_size': 3,
        'predictor_filter_size': 256,
        'predictor_dropout_rate': 0.2,
        
        # Post-net
        'postnet_layers': 5,
        'postnet_kernel_size': 5,
        'postnet_filters': 512,
    }


def build_fastspeech2_model() -> FastSpeech2:
    """Build FastSpeech 2 model with the specified configuration."""
    config = create_fastspeech2_config()
    model = FastSpeech2(config)
    return model 
