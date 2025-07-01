#!/usr/bin/env python3
"""
LJSpeech Dataset Learning Script (Speech Synthesis Version with Model Selection)
This script demonstrates how to load, work with, and train Text-to-Speech models on the LJSpeech dataset.
Supports multiple TTS model architectures including FastSpeech 2.
Enhanced with robust checkpoint and resume functionality.

Model Selection Features:
- FastSpeech 2: Advanced non-autoregressive TTS model with variance prediction
- Transformer TTS: Simple transformer-based TTS model for comparison
- Model-specific loss functions and training configurations
- Automatic checkpoint compatibility checking

Usage Examples:
  # Train with FastSpeech 2 (default)
  python ljspeech_demo.py --mode mini --epochs 100 --model fastspeech2
  
  # Train with simple Transformer TTS
  python ljspeech_demo.py --mode mini --epochs 100 --model transformer_tts
  
  # Full dataset training with FastSpeech 2
  python ljspeech_demo.py --mode full --epochs 2000 --model fastspeech2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
import json
import signal
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

# Set memory growth to avoid GPU memory issues
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s)")
    else:
        print("No GPU devices found, using CPU")
except Exception as e:
    print(f"Warning: Could not configure GPU memory growth: {e}")

# Model selection enumeration
class TTSModel(Enum):
    FASTSPEECH2 = "fastspeech2"
    TACOTRON2 = "tacotron2"  # Future implementation
    TRANSFORMER_TTS = "transformer_tts"  # Now implemented
    VITS = "vits"  # Now implemented

# Set the maximum number of frames for mel-spectrogram (for ~5 seconds audio)
MAX_FRAMES = 430  # (5 * 22050 / 256 ≈ 430)

# Base paths - will be updated based on model type
BASE_OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = None
MODEL_CHECKPOINT_PATH = None
TEXT_ENCODER_PATH = None
TRAINING_STATE_PATH = None
DATASET_CACHE_PATH = None

def setup_model_paths(model_type: TTSModel, limit_samples: int = None, mode: str = "mini"):
    """Setup model-specific paths based on the selected model type, sample count, and mode."""
    global CHECKPOINT_DIR, MODEL_CHECKPOINT_PATH, TEXT_ENCODER_PATH, TRAINING_STATE_PATH, DATASET_CACHE_PATH
    
    # Create sample-specific directory name
    if limit_samples is not None:
        sample_dir = f"samples_{limit_samples}"
    elif mode == 'full':
        sample_dir = "full_dataset"
    else:  # mini mode
        sample_dir = "samples_10"
    
    # Create hierarchical directory: outputs/model_type/sample_dir
    model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_type.value, sample_dir)
    CHECKPOINT_DIR = os.path.join(model_output_dir, "checkpoints")
    MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.keras")
    TEXT_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "text_encoder")
    TRAINING_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_state.json")
    DATASET_CACHE_PATH = os.path.join(CHECKPOINT_DIR, "dataset_processed.cache")
    
    # Create directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, "epoch_samples"), exist_ok=True)
    
    print(f"📁 モデル固有のディレクトリを設定: {model_output_dir}")
    print(f"📁 チェックポイントディレクトリ: {CHECKPOINT_DIR}")
    print(f"📁 サンプル設定: {sample_dir}")

    return model_output_dir

# Global variables for graceful shutdown
training_interrupted = False
current_model = None
current_text_encoder = None
current_epoch = 0

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C) gracefully."""
    global training_interrupted, current_model, current_text_encoder, current_epoch
    print(f"\n\n=== 中断シグナルを受信しました (Signal: {signum}) ===")
    print("トレーニングを安全に停止しています...")
    training_interrupted = True
    
    if current_model is not None and current_text_encoder is not None:
        print("現在の状態を保存中...")
        try:
            # Use default values for signal handler save
            save_training_state(current_epoch, current_text_encoder, current_model, TTSModel.FASTSPEECH2, 10, "mini")
            print("✅ チェックポイントが正常に保存されました")
        except Exception as e:
            print(f"❌ チェックポイント保存中にエラーが発生しました: {e}")
    
    print("プログラムを終了します...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


class FastSpeech2Loss(tf.keras.losses.Loss):
    """Custom loss function for FastSpeech 2 model."""
    
    def __init__(self, mel_loss_weight=1.0, duration_loss_weight=1.0, 
                 pitch_loss_weight=1.0, energy_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight
        
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
    
    def call(self, y_true, y_pred):
        """
        Compute loss for FastSpeech 2 model.
        
        y_true: mel-spectrogram (batch_size, time_steps, n_mels)
        y_pred: dict containing model outputs
        """
        # Mel-spectrogram loss (both before and after postnet)
        mel_loss = self.mse(y_true, y_pred['mel_output'])
        mel_postnet_loss = self.mse(y_true, y_pred['mel_output_refined'])
        total_mel_loss = mel_loss + mel_postnet_loss
        
        # For duration, pitch, and energy, we use simplified target (zeros for now)
        # In a real implementation, these would be extracted from the data
        batch_size = tf.shape(y_true)[0]
        seq_length = tf.shape(y_pred['duration_pred'])[1]
        
        # Dummy targets (in real implementation, these would be provided)
        duration_target = tf.ones((batch_size, seq_length, 1), dtype=tf.float32)
        pitch_target = tf.zeros((batch_size, seq_length, 1), dtype=tf.float32)
        energy_target = tf.zeros((batch_size, seq_length, 1), dtype=tf.float32)
        
        duration_loss = self.mae(duration_target, y_pred['duration_pred'])
        pitch_loss = self.mse(pitch_target, y_pred['pitch_pred'])
        energy_loss = self.mse(energy_target, y_pred['energy_pred'])
        
        total_loss = (self.mel_loss_weight * total_mel_loss +
                     self.duration_loss_weight * duration_loss +
                     self.pitch_loss_weight * pitch_loss +
                     self.energy_loss_weight * energy_loss)
        
        return total_loss


class BasicTTSLoss(tf.keras.losses.Loss):
    """Basic loss function for simple TTS models."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        """
        Compute basic MSE loss for TTS model.
        
        y_true: mel-spectrogram (batch_size, time_steps, n_mels)
        y_pred: predicted mel-spectrogram or dict containing model outputs
        """
        if isinstance(y_pred, dict):
            # If the model returns a dictionary, use the main output
            main_output_key = 'mel_output_refined' if 'mel_output_refined' in y_pred else 'mel_output'
            y_pred = y_pred.get(main_output_key, list(y_pred.values())[0])
        
        return self.mse(y_true, y_pred)

def load_ljspeech_dataset(
    split: str = "train", batch_size: int = 32, limit_samples: int = None
) -> tf.data.Dataset:
    """
    Load the LJSpeech dataset from TensorFlow Datasets.
    Optimized for efficient loading when limiting samples.
    """
    print(f"Loading LJSpeech dataset with split: {split}")
    
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        print(f"Error importing tensorflow_datasets: {e}")
        print("Please install tensorflow_datasets: pip install tensorflow-datasets")
        raise

    # Optimize split for limited samples to avoid processing entire dataset
    if limit_samples:
        print(f"🚀 効率的モード: 最初の{limit_samples}サンプルのみロード")
        optimized_split = f"{split}[:{limit_samples}]"
        print(f"📊 使用split: {optimized_split}")
    else:
        optimized_split = split

    dataset, info = tfds.load(
        "ljspeech",
        split=optimized_split,
        with_info=True,
        as_supervised=True,
        data_dir="./datasets",
    )

    print(f"Dataset info: {info}")
    if limit_samples:
        print(f"✅ 効率的ロード完了: {limit_samples}サンプル")
    else:
        print(f"Number of examples: {info.splits[split].num_examples}")

    return dataset.batch(batch_size)


def preprocess_audio(audio: tf.Tensor) -> tf.Tensor:
    """
    Preprocess audio data for machine learning.
    """
    audio = tf.cast(audio, tf.float32) / 32768.0
    if len(audio.shape) > 1 and audio.shape[-1] > 1:
        audio = tf.reduce_mean(audio, axis=-1)
    return audio


def extract_mel_spectrogram(
    audio: tf.Tensor,
    sample_rate: int = 22050,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tf.Tensor:
    """
    Extract mel-spectrogram features from audio.
    """
    audio_np = audio.numpy()
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T  # Transpose to [time, n_mels]
    return tf.convert_to_tensor(mel_spec_db, dtype=tf.float32)


def create_text_encoder(vocab_size: int = 10000) -> tf.keras.layers.TextVectorization:
    """
    Create a text encoder for processing transcriptions.
    """
    return tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=MAX_FRAMES,  # Use MAX_FRAMES for output sequence length
        standardize="lower_and_strip_punctuation",
    )


class TTSModelTrainer(tf.keras.Model):
    """Universal wrapper class for TTS model training with custom train_step."""
    
    def __init__(self, tts_model, model_type: TTSModel = TTSModel.FASTSPEECH2, **kwargs):
        super().__init__(**kwargs)
        self.tts_model = tts_model
        self.model_type = model_type
        
        # Select appropriate loss function based on model type
        if model_type == TTSModel.FASTSPEECH2:
            self.loss_fn = FastSpeech2Loss()
        elif model_type == TTSModel.TRANSFORMER_TTS:
            from simple_transformer_tts import SimpleTransformerTTSLoss
            self.loss_fn = SimpleTransformerTTSLoss()
        elif model_type == TTSModel.VITS:
            from simple_vits import SimpleVITSLoss
            self.loss_fn = SimpleVITSLoss()
        else:
            self.loss_fn = BasicTTSLoss()
    
    def call(self, inputs, training=None):
        return self.tts_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            model_outputs = self.tts_model(x, training=True)
            
            # Compute loss
            loss = self.loss_fn(y, model_outputs)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.tts_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.tts_model.trainable_variables))
        
        # For transformer_tts, skip metrics update due to shape mismatch
        # The loss function handles all necessary computations
        if self.model_type == TTSModel.TRANSFORMER_TTS:
            return {"loss": loss}
        
        # Get main output for metrics (for other models)
        if isinstance(model_outputs, dict):
            main_output = model_outputs.get('mel_output_refined', 
                                          model_outputs.get('mel_output', 
                                                          list(model_outputs.values())[0]))
        else:
            main_output = model_outputs
            
        # Update metrics (skip for transformer_tts to avoid shape issues)
        try:
            self.compiled_metrics.update_state(y, main_output)
            # Safely get metrics results
            metric_results = {}
            for m in self.metrics:
                try:
                    metric_results[m.name] = m.result()
                except:
                    # Skip metrics that can't be computed
                    pass
            return {"loss": loss, **metric_results}
        except Exception:
            # If metrics update fails, just return loss
            return {"loss": loss}


# Legacy alias for backward compatibility
FastSpeech2Trainer = TTSModelTrainer


def build_fastspeech2_model(
    vocab_size: int, mel_bins: int = 80, max_sequence_length: int = MAX_FRAMES
) -> tf.keras.Model:
    """
    Build FastSpeech 2 model for speech synthesis.
    """
    # Import FastSpeech 2 model
    from fastspeech2_model import FastSpeech2, create_fastspeech2_config
    
    # Create configuration for FastSpeech 2
    config = create_fastspeech2_config()
    config['vocab_size'] = vocab_size
    config['num_mels'] = mel_bins
    
    # Create FastSpeech 2 model
    fastspeech2_model = FastSpeech2(config)
    
    # Build the model by calling it with dummy input
    dummy_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
    _ = fastspeech2_model(dummy_input)
    
    # Wrap with trainer class
    model = TTSModelTrainer(fastspeech2_model, TTSModel.FASTSPEECH2)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        metrics=['mae']
    )
    
    return model


def build_simple_transformer_model(
    vocab_size: int, mel_bins: int = 80, max_sequence_length: int = MAX_FRAMES
) -> tf.keras.Model:
    """
    Build a simple Transformer-based TTS model for comparison.
    This is a basic implementation for demonstration purposes.
    """
    # Input layer
    text_input = tf.keras.layers.Input(shape=(max_sequence_length,), name='text_input')
    
    # Embedding layer
    embedding = tf.keras.layers.Embedding(vocab_size, 256)(text_input)
    
    # Positional encoding (simplified)
    pos_encoding = tf.keras.layers.Dense(256, activation='linear')(embedding)
    x = embedding + pos_encoding
    
    # Transformer blocks (simplified)
    for _ in range(4):
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=0.1
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attention)
        
        # Feed forward
        ffn = tf.keras.layers.Dense(1024, activation='relu')(x)
        ffn = tf.keras.layers.Dense(256)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
    
    # Output projection to mel-spectrogram
    mel_output = tf.keras.layers.Dense(mel_bins, activation='linear', name='mel_output')(x)
    
    # Create model
    simple_model = tf.keras.Model(inputs=text_input, outputs=mel_output)
    
    # Wrap with trainer class
    model = TTSModelTrainer(simple_model, TTSModel.TRANSFORMER_TTS)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        metrics=['mae']
    )
    
    return model


def build_text_to_spectrogram_model(
    vocab_size: int, 
    mel_bins: int = 80, 
    max_sequence_length: int = MAX_FRAMES,
    model_type: TTSModel = TTSModel.FASTSPEECH2
) -> tf.keras.Model:
    """
    Build TTS model based on the specified model type.
    
    Args:
        vocab_size: Size of the vocabulary
        mel_bins: Number of mel-spectrogram bins
        max_sequence_length: Maximum sequence length
        model_type: Type of TTS model to build
        
    Returns:
        Compiled TTS model ready for training
    """
    print(f"🏗️  構築中のモデル: {model_type.value}")
    
    if model_type == TTSModel.FASTSPEECH2:
        return build_fastspeech2_model(vocab_size, mel_bins, max_sequence_length)
    elif model_type == TTSModel.TRANSFORMER_TTS:
        # Import the simple transformer TTS model
        from simple_transformer_tts import build_simple_transformer_tts_model, create_simple_transformer_config
        config = create_simple_transformer_config()
        config['vocab_size'] = vocab_size
        config['n_mels'] = mel_bins
        return build_simple_transformer_tts_model(vocab_size, mel_bins, config)
    elif model_type == TTSModel.VITS:
        # Import the Simple VITS model
        from simple_vits import build_simple_vits_model, create_simple_vits_config
        config = create_simple_vits_config()
        config['vocab_size'] = vocab_size
        config['n_mels'] = mel_bins
        return build_simple_vits_model(vocab_size, mel_bins, config)
    elif model_type == TTSModel.TACOTRON2:
        # Placeholder for future Tacotron 2 implementation
        raise NotImplementedError("Tacotron 2 モデルは将来の実装予定です")
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")


def get_model_config(model_type: TTSModel) -> Dict[str, Any]:
    """Get model-specific configuration."""
    if model_type == TTSModel.FASTSPEECH2:
        from fastspeech2_model import create_fastspeech2_config
        return create_fastspeech2_config()
    elif model_type == TTSModel.TRANSFORMER_TTS:
        from simple_transformer_tts import create_simple_transformer_config
        return create_simple_transformer_config()
    elif model_type == TTSModel.VITS:
        from simple_vits import create_simple_vits_config
        return create_simple_vits_config()
    else:
        return {}


def visualize_audio_and_spectrogram(
    audio: tf.Tensor, text: str, sample_rate: int = 22050, save_path: str | None = None
):
    """
    Visualize audio waveform and mel-spectrogram.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    audio_np = audio.numpy()
    time = np.arange(len(audio_np)) / sample_rate
    ax1.plot(time, audio_np)
    ax1.set_title(f'Waveform - Text: "{text[:50]}..."')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    mel_spec_db = extract_mel_spectrogram(audio)
    mel_spec_db_np = mel_spec_db.numpy()
    librosa.display.specshow(
        mel_spec_db_np.T,  # Transpose back for display
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        ax=ax2,
    )
    ax2.set_title("Mel-Spectrogram")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()  # プロットを表示せずに閉じる


class TrainingPlotCallback(tf.keras.callbacks.Callback):
    """
    学習過程（epoch、loss、MAE）をリアルタイムでグラフ化するコールバック
    """
    
    def __init__(self, model_output_dir=None, model_type=TTSModel.FASTSPEECH2):
        super().__init__()
        self.model_type = model_type
        
        # グラフ保存ディレクトリの設定
        if model_output_dir:
            self.plot_dir = os.path.join(model_output_dir, "training_plots")
        else:
            self.plot_dir = os.path.join(BASE_OUTPUT_DIR, model_type.value, "training_plots")
        
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 学習履歴を保存するリスト
        self.epochs = []
        self.train_losses = []
        self.train_maes = []
        self.val_losses = []
        self.val_maes = []
        
        # グラフのスタイル設定
        plt.style.use('default')
        
        print(f"📊 学習過程グラフ化機能を初期化しました")
        print(f"📁 グラフ保存ディレクトリ: {self.plot_dir}")
    
    def on_epoch_end(self, epoch, logs=None):
        """エポック終了時に学習曲線を更新"""
        if logs is None:
            logs = {}
        
        # エポック番号を記録（1から始まる）
        current_epoch = epoch + 1
        self.epochs.append(current_epoch)
        
        # 損失とMAEを記録
        train_loss = logs.get('loss', 0)
        train_mae = logs.get('mae', logs.get('mean_absolute_error', 0))
        val_loss = logs.get('val_loss', None)
        val_mae = logs.get('val_mae', logs.get('val_mean_absolute_error', None))
        
        self.train_losses.append(train_loss)
        self.train_maes.append(train_mae)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_mae is not None:
            self.val_maes.append(val_mae)
        
        # グラフを生成・保存
        self._create_training_plots(current_epoch)
        
        print(f"📊 エポック {current_epoch}: Loss={train_loss:.4f}, MAE={train_mae:.4f}")
    
    def _create_training_plots(self, current_epoch):
        """学習曲線グラフを作成・保存"""
        try:
            # フィギュアサイズを設定
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.model_type.value.upper()} Training Progress - Epoch {current_epoch}', 
                        fontsize=16, fontweight='bold')
            
            # 1. Loss曲線
            ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if self.val_losses:
                ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            if len(self.epochs) > 1:
                ax1.set_xlim(1, max(self.epochs))
            
            # 2. MAE曲線
            ax2.plot(self.epochs, self.train_maes, 'g-', label='Training MAE', linewidth=2)
            if self.val_maes:
                ax2.plot(self.epochs, self.val_maes, 'orange', label='Validation MAE', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_title('Training and Validation MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(self.epochs) > 1:
                ax2.set_xlim(1, max(self.epochs))
            
            # 3. Loss最新20エポックの詳細
            recent_epochs = self.epochs[-20:] if len(self.epochs) > 20 else self.epochs
            recent_train_losses = self.train_losses[-20:] if len(self.train_losses) > 20 else self.train_losses
            recent_val_losses = self.val_losses[-20:] if len(self.val_losses) > 20 else self.val_losses
            
            ax3.plot(recent_epochs, recent_train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
            if recent_val_losses:
                ax3.plot(recent_epochs, recent_val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Recent Loss (Last 20 Epochs)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 学習統計情報
            ax4.axis('off')
            stats_text = f"""Training Statistics (Epoch {current_epoch})
             
Current Metrics:
• Training Loss: {self.train_losses[-1]:.6f}
• Training MAE: {self.train_maes[-1]:.6f}"""
            
            if self.val_losses:
                stats_text += f"\n• Validation Loss: {self.val_losses[-1]:.6f}"
            if self.val_maes:
                stats_text += f"\n• Validation MAE: {self.val_maes[-1]:.6f}"
            
            if len(self.train_losses) > 1:
                best_train_loss = min(self.train_losses)
                best_train_loss_epoch = self.epochs[self.train_losses.index(best_train_loss)]
                stats_text += f"""

Best Performance:
• Best Training Loss: {best_train_loss:.6f} (Epoch {best_train_loss_epoch})"""
                
                if len(self.train_maes) > 1:
                    best_train_mae = min(self.train_maes)
                    best_train_mae_epoch = self.epochs[self.train_maes.index(best_train_mae)]
                    stats_text += f"\n• Best Training MAE: {best_train_mae:.6f} (Epoch {best_train_mae_epoch})"
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # レイアウト調整
            plt.tight_layout()
            
            # グラフを保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"training_progress_epoch_{current_epoch:04d}_{timestamp}.png"
            plot_path = os.path.join(self.plot_dir, plot_filename)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # メモリリークを防ぐため
            
            # 最新のグラフを固定名でも保存（常に最新状態を確認できるように）
            latest_plot_path = os.path.join(self.plot_dir, "latest_training_progress.png")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.model_type.value.upper()} Training Progress - Epoch {current_epoch}', 
                        fontsize=16, fontweight='bold')
            
            # 同じグラフを再作成
            ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if self.val_losses:
                ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            if len(self.epochs) > 1:
                ax1.set_xlim(1, max(self.epochs))
            
            ax2.plot(self.epochs, self.train_maes, 'g-', label='Training MAE', linewidth=2)
            if self.val_maes:
                ax2.plot(self.epochs, self.val_maes, 'orange', label='Validation MAE', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_title('Training and Validation MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(self.epochs) > 1:
                ax2.set_xlim(1, max(self.epochs))
            
            ax3.plot(recent_epochs, recent_train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
            if recent_val_losses:
                ax3.plot(recent_epochs, recent_val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Recent Loss (Last 20 Epochs)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.axis('off')
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 学習曲線グラフを保存: {plot_filename}")
            
        except Exception as e:
            print(f"❌ グラフ作成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    def save_training_history(self):
        """学習履歴をJSONファイルとして保存"""
        try:
            history_data = {
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'train_maes': self.train_maes,
                'val_losses': self.val_losses,
                'val_maes': self.val_maes,
                'model_type': self.model_type.value,
                'saved_at': datetime.now().isoformat()
            }
            
            history_path = os.path.join(self.plot_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 学習履歴を保存: {history_path}")
            
        except Exception as e:
            print(f"❌ 学習履歴保存中にエラーが発生しました: {e}")

class SynthesisCallback(tf.keras.callbacks.Callback):
    def __init__(self, text_encoder, n_fft, hop_length, sample_rate=22050, inference_text=None, model_type=TTSModel.FASTSPEECH2, model_output_dir=None):
        super().__init__()
        self.text_encoder = text_encoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.model_type = model_type
        # 音声合成に使用するテスト用のテキスト（最初のデータセットのテキストを使用）
        self.inference_text = inference_text if inference_text else "This is a test of the model at the end of each epoch."
        
        # モデル固有のディレクトリを設定（model_output_dirが提供されている場合はそれを使用）
        if model_output_dir:
            self.epoch_samples_dir = os.path.join(model_output_dir, "epoch_samples")
        else:
            # 旧方式の互換性維持
            self.epoch_samples_dir = os.path.join(BASE_OUTPUT_DIR, model_type.value, "epoch_samples")
        
        os.makedirs(self.epoch_samples_dir, exist_ok=True)
        
        # 時間予測用の変数
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []  # 各エポックの実行時間を記録
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        """トレーニング開始時に開始時刻を記録し、総エポック数を設定"""
        self.training_start_time = time.time()
        # パラメータから総エポック数を取得
        self.total_epochs = self.params.get('epochs', 500)
        start_time_str = datetime.fromtimestamp(self.training_start_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n🚀 学習開始時刻: {start_time_str}")
        print(f"📊 総エポック数: {self.total_epochs}")

    def on_epoch_begin(self, epoch, logs=None):
        """Update global variables at the start of each epoch."""
        global current_epoch, current_model, current_text_encoder
        current_epoch = epoch
        current_model = self.model
        current_text_encoder = self.text_encoder
        
        # エポック開始時刻を記録
        self.epoch_start_time = time.time()
        epoch_start_str = datetime.fromtimestamp(self.epoch_start_time).strftime('%H:%M:%S')
        print(f"\n🚀 エポック {epoch + 1}/{self.total_epochs} を開始しています... (開始時刻: {epoch_start_str})")

    def on_epoch_end(self, epoch, logs=None):
        """Generate audio sample, save state, and predict completion time at the end of each epoch."""
        global training_interrupted
        
        # エポック終了時刻を記録
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        # 時間予測の計算
        completed_epochs = epoch + 1
        remaining_epochs = self.total_epochs - completed_epochs
        
        # 平均エポック時間を計算
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        # 残り時間を計算
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        estimated_completion_time = epoch_end_time + estimated_remaining_time
        
        # 時間情報を表示
        epoch_duration_min = epoch_duration / 60
        avg_epoch_time_min = avg_epoch_time / 60
        remaining_time_min = estimated_remaining_time / 60
        
        completion_time_str = datetime.fromtimestamp(estimated_completion_time).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n⏱️  === エポック {epoch + 1}/{self.total_epochs} 時間情報 ===")
        print(f"📈 今回のエポック実行時間: {epoch_duration_min:.1f}分")
        print(f"📊 平均エポック実行時間: {avg_epoch_time_min:.1f}分")
        
        if remaining_epochs > 0:
            print(f"⏳ 残りエポック数: {remaining_epochs}")
            print(f"🕐 推定残り時間: {remaining_time_min:.1f}分")
            print(f"🏁 推定完了時刻: {completion_time_str}")
        else:
            total_training_time = (epoch_end_time - self.training_start_time) / 60
            print(f"🎉 全学習完了！総学習時間: {total_training_time:.1f}分")
        
        print("=" * 50)
        
        if training_interrupted:
            print("\n⚠️  中断が要求されました。音声生成をスキップします。")
            return
            
        print(f"\n\n--- エポック {epoch + 1} 終了時の音声サンプル生成 ---")

        try:
            # テキストをベクトル化
            text_vec = self.text_encoder([self.inference_text])

            # モデルでメルスペクトログラムを予測
            model_output = self.model.predict(text_vec)
            
            # モデルタイプに応じて適切な出力を取得
            if isinstance(model_output, dict):
                if self.model_type in [TTSModel.FASTSPEECH2, TTSModel.TRANSFORMER_TTS, TTSModel.VITS]:
                    predicted_mel_spec = model_output['mel_output_refined']
                else:
                    # 他のモデルタイプの場合、利用可能な出力キーから選択
                    predicted_mel_spec = model_output.get('mel_output_refined', 
                                                        model_output.get('mel_output',
                                                                        list(model_output.values())[0]))
            else:
                predicted_mel_spec = model_output

            # バッチ次元を削除し、numpy配列に変換
            predicted_mel_spec_np = predicted_mel_spec[0]

            # スペクトログラムをデシベルからパワーに変換
            predicted_mel_spec_db_t = predicted_mel_spec_np.T
            power_spec = librosa.db_to_power(predicted_mel_spec_db_t)

            # Griffin-Limアルゴリズムで音声を復元
            generated_audio = librosa.feature.inverse.mel_to_audio(
                power_spec,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # 生成された音声を保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_audio_path = f"{self.epoch_samples_dir}/epoch_{epoch + 1}_{timestamp}.wav"
            sf.write(output_audio_path, generated_audio, self.sample_rate)
            print(f"🎵 エポック {epoch + 1} の音声サンプルを保存: {output_audio_path}")

        except Exception as e:
            print(f"❌ エポック {epoch + 1} の音声生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    def on_train_end(self, logs=None):
        """トレーニング終了時の最終情報を表示"""
        if self.training_start_time is not None:
            total_training_time = (time.time() - self.training_start_time) / 60
            end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n🎉 === 学習完了情報 ===")
            print(f"🏁 学習終了時刻: {end_time_str}")
            print(f"⏱️  総学習時間: {total_training_time:.1f}分")
            print(f"📊 実行されたエポック数: {len(self.epoch_times)}")
            if self.epoch_times:
                avg_time = sum(self.epoch_times) / len(self.epoch_times) / 60
                print(f"📈 平均エポック時間: {avg_time:.1f}分")
            print("=" * 50)


def save_training_state(epoch, text_encoder, model, model_type=TTSModel.FASTSPEECH2, limit_samples=None, mode="mini"):
    """Save training state including epoch number, text encoder, model type, and training configuration."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        print(f"💾 エポック {epoch} の状態を保存中...")
        
        # Save model weights instead of full model to avoid serialization issues
        model_weights_path = os.path.join(CHECKPOINT_DIR, "model.weights.h5")
        model.save_weights(model_weights_path)
        print(f"  ✅ モデル重みを保存: {model_weights_path}")

        # Save text encoder vocabulary
        vocab = text_encoder.get_vocabulary()
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        with open(vocab_path, "w", encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 語彙を保存: {vocab_path}")

        # Determine sample directory name
        if limit_samples is not None:
            sample_dir = f"samples_{limit_samples}"
        elif mode == 'full':
            sample_dir = "full_dataset"
        else:  # mini mode
            sample_dir = "samples_10"

        # Save training state with timestamp, model type, and training configuration
        training_state = {
            "epoch": epoch,
            "model_type": model_type.value,
            "vocab_size": text_encoder.vocabulary_size(),
            "max_tokens": text_encoder._max_tokens,
            "output_sequence_length": text_encoder._output_sequence_length,
            "standardize": text_encoder._standardize,
            "limit_samples": limit_samples,
            "mode": mode,
            "sample_dir": sample_dir,
            "saved_at": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "checkpoint_version": "3.2"
        }
        
        # Create backup of previous state
        if os.path.exists(TRAINING_STATE_PATH):
            backup_path = TRAINING_STATE_PATH + ".backup"
            os.rename(TRAINING_STATE_PATH, backup_path)
            print(f"  📋 前回の状態をバックアップ: {backup_path}")
        
        with open(TRAINING_STATE_PATH, "w", encoding='utf-8') as f:
            json.dump(training_state, f, ensure_ascii=False, indent=2)
        print(f"  ✅ トレーニング状態を保存: {TRAINING_STATE_PATH}")

        print(f"💾 エポック {epoch} の保存完了")
        
    except Exception as e:
        print(f"❌ 状態保存中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_training_state():
    """Load training state and return epoch number, text encoder, model, model type, and training config if available."""
    print("🔍 保存された状態を確認中...")
    
    if not os.path.exists(TRAINING_STATE_PATH):
        print("ℹ️  保存された状態が見つかりません。最初から開始します。")
        return 0, None, None, TTSModel.FASTSPEECH2, None

    try:
        # Load training state
        print(f"📂 トレーニング状態を読み込み中: {TRAINING_STATE_PATH}")
        with open(TRAINING_STATE_PATH, "r", encoding='utf-8') as f:
            training_state = json.load(f)
        
        # Validate checkpoint version compatibility
        checkpoint_version = training_state.get("checkpoint_version", "1.0")
        print(f"  📋 チェックポイントバージョン: {checkpoint_version}")
        
        # Get model type from saved state (default to FastSpeech2 for older checkpoints)
        model_type_str = training_state.get("model_type", "fastspeech2")
        try:
            model_type = TTSModel(model_type_str)
            print(f"  🏗️  保存されたモデルタイプ: {model_type.value}")
        except ValueError:
            print(f"  ⚠️  不明なモデルタイプ '{model_type_str}'、デフォルトのFastSpeech2を使用")
            model_type = TTSModel.FASTSPEECH2
        
        # Extract training configuration
        training_config = {
            'limit_samples': training_state.get('limit_samples'),
            'mode': training_state.get('mode', 'mini'),
            'sample_dir': training_state.get('sample_dir', 'samples_10')
        }
        
        if training_state.get("tensorflow_version"):
            print(f"  🔧 保存時のTensorFlowバージョン: {training_state['tensorflow_version']}")
            print(f"  🔧 現在のTensorFlowバージョン: {tf.__version__}")
        
        saved_at = training_state.get("saved_at", "不明")
        print(f"  ⏰ 保存時刻: {saved_at}")

        # Check if model weights file exists
        model_weights_path = os.path.join(CHECKPOINT_DIR, "model.weights.h5")
        if not os.path.exists(model_weights_path):
            print(f"❌ モデル重みファイルが見つかりません: {model_weights_path}")
            return 0, None, None, model_type, training_config

        # Load vocabulary
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        if not os.path.exists(vocab_path):
            print(f"❌ 語彙ファイルが見つかりません: {vocab_path}")
            return 0, None, None, model_type, training_config
            
        print(f"📖 語彙を読み込み中: {vocab_path}")
        with open(vocab_path, "r", encoding='utf-8') as f:
            vocab = json.load(f)

        # Recreate text encoder
        print("🔤 テキストエンコーダーを再構築中...")
        text_encoder = tf.keras.layers.TextVectorization(
            max_tokens=training_state["max_tokens"],
            output_sequence_length=training_state["output_sequence_length"],
            standardize=training_state["standardize"],
        )
        text_encoder.set_vocabulary(vocab)
        print(f"  ✅ 語彙サイズ: {len(vocab)}")

        # Create new model with same architecture
        print(f"🤖 モデルを再構築中...")
        model = build_text_to_spectrogram_model(
            vocab_size=training_state["vocab_size"],
            mel_bins=80,
            max_sequence_length=training_state["output_sequence_length"],
            model_type=model_type
        )
        
        # Load weights
        print(f"⚖️  モデル重みを読み込み中: {model_weights_path}")
        model.load_weights(model_weights_path)
        print(f"  ✅ モデル重み読み込み完了")

        epoch = training_state["epoch"]
        print(f"🎯 エポック {epoch} から再開します")
        print("=" * 50)
        
        return epoch, text_encoder, model, model_type, training_config

    except Exception as e:
        print(f"❌ チェックポイント読み込み中にエラーが発生しました: {e}")
        print("⚠️  最初から開始します...")
        import traceback
        traceback.print_exc()
        
        # Try to use backup if available
        backup_path = TRAINING_STATE_PATH + ".backup"
        if os.path.exists(backup_path):
            print(f"🔄 バックアップから復元を試行中: {backup_path}")
            try:
                os.rename(backup_path, TRAINING_STATE_PATH)
                return load_training_state()  # Recursive call with backup
            except Exception as backup_error:
                print(f"❌ バックアップからの復元も失敗しました: {backup_error}")
        
        return 0, None, None, TTSModel.FASTSPEECH2, None


def main():
    """Main function to run the LJSpeech learning script."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LJSpeech TTS Training Script with Model Selection')
    parser.add_argument('--mode', choices=['full', 'mini'], default='mini',
                       help='Training mode: full (entire dataset) or mini (first 10 samples)')
    parser.add_argument('--limit-samples', type=int, default=None,
                       help='Limit number of samples for training (overrides mode)')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of epochs to train (default: 2000)')
    parser.add_argument('--model', choices=['fastspeech2', 'transformer_tts'], default='fastspeech2',
                       help='TTS model type to use (default: fastspeech2)')
    args = parser.parse_args()
    
    # Convert model argument to enum
    model_type = TTSModel(args.model)
    
    # Set training parameters based on mode
    if args.limit_samples is not None:
        limit_samples = args.limit_samples
        mode_desc = f"カスタム({limit_samples}サンプル)"
    elif args.mode == 'full':
        limit_samples = None
        mode_desc = "フルデータセット"
    else:  # mini mode
        limit_samples = 10
        mode_desc = "ミニモード(10サンプル)"
    
    print("=== LJSpeech TTS 音声合成スクリプト ===")
    print(f"🏗️  使用モデル: {model_type.value}")
    print(f"📊 学習モード: {mode_desc}")
    print(f"🔄 エポック数: {args.epochs}")
    print(f"🕐 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Setup model-specific paths
        model_output_dir = setup_model_paths(model_type, limit_samples, args.mode)
        
        # Check for existing checkpoint
        start_epoch, text_encoder, model, saved_model_type, training_config = load_training_state()

        # Check if the saved model type matches the requested model type and training configuration
        if start_epoch > 0 and training_config:
            config_matches = True
            if saved_model_type != model_type:
                config_matches = False
                print(f"⚠️  保存されたモデルタイプ ({saved_model_type.value}) と要求されたモデルタイプ ({model_type.value}) が異なります")
            
            # Check if training configuration matches
            saved_limit_samples = training_config.get('limit_samples')
            saved_mode = training_config.get('mode', 'mini')
            
            if saved_limit_samples != limit_samples or saved_mode != args.mode:
                config_matches = False
                print(f"⚠️  保存された設定 (samples: {saved_limit_samples}, mode: {saved_mode}) と要求された設定 (samples: {limit_samples}, mode: {args.mode}) が異なります")
            
            if config_matches:
                print(f"🔄 チェックポイントが見つかりました。エポック {start_epoch + 1} から再開します")
                print(f"📋 保存されたモデルタイプ: {saved_model_type.value}")
                print(f"📋 保存された設定: samples={saved_limit_samples}, mode={saved_mode}")
            else:
                print("🆕 設定が異なるため、新規トレーニングを開始します（既存のチェックポイントは無視されます）")
                start_epoch, text_encoder, model = 0, None, None
        else:
            print("🆕 新規トレーニングを開始します")

        # Load dataset
        print("\n=== データセット読み込み ===")
        if limit_samples:
            print(f"📊 データセット制限: 最初の{limit_samples}サンプル")
        else:
            print("📊 データセット: フルデータセット使用")
        dataset = load_ljspeech_dataset(split="train", batch_size=1, limit_samples=limit_samples)
        
        # 最初のデータのテキストを取得
        print("📝 最初のデータのテキストを取得中...")
        first_text = None
        for text, audio in dataset.take(1):
            first_text = text[0].numpy().decode('utf-8')
            print(f"🎯 最初のサンプルテキスト: '{first_text}'")
            break
        
        # 推論用テキストの設定
        if args.mode == 'mini':
            inference_text = first_text
            print(f"🎤 ミニモード: 推論テキストは最初のサンプルと同じ")
        else:
            inference_text = first_text if first_text else f"This is a test of the {model_type.value} model."
            print(f"🎤 推論用テキスト: '{inference_text}'")

        # Print dataset size before preparation
        print("📊 データセットサイズを確認中...")
        num_before = (
            dataset.unbatch()
            .reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1)
            .numpy()
        )
        print(f"📈 前処理前のサンプル数: {num_before:,}")

        # Text processing setup
        print("\n=== テキスト処理 ===")
        if text_encoder is None:
            print("🔤 新しいテキストエンコーダーを作成中...")
            text_encoder = create_text_encoder(vocab_size=10000)
            # フィルタリングする前のデータセットで語彙を構築
            print("📚 語彙を構築中...")
            # 10サンプルしかないため、すべてを使用
            example_texts = dataset.unbatch().map(lambda text, audio: text)
            text_encoder.adapt(example_texts)
        else:
            print("♻️  保存されたテキストエンコーダーを使用")
            
        print(f"📖 語彙サイズ: {text_encoder.vocabulary_size():,}")

        # Build model
        print("\n=== モデル構築 ===")
        if model is None:
            print("🤖 新しいモデルを構築中...")
            model = build_text_to_spectrogram_model(
                vocab_size=text_encoder.vocabulary_size(),
                mel_bins=80,
                max_sequence_length=MAX_FRAMES,
                model_type=model_type
            )
        else:
            print("♻️  保存されたモデルを使用")
            
        print("📋 モデル構造:")
        model.summary()

        # --- Training Part ---
        print("\n=== トレーニング用データセット準備 ===")

        # スペクトログラム計算に必要な最小の音声長を定義
        N_FFT = 1024
        HOP_LENGTH = 256

        def filter_short_audio(text, audio):
            """
            音声の長さがn_fftより短いサンプルを除外するフィルタ関数。
            """
            return tf.shape(audio)[0] > N_FFT

        def py_extract_mel_spectrogram_wrapper(audio):
            # この関数はtf.py_function内で呼ばれる
            return extract_mel_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

        def prepare_for_training(text, audio):
            """
            Prepare data for text-to-spectrogram training.
            Input: text vector
            Output: mel-spectrogram
            """
            audio_processed = preprocess_audio(audio)
            mel_spec = tf.py_function(
                func=py_extract_mel_spectrogram_wrapper,
                inp=[audio_processed],
                Tout=tf.float32,
            )
            mel_spec.set_shape((None, 80))  # Set shape for Keras
            text_vec = text_encoder(text)
            # メルスペクトログラムをMAX_FRAMESフレームにパディング/切り詰め
            mel_spec = mel_spec[:MAX_FRAMES]  # 切り詰め
            mel_spec = tf.pad(
                mel_spec, [[0, MAX_FRAMES - tf.shape(mel_spec)[0]], [0, 0]]
            )  # パディング
            mel_spec.set_shape((MAX_FRAMES, 80))
            return text_vec, mel_spec

        print("⚙️  データセットパイプラインを構築中...")
        train_dataset = (
            dataset.unbatch()
            .filter(filter_short_audio)  # 短すぎる音声を除外
            .map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(buffer_size=1024)
            .padded_batch(
                batch_size=32,
                padded_shapes=(
                    tf.TensorShape([None]),  # Shape for text_vec
                    tf.TensorShape([None, 80]),  # Shape for mel_spec
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
        )

        # Print dataset size after preparation
        print("📊 前処理後のデータセットサイズを確認中...")
        num_after = 0
        for _ in train_dataset.unbatch():
            num_after += 1
        print(f"📈 前処理後のサンプル数: {num_after:,}")

        print("\n=== モデルトレーニング開始 ===")

        # エポックごとに音声を出力するためのコールバックを作成
        training_plot_callback = TrainingPlotCallback(model_output_dir=model_output_dir, model_type=model_type)
        synthesis_callback = SynthesisCallback(
            text_encoder=text_encoder, n_fft=N_FFT, hop_length=HOP_LENGTH, 
            inference_text=inference_text, model_type=model_type, model_output_dir=model_output_dir
        )

        # Create checkpoint callback (save weights only)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "model.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )

        # Create custom callback to save training state
        class TrainingStateCallback(tf.keras.callbacks.Callback):
            def __init__(self, text_encoder, model_type, limit_samples, mode, plot_callback):
                super().__init__()
                self.text_encoder = text_encoder
                self.model_type = model_type
                self.limit_samples = limit_samples
                self.mode = mode
                self.plot_callback = plot_callback

            def on_epoch_end(self, epoch, logs=None):
                """Save training state at the end of each epoch."""
                global training_interrupted
                
                if training_interrupted:
                    print("\n⚠️  中断が要求されたため、状態保存をスキップします。")
                    return
                
                try:
                    save_training_state(epoch + 1, self.text_encoder, self.model, self.model_type, self.limit_samples, self.mode)  # Save next epoch number
                except Exception as e:
                    print(f"❌ 状態保存中にエラーが発生しましたが、トレーニングを継続します: {e}")

            def on_batch_end(self, batch, logs=None):
                """Check for interruption during training."""
                global training_interrupted
                if training_interrupted:
                    print("\n⚠️  中断が要求されました。現在のエポックを完了後に停止します。")
                    self.model.stop_training = True
            
            def on_train_end(self, logs=None):
                """学習終了時に学習履歴を保存"""
                self.plot_callback.save_training_history()

        training_state_callback = TrainingStateCallback(text_encoder, model_type, limit_samples, args.mode, training_plot_callback)

        # Update global variables before training
        global current_model, current_text_encoder, current_epoch
        current_model = model
        current_text_encoder = text_encoder
        current_epoch = start_epoch

        print(f"🎯 エポック {start_epoch + 1} から {args.epochs} まで学習します")
        print("💡 Ctrl+C で安全に中断できます")
        print("=" * 50)
        
        # model.fitにcallbacks引数を追加
        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            initial_epoch=start_epoch,
            callbacks=[
                training_plot_callback,
                synthesis_callback,
                checkpoint_callback,
                training_state_callback,
            ],
        )

        # Check if training was interrupted
        global training_interrupted
        if training_interrupted:
            print("\n⚠️  トレーニングが中断されました")
            print("💾 最終状態を保存中...")
            try:
                save_training_state(current_epoch, text_encoder, model, model_type, limit_samples, args.mode)
                print("✅ 中断時の状態保存が完了しました")
            except Exception as e:
                print(f"❌ 中断時の状態保存に失敗しました: {e}")
            return  # Early return on interruption

        print("\n=== トレーニングが完了しました! ===")

        # --- Save the Trained Model ---
        print("\n=== 最終モデル保存 ===")
        model_save_path = os.path.join(BASE_OUTPUT_DIR, model_type.value, "ljspeech_synthesis_model.weights.h5")
        model.save_weights(model_save_path)
        print(f"💾 最終モデル重みを保存: {model_save_path}")

        # Save final training state
        save_training_state(args.epochs, text_encoder, model, model_type, limit_samples, args.mode)  # Final epoch

        # --- Perform Final Inference (Text-to-Speech) ---
        print("\n=== 最終推論実行 (テキストから音声) ===")

        # 推論に使用するテキスト
        print(f"🎤 最終合成用テキスト: '{inference_text}'")

        # テキストをベクトル化
        text_vec = text_encoder([inference_text])

        # モデルでメルスペクトログラムを予測
        model_output = model.predict(text_vec)
        
        # モデルタイプに応じて適切な出力を取得
        if isinstance(model_output, dict):
            if model_type == TTSModel.FASTSPEECH2:
                predicted_mel_spec = model_output['mel_output_refined']
            else:
                # 他のモデルタイプの場合、利用可能な出力キーから選択
                predicted_mel_spec = model_output.get('mel_output_refined', 
                                                    model_output.get('mel_output',
                                                                    list(model_output.values())[0]))
        else:
            predicted_mel_spec = model_output

        # バッチ次元を削除し、numpy配列に変換
        predicted_mel_spec_np = predicted_mel_spec[0]

        # スペクトログラムをデシベルからパワーに変換
        predicted_mel_spec_db_t = predicted_mel_spec_np.T
        power_spec = librosa.db_to_power(predicted_mel_spec_db_t)

        # Griffin-Limアルゴリズムで音声を復元
        print("🎵 Griffin-Limアルゴリズムで音声を合成中...")
        generated_audio = librosa.feature.inverse.mel_to_audio(
            power_spec, sr=22050, n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        # 生成された音声を保存
        output_audio_path = os.path.join(BASE_OUTPUT_DIR, model_type.value, "synthesized_audio_final.wav")
        sf.write(output_audio_path, generated_audio, 22050)
        print(f"🎵 最終合成音声を保存: {output_audio_path}")

        # 生成されたスペクトログラムと音声を可視化
        visualize_audio_and_spectrogram(
            tf.convert_to_tensor(generated_audio),
            inference_text,
            save_path=os.path.join(BASE_OUTPUT_DIR, model_type.value, "synthesis_visualization_final.png"),
        )

        print(f"\n🕐 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ スクリプトが正常に完了しました!")

    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーによって中断されました")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save current state if possible
        try:
            if 'current_model' in globals() and current_model is not None:
                print("🆘 エラー発生時の緊急状態保存を試行中...")
                # Use default values for emergency save if variables are not available
                emergency_limit_samples = locals().get('limit_samples', 10)
                emergency_mode = locals().get('args', type('', (), {'mode': 'mini'})).mode
                save_training_state(current_epoch, current_text_encoder, current_model, model_type, emergency_limit_samples, emergency_mode)
                print("✅ 緊急状態保存が完了しました")
        except Exception as save_error:
            print(f"❌ 緊急状態保存に失敗しました: {save_error}")


if __name__ == "__main__":
    main()
