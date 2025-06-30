#!/usr/bin/env python3
"""
M-AILABS Dataset Multi-Speaker Speech Synthesis Script
This script demonstrates how to load, work with, and train a Multi-Speaker Text-to-Speech model on the M-AILABS dataset.
Enhanced with robust checkpoint and resume functionality for multi-speaker synthesis.
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
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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

# Set the maximum number of frames for mel-spectrogram (for ~5 seconds audio)
MAX_FRAMES = 430  # (5 * 22050 / 256 ≈ 430)

# M-AILABS Dataset configuration
MAILABS_BASE_URL = "https://www.caito.de/data/Training/stt_tts/en_US.tgz"
DATASET_DIR = "/opt/datasets/mailabs"
AUDIO_DIR = os.path.join(DATASET_DIR, "en_US")

# Checkpoint paths
CHECKPOINT_DIR = "outputs/checkpoints"
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.keras")
TEXT_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "text_encoder")
SPEAKER_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "speaker_encoder")
TRAINING_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_state.json")
DATASET_CACHE_PATH = os.path.join(CHECKPOINT_DIR, "dataset_processed.cache")

# Global variables for graceful shutdown
training_interrupted = False
current_model = None
current_text_encoder = None
current_speaker_encoder = None
current_epoch = 0

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C) gracefully."""
    global training_interrupted, current_model, current_text_encoder, current_speaker_encoder, current_epoch
    print(f"\n\n=== 中断シグナルを受信しました (Signal: {signum}) ===")
    print("トレーニングを安全に停止しています...")
    training_interrupted = True
    
    if current_model is not None and current_text_encoder is not None:
        print("現在の状態を保存中...")
        try:
            save_training_state(current_epoch, current_text_encoder, current_speaker_encoder, current_model)
            print("✅ チェックポイントが正常に保存されました")
        except Exception as e:
            print(f"❌ チェックポイント保存中にエラーが発生しました: {e}")
    
    print("プログラムを終了します...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def download_mailabs_dataset():
    """Download and extract M-AILABS dataset."""
    print("=== M-AILABS データセットダウンロード ===")
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    dataset_path = os.path.join(DATASET_DIR, "en_US.tgz")
    
    # Check if already downloaded
    if os.path.exists(AUDIO_DIR) and os.listdir(AUDIO_DIR):
        print("✅ M-AILABS データセットは既にダウンロード済みです")
        return True
    
    try:
        print(f"📥 データセットをダウンロード中: {MAILABS_BASE_URL}")
        response = requests.get(MAILABS_BASE_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(dataset_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r進捗: {progress:.1f}%", end="", flush=True)
        
        print(f"\n📦 ダウンロード完了: {dataset_path}")
        
        # Extract the dataset
        print("📂 データセットを展開中...")
        import tarfile
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(DATASET_DIR)
        
        print("✅ データセット展開完了")
        
        # Clean up the tar file
        os.remove(dataset_path)
        
        return True
        
    except Exception as e:
        print(f"❌ データセットダウンロードに失敗しました: {e}")
        return False

def parse_mailabs_metadata(metadata_path: str) -> List[Dict]:
    """Parse M-AILABS metadata file."""
    metadata = []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 2:
                        filename = parts[0]
                        text = parts[1]
                        # Extract speaker info from path
                        speaker_id = filename.split('/')[0] if '/' in filename else "unknown"
                        
                        metadata.append({
                            'filename': filename,
                            'text': text,
                            'speaker_id': speaker_id
                        })
    except Exception as e:
        print(f"メタデータ解析エラー: {e}")
    
    return metadata

def load_mailabs_dataset(batch_size: int = 32) -> tf.data.Dataset:
    """Load the M-AILABS dataset."""
    print(f"Loading M-AILABS dataset...")
    
    # Download dataset if needed
    if not download_mailabs_dataset():
        raise RuntimeError("Failed to download M-AILABS dataset")
    
    # Find metadata files
    metadata_files = []
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith('metadata.csv') or file.endswith('metadata.txt'):
                metadata_files.append(os.path.join(root, file))
    
    if not metadata_files:
        raise RuntimeError("No metadata files found in M-AILABS dataset")
    
    # Parse all metadata
    all_metadata = []
    speakers = set()
    
    for metadata_file in metadata_files:
        print(f"📖 メタデータを読み込み中: {metadata_file}")
        metadata = parse_mailabs_metadata(metadata_file)
        all_metadata.extend(metadata)
        
        for item in metadata:
            speakers.add(item['speaker_id'])
    
    print(f"📊 総サンプル数: {len(all_metadata):,}")
    print(f"🎭 スピーカー数: {len(speakers)}")
    print(f"🎭 スピーカー一覧: {sorted(list(speakers))}")
    
    # Create speaker ID mapping
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
    
    def data_generator():
        for item in all_metadata:
            audio_path = os.path.join(AUDIO_DIR, item['filename'])
            if audio_path.endswith('.txt'):
                audio_path = audio_path.replace('.txt', '.wav')
            
            if os.path.exists(audio_path):
                try:
                    audio, sr = librosa.load(audio_path, sr=22050)
                    speaker_id = speaker_to_id[item['speaker_id']]
                    
                    yield {
                        'text': item['text'],
                        'audio': audio.astype(np.float32),
                        'speaker_id': speaker_id
                    }
                except Exception as e:
                    print(f"オーディオ読み込みエラー {audio_path}: {e}")
                    continue
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'text': tf.TensorSpec(shape=(), dtype=tf.string),
            'audio': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'speaker_id': tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size), len(speakers)


def preprocess_audio(audio: tf.Tensor) -> tf.Tensor:
    """Preprocess audio data for machine learning."""
    audio = tf.cast(audio, tf.float32)
    if len(audio.shape) > 1 and audio.shape[-1] > 1:
        audio = tf.reduce_mean(audio, axis=-1)
    # Normalize to [-1, 1] range
    max_val = tf.reduce_max(tf.abs(audio))
    if max_val > 0:
        audio = audio / max_val
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


def create_text_encoder(vocab_size: int = 1000) -> tf.keras.layers.TextVectorization:
    """Create a text encoder for processing transcriptions."""
    return tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=MAX_FRAMES,
        standardize="lower_and_strip_punctuation",
    )

def create_speaker_encoder(num_speakers: int) -> tf.keras.layers.CategoryEncoding:
    """Create a speaker encoder for multi-speaker support."""
    return tf.keras.layers.CategoryEncoding(
        num_tokens=num_speakers,
        output_mode="one_hot"
    )


def build_multispeaker_text_to_spectrogram_model(
    vocab_size: int, 
    num_speakers: int,
    mel_bins: int = 80, 
    max_sequence_length: int = MAX_FRAMES,
    embedding_dim: int = 128,
    speaker_embedding_dim: int = 64
) -> tf.keras.Model:
    """Build a Multi-Speaker Text-to-Spectrogram model for speech synthesis."""
    
    # Text input
    text_input = tf.keras.layers.Input(shape=(None,), name="text_input")
    # Speaker input
    speaker_input = tf.keras.layers.Input(shape=(num_speakers,), name="speaker_input")
    
    # Text embedding and encoding
    text_features = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
    text_features = tf.keras.layers.LSTM(256, return_sequences=True)(text_features)
    text_features = tf.keras.layers.LSTM(256, return_sequences=True)(text_features)
    
    # Speaker embedding
    speaker_features = tf.keras.layers.Dense(speaker_embedding_dim, activation='relu')(speaker_input)
    speaker_features = tf.keras.layers.RepeatVector(max_sequence_length)(speaker_features)
    
    # Concatenate text and speaker features
    combined_features = tf.keras.layers.Concatenate(axis=-1)([text_features, speaker_features])
    
    # Additional processing layers
    combined_features = tf.keras.layers.LSTM(512, return_sequences=True)(combined_features)
    combined_features = tf.keras.layers.LSTM(512, return_sequences=True)(combined_features)
    
    # Output layer to predict mel-spectrogram
    mel_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(mel_bins, activation="linear")
    )(combined_features)
    
    model = tf.keras.Model(inputs=[text_input, speaker_input], outputs=mel_output)
    
    # Use Mean Squared Error to measure the difference between predicted and actual spectrograms
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def visualize_audio_and_spectrogram(
    audio: tf.Tensor, text: str, speaker_id: int, sample_rate: int = 22050, save_path: str | None = None
):
    """Visualize audio waveform and mel-spectrogram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    audio_np = audio.numpy()
    time = np.arange(len(audio_np)) / sample_rate
    ax1.plot(time, audio_np)
    ax1.set_title(f'Waveform - Speaker {speaker_id} - Text: "{text[:50]}..."')
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
    ax2.set_title(f"Mel-Spectrogram - Speaker {speaker_id}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


class MultiSpeakerSynthesisCallback(tf.keras.callbacks.Callback):
    """Callback to generate audio at the end of each epoch for multi-speaker model."""
    
    def __init__(self, text_encoder, speaker_encoder, num_speakers, n_fft, hop_length, sample_rate=22050):
        super().__init__()
        self.text_encoder = text_encoder
        self.speaker_encoder = speaker_encoder
        self.num_speakers = num_speakers
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        # Test texts for different scenarios
        self.inference_texts = [
            "This is a test of the multi-speaker model at the end of each epoch.",
            "Hello, how are you today?",
            "The weather is beautiful."
        ]
        os.makedirs("outputs/epoch_samples", exist_ok=True)
        
        # Time prediction variables
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        """トレーニング開始時に開始時刻を記録し、総エポック数を設定"""
        self.training_start_time = time.time()
        # パラメータから総エポック数を取得
        self.total_epochs = self.params.get('epochs', 3)
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
        """Generate multi-speaker audio samples, save state, and predict completion time."""
        global training_interrupted
        
        # Record epoch end time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        # Time prediction calculations
        completed_epochs = epoch + 1
        remaining_epochs = self.total_epochs - completed_epochs
        
        # Calculate average epoch time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        # Calculate remaining time
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        estimated_completion_time = epoch_end_time + estimated_remaining_time
        
        # Display time information
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
            
        print(f"\n🎵 エポック {epoch + 1} 完了 - マルチスピーカーサンプル音声を生成中...")

        try:
            # Test with first few speakers and texts
            test_speaker_ids = [0, min(1, self.num_speakers-1)]  # Test with available speakers
            test_texts = self.inference_texts[:1]  # Use first text only for speed
            
            for text_idx, test_text in enumerate(test_texts):
                for speaker_idx in test_speaker_ids:
                    if speaker_idx >= self.num_speakers:
                        continue
                        
                    try:
                        # Prepare inputs
                        text_vec = self.text_encoder([test_text])
                        speaker_vec = tf.one_hot([speaker_idx], depth=self.num_speakers)
                        
                        # Generate mel-spectrogram
                        predicted_mel_spec = self.model.predict([text_vec, speaker_vec], verbose=0)
                        predicted_mel_spec_np = predicted_mel_spec[0]
                        
                        # Convert to audio using Griffin-Lim
                        predicted_mel_spec_db_t = predicted_mel_spec_np.T
                        power_spec = librosa.db_to_power(predicted_mel_spec_db_t)
                        generated_audio = librosa.feature.inverse.mel_to_audio(
                            power_spec, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
                        )
                        
                        # Save audio
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"outputs/epoch_samples/epoch_{epoch + 1}_speaker_{speaker_idx}_text_{text_idx}_{timestamp}.wav"
                        sf.write(output_path, generated_audio, self.sample_rate)
                        
                        print(f"  💾 保存: {output_path}")
                        
                    except Exception as e:
                        print(f"  ❌ スピーカー {speaker_idx} の音声生成エラー: {e}")
                        continue
            
            print(f"✅ エポック {epoch + 1} のマルチスピーカーサンプル音声生成完了")

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


def save_training_state(epoch, text_encoder, speaker_encoder, model):
    """Save training state including epoch number and text encoder."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        print(f"💾 エポック {epoch} の状態を保存中...")
        
        # Save model
        model.save(MODEL_CHECKPOINT_PATH)
        print(f"  ✅ モデルを保存: {MODEL_CHECKPOINT_PATH}")

        # Save text encoder vocabulary
        vocab = text_encoder.get_vocabulary()
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        with open(vocab_path, "w", encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 語彙を保存: {vocab_path}")

        # Save speaker encoder config
        if speaker_encoder is not None:
            speaker_config = speaker_encoder.get_config()
            speaker_config_path = os.path.join(CHECKPOINT_DIR, "speaker_config.json")
            with open(speaker_config_path, "w", encoding='utf-8') as f:
                json.dump(speaker_config, f, ensure_ascii=False, indent=2)
            print(f"  ✅ スピーカー設定を保存: {speaker_config_path}")

        # Save training state with timestamp
        training_state = {
            "epoch": epoch,
            "vocab_size": text_encoder.vocabulary_size(),
            "max_tokens": text_encoder._max_tokens,
            "output_sequence_length": text_encoder._output_sequence_length,
            "standardize": text_encoder._standardize,
            "num_speakers": speaker_encoder.num_tokens if speaker_encoder else 0,
            "saved_at": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "checkpoint_version": "3.0"  # Updated for multi-speaker
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
    """Load training state and return epoch number, text encoder, speaker encoder, and model if available."""
    print("🔍 保存された状態を確認中...")
    
    if not os.path.exists(TRAINING_STATE_PATH):
        print("ℹ️  保存された状態が見つかりません。最初から開始します。")
        return 0, None, None, None

    try:
        # Load training state
        print(f"📂 トレーニング状態を読み込み中: {TRAINING_STATE_PATH}")
        with open(TRAINING_STATE_PATH, "r", encoding='utf-8') as f:
            training_state = json.load(f)
        
        # Validate checkpoint version compatibility
        checkpoint_version = training_state.get("checkpoint_version", "1.0")
        print(f"  📋 チェックポイントバージョン: {checkpoint_version}")
        
        if training_state.get("tensorflow_version"):
            print(f"  🔧 保存時のTensorFlowバージョン: {training_state['tensorflow_version']}")
            print(f"  🔧 現在のTensorFlowバージョン: {tf.__version__}")
        
        saved_at = training_state.get("saved_at", "不明")
        print(f"  ⏰ 保存時刻: {saved_at}")

        # Check if model file exists
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            print(f"❌ モデルファイルが見つかりません: {MODEL_CHECKPOINT_PATH}")
            return 0, None, None, None

        # Load vocabulary
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        if not os.path.exists(vocab_path):
            print(f"❌ 語彙ファイルが見つかりません: {vocab_path}")
            return 0, None, None, None
            
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

        # Recreate speaker encoder
        speaker_encoder = None
        num_speakers = training_state.get("num_speakers", 0)
        if num_speakers > 0:
            print("🎭 スピーカーエンコーダーを再構築中...")
            speaker_encoder = create_speaker_encoder(num_speakers)
            print(f"  ✅ スピーカー数: {num_speakers}")
        else:
            print("⚠️  スピーカー情報が見つかりません。新規作成が必要です。")

        # Load model
        print(f"🤖 モデルを読み込み中: {MODEL_CHECKPOINT_PATH}")
        model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)
        print(f"  ✅ モデル読み込み完了")

        epoch = training_state["epoch"]
        print(f"🎯 エポック {epoch} から再開します")
        print("=" * 50)
        
        return epoch, text_encoder, speaker_encoder, model

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
        
        return 0, None, None, None


def main():
    """Main function to run the M-AILABS multi-speaker learning script."""
    print("=== M-AILABS マルチスピーカー音声合成スクリプト ===")
    print(f"🕐 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Check for existing checkpoint
        start_epoch, text_encoder, speaker_encoder, model = load_training_state()

        if start_epoch > 0:
            print(f"🔄 チェックポイントが見つかりました。エポック {start_epoch + 1} から再開します")
        else:
            print("🆕 新規トレーニングを開始します")

        # Load dataset
        print("\n=== データセット読み込み ===")
        dataset, num_speakers = load_mailabs_dataset(batch_size=1)
        os.makedirs("outputs", exist_ok=True)
        
        print(f"🎭 検出されたスピーカー数: {num_speakers}")

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
            text_encoder = create_text_encoder(vocab_size=1000)
            # Build vocabulary from dataset
            print("📚 語彙を構築中...")
            example_texts = dataset.unbatch().map(lambda x: x['text']).take(5000)
            text_encoder.adapt(example_texts)
        else:
            print("♻️  保存されたテキストエンコーダーを使用")
            
        print(f"📖 語彙サイズ: {text_encoder.vocabulary_size():,}")

        # Speaker processing setup
        print("\n=== スピーカー処理 ===")
        if speaker_encoder is None:
            print("🎭 新しいスピーカーエンコーダーを作成中...")
            speaker_encoder = create_speaker_encoder(num_speakers)
        else:
            print("♻️  保存されたスピーカーエンコーダーを使用")
            
        print(f"🎭 スピーカーエンコーダー設定完了: {num_speakers} スピーカー")

        # Build model
        print("\n=== モデル構築 ===")
        if model is None:
            print("🤖 新しいマルチスピーカーモデルを構築中...")
            model = build_multispeaker_text_to_spectrogram_model(
                vocab_size=text_encoder.vocabulary_size(),
                num_speakers=num_speakers,
                mel_bins=80,
                max_sequence_length=MAX_FRAMES,
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

        def filter_short_audio(data):
            """Filter out samples with audio shorter than n_fft."""
            return tf.shape(data['audio'])[0] > N_FFT

        def py_extract_mel_spectrogram_wrapper(audio):
            """Wrapper for mel-spectrogram extraction in tf.py_function."""
            return extract_mel_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

        def prepare_for_training(data):
            """Prepare data for multi-speaker text-to-spectrogram training."""
            audio_processed = preprocess_audio(data['audio'])
            mel_spec = tf.py_function(
                func=py_extract_mel_spectrogram_wrapper,
                inp=[audio_processed],
                Tout=tf.float32,
            )
            mel_spec.set_shape((None, 80))  # Set shape for Keras
            
            text_vec = text_encoder(data['text'])
            speaker_vec = tf.one_hot(data['speaker_id'], depth=num_speakers)
            
            # Pad/truncate mel-spectrogram to MAX_FRAMES
            mel_spec = mel_spec[:MAX_FRAMES]  # Truncate
            mel_spec = tf.pad(
                mel_spec, [[0, MAX_FRAMES - tf.shape(mel_spec)[0]], [0, 0]]
            )  # Pad
            mel_spec.set_shape((MAX_FRAMES, 80))
            
            return (text_vec, speaker_vec), mel_spec

        print("⚙️  データセットパイプラインを構築中...")
        train_dataset = (
            dataset.unbatch()
            .filter(filter_short_audio)  # Filter short audio
            .map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(buffer_size=1024)
            .padded_batch(
                batch_size=16,  # Smaller batch size for multi-speaker model
                padded_shapes=(
                    (tf.TensorShape([None]), tf.TensorShape([None])),  # (text_vec, speaker_vec)
                    tf.TensorShape([None, 80]),  # mel_spec
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

        # Create callbacks
        synthesis_callback = MultiSpeakerSynthesisCallback(
            text_encoder=text_encoder, 
            speaker_encoder=speaker_encoder,
            num_speakers=num_speakers,
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
        )

        # Create custom callback to save training state
        class TrainingStateCallback(tf.keras.callbacks.Callback):
            def __init__(self, text_encoder, speaker_encoder):
                super().__init__()
                self.text_encoder = text_encoder
                self.speaker_encoder = speaker_encoder

            def on_epoch_end(self, epoch, logs=None):
                """Save training state at the end of each epoch."""
                global training_interrupted
                
                if training_interrupted:
                    print("\n⚠️  中断が要求されたため、状態保存をスキップします。")
                    return
                
                try:
                    save_training_state(epoch + 1, self.text_encoder, self.speaker_encoder, self.model)
                except Exception as e:
                    print(f"❌ 状態保存中にエラーが発生しましたが、トレーニングを継続します: {e}")

            def on_batch_end(self, batch, logs=None):
                """Check for interruption during training."""
                global training_interrupted
                if training_interrupted:
                    print("\n⚠️  中断が要求されました。現在のエポックを完了後に停止します。")
                    self.model.stop_training = True

        training_state_callback = TrainingStateCallback(text_encoder, speaker_encoder)

        # Update global variables before training
        global current_model, current_text_encoder, current_speaker_encoder, current_epoch
        current_model = model
        current_text_encoder = text_encoder
        current_speaker_encoder = speaker_encoder
        current_epoch = start_epoch

        print(f"🎯 エポック {start_epoch + 1} から {5} まで学習します")
        print("💡 Ctrl+C で安全に中断できます")
        print("=" * 50)
        
        # model.fitにcallbacks引数を追加
        history = model.fit(
            train_dataset,
            epochs=5,
            initial_epoch=start_epoch,
            callbacks=[
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
                save_training_state(current_epoch, text_encoder, speaker_encoder, model)
                print("✅ 中断時の状態保存が完了しました")
            except Exception as e:
                print(f"❌ 中断時の状態保存に失敗しました: {e}")
            return  # Early return on interruption

        print("\n=== トレーニングが完了しました! ===")

        # --- Save the Trained Model ---
        print("\n=== 最終モデル保存 ===")
        model_save_path = "outputs/mailabs_multispeaker_synthesis_model.keras"
        model.save(model_save_path)
        print(f"💾 最終モデルを保存: {model_save_path}")

        # Save final training state
        save_training_state(5, text_encoder, speaker_encoder, model)  # Final epoch

        # --- Perform Final Inference (Multi-Speaker Text-to-Speech) ---
        print("\n=== 最終推論実行 (マルチスピーカー音声合成) ===")

        inference_texts = [
            "Hello, this is a test of multi-speaker synthesis.",
            "The weather is beautiful today.",
            "Thank you for using our speech synthesis system."
        ]

        test_speaker_ids = [0, min(1, num_speakers-1), min(2, num_speakers-1)]  # Test with available speakers

        for text_idx, inference_text in enumerate(inference_texts):
            for speaker_id in test_speaker_ids:
                if speaker_id >= num_speakers:
                    continue
                    
                print(f"🎤 合成中 - スピーカー {speaker_id}: '{inference_text}'")

                try:
                    # Prepare inputs
                    text_vec = text_encoder([inference_text])
                    speaker_vec = tf.one_hot([speaker_id], depth=num_speakers)

                    # Generate mel-spectrogram
                    predicted_mel_spec = model.predict([text_vec, speaker_vec], verbose=0)
                    predicted_mel_spec_np = predicted_mel_spec[0]

                    # Convert to audio using Griffin-Lim
                    predicted_mel_spec_db_t = predicted_mel_spec_np.T
                    power_spec = librosa.db_to_power(predicted_mel_spec_db_t)
                    generated_audio = librosa.feature.inverse.mel_to_audio(
                        power_spec, sr=22050, n_fft=N_FFT, hop_length=HOP_LENGTH
                    )

                    # Save generated audio
                    output_audio_path = f"outputs/final_synthesis_speaker_{speaker_id}_text_{text_idx}.wav"
                    sf.write(output_audio_path, generated_audio, 22050)
                    print(f"🎵 音声保存: {output_audio_path}")

                    # Visualize for first example only
                    if text_idx == 0 and speaker_id == 0:
                        visualize_audio_and_spectrogram(
                            tf.convert_to_tensor(generated_audio),
                            inference_text,
                            speaker_id,
                            save_path="outputs/synthesis_visualization_final.png",
                        )

                except Exception as e:
                    print(f"❌ スピーカー {speaker_id} の音声合成エラー: {e}")

        print(f"\n🕐 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ M-AILABS マルチスピーカー音声合成が完了しました!")
        print(f"🕐 終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
                save_training_state(current_epoch, current_text_encoder, current_speaker_encoder, current_model)
                print("✅ 緊急状態保存が完了しました")
        except Exception as save_error:
            print(f"❌ 緊急状態保存に失敗しました: {save_error}")


if __name__ == "__main__":
    main()
