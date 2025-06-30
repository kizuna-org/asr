#!/usr/bin/env python3
"""
LJSpeech Dataset Learning Script (Speech Synthesis Version with Epoch Callback)
This script demonstrates how to load, work with, and train a Text-to-Speech model on the LJSpeech dataset.
This version has been modified to generate audio from text and includes a callback to output
a sample audio file at the end of each training epoch.
Enhanced with robust checkpoint and resume functionality.
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

# Checkpoint paths
CHECKPOINT_DIR = "outputs/checkpoints"
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.keras")
TEXT_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "text_encoder")
TRAINING_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_state.json")
DATASET_CACHE_PATH = os.path.join(CHECKPOINT_DIR, "dataset_processed.cache")

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
            save_training_state(current_epoch, current_text_encoder, current_model)
            print("✅ チェックポイントが正常に保存されました")
        except Exception as e:
            print(f"❌ チェックポイント保存中にエラーが発生しました: {e}")
    
    print("プログラムを終了します...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def load_ljspeech_dataset(
    split: str = "train", batch_size: int = 32
) -> tf.data.Dataset:
    """
    Load the LJSpeech dataset from TensorFlow Datasets.
    """
    print(f"Loading LJSpeech dataset with split: {split}")

    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        print(f"Error importing tensorflow_datasets: {e}")
        print("Please install tensorflow_datasets: pip install tensorflow-datasets")
        raise

    dataset, info = tfds.load(
        "ljspeech",
        split=split,
        with_info=True,
        as_supervised=True,
        data_dir="/opt/datasets",
    )

    print(f"Dataset info: {info}")
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


def create_text_encoder(vocab_size: int = 1000) -> tf.keras.layers.TextVectorization:
    """
    Create a text encoder for processing transcriptions.
    """
    return tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=MAX_FRAMES,  # Use MAX_FRAMES for output sequence length
        standardize="lower_and_strip_punctuation",
    )


def build_text_to_spectrogram_model(
    vocab_size: int, mel_bins: int = 80, max_sequence_length: int = MAX_FRAMES
) -> tf.keras.Model:
    """
    Build a simple Text-to-Spectrogram model for speech synthesis.
    """
    text_input = tf.keras.layers.Input(shape=(None,), name="text_input")

    # Text Embedding and Encoding
    text_features = tf.keras.layers.Embedding(vocab_size, 2, mask_zero=True)(
        text_input
    )
    text_features = tf.keras.layers.LSTM(4, return_sequences=True)(text_features)
    text_features = tf.keras.layers.LSTM(4, return_sequences=True)(text_features)

    # Output layer to predict mel-spectrogram
    mel_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(mel_bins, activation="linear")
    )(text_features)

    model = tf.keras.Model(inputs=text_input, outputs=mel_output)

    # Use Mean Squared Error to measure the difference between predicted and actual spectrograms
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


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
    plt.show()


class SynthesisCallback(tf.keras.callbacks.Callback):
    def __init__(self, text_encoder, n_fft, hop_length, sample_rate=22050):
        super().__init__()
        self.text_encoder = text_encoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        # 音声合成に使用するテスト用のテキスト
        self.inference_text = "This is a test of the model at the end of each epoch."
        os.makedirs("outputs/epoch_samples", exist_ok=True)
        
        # 時間予測用の変数
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []  # 各エポックの実行時間を記録
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
            predicted_mel_spec = self.model.predict(text_vec)

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
            output_audio_path = f"outputs/epoch_samples/epoch_{epoch + 1}_{timestamp}.wav"
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


def save_training_state(epoch, text_encoder, model):
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

        # Save training state with timestamp
        training_state = {
            "epoch": epoch,
            "vocab_size": text_encoder.vocabulary_size(),
            "max_tokens": text_encoder._max_tokens,
            "output_sequence_length": text_encoder._output_sequence_length,
            "standardize": text_encoder._standardize,
            "saved_at": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "checkpoint_version": "2.0"
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
    """Load training state and return epoch number, text encoder, and model if available."""
    print("🔍 保存された状態を確認中...")
    
    if not os.path.exists(TRAINING_STATE_PATH):
        print("ℹ️  保存された状態が見つかりません。最初から開始します。")
        return 0, None, None

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
            return 0, None, None

        # Load vocabulary
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        if not os.path.exists(vocab_path):
            print(f"❌ 語彙ファイルが見つかりません: {vocab_path}")
            return 0, None, None
            
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

        # Load model
        print(f"🤖 モデルを読み込み中: {MODEL_CHECKPOINT_PATH}")
        model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)
        print(f"  ✅ モデル読み込み完了")

        epoch = training_state["epoch"]
        print(f"🎯 エポック {epoch} から再開します")
        print("=" * 50)
        
        return epoch, text_encoder, model

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
        
        return 0, None, None


def main():
    """Main function to run the LJSpeech learning script."""
    print("=== LJSpeech 音声合成スクリプト ===")
    print(f"🕐 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Check for existing checkpoint
        start_epoch, text_encoder, model = load_training_state()

        if start_epoch > 0:
            print(f"🔄 チェックポイントが見つかりました。エポック {start_epoch + 1} から再開します")
        else:
            print("🆕 新規トレーニングを開始します")

        # Load dataset
        print("\n=== データセット読み込み ===")
        dataset = load_ljspeech_dataset(split="train", batch_size=1)
        os.makedirs("outputs", exist_ok=True)

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
            # フィルタリングする前のデータセットで語彙を構築
            print("📚 語彙を構築中...")
            example_texts = dataset.unbatch().map(lambda text, audio: text).take(5000)
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
        synthesis_callback = SynthesisCallback(
            text_encoder=text_encoder, n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        # Create checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
        )

        # Create custom callback to save training state
        class TrainingStateCallback(tf.keras.callbacks.Callback):
            def __init__(self, text_encoder):
                super().__init__()
                self.text_encoder = text_encoder

            def on_epoch_end(self, epoch, logs=None):
                """Save training state at the end of each epoch."""
                global training_interrupted
                
                if training_interrupted:
                    print("\n⚠️  中断が要求されたため、状態保存をスキップします。")
                    return
                
                try:
                    save_training_state(epoch + 1, self.text_encoder, self.model)  # Save next epoch number
                except Exception as e:
                    print(f"❌ 状態保存中にエラーが発生しましたが、トレーニングを継続します: {e}")

            def on_batch_end(self, batch, logs=None):
                """Check for interruption during training."""
                global training_interrupted
                if training_interrupted:
                    print("\n⚠️  中断が要求されました。現在のエポックを完了後に停止します。")
                    self.model.stop_training = True

        training_state_callback = TrainingStateCallback(text_encoder)

        # Update global variables before training
        global current_model, current_text_encoder, current_epoch
        current_model = model
        current_text_encoder = text_encoder
        current_epoch = start_epoch

        print(f"🎯 エポック {start_epoch + 1} から {3} まで学習します")
        print("💡 Ctrl+C で安全に中断できます")
        print("=" * 50)
        
        # model.fitにcallbacks引数を追加
        history = model.fit(
            train_dataset,
            epochs=3,
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
                save_training_state(current_epoch, text_encoder, model)
                print("✅ 中断時の状態保存が完了しました")
            except Exception as e:
                print(f"❌ 中断時の状態保存に失敗しました: {e}")
            return  # Early return on interruption

        print("\n=== トレーニングが完了しました! ===")

        # --- Save the Trained Model ---
        print("\n=== 最終モデル保存 ===")
        model_save_path = "outputs/ljspeech_synthesis_model.keras"
        model.save(model_save_path)
        print(f"💾 最終モデルを保存: {model_save_path}")

        # Save final training state
        save_training_state(3, text_encoder, model)  # Final epoch

        # --- Perform Final Inference (Text-to-Speech) ---
        print("\n=== 最終推論実行 (テキストから音声) ===")

        # 推論に使用するテキスト
        inference_text = (
            "Hello, this is a final test of the new speech synthesis model."
        )
        print(f"🎤 合成用テキスト: '{inference_text}'")

        # テキストをベクトル化
        text_vec = text_encoder([inference_text])

        # モデルでメルスペクトログラムを予測
        predicted_mel_spec = model.predict(text_vec)

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
        output_audio_path = "outputs/synthesized_audio_final.wav"
        sf.write(output_audio_path, generated_audio, 22050)
        print(f"🎵 最終合成音声を保存: {output_audio_path}")

        # 生成されたスペクトログラムと音声を可視化
        visualize_audio_and_spectrogram(
            tf.convert_to_tensor(generated_audio),
            inference_text,
            save_path="outputs/synthesis_visualization_final.png",
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
                save_training_state(current_epoch, current_text_encoder, current_model)
                print("✅ 緊急状態保存が完了しました")
        except Exception as save_error:
            print(f"❌ 緊急状態保存に失敗しました: {save_error}")


if __name__ == "__main__":
    main()
