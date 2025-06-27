#!/usr/bin/env python3
"""
LJSpeech Dataset Learning Script (Speech Synthesis Version with Epoch Callback)
This script demonstrates how to load, work with, and train a Text-to-Speech model on the LJSpeech dataset.
This version has been modified to generate audio from text and includes a callback to output
a sample audio file at the end of each training epoch.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import soundfile as sf
import json

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

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n\n--- Generating audio sample at end of epoch {epoch + 1} ---")

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

        # 生成された音声をエポック番号付きで保存
        output_audio_path = (
            f"outputs/epoch_samples/synthesized_epoch_{epoch + 1:02d}.wav"
        )
        sf.write(output_audio_path, generated_audio, self.sample_rate)
        print(
            f"--- Synthesized audio for epoch {epoch + 1} saved to: {output_audio_path} ---\n"
        )


def save_training_state(epoch, text_encoder, model):
    """Save training state including epoch number and text encoder."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save model
    model.save(MODEL_CHECKPOINT_PATH)

    # Save text encoder vocabulary
    vocab = text_encoder.get_vocabulary()
    vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    # Save training state
    training_state = {
        "epoch": epoch,
        "vocab_size": text_encoder.vocabulary_size(),
        "max_tokens": text_encoder._max_tokens,
        "output_sequence_length": text_encoder._output_sequence_length,
        "standardize": text_encoder._standardize,
    }
    with open(TRAINING_STATE_PATH, "w") as f:
        json.dump(training_state, f)

    print(f"Training state saved at epoch {epoch}")


def load_training_state():
    """Load training state and return epoch number, text encoder, and model if available."""
    if not os.path.exists(TRAINING_STATE_PATH):
        return 0, None, None

    try:
        # Load training state
        with open(TRAINING_STATE_PATH, "r") as f:
            training_state = json.load(f)

        # Load vocabulary
        vocab_path = os.path.join(CHECKPOINT_DIR, "vocabulary.json")
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        # Recreate text encoder
        text_encoder = tf.keras.layers.TextVectorization(
            max_tokens=training_state["max_tokens"],
            output_sequence_length=training_state["output_sequence_length"],
            standardize=training_state["standardize"],
        )
        text_encoder.set_vocabulary(vocab)

        # Load model
        model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)

        print(f"Resuming from epoch {training_state['epoch']}")
        return training_state["epoch"], text_encoder, model

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, None, None


def main():
    """Main function to run the LJSpeech learning script."""
    print("=== LJSpeech Speech Synthesis Script ===")
    try:
        # Check for existing checkpoint
        start_epoch, text_encoder, model = load_training_state()

        if start_epoch > 0:
            print(f"Found checkpoint, resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

        # Load dataset
        dataset = load_ljspeech_dataset(split="train", batch_size=1)
        os.makedirs("outputs", exist_ok=True)

        # Print dataset size before preparation
        num_before = (
            dataset.unbatch()
            .reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1)
            .numpy()
        )
        print(f"Number of samples before preparation: {num_before}")

        # Text processing setup
        print("\n=== Text Processing ===")
        if text_encoder is None:
            text_encoder = create_text_encoder(vocab_size=1000)
            # フィルタリングする前のデータセットで語彙を構築
            example_texts = dataset.unbatch().map(lambda text, audio: text).take(5000)
            text_encoder.adapt(example_texts)
        print(f"Vocabulary size: {text_encoder.vocabulary_size()}")

        # Build model
        print("\n=== Model Architecture ===")
        if model is None:
            model = build_text_to_spectrogram_model(
                vocab_size=text_encoder.vocabulary_size(),
                mel_bins=80,
                max_sequence_length=MAX_FRAMES,
            )
        model.summary()

        # --- Training Part ---
        print("\n=== Preparing Dataset for Training ===")

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
        num_after = 0
        for _ in train_dataset.unbatch():
            num_after += 1
        print(f"Number of samples after preparation: {num_after}")

        print("\n=== Starting Model Training ===")

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
                save_training_state(epoch, self.text_encoder, self.model)

        training_state_callback = TrainingStateCallback(text_encoder)

        # model.fitにcallbacks引数を追加
        model.fit(
            train_dataset,
            epochs=3,
            initial_epoch=start_epoch,
            callbacks=[
                synthesis_callback,
                checkpoint_callback,
                training_state_callback,
            ],
        )

        print("\n=== Training completed! ===")

        # --- Save the Trained Model ---
        print("\n=== Saving Trained Model ===")
        model_save_path = "outputs/ljspeech_synthesis_model.keras"
        model.save(model_save_path)
        print(f"Model saved successfully to {model_save_path}")

        # --- Perform Final Inference (Text-to-Speech) ---
        print("\n=== Performing Final Inference (Text to Speech) ===")

        # 推論に使用するテキスト
        inference_text = (
            "Hello, this is a final test of the new speech synthesis model."
        )
        print(f"Input text for synthesis: '{inference_text}'")

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
        print("Synthesizing audio from spectrogram using Griffin-Lim algorithm...")
        generated_audio = librosa.feature.inverse.mel_to_audio(
            power_spec, sr=22050, n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        # 生成された音声を保存
        output_audio_path = "outputs/synthesized_audio_final.wav"
        sf.write(output_audio_path, generated_audio, 22050)
        print(f"Synthesized audio saved to: {output_audio_path}")

        # 生成されたスペクトログラムと音声を可視化
        visualize_audio_and_spectrogram(
            tf.convert_to_tensor(generated_audio),
            inference_text,
            save_path="outputs/synthesis_visualization_final.png",
        )

        print("\n=== Script completed successfully! ===")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
