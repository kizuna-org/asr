import torch
import numpy as np
import librosa
import soundfile as sf
import wave
import threading
import queue
import time
from typing import Optional, List, Tuple
import os
import json
from app.model import ID_TO_CHAR

# ALSAエラーを抑制
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_CONFIG_PATH'] = '/dev/null'
os.environ['ALSA_PCM_NAME'] = 'null'
os.environ['PYTHONWARNINGS'] = 'ignore'

# PyAudioを遅延インポート（エラーを抑制）
try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("PyAudioが利用できません")


class AudioRecorder:
    """リアルタイム音声録音クラス"""
    
    def __init__(self, 
                 sample_rate=16000,
                 chunk_size=1024,
                 channels=1,
                 format=None):
        # pyaudioが利用できない場合はデフォルト値を設定
        if pyaudio is None:
            format = 1  # paFloat32の値
        else:
            format = format or pyaudio.paFloat32
        # ALSAエラーを強力に抑制
        import os
        os.environ['ALSA_PCM_CARD'] = '0'
        os.environ['ALSA_PCM_DEVICE'] = '0'
        os.environ['ALSA_CONFIG_PATH'] = '/dev/null'
        os.environ['ALSA_PCM_NAME'] = 'null'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        # 標準エラー出力を一時的にリダイレクト
        import sys
        import contextlib
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        # PyAudioの初期化を試行（エラー出力を抑制）
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"PyAudio初期化エラー: {e}")
            self.audio = None
            
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
    
    def start_recording(self):
        """録音開始"""
        if self.audio is None:
            print("PyAudioが初期化されていません")
            return False
            
        try:
            # ALSAエラーを強力に抑制
            import os
            import contextlib
            os.environ['ALSA_PCM_CARD'] = '0'
            os.environ['ALSA_PCM_DEVICE'] = '0'
            os.environ['ALSA_CONFIG_PATH'] = '/dev/null'
            os.environ['ALSA_PCM_NAME'] = 'null'
            
            self.is_recording = True
            
            # エラー出力を抑制してストリームを開く
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    self.stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=self._audio_callback
                    )
                    self.stream.start_stream()
            
            print("Recording started...")
            return True
        except OSError as e:
            print(f"マイクアクセスエラー: {e}")
            print("Dockerコンテナ内ではマイクアクセスが制限されています")
            return False
        except Exception as e:
            print(f"録音開始エラー: {e}")
            return False
    
    def stop_recording(self):
        """録音停止"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("Recording stopped.")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音声データのコールバック"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def get_audio_data(self, duration_seconds: float) -> np.ndarray:
        """指定時間の音声データを取得"""
        audio_chunks = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
            except queue.Empty:
                break
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            return np.array([])
    
    def close(self):
        """リソースの解放"""
        self.stop_recording()
        if self.audio is not None:
            self.audio.terminate()


class RealTimeASR:
    """リアルタイム音声認識クラス"""
    
    def __init__(self, 
                 model,
                 audio_preprocessor,
                 text_preprocessor,
                 device='cpu',
                 buffer_duration=3.0,
                 overlap_duration=1.0):
        self.model = model.to(device)
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.device = device
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        
        # 音声バッファ
        self.audio_buffer = np.array([])
        self.sample_rate = audio_preprocessor.sample_rate
        
        # 録音機
        self.recorder = AudioRecorder(sample_rate=self.sample_rate)
    
    def start_realtime_recognition(self):
        """リアルタイム認識開始"""
        self.recorder.start_recording()
        
        try:
            while True:
                # 音声データを取得
                audio_data = self.recorder.get_audio_data(self.buffer_duration)
                
                if len(audio_data) > 0:
                    # 音声認識を実行
                    text = self.recognize_audio(audio_data)
                    if text.strip():
                        print(f"Recognized: {text}")
                
                # オーバーラップ部分を保持
                overlap_samples = int(self.overlap_duration * self.sample_rate)
                if len(audio_data) > overlap_samples:
                    self.audio_buffer = audio_data[-overlap_samples:]
                else:
                    self.audio_buffer = audio_data
                    
        except KeyboardInterrupt:
            print("Stopping real-time recognition...")
        finally:
            self.recorder.close()
    
    def recognize_audio(self, audio_data: np.ndarray) -> str:
        """音声データを認識"""
        if len(audio_data) == 0:
            return ""
        
        # 音声の前処理
        audio_features = self.audio_preprocessor.preprocess_audio_from_array(
            audio_data, self.sample_rate
        )
        
        # バッチ次元を追加
        audio_features = audio_features.unsqueeze(0).to(self.device)
        
        # 推論
        with torch.no_grad():
            logits = self.model(audio_features)
            decoded_sequences = self.model.decode(logits)
        
        # テキストに変換
        if decoded_sequences:
            text_ids = decoded_sequences[0]
            text = self.text_preprocessor.ids_to_text(text_ids)
            return text
        
        return ""


class AudioProcessor:
    """音声処理ユーティリティ"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """音声ファイルを読み込み"""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([]), target_sr
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sr: int = 16000):
        """音声ファイルを保存"""
        try:
            sf.write(file_path, audio, sr)
            print(f"Audio saved to {file_path}")
        except Exception as e:
            print(f"Error saving audio file {file_path}: {e}")
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """音声の正規化"""
        if len(audio) == 0:
            return audio
        
        # RMS正規化
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms * 0.1
        
        return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """無音部分の除去"""
        if len(audio) == 0:
            return audio
        
        # デシベルに変換
        db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # 閾値以上の部分を検出
        mask = db > threshold_db
        
        # 開始と終了位置を検出
        start = np.argmax(mask)
        end = len(audio) - np.argmax(mask[::-1])
        
        return audio[start:end]


class ModelManager:
    """モデル管理クラス"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model_info(self, model, model_name: str, info: dict):
        """モデル情報を保存"""
        info_path = os.path.join(self.model_dir, f"{model_name}_info.json")
        
        model_info = {
            "model_name": model_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **info
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to {info_path}")
    
    def load_model_info(self, model_name: str) -> dict:
        """モデル情報を読み込み"""
        info_path = os.path.join(self.model_dir, f"{model_name}_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def list_models(self) -> List[str]:
        """保存されたモデルの一覧を取得"""
        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pth'):
                models.append(file)
        return models


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.inference_times = []
        self.audio_lengths = []
    
    def record_inference(self, inference_time: float, audio_length: float):
        """推論時間を記録"""
        self.inference_times.append(inference_time)
        self.audio_lengths.append(audio_length)
    
    def get_statistics(self) -> dict:
        """統計情報を取得"""
        if not self.inference_times:
            return {}
        
        inference_times = np.array(self.inference_times)
        audio_lengths = np.array(self.audio_lengths)
        
        # リアルタイム比を計算
        realtime_ratios = audio_lengths / inference_times
        
        return {
            "total_inferences": len(self.inference_times),
            "avg_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "avg_realtime_ratio": np.mean(realtime_ratios),
            "min_realtime_ratio": np.min(realtime_ratios),
            "max_realtime_ratio": np.max(realtime_ratios)
        }
    
    def print_statistics(self):
        """統計情報を表示"""
        stats = self.get_statistics()
        if not stats:
            print("No inference data recorded.")
            return
        
        print("\n=== Performance Statistics ===")
        print(f"Total inferences: {stats['total_inferences']}")
        print(f"Average inference time: {stats['avg_inference_time']:.4f}s")
        print(f"Inference time std: {stats['std_inference_time']:.4f}s")
        print(f"Inference time range: {stats['min_inference_time']:.4f}s - {stats['max_inference_time']:.4f}s")
        print(f"Average real-time ratio: {stats['avg_realtime_ratio']:.2f}x")
        print(f"Real-time ratio range: {stats['min_realtime_ratio']:.2f}x - {stats['max_realtime_ratio']:.2f}x")
        print("=" * 30)


def create_sample_audio_data(num_samples: int = 10, duration: float = 3.0) -> List[Tuple[np.ndarray, str]]:
    """サンプル音声データを生成（テスト用）"""
    sample_rate = 16000
    samples = []
    
    # サンプルテキスト（より短く、認識しやすいもの）
    texts = [
        "hello",
        "world",
        "test",
        "audio",
        "speech",
        "recognition",
        "model",
        "training",
        "data",
        "sample"
    ]
    
    for i in range(min(num_samples, len(texts))):
        # より現実的な音声波形を生成
        audio_length = int(duration * sample_rate)
        
        # 複数の周波数成分を持つ音声を生成
        t = np.linspace(0, duration, audio_length)
        
        # 基本周波数（人間の声の範囲）
        base_freq = np.random.uniform(100, 300)  # 100-300Hz
        
        # 複数の調波を追加
        audio = np.zeros(audio_length)
        for harmonic in range(1, 6):  # 1次から5次調波
            freq = base_freq * harmonic
            amplitude = 0.1 / harmonic  # 高調波ほど振幅が小さくなる
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # ノイズを追加（現実的な音声に近づける）
        noise = np.random.randn(audio_length) * 0.02
        audio += noise
        
        # エンベロープを適用（音声の開始と終了を滑らかに）
        envelope = np.exp(-t / (duration * 0.1))  # 指数減衰
        audio *= envelope
        
        # 正規化
        audio = AudioProcessor.normalize_audio(audio)
        
        # 振幅を調整
        audio *= 0.3
        
        samples.append((audio, texts[i]))
    
    return samples


def save_sample_dataset(samples: List[Tuple[np.ndarray, str]], output_dir: str):
    """サンプルデータセットを保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # メタデータファイル
    metadata = []
    
    for i, (audio, text) in enumerate(samples):
        # 音声ファイルを保存
        audio_path = f"sample_{i:03d}.wav"
        full_audio_path = os.path.join(output_dir, audio_path)
        AudioProcessor.save_audio(audio, full_audio_path)
        
        # メタデータに追加
        metadata.append({
            "audio": audio_path,
            "text": text
        })
    
    # メタデータファイルを保存
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sample dataset saved to {output_dir}")
    print(f"Generated {len(samples)} audio files")
