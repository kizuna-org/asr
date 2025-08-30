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
        if pyaudio is not None:
            try:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stderr(devnull):
                        self.audio = pyaudio.PyAudio()
            except Exception as e:
                print(f"PyAudio初期化エラー: {e}")
                self.audio = None
        else:
            self.audio = None
            
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
    
    def start_recording(self):
        """録音開始"""
        if self.audio is None:
            return False
        
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            return True
        except Exception as e:
            print(f"録音開始エラー: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音声データのコールバック"""
        if self.is_recording:
            try:
                # バイトデータをfloat32に変換
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.audio_queue.put(audio_data)
            except Exception as e:
                print(f"音声データ処理エラー: {e}")
        return (None, pyaudio.paContinue)
    
    def get_audio_data(self, duration: float) -> np.ndarray:
        """指定時間分の音声データを取得"""
        if not self.is_recording:
            return np.array([])
        
        target_samples = int(duration * self.sample_rate)
        audio_chunks = []
        total_samples = 0
        
        # 指定時間分のデータを収集
        while total_samples < target_samples and self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
                total_samples += len(chunk)
            except queue.Empty:
                break
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            return np.array([])
    
    def stop_recording(self):
        """録音停止"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def close(self):
        """リソース解放"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


class RealTimeASR:
    """リアルタイム音声認識クラス（改善版）"""
    
    def __init__(self, 
                 model,
                 audio_preprocessor,
                 text_preprocessor,
                 device='cpu',
                 buffer_duration=3.0,
                 overlap_duration=1.0,
                 min_confidence=0.1):
        self.model = model.to(device)
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.device = device
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self.min_confidence = min_confidence
        
        # 音声バッファ
        self.audio_buffer = np.array([])
        self.sample_rate = audio_preprocessor.sample_rate
        
        # 認識結果の履歴
        self.recognition_history = []
        self.max_history = 10
        
        # 録音機
        self.recorder = AudioRecorder(sample_rate=self.sample_rate)
    
    def _apply_voice_activity_detection(self, audio_data: np.ndarray) -> bool:
        """
        音声活動検出（VAD）
        """
        if len(audio_data) == 0:
            return False
        
        # エネルギーベースのVAD
        energy = np.mean(audio_data ** 2)
        threshold = 0.001  # 調整可能
        
        # ゼロクロスレートベースのVAD
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zcr_threshold = len(audio_data) * 0.1  # 調整可能
        
        is_speech = energy > threshold and zero_crossings > zcr_threshold
        
        return is_speech
    
    def _apply_confidence_filtering(self, text: str, logits: torch.Tensor) -> bool:
        """
        信頼度フィルタリング
        """
        if not text.strip():
            return False
        
        # ロジットの最大確率を信頼度として使用
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        avg_confidence = torch.mean(max_probs).item()
        
        return avg_confidence > self.min_confidence
    
    def _apply_result_smoothing(self, text: str) -> str:
        """
        認識結果の平滑化
        """
        if not text.strip():
            return ""
        
        # 履歴に追加
        self.recognition_history.append(text)
        if len(self.recognition_history) > self.max_history:
            self.recognition_history.pop(0)
        
        # 履歴が少ない場合はそのまま返す
        if len(self.recognition_history) < 3:
            return text
        
        # 最近の結果を比較して、一貫性をチェック
        recent_results = self.recognition_history[-3:]
        
        # 最も頻繁に出現する結果を選択
        from collections import Counter
        counter = Counter(recent_results)
        most_common = counter.most_common(1)[0]
        
        # 信頼度が高い場合のみ平滑化を適用
        if most_common[1] >= 2:  # 3回中2回以上同じ結果
            smoothed_text = most_common[0]
            return smoothed_text
        
        return text
    
    def start_realtime_recognition(self):
        """リアルタイム認識開始"""
        self.recorder.start_recording()
        
        try:
            while True:
                # 音声データを取得
                audio_data = self.recorder.get_audio_data(self.buffer_duration)
                
                if len(audio_data) > 0:
                    # 音声活動検出
                    if self._apply_voice_activity_detection(audio_data):
                        # 音声認識を実行
                        text = self.recognize_audio(audio_data)
                        if text.strip():
                            # 信頼度フィルタリングと平滑化
                            if self._apply_confidence_filtering(text, self.model.last_logits):
                                smoothed_text = self._apply_result_smoothing(text)
                                print(f"Recognized: {smoothed_text}")
                
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
        """音声データを認識（改善版）"""
        if len(audio_data) == 0:
            return ""
        
        # モデルが学習済みかチェック
        if hasattr(self.model, 'is_trained') and not self.model.is_trained():
            print("⚠️ 警告: モデルが学習されていません。認識結果は不正確です。")
            return "[未学習モデル]"
        
        # 音声の前処理
        audio_features = self.audio_preprocessor.preprocess_audio_from_array(
            audio_data, self.sample_rate
        )
        
        # バッチ次元を追加
        audio_features = audio_features.unsqueeze(0).to(self.device)
        
        # 推論
        with torch.no_grad():
            logits = self.model(audio_features)
            # ロジットを保存（信頼度計算用）
            self.model.last_logits = logits
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
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """音声の正規化"""
        if len(audio) == 0:
            return audio
        
        # 最大値で正規化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    @staticmethod
    def apply_preemphasis(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
        """プリエンファシスフィルタ"""
        if len(audio) < 2:
            return audio
        
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]
        for i in range(1, len(audio)):
            emphasized[i] = audio[i] - coefficient * audio[i-1]
        
        return emphasized
    
    @staticmethod
    def remove_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """無音部分の除去"""
        if len(audio) == 0:
            return audio
        
        # エネルギーベースの無音検出
        energy = np.mean(audio ** 2)
        if energy < threshold:
            return np.array([])
        
        # 無音部分の検出と除去
        frame_length = 1024
        hop_length = 512
        
        non_silent_frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_energy = np.mean(frame ** 2)
            if frame_energy > threshold:
                non_silent_frames.append(i)
        
        if non_silent_frames:
            start_idx = non_silent_frames[0]
            end_idx = non_silent_frames[-1] + frame_length
            return audio[start_idx:end_idx]
        
        return audio


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.inference_times = []
        self.audio_durations = []
        self.realtime_ratios = []
        self.max_history = 100
    
    def record_inference(self, inference_time: float, audio_duration: float):
        """推論時間を記録"""
        self.inference_times.append(inference_time)
        self.audio_durations.append(audio_duration)
        
        # リアルタイム比を計算
        if inference_time > 0:
            realtime_ratio = audio_duration / inference_time
            self.realtime_ratios.append(realtime_ratio)
        
        # 履歴を制限
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)
            self.audio_durations.pop(0)
            if self.realtime_ratios:
                self.realtime_ratios.pop(0)
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        if not self.inference_times:
            return {
                "avg_inference_time": 0.0,
                "avg_audio_duration": 0.0,
                "avg_realtime_ratio": 0.0,
                "total_inferences": 0
            }
        
        return {
            "avg_inference_time": np.mean(self.inference_times),
            "avg_audio_duration": np.mean(self.audio_durations),
            "avg_realtime_ratio": np.mean(self.realtime_ratios) if self.realtime_ratios else 0.0,
            "total_inferences": len(self.inference_times)
        }


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
        base_freq = np.random.uniform(150, 250)  # 150-250Hz（より現実的な声の周波数）
        
        # 複数の調波を追加
        audio = np.zeros(audio_length)
        for harmonic in range(1, 8):  # 1次から7次調波
            freq = base_freq * harmonic
            amplitude = 0.15 / harmonic  # 高調波ほど振幅が小さくなる
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # フォルマント（母音の特徴的な周波数）を追加
        formant_freqs = [500, 1500, 2500]  # 一般的な母音のフォルマント
        for formant_freq in formant_freqs:
            formant_amp = 0.05
            audio += formant_amp * np.sin(2 * np.pi * formant_freq * t)
        
        # ノイズを追加（現実的な音声に近づける）
        noise = np.random.randn(audio_length) * 0.01
        audio += noise
        
        # エンベロープを適用（音声の開始と終了を滑らかに）
        # アタック時間（音の立ち上がり）
        attack_time = 0.05  # 50ms
        attack_samples = int(attack_time * sample_rate)
        
        # リリース時間（音の減衰）
        release_time = 0.1  # 100ms
        release_samples = int(release_time * sample_rate)
        
        # エンベロープ作成
        envelope = np.ones(audio_length)
        
        # アタック部分（線形増加）
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # リリース部分（指数減衰）
        if release_samples > 0:
            release_start = audio_length - release_samples
            if release_start > attack_samples:
                envelope[release_start:] = np.exp(-np.linspace(0, 3, release_samples))
        
        audio *= envelope
        
        # 正規化
        audio = AudioProcessor.normalize_audio(audio)
        
        # 振幅を調整（より現実的な音量）
        audio *= 0.4
        
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
