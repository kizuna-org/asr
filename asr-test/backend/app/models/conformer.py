# backend/app/models/conformer.py
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, Any, List

from .interface import BaseASRModel

class ConformerASRModel(BaseASRModel):
    """Hugging FaceのWav2Vec2-Conformerモデルをラップするクラス"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = config.get("huggingface_model_name", "facebook/wav2vec2-conformer-rel-pos-large-960h-ft-librispeech-ft")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        waveforms: (Batch, Time)
        texts: List[str]
        """
        waveforms = batch["waveforms"]
        texts: List[str] = batch["texts"]

        # Processor に渡す前に、各サンプルの1次元配列(list of 1D)へ明示変換する
        # これにより、Processor が (batch, time) ではなく「単一サンプルの2次元」だと誤解して
        # 4次元入力になってしまう問題を防ぐ
        if isinstance(waveforms, torch.Tensor):
            # waveforms は (batch, time) であることを想定
            waveform_list = [wf.detach().cpu().numpy() for wf in waveforms]
        else:
            waveform_list = waveforms

        # Processorで前処理とラベルエンコード（CPUで実行）
        processed = self.processor(
            waveform_list, sampling_rate=16000, return_tensors="pt", padding=True
        )

        with self.processor.as_target_processor():
            labels = self.processor(texts, return_tensors="pt", padding=True).input_ids

        # 入力テンソルをモデルと同じデバイスへ
        input_values = processed.input_values.to(self.model.device)
        labels = labels.to(self.model.device)

        outputs = self.model(
            input_values=input_values,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def inference(self, waveform: torch.Tensor) -> str:
        """単一の音声波形から文字起こしを行う。"""
        # 入力は1Dテンソル想定。2D(チャネル, 長さ)の可能性があればモノラル化
        if waveform.dim() == 2:
            # (channels, time) -> mono 1D
            waveform = waveform.mean(dim=0)

        # モデルへの入力は list[np.ndarray] で (time,) となるようにする
        if isinstance(waveform, torch.Tensor):
            waveform_list = [waveform.detach().cpu().numpy()]
        else:
            waveform_list = [waveform]

        processed = self.processor(waveform_list, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = processed.input_values.to(self.model.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int):
        """Hugging Faceモデルの保存形式に合わせる"""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        super().save_checkpoint(f"{path}/optimizer.pt", optimizer, epoch)

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """Hugging Faceモデルの読み込み形式に合わせる"""
        self.model = Wav2Vec2ForCTC.from_pretrained(path)
        self.processor = Wav2Vec2Processor.from_pretrained(path)
        try:
            return super().load_checkpoint(f"{path}/optimizer.pt", optimizer)
        except FileNotFoundError:
            return 0
