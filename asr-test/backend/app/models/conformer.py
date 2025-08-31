# backend/app/models/conformer.py
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, Any

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
        バッチデータを受け取り、CTC損失を計算して返す。
        モデル内部で損失計算が行われる。
        """
        # Hugging Faceモデルは、labelsが渡されると自動的に損失を計算する
        # processorでテキストをIDに変換する必要があるが、dataloaderで処理済み
        outputs = self.model(
            input_values=batch["waveforms"],
            attention_mask=None, # マスクは通常不要
            labels=batch["token_ids"]
        )
        return outputs.loss

    @torch.no_grad()
    def inference(self, waveform: torch.Tensor) -> str:
        """
        単一の音声波形から文字起こしを行う。
        """
        # 入力はバッチ形式である必要があるため、次元を追加
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # モデルによる推論
        logits = self.model(waveform).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # IDをテキストにデコード
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int):
        """Hugging Faceモデルの保存形式に合わせる"""
        # モデルとプロセッサを保存
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        # オプティマイザなどの追加情報を保存
        super().save_checkpoint(f"{path}/optimizer.pt", optimizer, epoch)

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """Hugging Faceモデルの読み込み形式に合わせる"""
        # モデルとプロセッサを読み込み
        self.model = Wav2Vec2ForCTC.from_pretrained(path)
        self.processor = Wav2Vec2Processor.from_pretrained(path)

        # オプティマイザなどの追加情報を読み込み
        try:
            return super().load_checkpoint(f"{path}/optimizer.pt", optimizer)
        except FileNotFoundError:
            # optimizer.pt がない場合はエポック0から開始
            return 0
