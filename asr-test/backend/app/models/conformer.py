# backend/app/models/conformer.py
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, Any, List
import logging
import traceback

from .interface import BaseASRModel

# モデル専用のロガー
logger = logging.getLogger("model")

class ConformerASRModel(BaseASRModel):
    """Hugging FaceのWav2Vec2-Conformerモデルをラップするクラス"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = config.get("huggingface_model_name", "facebook/wav2vec2-conformer-rel-pos-large-960h-ft-librispeech-ft")

        logger.info("Initializing ConformerASRModel", 
                   extra={"extra_fields": {"component": "model", "action": "init", "model_name": model_name}})
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        logger.info("ConformerASRModel initialized successfully", 
                   extra={"extra_fields": {"component": "model", "action": "init_complete", "model_name": model_name}})

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
        logger.info("Starting model inference", 
                   extra={"extra_fields": {"component": "model", "action": "inference_start", 
                                         "waveform_shape": waveform.shape, "dtype": str(waveform.dtype)}})
        
        # 入力は1Dテンソル想定。2D(チャネル, 長さ)の可能性があればモノラル化
        if waveform.dim() == 2:
            # (channels, time) -> mono 1D
            waveform = waveform.mean(dim=0)
            logger.debug("Converted to mono", 
                        extra={"extra_fields": {"component": "model", "action": "mono_conversion", 
                                              "new_shape": waveform.shape}})

        # 音声データが短すぎる場合は空文字を返す
        if waveform.numel() < 1600:  # 0.1秒未満
            logger.debug("Audio too short for inference", 
                        extra={"extra_fields": {"component": "model", "action": "audio_too_short", 
                                              "samples": waveform.numel(), "duration_sec": waveform.numel() / 16000}})
            return ""

        # モデルへの入力は list[np.ndarray] で (time,) となるようにする
        if isinstance(waveform, torch.Tensor):
            waveform_list = [waveform.detach().cpu().numpy()]
        else:
            waveform_list = [waveform]
        
        logger.debug("Waveform prepared for processing", 
                    extra={"extra_fields": {"component": "model", "action": "waveform_prepared", 
                                          "list_length": len(waveform_list), "array_shape": waveform_list[0].shape}})

        try:
            processed = self.processor(waveform_list, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = processed.input_values.to(self.model.device)
            
            logger.debug("Input processed", 
                        extra={"extra_fields": {"component": "model", "action": "input_processed", 
                                              "input_shape": input_values.shape}})

            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            logger.debug("Model forward pass completed", 
                        extra={"extra_fields": {"component": "model", "action": "forward_pass", 
                                              "logits_shape": logits.shape, "predicted_ids_shape": predicted_ids.shape}})
            
            # 空のトークンを除去
            predicted_ids = predicted_ids[predicted_ids != self.processor.tokenizer.pad_token_id]
            if len(predicted_ids) == 0:
                logger.debug("No valid tokens found", 
                            extra={"extra_fields": {"component": "model", "action": "no_valid_tokens"}})
                return ""
                
            transcription = self.processor.batch_decode(predicted_ids.unsqueeze(0))[0]
            
            logger.info("Inference completed successfully", 
                       extra={"extra_fields": {"component": "model", "action": "inference_complete", 
                                             "transcription": transcription, "tokens_count": len(predicted_ids)}})
            return transcription
        except Exception as e:
            logger.error("Error during inference", 
                        extra={"extra_fields": {"component": "model", "action": "inference_error", 
                                              "error": str(e), "traceback": traceback.format_exc()}})
            return ""

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
