import json
import asyncio
from typing import List, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import torch
import torchaudio
from . import config_loader

router = APIRouter()
# 既存の API ロガーに合わせる
logger = logging.getLogger("asr-api")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """接続している全てのクライアントにメッセージを送信する"""
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

    def broadcast_sync(self, message: Dict):
        """同期関数からブロードキャストを実行するためのラッパー"""
        try:
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
            else:
                self.loop.run_until_complete(self.broadcast(message))
        except Exception as e:
            # エラーが発生しても学習プロセスを継続する
            print(f"WebSocket broadcast error: {e}")
            pass

# ConnectionManagerのインスタンスを作成
manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続のエンドポイント

    拡張: クライアントからの制御メッセージ(JSON)と音声バイナリ(PCM/float)を受信し、
    バッファリングして一定間隔で部分的な文字起こし結果を返す。
    プロトコル(暫定):
      - Text(JSON): {"type":"start","model_name":"conformer","sample_rate":48000,"format":"f32"}
      - Text(JSON): {"type":"stop"}
      - Binary: 音声フレーム(PCM16LE もしくは float32 リトルエンディアン)。
    サーバ側は 16kHz mono にリサンプリングしてバッファに蓄積し、
    約1秒毎にモデル推論を走らせて partial を返す。
    """
    await manager.connect(websocket)
    logger.info("WS connected: client=%s", getattr(websocket, "client", None))
    print("[WS] connected", getattr(websocket, "client", None))

    # ストリーミング状態
    model = None  # type: Optional[torch.nn.Module]
    input_sample_rate = 16000
    input_dtype = "i16"  # "i16" or "f32"
    resampler = None  # type: Optional[torchaudio.transforms.Resample]
    buffer = []  # list[torch.Tensor]
    last_infer_time = 0.0

    def get_model_for_inference(model_name: str):
        # api.py との循環依存を避けるため、ここで簡易実装
        from .state import _model_cache
        if model_name not in _model_cache:
            model_config = config_loader.get_model_config(model_name)
            if not model_config:
                logger.error("Model config not found: %s", model_name)
                raise RuntimeError(f"Model '{model_name}' not found in config.")
            import importlib
            class_name = f"{model_name.capitalize()}ASRModel"
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), class_name)
            logger.info("Loading model class: %s", class_name)
            m = ModelClass(model_config)
            m.eval()
            _model_cache[model_name] = m
        return _model_cache[model_name]

    try:
        while True:
            message = await websocket.receive()
            logger.debug("WS received message type=%s keys=%s", message.get("type"), list(message.keys()))
            mtype = message.get("type")

            if mtype == "websocket.disconnect":
                break

            if mtype == "websocket.receive":
                if "text" in message and message["text"] is not None:
                    try:
                        data = json.loads(message["text"])
                    except Exception:
                        logger.exception("Invalid JSON from client")
                        await websocket.send_text(json.dumps({"type": "error", "payload": {"message": "Invalid JSON"}}))
                        continue

                    if data.get("type") == "start":
                        model_name = data.get("model_name", "conformer")
                        input_sample_rate = int(data.get("sample_rate", 16000))
                        input_dtype = data.get("format", "i16")
                        if input_dtype not in ("i16", "f32"):
                            input_dtype = "i16"
                        # モデル・リサンプラ初期化
                        logger.info("Start streaming: model=%s sr=%s fmt=%s", model_name, input_sample_rate, input_dtype)
                        print(f"[WS] start: model={model_name} sr={input_sample_rate} fmt={input_dtype}")
                        model = get_model_for_inference(model_name)
                        if input_sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=16000)
                            logger.debug("Resampler initialized: %s -> 16000", input_sample_rate)
                            print(f"[WS] resampler: {input_sample_rate}->16000")
                        else:
                            resampler = None
                        buffer.clear()
                        last_infer_time = asyncio.get_event_loop().time()
                        await websocket.send_text(json.dumps({"type": "status", "payload": {"status": "ready"}}))
                        continue

                    if data.get("type") == "stop":
                        # 最終推論を実行
                        if model is not None and len(buffer) > 0:
                            waveform = torch.cat(buffer) if len(buffer) > 1 else buffer[0]
                            try:
                                transcription = model.inference(waveform)
                                await websocket.send_text(json.dumps({"type": "final", "payload": {"text": transcription}}))
                            except Exception as e:
                                logger.exception("Final inference error")
                                await websocket.send_text(json.dumps({"type": "error", "payload": {"message": str(e)}}))
                        buffer.clear()
                        model = None
                        logger.info("Stop streaming and cleared state")
                        print("[WS] stop streaming")
                        await websocket.send_text(json.dumps({"type": "status", "payload": {"status": "stopped"}}))
                        continue

                if "bytes" in message and message["bytes"] is not None:
                    # 音声フレーム受信
                    if model is None:
                        # start 前のフレームは無視
                        logger.debug("Audio frame ignored because streaming not started")
                        continue
                    raw = message["bytes"]
                    if not raw:
                        continue
                    logger.debug("Received audio chunk: %d bytes (fmt=%s)", len(raw), input_dtype)
                    # print too to ensure visibility in container logs
                    print(f"[WS] audio chunk {len(raw)} bytes fmt={input_dtype}")
                    # バイナリをテンソル化
                    if input_dtype == "i16":
                        tensor = torch.frombuffer(raw, dtype=torch.int16).to(torch.float32) / 32768.0
                    elif input_dtype == "f32":
                        tensor = torch.frombuffer(raw, dtype=torch.float32)
                    else:
                        # フォーマット未指定時は自動判別: 2バイト単位ならi16、4バイト単位ならf32
                        if len(raw) % 4 == 0:
                            tensor = torch.frombuffer(raw, dtype=torch.float32)
                            input_dtype = "f32"
                        else:
                            tensor = torch.frombuffer(raw, dtype=torch.int16).to(torch.float32) / 32768.0
                            input_dtype = "i16"

                    # mono 前提。必要ならチャネル分離はクライアント側で実施。
                    if resampler is not None:
                        tensor = resampler(tensor.unsqueeze(0)).squeeze(0)

                    buffer.append(tensor)
                    total_samples = sum(t.numel() for t in buffer)
                    logger.debug("Buffered samples: %d", total_samples)
                    print(f"[WS] buffered samples={total_samples}")

                    # 一定間隔で部分推論
                    now = asyncio.get_event_loop().time()
                    if now - last_infer_time >= 1.0:
                        last_infer_time = now
                        try:
                            waveform = torch.cat(buffer) if len(buffer) > 1 else buffer[0]
                            logger.debug("Running partial inference on %d samples", waveform.numel())
                            print(f"[WS] partial inference on samples={waveform.numel()}")
                            transcription = model.inference(waveform)
                            await websocket.send_text(json.dumps({"type": "partial", "payload": {"text": transcription}}))
                        except Exception as e:
                            logger.exception("Partial inference error")
                            await websocket.send_text(json.dumps({"type": "error", "payload": {"message": str(e)}}))
                        # バッファは継続的に蓄積（全体での最良仮説前提）。必要に応じて末尾数秒に切り詰め可能。
                        # 例: 最新 20 秒に制限
                        max_seconds = 20
                        max_samples = 16000 * max_seconds
                        total = sum(t.numel() for t in buffer)
                        if total > max_samples:
                            # 先頭から削る
                            remain = total - max_samples
                            new_buf = []
                            for t in buffer:
                                if remain <= 0:
                                    new_buf.append(t)
                                    continue
                                if t.numel() <= remain:
                                    remain -= t.numel()
                                    continue
                                new_buf.append(t[remain:])
                                remain = 0
                            buffer = new_buf
                            logger.debug("Trimmed buffer to last %d seconds", max_seconds)
                        continue
    except WebSocketDisconnect:
        logger.info("WS disconnected: client=%s", getattr(websocket, "client", None))
    finally:
        manager.disconnect(websocket)
