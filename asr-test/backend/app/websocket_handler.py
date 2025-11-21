# backend/app/websocket_handler.py
import asyncio
import json
import logging
import base64
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    from .realtime_processor import RealtimeAudioProcessor
except ImportError as e:
    logger.error(f"Failed to import RealtimeAudioProcessor: {e}")
    # フォールバック実装
    class RealtimeAudioProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def add_audio_chunk(self, *args, **kwargs):
            return True
        def get_audio_buffer(self):
            return None
        def clear_buffer(self):
            pass
from .models.interface import BaseASRModel
from .state import _model_cache

logger = logging.getLogger("websocket")

class RealtimeWebSocketHandler:
    """リアルタイム音声認識用のWebSocketハンドラー"""
    
    def __init__(self, model_name: str = "conformer"):
        self.model_name = model_name
        try:
            self.processor = RealtimeAudioProcessor()
        except Exception as e:
            logger.error(f"Failed to initialize RealtimeAudioProcessor: {e}")
            self.processor = None
        self.model: Optional[BaseASRModel] = None
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_buffers: Dict[str, list] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """WebSocket接続を確立"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_buffers[client_id] = []
        
        logger.info(f"WebSocket connection established for client: {client_id}")
        
        # モデルをロード（初回のみ）
        if self.model is None:
            await self._load_model()
    
    async def disconnect(self, client_id: str):
        """WebSocket接続を切断"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_buffers:
            del self.connection_buffers[client_id]
            
        logger.info(f"WebSocket connection closed for client: {client_id}")
    
    async def _load_model(self):
        """モデルをロード"""
        try:
            from .api import get_model_for_inference
            self.model = get_model_for_inference(self.model_name)
            logger.info(f"Model {self.model_name} loaded for realtime inference")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def handle_audio_chunk(self, client_id: str, audio_data: str):
        """音声チャンクを処理"""
        try:
            # Base64デコード
            audio_bytes = base64.b64decode(audio_data)
            
            # WebM/Opus形式の場合は、まずPCMに変換する必要がある
            # 簡易的な実装として、バイトデータを直接テンソルに変換
            if len(audio_bytes) > 0:
                # バイトデータをテンソルに変換（16-bit PCM想定）
                audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
            else:
                logger.warning(f"Empty audio data for client {client_id}")
                return
            
            # バッファに追加
            self.connection_buffers[client_id].append(audio_tensor)
            
            # バッファサイズを制限（メモリリーク防止）
            max_buffer_size = 10  # 10秒分
            if len(self.connection_buffers[client_id]) > max_buffer_size:
                self.connection_buffers[client_id] = self.connection_buffers[client_id][-max_buffer_size:]
            
            # 推論実行
            await self._process_audio_buffer(client_id)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for client {client_id}: {e}")
            await self._send_error(client_id, f"Audio processing error: {str(e)}")
    
    async def _process_audio_buffer(self, client_id: str):
        """音声バッファを処理して推論実行"""
        if not self.model or client_id not in self.connection_buffers:
            return
            
        buffer = self.connection_buffers[client_id]
        if len(buffer) < 1:  # 最低1秒分のデータが必要
            return
        
        try:
            # バッファを結合
            combined_audio = torch.cat(buffer, dim=0)
            
            # 音声の長さをチェック
            if combined_audio.numel() < 8000:  # 0.5秒未満はスキップ
                return
            
            # 推論実行
            start_time = time.time()
            transcription = self.model.inference(combined_audio)
            inference_time = (time.time() - start_time) * 1000
            
            # 結果を送信
            if transcription.strip():
                await self._send_result(client_id, transcription, inference_time)
                
        except Exception as e:
            logger.error(f"Error during inference for client {client_id}: {e}")
            await self._send_error(client_id, f"Inference error: {str(e)}")
    
    async def _send_result(self, client_id: str, transcription: str, inference_time: float):
        """推論結果を送信"""
        if client_id not in self.active_connections:
            return
            
        message = {
            "type": "transcription",
            "text": transcription,
            "timestamp": time.time(),
            "inference_time_ms": inference_time
        }
        
        try:
            await self.active_connections[client_id].send_text(json.dumps(message))
            logger.debug(f"Sent transcription to client {client_id}: {transcription}")
        except Exception as e:
            logger.error(f"Failed to send result to client {client_id}: {e}")
    
    async def _send_error(self, client_id: str, error_message: str):
        """エラーメッセージを送信"""
        if client_id not in self.active_connections:
            return
            
        message = {
            "type": "error",
            "message": error_message,
            "timestamp": time.time()
        }
        
        try:
            await self.active_connections[client_id].send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send error to client {client_id}: {e}")
    
    async def _send_status(self, client_id: str, status: str):
        """ステータスメッセージを送信"""
        if client_id not in self.active_connections:
            return
            
        message = {
            "type": "status",
            "status": status,
            "timestamp": time.time()
        }
        
        try:
            await self.active_connections[client_id].send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send status to client {client_id}: {e}")

# グローバルハンドラーインスタンス
realtime_handler = RealtimeWebSocketHandler()

async def handle_realtime_websocket(websocket: WebSocket):
    """リアルタイム音声認識のWebSocketエンドポイント"""
    client_id = f"client_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"WebSocket connection attempt from client: {client_id}")
        await realtime_handler.connect(websocket, client_id)
        await realtime_handler._send_status(client_id, "connected")
        logger.info(f"WebSocket connection established for client: {client_id}")
        
        while True:
            # メッセージを受信
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio_chunk":
                audio_data = message.get("data")
                if audio_data:
                    await realtime_handler.handle_audio_chunk(client_id, audio_data)
            elif message.get("type") == "ping":
                await realtime_handler._send_status(client_id, "pong")
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        await realtime_handler.disconnect(client_id)
