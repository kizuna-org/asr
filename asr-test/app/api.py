from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import numpy as np
import json
import base64
import asyncio
import time
from typing import List, Dict, Any
import logging

from app.model import LightweightASRModel, FastASRModel, CHAR_TO_ID, ID_TO_CHAR
from app.dataset import AudioPreprocessor, TextPreprocessor
from app.utils import AudioProcessor, PerformanceMonitor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="リアルタイム音声認識API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
model = None
audio_preprocessor = None
text_preprocessor = None
performance_monitor = PerformanceMonitor()
device = "cpu"

# WebSocket接続管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket接続確立: {len(self.active_connections)}接続")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket接続切断: {len(self.active_connections)}接続")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 接続が切れている場合は削除
                self.active_connections.remove(connection)

manager = ConnectionManager()

def initialize_model(model_type: str = "FastASRModel", hidden_dim: int = 64):
    """モデルの初期化"""
    global model, audio_preprocessor, text_preprocessor, device
    
    try:
        # デバイス設定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用デバイス: {device}")
        
        # 前処理器の初期化
        audio_preprocessor = AudioPreprocessor()
        text_preprocessor = TextPreprocessor()
        
        # モデルの初期化
        if model_type.startswith("Fast"):
            model = FastASRModel(
                hidden_dim=hidden_dim,
                num_classes=len(CHAR_TO_ID)
            )
        else:
            model = LightweightASRModel(
                hidden_dim=hidden_dim,
                num_layers=2,
                num_classes=len(CHAR_TO_ID)
            )
        
        model = model.to(device)
        model.eval()
        
        # パラメータ数をログ出力
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"モデル初期化完了: {params:,}パラメータ")
        
    except Exception as e:
        logger.error(f"モデル初期化エラー: {e}")
        import traceback
        logger.error(f"詳細エラー: {traceback.format_exc()}")
        # エラー時はNoneを設定
        model = None
        audio_preprocessor = None
        text_preprocessor = None

def load_model(model_path: str):
    """保存されたモデルの読み込み"""
    global model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"モデル読み込み完了: {model_path}")
        return True
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return False

def recognize_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
    """音声認識の実行（改善版）"""
    global model, audio_preprocessor, text_preprocessor, performance_monitor
    
    if model is None:
        return {"error": "モデルが初期化されていません"}
    
    try:
        start_time = time.time()
        
        # 入力データの安全チェック
        if audio_data is None or len(audio_data) == 0:
            return {
                "text": "",
                "error": "空の音声データが入力されました",
                "inference_time": 0.0,
                "audio_duration": 0.0,
                "realtime_ratio": 0.0
            }
        
        # モデルが学習済みかチェック
        if hasattr(model, 'is_trained') and not model.is_trained():
            logger.warning("⚠️ 警告: モデルが学習されていません。認識結果は不正確です。")
            return {
                "text": "[未学習モデル]",
                "warning": "モデルが学習されていません。まず学習を実行してください。",
                "inference_time": 0.0,
                "audio_duration": len(audio_data) / sample_rate,
                "realtime_ratio": 0.0
            }
        
        # 音声の前処理
        try:
            audio_features = audio_preprocessor.preprocess_audio_from_array(
                audio_data, sample_rate
            )
        except Exception as e:
            logger.error(f"音声前処理エラー: {e}")
            return {
                "text": "",
                "error": f"音声前処理に失敗しました: {str(e)}",
                "inference_time": 0.0,
                "audio_duration": len(audio_data) / sample_rate,
                "realtime_ratio": 0.0
            }
        
        # バッチ次元を追加
        audio_features = audio_features.unsqueeze(0).to(device)
        
        # 推論
        try:
            with torch.no_grad():
                logits = model(audio_features)
                # 複数のデコード方法を試行
                decoded_sequences = model.decode(logits, beam_size=5)
                
                # ビームサーチが失敗した場合は貪欲法を試行
                if not decoded_sequences or not decoded_sequences[0]:
                    decoded_sequences = model.decode(logits, beam_size=1)
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return {
                "text": "",
                "error": f"推論に失敗しました: {str(e)}",
                "inference_time": 0.0,
                "audio_duration": len(audio_data) / sample_rate,
                "realtime_ratio": 0.0
            }
        
        inference_time = time.time() - start_time
        
        # テキストに変換
        try:
            if decoded_sequences and len(decoded_sequences) > 0:
                text_ids = decoded_sequences[0]
                if text_ids is None or len(text_ids) == 0:
                    text = ""
                else:
                    text = text_preprocessor.ids_to_text(text_ids)
                    
                    # 信頼度計算
                    probs = torch.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    avg_confidence = torch.mean(max_probs).item()
                    
                    # 信頼度が低い場合は警告
                    if avg_confidence < 0.3:
                        text += " [低信頼度]"
        except Exception as e:
            logger.error(f"テキスト変換エラー: {e}")
            text = ""
        
        # パフォーマンス記録
        audio_duration = len(audio_data) / sample_rate
        performance_monitor.record_inference(inference_time, audio_duration)
        
        return {
            "text": text,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "realtime_ratio": audio_duration / inference_time if inference_time > 0 else 0,
            "confidence": avg_confidence if 'avg_confidence' in locals() else 0.0
        }
    
    except Exception as e:
        logger.error(f"音声認識エラー: {e}")
        import traceback
        logger.error(f"詳細エラー: {traceback.format_exc()}")
        return {
            "text": "",
            "error": f"音声認識に失敗しました: {str(e)}",
            "inference_time": 0.0,
            "audio_duration": len(audio_data) / sample_rate if audio_data is not None else 0.0,
            "realtime_ratio": 0.0
        }

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("アプリケーション起動中...")
    initialize_model()
    logger.info("アプリケーション起動完了")

@app.get("/")
async def get_root():
    """ルートエンドポイント - メインページを提供"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"message": "リアルタイム音声認識API", "status": "running", "note": "static/index.html not found"}

@app.get("/main")
async def get_main_page():
    """メインページの提供"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="メインページが見つかりません")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "active_connections": len(manager.active_connections)
    }

@app.post("/initialize")
async def initialize_model_endpoint(model_config: Dict[str, Any]):
    """モデルの初期化API"""
    try:
        model_type = model_config.get("model_type", "FastASRModel")
        hidden_dim = model_config.get("hidden_dim", 64)
        
        initialize_model(model_type, hidden_dim)
        
        return {
            "status": "success",
            "message": f"モデル初期化完了: {model_type}",
            "parameters": sum(p.numel() for p in model.parameters())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
async def load_model_endpoint(model_path: str):
    """モデルの読み込みAPI"""
    try:
        success = load_model(model_path)
        if success:
            return {"status": "success", "message": f"モデル読み込み完了: {model_path}"}
        else:
            raise HTTPException(status_code=404, detail="モデルファイルが見つかりません")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize")
async def recognize_audio_endpoint(audio_data: Dict[str, Any]):
    """音声認識API（REST）"""
    try:
        # Base64エンコードされた音声データをデコード
        audio_base64 = audio_data.get("audio")
        sample_rate = audio_data.get("sample_rate", 16000)
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="音声データが提供されていません")
        
        # Base64デコード
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # 音声認識実行
        result = recognize_audio(audio_array, sample_rate)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """WebSocket経由のリアルタイム音声認識"""
    await manager.connect(websocket)
    
    try:
        while True:
            # クライアントからのメッセージを受信
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "audio":
                # 音声データの処理
                audio_base64 = message.get("audio")
                sample_rate = message.get("sample_rate", 16000)
                
                if audio_base64:
                    try:
                        # Base64デコード
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                        
                        # 音声認識実行
                        result = recognize_audio(audio_array, sample_rate)
                        
                        # 結果をクライアントに送信
                        await websocket.send_text(json.dumps({
                            "type": "recognition_result",
                            "data": result
                        }))
                    
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
            
            elif message_type == "ping":
                # 接続確認
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket エラー: {e}")
        manager.disconnect(websocket)

@app.get("/performance")
async def get_performance_stats():
    """パフォーマンス統計の取得"""
    stats = performance_monitor.get_stats()
    return {
        "performance": stats,
        "model_info": {
            "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
            "device": device
        }
    }

@app.get("/model_info")
async def get_model_info():
    """モデル情報の取得"""
    if model is None:
        return {"error": "モデルが初期化されていません"}
    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        "model_type": model.__class__.__name__,
        "parameters": params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "device": device,
        "vocab_size": len(CHAR_TO_ID)
    }

# 静的ファイルの提供（HTML、JS、CSS）
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
