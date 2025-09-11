# backend/app/main.py
from fastapi import FastAPI
import logging
import sys
import json
from datetime import datetime
from .api import router as api_router
from .websocket import router as websocket_router

app = FastAPI(
    title="ASR Training POC",
    description="ASRモデルの学習と推論を行うためのAPI",
    version="0.1.0"
)

class StructuredFormatter(logging.Formatter):
    """構造化ログフォーマッター"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 例外情報がある場合は追加
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 追加のフィールドがある場合は追加
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging():
    """ログ設定を初期化"""
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソールハンドラーの設定
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(StructuredFormatter())
    
    # ファイルハンドラーの設定（オプション）
    try:
        import os
        log_dir = '/app/logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'asr-api.log'), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    root_logger.addHandler(console_handler)
    
    # 特定のロガーのレベル設定
    logging.getLogger("asr-api").setLevel(logging.DEBUG)
    logging.getLogger("websocket").setLevel(logging.INFO)
    logging.getLogger("model").setLevel(logging.DEBUG)
    logging.getLogger("audio").setLevel(logging.INFO)
    
    # 外部ライブラリのログレベル調整
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

# ログ設定を初期化
setup_logging()
logger = logging.getLogger("asr-api")

# HTTP APIエンドポイントをインクルード
print(f"DEBUG: Including API router with prefix '/api'")  # デバッグ用
app.include_router(api_router, prefix="/api")
print(f"DEBUG: API router included successfully")  # デバッグ用

# WebSocketエンドポイントをインクルード
app.include_router(websocket_router)

# デバッグ用：登録されたルートを確認
print(f"DEBUG: Registered routes:")
for route in app.routes:
    if hasattr(route, 'path'):
        print(f"  - {route.path}")
    elif hasattr(route, 'routes'):
        for sub_route in route.routes:
            if hasattr(sub_route, 'path'):
                print(f"  - {sub_route.path}")

@app.get("/", tags=["Default"], summary="ヘルスチェック")
def read_root():
    """ルートパスへのGETリクエストに応答し、アプリケーションが正常に動作していることを示す。"""
    return {"message": "Welcome to ASR Training POC API"}

@app.get("/debug", tags=["Debug"], summary="デバッグ用エンドポイント")
def debug_info():
    """デバッグ情報を返す"""
    return {
        "message": "Debug endpoint is working",
        "routes": [route.path for route in app.routes if hasattr(route, 'path')],
        "api_routes": [route.path for route in api_router.routes if hasattr(route, 'path')]
    }
