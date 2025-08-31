# backend/app/main.py
from fastapi import FastAPI
from .api import router as api_router
from .websocket import router as websocket_router

app = FastAPI(
    title="ASR Training POC",
    description="ASRモデルの学習と推論を行うためのAPI",
    version="0.1.0"
)

# HTTP APIエンドポイントをインクルード
app.include_router(api_router, prefix="/api")

# WebSocketエンドポイントをインクルード
app.include_router(websocket_router)

@app.get("/", tags=["Default"], summary="ヘルスチェック")
def read_root():
    """ルートパスへのGETリクエストに応答し、アプリケーションが正常に動作していることを示す。"""
    return {"message": "Welcome to ASR Training POC API"}
