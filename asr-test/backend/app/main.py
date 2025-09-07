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
