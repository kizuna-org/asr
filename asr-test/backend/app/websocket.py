# backend/app/websocket.py
import json
from typing import List, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """接続している全てのクライアントにメッセージを送信する"""
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

# ConnectionManagerのインスタンスを作成
manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続のエンドポイント"""
    await manager.connect(websocket)
    try:
        while True:
            # クライアントからのメッセージを待機（現在は使わない）
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
