import json
import asyncio
from typing import List, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

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
    """WebSocket接続のエンドポイント"""
    await manager.connect(websocket)
    try:
        while True:
            # クライアントからのメッセージを待機（現在は使わない）
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
