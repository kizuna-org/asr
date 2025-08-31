# backend/app/api.py
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict

router = APIRouter()

# --- グローバルな状態管理 (仮) ---
# 本来はRedisなどを使うべきだが、PoCのためインメモリで管理
training_status = {
    "is_training": False,
    "process": None
}

@router.post("/train/start", summary="学習開始")
def start_training(params: Dict, background_tasks: BackgroundTasks):
    """学習プロセスをバックグラウンドで開始する"""
    # ここにロジックを実装
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    # background_tasks.add_task(trainer.start_training, params)
    return {"message": "Training started in background."}

@router.post("/train/stop", summary="学習停止")
def stop_training():
    """学習プロセスを停止する"""
    # ここにロジックを実装
    if not training_status["is_training"]:
        raise HTTPException(status_code=404, detail="No training process is running.")
    return {"message": "Stop signal sent to training process."}

@router.post("/inference", summary="音声ファイルによる推論")
async def inference(file: UploadFile = File(...)):
    """アップロードされた音声ファイルで推論を実行する"""
    # ここにロジックを実装
    return {"transcription": "hello world"}

@router.get("/config", summary="設定情報の取得")
def get_config():
    """利用可能なモデルやデータセットなどの設定情報を返す"""
    # ここにロジックを実装
    return {"available_models": ["conformer"], "available_datasets": ["ljspeech"]}

@router.get("/status", summary="現在のステータスを取得")
def get_status():
    """現在の学習ステータスを返す"""
    return {"is_training": training_status["is_training"]}
