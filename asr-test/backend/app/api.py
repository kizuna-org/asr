from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict
import torch
import torchaudio

from . import trainer, config_loader
from .models.interface import BaseASRModel

router = APIRouter()

# --- グローバルな状態管理 ---
training_status = {
    "is_training": False,
}

# --- モデルのキャッシュ ---
# PoCのため、グローバル変数でモデルを保持
_model_cache: Dict[str, BaseASRModel] = {}

def get_model_for_inference(model_name: str) -> BaseASRModel:
    """推論用のモデルをロードまたはキャッシュから取得する"""
    if model_name not in _model_cache:
        model_config = config_loader.get_model_config(model_name)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in config.")
        
        # 動的にモデルクラスをインポート
        import importlib
        ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        
        model = ModelClass(model_config)
        # TODO: 最新のチェックポイントをロードするロジック
        model.eval() # 推論モード
        _model_cache[model_name] = model
    return _model_cache[model_name]

@router.post("/train/start", summary="学習開始")
def start_training(params: Dict, background_tasks: BackgroundTasks):
    """学習プロセスをバックグラウンドで開始する"""
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    
    model_name = params.get("model_name")
    if not model_name or not config_loader.get_model_config(model_name):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not a valid model name.")

    background_tasks.add_task(trainer.start_training, params)
    return {"message": "Training started in background."}

@router.post("/train/stop", summary="学習停止")
def stop_training():
    """学習プロセスを停止する"""
    if not training_status["is_training"]:
        raise HTTPException(status_code=404, detail="No training process is running.")
    # TODO: 停止フラグを立てるロジックを実装
    trainer.stop_training_flag = True # trainer.py にフラグを追加する必要がある
    return {"message": "Stop signal sent to training process."}

@router.post("/inference", summary="音声ファイルによる推論")
async def inference(file: UploadFile = File(...), model_name: str = "conformer"):
    """アップロードされた音声ファイルで推論を実行する"""
    try:
        # モデルを取得
        model = get_model_for_inference(model_name)
        
        # 音声ファイルを読み込み、リサンプリング
        waveform, sample_rate = torchaudio.load(file.file)
        
        # HuggingFaceモデルが要求する16kHzにリサンプリング
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        resampled_waveform = resampler(waveform)

        # 推論実行
        transcription = model.inference(resampled_waveform.squeeze(0))
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", summary="設定情報の取得")
def get_config():
    """利用可能なモデルやデータセットなどの設定情報を返す"""
    config = config_loader.load_config()
    return {
        "available_models": config.get("available_models", []),
        "available_datasets": config.get("available_datasets", []),
        "training_config": config.get("training", {})
    }

@router.get("/status", summary="現在のステータスを取得")
def get_status():
    """現在の学習ステータスを返す"""
    return {"is_training": training_status["is_training"]}
