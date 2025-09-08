from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict
import torch
import torchaudio

from . import trainer, config_loader
from .models.interface import BaseASRModel
from .state import training_status, _model_cache

router = APIRouter()
print(f"DEBUG: API router created")  # デバッグ用

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
    dataset_name = params.get("dataset_name")
    
    if not model_name or not config_loader.get_model_config(model_name):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not a valid model name.")
    
    if not dataset_name or not config_loader.get_dataset_config(dataset_name):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' is not a valid dataset name.")

    print(f"DEBUG: Starting training with params: {params}")
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

@router.get("/progress", summary="学習進捗を取得")
def get_progress():
    """現在の学習進捗を返す"""
    print(f"DEBUG: /progress endpoint called, training_status: {training_status}")  # デバッグ用
    return training_status

@router.post("/dataset/download", summary="データセットダウンロード")
def download_dataset(params: Dict):
    """指定されたデータセットをダウンロードする"""
    dataset_name = params.get("dataset_name")
    try:
        if dataset_name == "ljspeech":
            # LJSpeechデータセットのダウンロード処理
            import subprocess
            import os
            
            data_dir = "/app/data"
            ljspeech_dir = os.path.join(data_dir, "ljspeech")
            
            if not os.path.exists(ljspeech_dir):
                os.makedirs(ljspeech_dir, exist_ok=True)
            
            # ダウンロードスクリプトを実行
            download_script = "/app/download_ljspeech.py"
            if os.path.exists(download_script):
                result = subprocess.run(["python3", download_script], capture_output=True, text=True, cwd=data_dir)
                if result.returncode == 0:
                    return {"message": f"Dataset '{dataset_name}' downloaded successfully"}
                else:
                    raise HTTPException(status_code=500, detail=f"Download failed: {result.stderr}")
            else:
                raise HTTPException(status_code=404, detail="Download script not found")
        else:
            raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' is not supported for download")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test", summary="テスト用エンドポイント")
def test_endpoint():
    """テスト用のエンドポイント"""
    return {"message": "Test endpoint is working", "endpoints": ["/config", "/status", "/progress", "/train/start", "/train/stop", "/dataset/download"]}
