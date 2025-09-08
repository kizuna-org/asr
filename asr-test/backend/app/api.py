from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict
import os
import tarfile
from pathlib import Path
import requests
import torch
import torchaudio
import subprocess

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
            # スクリプトをサブプロセス経由で実行してダウンロード
            script_path = "/app/download_ljspeech.py"
            data_root = "/app/data"
            ljspeech_root = os.path.join(data_root, "ljspeech")
            final_dir = os.path.join(ljspeech_root, "LJSpeech-1.1")

            os.makedirs(ljspeech_root, exist_ok=True)

            # 既に展開済みならスキップ
            if os.path.exists(final_dir):
                wav_count = len(list(Path(final_dir).glob("wavs/*.wav")))
                return {"message": f"Dataset '{dataset_name}' already exists", "path": final_dir, "num_wavs": wav_count}

            try:
                # 環境の python パスで実行（compose/uvicorn 環境に追従）
                python_exe = os.environ.get("PYTHON", "python3")
                result = subprocess.run([
                    python_exe, script_path
                ], capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired as e:
                raise HTTPException(status_code=504, detail=f"Download script timeout: {str(e)}")

            if result.returncode != 0:
                # スクリプトの出力をすべて返す（クライアント側でスタックトレースを確認可能にする）
                stdout_text = result.stdout or ""
                stderr_text = result.stderr or ""
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Download script failed (exit={result.returncode})\n"
                        f"STDOUT:\n{stdout_text}\n"
                        f"STDERR:\n{stderr_text}"
                    ),
                )

            # 成功時の検査
            if not os.path.exists(final_dir):
                raise HTTPException(status_code=500, detail=f"Download finished but target dir not found: {final_dir}")

            wav_count = len(list(Path(final_dir).glob("wavs/*.wav")))
            return {
                "message": f"Dataset '{dataset_name}' downloaded successfully",
                "path": final_dir,
                "num_wavs": wav_count,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' is not supported for download")
    except requests.exceptions.Timeout as e:
        raise HTTPException(status_code=500, detail=f"Download timeout: {str(e)}")
    except tarfile.TarError as e:
        raise HTTPException(status_code=500, detail=f"Archive extract error: {str(e)}")
    except Exception as e:
        import traceback
        # エラー発生時は常にスタックトレースを含めて返す
        error_detail = (
            f"Unexpected error: {str(e)}\n"
            f"Traceback (most recent call last):\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/test", summary="テスト用エンドポイント")
def test_endpoint():
    """テスト用のエンドポイント"""
    return {"message": "Test endpoint is working", "endpoints": ["/config", "/status", "/progress", "/train/start", "/train/stop", "/dataset/download"]}
