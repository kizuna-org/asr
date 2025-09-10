from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from typing import Dict
import logging
import os
import tarfile
from pathlib import Path
import requests
import torch
import torchaudio
import subprocess
import time
import tempfile
import shutil
import traceback

from . import trainer, config_loader
from .models.interface import BaseASRModel
from .state import training_status, _model_cache
from .websocket import manager as websocket_manager

router = APIRouter()
logger = logging.getLogger("asr-api")

# ログ出力の改善
logger.info("API router initialized", extra={"extra_fields": {"component": "api", "action": "init"}})

def get_model_for_inference(model_name: str) -> BaseASRModel:
    """推論用のモデルをロードまたはキャッシュから取得する"""
    logger.info(f"Getting model for inference: {model_name}", 
                extra={"extra_fields": {"component": "api", "action": "get_model", "model_name": model_name}})
    
    if model_name not in _model_cache:
        logger.info(f"Model {model_name} not in cache, loading from config", 
                   extra={"extra_fields": {"component": "api", "action": "load_model", "model_name": model_name}})
        
        model_config = config_loader.get_model_config(model_name)
        if not model_config:
            logger.error(f"Model config not found: {model_name}", 
                        extra={"extra_fields": {"component": "api", "action": "error", "model_name": model_name, "error_type": "config_not_found"}})
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in config.")
        
        # 動的にモデルクラスをインポート
        import importlib
        if model_name == "realtime":
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), "RealtimeASRModel")
        else:
            ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        
        logger.info(f"Loading model class: {ModelClass.__name__}", 
                   extra={"extra_fields": {"component": "api", "action": "load_model_class", "model_name": model_name, "class_name": ModelClass.__name__}})
        
        model = ModelClass(model_config)
        # TODO: 最新のチェックポイントをロードするロジック
        model.eval() # 推論モード
        _model_cache[model_name] = model
        
        logger.info(f"Model {model_name} loaded and cached successfully", 
                   extra={"extra_fields": {"component": "api", "action": "model_cached", "model_name": model_name}})
    else:
        logger.debug(f"Using cached model: {model_name}", 
                    extra={"extra_fields": {"component": "api", "action": "use_cached_model", "model_name": model_name}})
    
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

    logger.info(f"/train/start called with params: {params}")
    # 即時にフラグを立て、初期状態をセットしてリロード時の不一致を防止
    training_status["is_training"] = True
    training_status["current_epoch"] = 0
    training_status["current_step"] = 0
    training_status["current_loss"] = 0.0
    training_status["current_learning_rate"] = 0.0
    training_status["progress"] = 0.0
    training_status.setdefault("latest_logs", [])
    training_status.pop("latest_error", None)
    try:
        websocket_manager.broadcast_sync({
            "type": "status",
            "payload": {
                "status": "starting",
                "message": f"学習開始リクエスト受理: model={model_name}, dataset={dataset_name}"
            }
        })
    except Exception:
        pass
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
    # 音声入力開始時刻（APIリクエスト受信時点）
    input_start_time = time.time()
    
    logger.info(f"Starting inference request", 
                extra={"extra_fields": {"component": "api", "action": "inference_start", "model_name": model_name, "filename": file.filename}})
    
    try:
        # モデルを取得
        model = get_model_for_inference(model_name)

        # 音声ファイルを一時ファイルに保存してから読み込み（バックエンド間の互換性向上）
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            file.file.seek(0)
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Audio file saved to temporary path: {tmp_path}", 
                   extra={"extra_fields": {"component": "api", "action": "file_saved", "tmp_path": tmp_path}})

        # torchaudio で読み込み
        waveform, sample_rate = torchaudio.load(tmp_path)
        
        logger.info(f"Audio loaded successfully", 
                   extra={"extra_fields": {"component": "api", "action": "audio_loaded", 
                                         "original_shape": waveform.shape, "sample_rate": sample_rate}})

        # HuggingFaceモデルが要求する16kHzにリサンプリング
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        resampled_waveform = resampler(waveform)

        # モノラル化（複数チャネルの場合は平均化）し、1Dテンソルへ整形
        if resampled_waveform.dim() == 2 and resampled_waveform.size(0) > 1:
            resampled_waveform = resampled_waveform.mean(dim=0, keepdim=False)
        else:
            resampled_waveform = resampled_waveform.squeeze(0)

        logger.info(f"Audio preprocessed", 
                   extra={"extra_fields": {"component": "api", "action": "audio_preprocessed", 
                                         "final_shape": resampled_waveform.shape, "dtype": str(resampled_waveform.dtype)}})

        # 推論実行（3種類の時間計測）
        # 2. 推論開始時刻
        inference_start_time = time.time()
        transcription = model.inference(resampled_waveform)
        inference_end_time = time.time()
        
        # 3. 推論完了時刻（レスポンス準備完了時点）
        output_end_time = time.time()
        
        # 時間計算
        first_token_time_ms = (inference_start_time - input_start_time) * 1000  # 音声入力から推論開始まで
        inference_time_ms = (inference_end_time - inference_start_time) * 1000  # 推論時間
        total_time_ms = (output_end_time - input_start_time) * 1000  # 音声入力から最終出力まで
        
        logger.info(f"Inference completed successfully", 
                   extra={"extra_fields": {"component": "api", "action": "inference_complete", 
                                         "transcription": transcription, 
                                         "first_token_time_ms": first_token_time_ms,
                                         "inference_time_ms": inference_time_ms,
                                         "total_time_ms": total_time_ms}})
        
        # 一時ファイルを削除
        try:
            os.unlink(tmp_path)
            logger.debug(f"Temporary file deleted: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")
        
        return {
            "transcription": transcription, 
            "first_token_time_ms": first_token_time_ms,
            "inference_time_ms": inference_time_ms,
            "total_time_ms": total_time_ms
        }
    except Exception as e:
        # 可能な限り詳細な情報を返す（フロントでログ表示）
        error_detail = (
            f"Inference failed: {str(e)}\n"
            f"Traceback (most recent call last):\n{traceback.format_exc()}"
        )
        logger.error(f"Inference failed", 
                    extra={"extra_fields": {"component": "api", "action": "inference_error", 
                                          "model_name": model_name, "filename": file.filename, 
                                          "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=error_detail)

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
    """現在の学習進捗を返す（安定化のためコピー＆型整形）"""
    logger.debug(f"/progress called. status: {training_status}")
    # 直接の参照を返さずコピーして返す
    data = dict(training_status)
    # 型の安定化（フロント側の描画がこけないように float へ）
    for key in ["current_loss", "current_learning_rate", "progress"]:
        value = data.get(key)
        try:
            data[key] = float(value) if value is not None else 0.0
        except Exception:
            data[key] = 0.0
    # 追加情報: サーバー側の時刻
    data["server_time"] = time.time()
    return data

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

@router.get("/models", summary="学習済みモデル一覧取得")
def get_models():
    """学習済みモデルの一覧を取得する"""
    try:
        checkpoints_dir = Path("/app/checkpoints")
        if not checkpoints_dir.exists():
            return {"models": []}
        
        models = []
        for model_path in checkpoints_dir.iterdir():
            if model_path.is_dir():
                # モデルディレクトリ内のファイルを確認
                model_files = list(model_path.glob("*"))
                if model_files:  # ファイルが存在する場合のみ追加
                    # モデル名からエポック情報を抽出
                    model_name = model_path.name
                    epoch_info = None
                    if "-epoch-" in model_name:
                        try:
                            epoch_part = model_name.split("-epoch-")[1]
                            if epoch_part.endswith(".pt"):
                                epoch_info = epoch_part[:-3]  # .ptを除去
                        except:
                            pass
                    
                    # ファイルサイズを計算
                    total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    # 作成日時を取得
                    created_time = model_path.stat().st_ctime
                    
                    models.append({
                        "name": model_name,
                        "path": str(model_path),
                        "epoch": epoch_info,
                        "size_mb": round(size_mb, 2),
                        "file_count": len(model_files),
                        "created_at": created_time,
                        "files": [f.name for f in model_files if f.is_file()]
                    })
        
        # 作成日時でソート（新しい順）
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        logger.info(f"Found {len(models)} trained models", 
                   extra={"extra_fields": {"component": "api", "action": "list_models", "count": len(models)}})
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models", 
                    extra={"extra_fields": {"component": "api", "action": "list_models_error", 
                                          "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.delete("/models/{model_name}", summary="学習済みモデル削除")
def delete_model(model_name: str):
    """指定された学習済みモデルを削除する"""
    try:
        # セキュリティチェック: パストラバーサル攻撃を防ぐ
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        model_path = Path("/app/checkpoints") / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        if not model_path.is_dir():
            raise HTTPException(status_code=400, detail=f"'{model_name}' is not a valid model directory")
        
        # モデルディレクトリを削除
        import shutil
        shutil.rmtree(model_path)
        
        logger.info(f"Model deleted successfully", 
                   extra={"extra_fields": {"component": "api", "action": "delete_model", 
                                         "model_name": model_name, "path": str(model_path)}})
        
        return {"message": f"Model '{model_name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model", 
                    extra={"extra_fields": {"component": "api", "action": "delete_model_error", 
                                          "model_name": model_name, "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.get("/test", summary="テスト用エンドポイント")
def test_endpoint():
    """テスト用のエンドポイント"""
    return {"message": "Test endpoint is working", "endpoints": ["/config", "/status", "/progress", "/train/start", "/train/stop", "/dataset/download", "/models"]}
