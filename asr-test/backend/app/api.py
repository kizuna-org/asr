from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, List
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
import zipfile
import io
from datetime import datetime

from . import trainer, config_loader
from .models.interface import BaseASRModel
from .state import training_status, _model_cache

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
        ModelClass = getattr(importlib.import_module(f".models.{model_name}", "app"), f"{model_name.capitalize()}ASRModel")
        
        logger.info(f"Loading model class: {ModelClass.__name__}", 
                   extra={"extra_fields": {"component": "api", "action": "load_model_class", "model_name": model_name, "class_name": ModelClass.__name__}})
        
        model = ModelClass(model_config)
        
        # 最新のチェックポイントをロードするロジック
        from .trainer import get_latest_checkpoint
        latest_checkpoint_path = get_latest_checkpoint(model_name, "ljspeech")
        if latest_checkpoint_path:
            logger.info(f"Loading checkpoint: {latest_checkpoint_path}", 
                       extra={"extra_fields": {"component": "api", "action": "load_checkpoint", 
                                             "model_name": model_name, "checkpoint_path": latest_checkpoint_path}})
            try:
                model.load_checkpoint(latest_checkpoint_path)
                logger.info(f"Checkpoint loaded successfully", 
                           extra={"extra_fields": {"component": "api", "action": "checkpoint_loaded", 
                                                 "model_name": model_name}})
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}", 
                              extra={"extra_fields": {"component": "api", "action": "checkpoint_load_failed", 
                                                    "model_name": model_name, "error": str(e)}})
        else:
            logger.info(f"No checkpoint found for model: {model_name}", 
                       extra={"extra_fields": {"component": "api", "action": "no_checkpoint", 
                                             "model_name": model_name}})
        
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
    resume_from_checkpoint = params.get("resume_from_checkpoint", True)  # デフォルトで再開を有効
    specific_checkpoint = params.get("specific_checkpoint")  # 特定のチェックポイントを指定
    
    if not model_name or not config_loader.get_model_config(model_name):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not a valid model name.")
    
    if not dataset_name or not config_loader.get_dataset_config(dataset_name):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' is not a valid dataset name.")

    # 特定のチェックポイントが指定された場合の検証
    if specific_checkpoint:
        checkpoint_path = Path("/app/checkpoints") / specific_checkpoint
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Specified checkpoint '{specific_checkpoint}' not found.")
        if not checkpoint_path.is_dir():
            raise HTTPException(status_code=400, detail=f"'{specific_checkpoint}' is not a valid checkpoint directory.")

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
    
    # 学習再開の情報をログに記録
    if resume_from_checkpoint:
        if specific_checkpoint:
            logger.info(f"Training will resume from specific checkpoint: {specific_checkpoint}")
        else:
            logger.info("Training will resume from latest checkpoint if available")
    else:
        logger.info("Training will start from scratch (no checkpoint resume)")
    
    background_tasks.add_task(trainer.start_training, params)
    return {"message": "Training started in background."}

@router.post("/train/resume", summary="学習再開")
def resume_training(params: Dict, background_tasks: BackgroundTasks):
    """学習をチェックポイントから再開する"""
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    
    model_name = params.get("model_name")
    dataset_name = params.get("dataset_name")
    specific_checkpoint = params.get("specific_checkpoint")  # 特定のチェックポイントを指定
    
    if not model_name or not config_loader.get_model_config(model_name):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not a valid model name.")
    
    if not dataset_name or not config_loader.get_dataset_config(dataset_name):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' is not a valid dataset name.")

    # チェックポイントの検証
    if specific_checkpoint:
        checkpoint_path = Path("/app/checkpoints") / specific_checkpoint
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Specified checkpoint '{specific_checkpoint}' not found.")
        if not checkpoint_path.is_dir():
            raise HTTPException(status_code=400, detail=f"'{specific_checkpoint}' is not a valid checkpoint directory.")
    else:
        # 最新のチェックポイントを探す
        from .trainer import get_latest_checkpoint
        latest_checkpoint = get_latest_checkpoint(model_name, dataset_name)
        if not latest_checkpoint:
            raise HTTPException(status_code=404, detail=f"No checkpoint found for model '{model_name}' and dataset '{dataset_name}'.")

    # 学習再開パラメータを設定
    resume_params = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "resume_from_checkpoint": True,
        "specific_checkpoint": specific_checkpoint,
        "epochs": params.get("epochs", 10),
        "batch_size": params.get("batch_size", 32),
        "lightweight": params.get("lightweight", False),
        "limit_samples": params.get("limit_samples")
    }

    logger.info(f"/train/resume called with params: {resume_params}")
    
    # 即時にフラグを立て、初期状態をセット
    training_status["is_training"] = True
    training_status["current_epoch"] = 0
    training_status["current_step"] = 0
    training_status["current_loss"] = 0.0
    training_status["current_learning_rate"] = 0.0
    training_status["progress"] = 0.0
    training_status.setdefault("latest_logs", [])
    training_status.pop("latest_error", None)
    
    
    background_tasks.add_task(trainer.start_training, resume_params)
    return {"message": "Training resumed from checkpoint in background."}

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

@router.get("/datasets", summary="データセット一覧取得")
def get_datasets():
    """利用可能なデータセットの一覧と状態を取得"""
    try:
        config = config_loader.load_config()
        available_datasets = config.get("available_datasets", [])
        datasets_info = []
        
        for dataset_name in available_datasets:
            dataset_config = config_loader.get_dataset_config(dataset_name)
            if not dataset_config:
                continue
            
            dataset_path = dataset_config.get("path", f"/app/data/{dataset_name}")
            status = "not_downloaded"
            num_files = 0
            size_mb = 0.0
            path = None
            
            # データセットの状態を確認
            if dataset_name == "ljspeech":
                final_dir = os.path.join(dataset_path, "LJSpeech-1.1")
                if os.path.exists(final_dir):
                    wav_files = list(Path(final_dir).glob("wavs/*.wav"))
                    if wav_files:
                        status = "downloaded"
                        num_files = len(wav_files)
                        path = final_dir
                        # ディレクトリサイズを計算
                        for file_path in Path(final_dir).rglob("*"):
                            if file_path.is_file():
                                size_mb += file_path.stat().st_size / (1024 * 1024)
            elif dataset_name == "jsut":
                final_dir = os.path.join(dataset_path, "jsut_ver1.1")
                if os.path.exists(final_dir):
                    wav_files = list(Path(final_dir).glob("**/wav/*.wav"))
                    if wav_files:
                        status = "downloaded"
                        num_files = len(wav_files)
                        path = final_dir
                        # ディレクトリサイズを計算
                        for file_path in Path(final_dir).rglob("*"):
                            if file_path.is_file():
                                size_mb += file_path.stat().st_size / (1024 * 1024)
            
            datasets_info.append({
                "name": dataset_name,
                "status": status,
                "num_files": num_files,
                "size_mb": round(size_mb, 2),
                "path": path,
                "config": dataset_config
            })
        
        return {"datasets": datasets_info}
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")

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
        elif dataset_name == "jsut":
            # スクリプトをサブプロセス経由で実行してダウンロード
            script_path = "/app/download_jsut.py"
            data_root = "/app/data"
            jsut_root = os.path.join(data_root, "jsut")
            final_dir = os.path.join(jsut_root, "jsut_ver1.1")

            os.makedirs(jsut_root, exist_ok=True)

            # 既に展開済みならスキップ
            if os.path.exists(final_dir):
                wav_count = len(list(Path(final_dir).glob("**/wav/*.wav")))
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

            wav_count = len(list(Path(final_dir).glob("**/wav/*.wav")))
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

@router.get("/checkpoints", summary="チェックポイント一覧取得")
def get_checkpoints(model_name: str = None, dataset_name: str = None):
    """利用可能なチェックポイントの一覧を取得する"""
    try:
        checkpoints_dir = Path("/app/checkpoints")
        if not checkpoints_dir.exists():
            return {"checkpoints": []}
        
        checkpoints = []
        for checkpoint_path in checkpoints_dir.iterdir():
            if checkpoint_path.is_dir():
                # チェックポイント名からモデル名、データセット名、エポック情報を抽出
                checkpoint_name = checkpoint_path.name
                
                # パターン: {model_name}-{dataset_name}-epoch-{epoch}.pt
                if "-epoch-" in checkpoint_name:
                    try:
                        parts = checkpoint_name.split("-")
                        if len(parts) >= 4:
                            # モデル名とデータセット名を抽出
                            model_part = parts[0]
                            dataset_part = parts[1]
                            epoch_part = parts[3].replace(".pt", "")
                            
                            # フィルタリング
                            if model_name and model_part != model_name:
                                continue
                            if dataset_name and dataset_part != dataset_name:
                                continue
                            
                            # チェックポイントディレクトリ内のファイルを確認
                            checkpoint_files = list(checkpoint_path.glob("*"))
                            if checkpoint_files:  # ファイルが存在する場合のみ追加
                                # ファイルサイズを計算
                                total_size = sum(f.stat().st_size for f in checkpoint_files if f.is_file())
                                size_mb = total_size / (1024 * 1024)
                                
                                # 作成日時を取得
                                created_time = checkpoint_path.stat().st_ctime
                                
                                checkpoints.append({
                                    "name": checkpoint_name,
                                    "model_name": model_part,
                                    "dataset_name": dataset_part,
                                    "epoch": int(epoch_part),
                                    "path": str(checkpoint_path),
                                    "size_mb": round(size_mb, 2),
                                    "file_count": len(checkpoint_files),
                                    "created_at": created_time,
                                    "files": [f.name for f in checkpoint_files if f.is_file()]
                                })
                    except (ValueError, IndexError):
                        # パースに失敗した場合はスキップ
                        continue
        
        # エポック番号でソート（新しい順）
        checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
        
        logger.info(f"Found {len(checkpoints)} checkpoints", 
                   extra={"extra_fields": {"component": "api", "action": "list_checkpoints", 
                                         "count": len(checkpoints), "model_name": model_name, "dataset_name": dataset_name}})
        
        return {"checkpoints": checkpoints}
        
    except Exception as e:
        logger.error(f"Error listing checkpoints", 
                    extra={"extra_fields": {"component": "api", "action": "list_checkpoints_error", 
                                          "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=f"Failed to list checkpoints: {str(e)}")

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
                    dataset_info = None
                    # パターン: {model_name}-{dataset_name}-epoch-{epoch}.pt
                    if "-epoch-" in model_name:
                        try:
                            parts = model_name.split("-")
                            if len(parts) >= 4:
                                # エポック部分を抽出（parts[3]が"10.pt"の形式）
                                epoch_part = parts[3].replace(".pt", "")
                                epoch_info = epoch_part
                                # データセット名を抽出（parts[1]）
                                dataset_info = parts[1]
                        except (ValueError, IndexError):
                            # パースに失敗した場合はフォールバック
                            try:
                                epoch_part = model_name.split("-epoch-")[1]
                                if epoch_part.endswith(".pt"):
                                    epoch_info = epoch_part[:-3]  # .ptを除去
                                elif epoch_part:
                                    epoch_info = epoch_part
                            except:
                                pass
                    
                    # 学習メタデータを読み込む
                    metadata_file = model_path / "training_metadata.json"
                    training_metadata = None
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                training_metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load training metadata from {metadata_file}: {e}")
                    
                    # メタデータから情報を取得（フォールバックとしてファイル名から抽出した情報を使用）
                    if training_metadata:
                        dataset_info = training_metadata.get("dataset_name", dataset_info)
                        epoch_info = training_metadata.get("current_epoch", epoch_info)
                        if isinstance(epoch_info, int):
                            epoch_info = str(epoch_info)
                    
                    # ファイルサイズを計算
                    total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    # 作成日時を取得
                    created_time = model_path.stat().st_ctime
                    
                    model_data = {
                        "name": model_name,
                        "path": str(model_path),
                        "epoch": epoch_info,
                        "dataset_name": dataset_info,
                        "size_mb": round(size_mb, 2),
                        "file_count": len(model_files),
                        "created_at": created_time,
                        "files": [f.name for f in model_files if f.is_file()]
                    }
                    
                    # メタデータがある場合は追加情報を設定
                    if training_metadata:
                        model_data["training_metadata"] = training_metadata
                        # 学習開始・終了時刻
                        if "training_start_time" in training_metadata:
                            model_data["training_start_time"] = training_metadata["training_start_time"]
                        if "training_end_time" in training_metadata:
                            model_data["training_end_time"] = training_metadata["training_end_time"]
                        # 学習パラメータ
                        if "training_params" in training_metadata:
                            model_data["training_params"] = training_metadata["training_params"]
                    
                    models.append(model_data)
        
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

@router.delete("/models", summary="学習済みモデル一括削除")
def delete_models_bulk(model_names: List[str]):
    """指定された複数の学習済みモデルを一括削除する（リクエストボディで受け取る）"""
    try:
        checkpoints_dir = Path("/app/checkpoints")
        deleted_models = []
        failed_models = []
        
        for model_name in model_names:
            try:
                # セキュリティチェック: パストラバーサル攻撃を防ぐ
                if ".." in model_name or "/" in model_name or "\\" in model_name:
                    failed_models.append({"model_name": model_name, "error": "Invalid model name"})
                    continue
                
                model_path = checkpoints_dir / model_name
                if not model_path.exists():
                    failed_models.append({"model_name": model_name, "error": "Model not found"})
                    continue
                
                if not model_path.is_dir():
                    failed_models.append({"model_name": model_name, "error": "Not a valid model directory"})
                    continue
                
                # モデルディレクトリを削除
                shutil.rmtree(model_path)
                deleted_models.append(model_name)
                
                logger.info(f"Model deleted successfully", 
                           extra={"extra_fields": {"component": "api", "action": "delete_model_bulk", 
                                                 "model_name": model_name}})
            except Exception as e:
                failed_models.append({"model_name": model_name, "error": str(e)})
                logger.error(f"Error deleting model in bulk", 
                            extra={"extra_fields": {"component": "api", "action": "delete_model_bulk_error", 
                                                  "model_name": model_name, "error": str(e)}})
        
        return {
            "deleted": deleted_models,
            "failed": failed_models,
            "message": f"Deleted {len(deleted_models)} model(s), {len(failed_models)} failed"
        }
        
    except Exception as e:
        logger.error(f"Error in bulk delete", 
                    extra={"extra_fields": {"component": "api", "action": "delete_models_bulk_error", 
                                          "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=f"Failed to delete models: {str(e)}")

@router.get("/models/bulk-download", summary="学習済みモデル一括ダウンロード")
def download_models_bulk(model_names: List[str] = Query(...)):
    """指定された複数の学習済みモデルをZIPファイルとして一括ダウンロードする"""
    try:
        checkpoints_dir = Path("/app/checkpoints")
        
        # 一時的なZIPファイルを作成
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for model_name in model_names:
                try:
                    # セキュリティチェック: パストラバーサル攻撃を防ぐ
                    if ".." in model_name or "/" in model_name or "\\" in model_name:
                        continue
                    
                    model_path = checkpoints_dir / model_name
                    if not model_path.exists() or not model_path.is_dir():
                        continue
                    
                    # モデルディレクトリ内のすべてのファイルをZIPに追加
                    for file_path in model_path.rglob("*"):
                        if file_path.is_file():
                            # ZIP内のパスは model_name/ファイル名 の形式
                            arcname = model_name / file_path.relative_to(model_path)
                            zip_file.write(file_path, arcname)
                    
                    logger.info(f"Model added to zip", 
                               extra={"extra_fields": {"component": "api", "action": "download_model_bulk", 
                                                     "model_name": model_name}})
                except Exception as e:
                    logger.error(f"Error adding model to zip", 
                                extra={"extra_fields": {"component": "api", "action": "download_model_bulk_error", 
                                                      "model_name": model_name, "error": str(e)}})
        
        zip_buffer.seek(0)
        
        # ZIPファイル名を生成（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"models_{timestamp}.zip"
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error in bulk download", 
                    extra={"extra_fields": {"component": "api", "action": "download_models_bulk_error", 
                                          "error": str(e), "traceback": traceback.format_exc()}})
        raise HTTPException(status_code=500, detail=f"Failed to download models: {str(e)}")

@router.get("/test", summary="テスト用エンドポイント")
def test_endpoint():
    """テスト用のエンドポイント"""
    return {
        "message": "Test endpoint is working", 
        "endpoints": [
            "/config", 
            "/status", 
            "/progress", 
            "/train/start", 
            "/train/resume", 
            "/train/stop", 
            "/dataset/download", 
            "/models",
            "/checkpoints"
        ]
    }
