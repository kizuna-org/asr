import streamlit as st
import pandas as pd
import requests
import asyncio
import websockets
import json
from typing import Dict, Any
import traceback
import os
import threading
import queue
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import logging
import sys
from datetime import datetime

# --- ログ設定 ---
class StructuredFormatter(logging.Formatter):
    """構造化ログフォーマッター（フロントエンド用）"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # 例外情報がある場合は追加
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 追加のフィールドがある場合は追加
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)

def setup_frontend_logging():
    """フロントエンドのログ設定を初期化"""
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # コンソールハンドラーの設定
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)

    # 特定のロガーのレベル設定
    logging.getLogger("ui-rt").setLevel(logging.INFO)
    logging.getLogger("audio_puller").setLevel(logging.INFO)
    logging.getLogger("websocket_loop").setLevel(logging.INFO)
    logging.getLogger("websocket_sender").setLevel(logging.INFO)

# ログ設定を初期化
setup_frontend_logging()

# --- 設定 ---
# 環境変数からバックエンドURLを取得、デフォルトはローカルホスト
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "58081")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api"
WEBSOCKET_URL = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws"

# プロキシ設定
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1,asr-api")

# プロキシ設定を辞書形式で準備
proxies = {}
if HTTP_PROXY:
    proxies["http"] = HTTP_PROXY
if HTTPS_PROXY:
    proxies["https"] = HTTPS_PROXY

# NO_PROXYの処理（簡易版）
def should_use_proxy(url):
    """URLがプロキシを使用すべきかどうかを判定"""
    if not proxies:
        return False

    no_proxy_hosts = [host.strip() for host in NO_PROXY.split(",")]
    for host in no_proxy_hosts:
        if host in url:
            return False
    return True

# --- 状態管理の初期化 ---
def init_session_state():
    defaults = {
        "logs": ["ダッシュボードへようこそ！"],
        "progress_df": pd.DataFrame(columns=["epoch", "step", "loss"]).astype({"epoch": int, "step": int, "loss": float}),
        "validation_df": pd.DataFrame(columns=["epoch", "val_loss"]).astype({"epoch": int, "val_loss": float}),
        "lr_df": pd.DataFrame(columns=["step", "learning_rate"]).astype({"step": int, "learning_rate": float}),
        "is_training": False,
        "available_models": [],
        "available_datasets": [],
        "current_progress": 0,
        "progress_text": "待機中",
        "current_epoch": 0,
        "current_step": 0,
        "total_epochs": 0,
        "total_steps": 0,
        "last_progress_update": 0,
        "initial_load": False,
        "last_rerun_time": 0,
        "consecutive_errors": 0,
        "max_consecutive_errors": 3
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 推論API呼び出し ---
def run_inference(file_bytes: bytes, filename: str, model_name: str) -> Dict[str, Any]:
    """音声ファイルをアップロードして推論を実行し、結果と3種類の時間(ms)を返す"""
    try:
        import time
        st.session_state.logs.append(f"🧪 推論リクエスト送信中... URL: {BACKEND_URL}/inference")
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        files = {
            "file": (filename, file_bytes, "application/octet-stream"),
        }
        params = {"model_name": model_name} if model_name else None
        start_time = time.perf_counter()
        response = requests.post(
            f"{BACKEND_URL}/inference",
            files=files,
            params=params,
            timeout=120,
            proxies=request_proxies,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.raise_for_status()
        data = response.json()
        transcription = data.get("transcription", "")

        # バックエンドから3種類の時間を取得
        first_token_time_ms = data.get("first_token_time_ms")
        inference_time_ms = data.get("inference_time_ms")
        total_time_ms = data.get("total_time_ms")

        # バックエンドが時間を返していない場合はフォールバック
        if first_token_time_ms is None:
            first_token_time_ms = elapsed_ms * 0.1  # 仮の値
        if inference_time_ms is None:
            inference_time_ms = elapsed_ms * 0.8  # 仮の値
        if total_time_ms is None:
            total_time_ms = elapsed_ms

        st.session_state.logs.append(f"✅ 推論が完了しました")
        st.session_state.logs.append(f"   📊 時間計測結果:")
        st.session_state.logs.append(f"   - 最初の出力まで: {first_token_time_ms:.0f} ms")
        st.session_state.logs.append(f"   - 推論時間: {inference_time_ms:.0f} ms")
        st.session_state.logs.append(f"   - 総時間: {total_time_ms:.0f} ms")

        return {
            "transcription": transcription,
            "first_token_time_ms": first_token_time_ms,
            "inference_time_ms": inference_time_ms,
            "total_time_ms": total_time_ms
        }
    except requests.exceptions.RequestException as e:
        log_detailed_error("推論実行", e, getattr(e, "response", None))
        return {"transcription": "", "first_token_time_ms": None, "inference_time_ms": None, "total_time_ms": None}
    except Exception as e:
        log_detailed_error("推論実行", e)
        return {"transcription": "", "first_token_time_ms": None, "inference_time_ms": None, "total_time_ms": None}

# --- 詳細エラーログ関数 ---
def log_detailed_error(operation: str, error: Exception, response=None):
    """詳細なエラー情報をログに記録"""
    error_msg = f"❌ {operation} エラー:"

    # 基本エラー情報
    error_msg += f"\n   - エラータイプ: {type(error).__name__}"
    error_msg += f"\n   - エラーメッセージ: {str(error)}"

    # レスポンス情報がある場合
    if response is not None:
        error_msg += f"\n   - ステータスコード: {response.status_code}"
        error_msg += f"\n   - レスポンスヘッダー: {dict(response.headers)}"
        try:
            error_msg += f"\n   - レスポンスボディ: {response.text}"
        except:
            error_msg += f"\n   - レスポンスボディ: 読み取り不可"

    # 接続エラーの詳細
    if isinstance(error, requests.exceptions.ConnectionError):
        error_msg += f"\n   - 接続先: {BACKEND_URL}"
        error_msg += f"\n   - 接続エラー詳細: バックエンドサービスに接続できません"
        error_msg += f"\n   - 確認事項:"
        error_msg += f"\n     * バックエンドサービスが起動しているか"
        error_msg += f"\n     * Dockerコンテナが正常に動作しているか"
        error_msg += f"\n     * ネットワーク設定が正しいか"
    elif isinstance(error, requests.exceptions.Timeout):
        error_msg += f"\n   - タイムアウト詳細: リクエストがタイムアウトしました"
    elif isinstance(error, requests.exceptions.HTTPError):
        error_msg += f"\n   - HTTPエラー詳細: HTTPステータスエラー"

    # スタックトレース（開発用）
    error_msg += f"\n   - スタックトレース: {traceback.format_exc()}"

    st.session_state.logs.append(error_msg)

# --- バックエンドAPI通信 ---
def get_config():
    """設定情報を取得"""
    try:
        st.session_state.logs.append(f"🔍 設定情報を取得中... URL: {BACKEND_URL}/config")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/config", timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            config = response.json()
            st.session_state.available_models = config.get("available_models", [])
            st.session_state.available_datasets = config.get("available_datasets", [])
            st.session_state.logs.append("✅ 設定情報を取得しました")
        else:
            log_detailed_error("設定取得", Exception(f"HTTP {response.status_code}"), response)

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("設定取得", e)
    except requests.exceptions.Timeout as e:
        log_detailed_error("設定取得", e)
    except requests.exceptions.RequestException as e:
        log_detailed_error("設定取得", e)
    except Exception as e:
        log_detailed_error("設定取得", e)

def get_status():
    """現在のステータスを取得"""
    try:
        st.session_state.logs.append(f"🔍 ステータスを取得中... URL: {BACKEND_URL}/status")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/status", timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            status = response.json()
            st.session_state.is_training = status.get("is_training", False)
            st.session_state.logs.append("✅ ステータスを取得しました")
            # 成功時は連続エラーカウンターをリセット
            st.session_state.consecutive_errors = 0
        else:
            log_detailed_error("ステータス取得", Exception(f"HTTP {response.status_code}"), response)
            st.session_state.consecutive_errors += 1

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("ステータス取得", e)
        st.session_state.consecutive_errors += 1
    except requests.exceptions.Timeout as e:
        log_detailed_error("ステータス取得", e)
        st.session_state.consecutive_errors += 1
    except requests.exceptions.RequestException as e:
        log_detailed_error("ステータス取得", e)
        st.session_state.consecutive_errors += 1
    except Exception as e:
        log_detailed_error("ステータス取得", e)
        st.session_state.consecutive_errors += 1

def start_training(model_name: str, dataset_name: str, epochs: int, batch_size: int, lightweight: bool = False, limit_samples: int = 0, resume_from_checkpoint: bool = True, specific_checkpoint: str = None):
    """学習を開始"""
    try:
        params = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "resume_from_checkpoint": resume_from_checkpoint
        }
        # 軽量モード/サンプル制限の付与
        if lightweight:
            params["lightweight"] = True
        if isinstance(limit_samples, int) and limit_samples > 0:
            params["limit_samples"] = int(limit_samples)
        if specific_checkpoint:
            params["specific_checkpoint"] = specific_checkpoint

        st.session_state.logs.append(f"🚀 学習開始リクエスト送信中... URL: {BACKEND_URL}/train/start")
        if resume_from_checkpoint:
            if specific_checkpoint:
                st.session_state.logs.append(f"📂 チェックポイントから再開: {specific_checkpoint}")
            else:
                st.session_state.logs.append("📂 最新のチェックポイントから再開")
        else:
            st.session_state.logs.append("🆕 最初から学習を開始")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/train/start", json=params, timeout=30, proxies=request_proxies)

        if response.status_code == 200:
            st.session_state.is_training = True
            # 学習再開情報を保存
            if resume_from_checkpoint:
                if specific_checkpoint:
                    st.session_state.resume_info = f"Epoch {specific_checkpoint.split('-epoch-')[1].replace('.pt', '')}"
                else:
                    st.session_state.resume_info = "最新"
            else:
                st.session_state.resume_info = "新規"
            st.session_state.logs.append("✅ 学習を開始しました")
            return True
        else:
            log_detailed_error("学習開始", Exception(f"HTTP {response.status_code}"), response)
            return False

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("学習開始", e)
        return False
    except requests.exceptions.Timeout as e:
        log_detailed_error("学習開始", e)
        return False
    except requests.exceptions.RequestException as e:
        log_detailed_error("学習開始", e)
        return False
    except Exception as e:
        log_detailed_error("学習開始", e)
        return False

def resume_training(model_name: str, dataset_name: str, epochs: int, batch_size: int, specific_checkpoint: str = None, lightweight: bool = False, limit_samples: int = 0):
    """学習を再開"""
    try:
        params = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size
        }
        if specific_checkpoint:
            params["specific_checkpoint"] = specific_checkpoint
        if lightweight:
            params["lightweight"] = True
        if isinstance(limit_samples, int) and limit_samples > 0:
            params["limit_samples"] = int(limit_samples)

        st.session_state.logs.append(f"🔄 学習再開リクエスト送信中... URL: {BACKEND_URL}/train/resume")
        if specific_checkpoint:
            st.session_state.logs.append(f"📂 指定されたチェックポイントから再開: {specific_checkpoint}")
        else:
            st.session_state.logs.append("📂 最新のチェックポイントから再開")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/train/resume", json=params, timeout=30, proxies=request_proxies)

        if response.status_code == 200:
            st.session_state.is_training = True
            # 学習再開情報を保存
            if specific_checkpoint:
                st.session_state.resume_info = f"Epoch {specific_checkpoint.split('-epoch-')[1].replace('.pt', '')}"
            else:
                st.session_state.resume_info = "最新"
            st.session_state.logs.append("✅ 学習を再開しました")
            return True
        else:
            log_detailed_error("学習再開", Exception(f"HTTP {response.status_code}"), response)
            return False

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("学習再開", e)
        return False
    except requests.exceptions.Timeout as e:
        log_detailed_error("学習再開", e)
        return False
    except requests.exceptions.RequestException as e:
        log_detailed_error("学習再開", e)
        return False
    except Exception as e:
        log_detailed_error("学習再開", e)
        return False

def stop_training():
    """学習を停止"""
    try:
        st.session_state.logs.append(f"🛑 学習停止リクエスト送信中... URL: {BACKEND_URL}/train/stop")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/train/stop", timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            st.session_state.is_training = False
            # 学習再開情報をクリア
            if "resume_info" in st.session_state:
                del st.session_state.resume_info
            st.session_state.logs.append("✅ 学習を停止しました")
            return True
        else:
            log_detailed_error("学習停止", Exception(f"HTTP {response.status_code}"), response)
            return False

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("学習停止", e)
        return False
    except requests.exceptions.Timeout as e:
        log_detailed_error("学習停止", e)
        return False
    except requests.exceptions.RequestException as e:
        log_detailed_error("学習停止", e)
        return False
    except Exception as e:
        log_detailed_error("学習停止", e)
        return False

def download_dataset(dataset_name: str):
    """データセットをダウンロード"""
    try:
        st.session_state.logs.append(f"📥 データセットダウンロード開始: {dataset_name}")

        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/dataset/download", json={"dataset_name": dataset_name}, timeout=300, proxies=request_proxies)

        if response.status_code == 200:
            result = response.json()
            st.session_state.logs.append(f"✅ データセット '{dataset_name}' のダウンロードが完了しました")

            # サーバーからの詳細情報をログに追加
            if "stdout" in result and result["stdout"]:
                st.session_state.logs.append(f"📋 ダウンロード詳細:\n{result['stdout']}")
            if "stderr" in result and result["stderr"]:
                st.session_state.logs.append(f"⚠️ 警告メッセージ:\n{result['stderr']}")

            return True
        else:
            # より詳細なエラー情報を表示
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    st.session_state.logs.append(f"❌ ダウンロードエラー詳細:\n{error_detail['detail']}")
                else:
                    st.session_state.logs.append(f"❌ ダウンロードエラー: {response.text}")
            except:
                st.session_state.logs.append(f"❌ ダウンロードエラー: HTTP {response.status_code}")

            log_detailed_error("データセットダウンロード", Exception(f"HTTP {response.status_code}"), response)
            return False

    except requests.exceptions.ConnectionError as e:
        log_detailed_error("データセットダウンロード", e)
        return False
    except requests.exceptions.Timeout as e:
        log_detailed_error("データセットダウンロード", e)
        return False
    except requests.exceptions.RequestException as e:
        log_detailed_error("データセットダウンロード", e)
        return False
    except Exception as e:
        log_detailed_error("データセットダウンロード", e)
        return False

# --- 進捗取得関数 ---
def get_training_progress():
    """バックエンドから学習進捗を取得"""
    try:
        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/progress", timeout=5, proxies=request_proxies)

        if response.status_code == 200:
            progress_data = response.json()
            return progress_data
        elif response.status_code == 404:
            # 404エラーの場合は進捗エンドポイントが存在しない可能性
            st.session_state.logs.append(f"⚠️ 進捗エンドポイントが見つかりません: {response.status_code}")
            return None
        else:
            st.session_state.logs.append(f"⚠️ 進捗取得エラー: HTTP {response.status_code}")
            return None
    except requests.exceptions.ConnectionError as e:
        # 一時的な接続エラーでは学習状態は変更しない
        st.session_state.logs.append(f"❌ バックエンド接続エラー: {e}")
        return None
    except requests.exceptions.Timeout as e:
        # タイムアウトの場合はログに記録するが、学習状態は維持
        st.session_state.logs.append(f"⏰ 進捗取得タイムアウト: {e}")
        return None
    except Exception as e:
        st.session_state.logs.append(f"❌ 進捗取得エラー: {e}")
        return None

def update_progress_from_backend():
    """バックエンドから進捗を取得して更新"""
    # 連続エラーが多すぎる場合は進捗更新をスキップ
    if st.session_state.consecutive_errors >= st.session_state.max_consecutive_errors:
        st.session_state.logs.append("⚠️ 連続エラーが多すぎるため、進捗更新を一時停止します")
        return False

    progress_data = get_training_progress()
    # ポーリング時刻を記録（可視化用）
    import time
    st.session_state["last_poll_at"] = time.time()
    if progress_data:
        # 進捗データを更新
        if "current_epoch" in progress_data and "current_step" in progress_data:
            st.session_state.current_epoch = progress_data.get("current_epoch", 0)
            st.session_state.current_step = progress_data.get("current_step", 0)
            st.session_state.total_epochs = progress_data.get("total_epochs", 0)
            st.session_state.total_steps = progress_data.get("total_steps", 0)
            st.session_state.current_progress = progress_data.get("progress", 0)
            st.session_state.progress_text = f"Epoch {progress_data['current_epoch']}/{progress_data.get('total_epochs', '?')}, Step {progress_data['current_step']}/{progress_data.get('total_steps', '?')}"

        # ロスデータを更新（重複を避ける）
        if "current_loss" in progress_data and progress_data.get("current_step", 0) > 0:
            current_step = progress_data.get("current_step", 0)
            # 既に同じステップのデータがあるかチェック
            if not st.session_state.progress_df.empty:
                last_step = st.session_state.progress_df.iloc[-1]["step"]
                if current_step > last_step:
                    st.session_state.progress_df.loc[len(st.session_state.progress_df)] = {
                        "epoch": progress_data.get("current_epoch", 0),
                        "step": current_step,
                        "loss": progress_data["current_loss"]
                    }
            else:
                st.session_state.progress_df.loc[len(st.session_state.progress_df)] = {
                    "epoch": progress_data.get("current_epoch", 0),
                    "step": current_step,
                    "loss": progress_data["current_loss"]
                }

        # 学習率データを更新（重複を避ける）
        if "current_learning_rate" in progress_data and progress_data.get("current_step", 0) > 0:
            current_step = progress_data.get("current_step", 0)
            # 既に同じステップのデータがあるかチェック
            if not st.session_state.lr_df.empty:
                last_step = st.session_state.lr_df.iloc[-1]["step"]
                if current_step > last_step:
                    st.session_state.lr_df.loc[len(st.session_state.lr_df)] = {
                        "step": current_step,
                        "learning_rate": progress_data["current_learning_rate"]
                    }
            else:
                st.session_state.lr_df.loc[len(st.session_state.lr_df)] = {
                    "step": current_step,
                    "learning_rate": progress_data["current_learning_rate"]
                }

        # ログメッセージを更新
        if "latest_logs" in progress_data:
            for log in progress_data["latest_logs"]:
                if log not in st.session_state.logs:
                    st.session_state.logs.append(log)

        # 成功時は連続エラーカウンターをリセット
        st.session_state.consecutive_errors = 0
        return True
    else:
        st.session_state.consecutive_errors += 1
        return False

# --- モデル管理機能 ---
def get_checkpoints(model_name: str = None, dataset_name: str = None):
    """チェックポイント一覧を取得"""
    try:
        params = {}
        if model_name:
            params["model_name"] = model_name
        if dataset_name:
            params["dataset_name"] = dataset_name

        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/checkpoints", params=params, timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            return response.json().get("checkpoints", [])
        else:
            st.error(f"チェックポイント一覧の取得に失敗しました: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("バックエンドに接続できません。バックエンドサービスが起動しているか確認してください。")
        return []
    except requests.exceptions.Timeout:
        st.error("リクエストがタイムアウトしました。")
        return []
    except Exception as e:
        st.error(f"チェックポイント一覧の取得中にエラーが発生しました: {str(e)}")
        return []

def get_models():
    """学習済みモデル一覧を取得"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/models", timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            st.error(f"モデル一覧の取得に失敗しました: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("バックエンドに接続できません。バックエンドサービスが起動しているか確認してください。")
        return []
    except requests.exceptions.Timeout:
        st.error("リクエストがタイムアウトしました。")
        return []
    except Exception as e:
        st.error(f"モデル一覧の取得中にエラーが発生しました: {str(e)}")
        return []

def delete_model(model_name):
    """指定されたモデルを削除"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.delete(f"{BACKEND_URL}/models/{model_name}", timeout=30, proxies=request_proxies)

        if response.status_code == 200:
            return True, "モデルが正常に削除されました。"
        else:
            error_detail = response.json().get("detail", "不明なエラー")
            return False, f"削除に失敗しました: {error_detail}"
    except requests.exceptions.ConnectionError:
        return False, "バックエンドに接続できません。"
    except requests.exceptions.Timeout:
        return False, "リクエストがタイムアウトしました。"
    except Exception as e:
        return False, f"削除中にエラーが発生しました: {str(e)}"

def format_file_size(size_mb):
    """ファイルサイズを適切な単位でフォーマット"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"

def format_timestamp(timestamp):
    """タイムスタンプを読みやすい形式でフォーマット"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

# --- UI描画 ---
st.set_page_config(
    page_title="ASR学習ダッシュボード",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_session_state()

if not st.session_state.initial_load:
    # 初期化時は設定とステータスを取得
    get_config()
    get_status()
    st.session_state.initial_load = True
elif st.session_state.is_training:
    # 学習中の場合のみ、ステータスを再確認（リロード時の状態復元）
    import time
    current_time = time.time()
    if "last_status_check" not in st.session_state:
        st.session_state.last_status_check = 0

    # ステータス確認の頻度を制限（30秒ごと）
    if current_time - st.session_state.last_status_check >= 30:
        get_status()
        st.session_state.last_status_check = current_time

# タイトル
st.title("ASR 学習ダッシュボード")

# ナビゲーション
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("🏠 メインダッシュボード", use_container_width=True, key="nav_main_top"):
        st.session_state.current_page = "main"
        st.rerun()
with col_nav2:
    if st.button("🤖 モデル管理", use_container_width=True, key="nav_model_top"):
        st.session_state.current_page = "model_management"
        st.rerun()
with col_nav3:
    current_page = st.session_state.get("current_page", "main")
    if current_page == "main":
        page_name = "メインダッシュボード"
    elif current_page == "model_management":
        page_name = "モデル管理"
    elif current_page == "checkpoint_management":
        page_name = "チェックポイント管理"
    else:
        page_name = "不明"
    st.markdown(f"### 📊 現在のページ: {page_name}")
st.markdown("---")

# サイドバー - 学習制御
with st.sidebar:
    st.header("📋 ナビゲーション")

    # ページ間のナビゲーション
    current_page = st.session_state.get("current_page", "main")
    if st.button("🏠 メインダッシュボード", use_container_width=True, disabled=(current_page == "main"), key="nav_main_sidebar"):
        st.session_state.current_page = "main"
        st.rerun()
    if st.button("🤖 モデル管理", use_container_width=True, disabled=(current_page == "model_management"), key="nav_model_sidebar"):
        st.session_state.current_page = "model_management"
        st.rerun()
    if st.button("📂 チェックポイント管理", use_container_width=True, disabled=(current_page == "checkpoint_management"), key="nav_checkpoint_sidebar"):
        st.session_state.current_page = "checkpoint_management"
        st.rerun()

    st.markdown("---")
    st.header("🎯 学習制御")

    # データセット選択
    dataset_name = st.selectbox(
        "データセット",
        st.session_state.available_datasets,
        index=0 if st.session_state.available_datasets else None
    )

    # データセットダウンロードボタン
    if st.button("データセットをダウンロード", disabled=st.session_state.is_training):
        if dataset_name:
            with st.spinner(f"データセット '{dataset_name}' をダウンロード中..."):
                success = download_dataset(dataset_name)
                if success:
                    st.success(f"データセット '{dataset_name}' のダウンロードが完了しました")
                    # 設定を再取得して最新の状態を反映
                    get_config()
                else:
                    st.error(f"データセット '{dataset_name}' のダウンロードに失敗しました")
        else:
            st.error("データセットを選択してください")

    # 学習用モデル選択
    training_model_name = st.selectbox(
        "学習用モデル",
        st.session_state.available_models,
        index=0 if st.session_state.available_models else None,
        key="training_model_selector"
    )

    # 学習パラメータ
    epochs = st.number_input("エポック数", min_value=1, value=10)
    batch_size = st.number_input("バッチサイズ", min_value=1, value=4)
    lightweight = st.checkbox("軽量(先頭10件)でテスト実行", value=True)
    limit_samples = st.number_input("使用サンプル数を制限 (0で無効)", min_value=0, value=0)

    # 学習再開オプション
    st.subheader("🔄 学習再開オプション")
    resume_from_checkpoint = st.checkbox("チェックポイントから再開", value=True, help="チェックポイントが存在する場合、自動的に再開します")

    # チェックポイント選択
    specific_checkpoint = None
    if resume_from_checkpoint and training_model_name and dataset_name:
        checkpoints = get_checkpoints(training_model_name, dataset_name)
        if checkpoints:
            checkpoint_options = ["最新のチェックポイントから再開"] + [f"{cp['name']} (Epoch {cp['epoch']})" for cp in checkpoints]
            selected_checkpoint = st.selectbox(
                "再開するチェックポイントを選択",
                checkpoint_options,
                help="特定のチェックポイントから再開する場合は選択してください"
            )
            if selected_checkpoint != "最新のチェックポイントから再開":
                # チェックポイント名を抽出
                for cp in checkpoints:
                    if f"{cp['name']} (Epoch {cp['epoch']})" == selected_checkpoint:
                        specific_checkpoint = cp['name']
                        break
        else:
            st.info("利用可能なチェックポイントがありません。最初から学習を開始します。")

    # 学習制御ボタン
    st.subheader("🎮 学習制御")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🆕 新規学習開始", disabled=st.session_state.is_training, help="最初から学習を開始します"):
            if training_model_name and dataset_name:
                success = start_training(training_model_name, dataset_name, epochs, batch_size,
                                       lightweight=lightweight, limit_samples=limit_samples,
                                       resume_from_checkpoint=False)
                if not success:
                    st.error("学習の開始に失敗しました。ログを確認してください。")
            else:
                st.error("モデルとデータセットを選択してください")

    with col2:
        if st.button("🔄 学習再開", disabled=st.session_state.is_training, help="チェックポイントから学習を再開します"):
            if training_model_name and dataset_name:
                if resume_from_checkpoint:
                    success = resume_training(training_model_name, dataset_name, epochs, batch_size,
                                            specific_checkpoint=specific_checkpoint,
                                            lightweight=lightweight, limit_samples=limit_samples)
                else:
                    success = start_training(training_model_name, dataset_name, epochs, batch_size,
                                           lightweight=lightweight, limit_samples=limit_samples,
                                           resume_from_checkpoint=True, specific_checkpoint=specific_checkpoint)
                if not success:
                    st.error("学習の再開に失敗しました。ログを確認してください。")
            else:
                st.error("モデルとデータセットを選択してください")

    with col3:
        if st.button("🛑 学習停止", disabled=not st.session_state.is_training, help="現在の学習を停止します"):
            stop_training()

    # 進捗表示
    if st.session_state.is_training:
        st.progress(st.session_state.current_progress)
        st.text(st.session_state.progress_text)

# メインコンテンツ
current_page = st.session_state.get("current_page", "main")
if current_page == "main":
    col1, col2 = st.columns(2)

    # 推論テストセクション
    st.header("推論テスト（音声アップロード）")

    # 推論用モデル選択
    st.subheader("📋 推論設定")
    col_model_select, col_model_info = st.columns([1, 2])

    with col_model_select:
        # 利用可能なモデル一覧を取得
        available_models = st.session_state.available_models
        if available_models:
            selected_inference_model = st.selectbox(
                "推論に使用するモデル:",
                available_models,
                index=0,
                key="inference_model_selector",
                help="推論に使用するモデルを選択してください"
            )
        else:
            st.warning("利用可能なモデルがありません")
            selected_inference_model = None

    with col_model_info:
        if selected_inference_model:
            st.info(f"選択されたモデル: **{selected_inference_model}**")
        else:
            st.warning("モデルを選択してください")

    # 音声ファイルアップロードと推論実行
    st.subheader("🎵 音声ファイルアップロード")
    inf_col1, inf_col2 = st.columns([2, 1])

    with inf_col1:
        uploaded = st.file_uploader(
            "音声ファイルを選択 (WAV/FLAC/MP3/M4A/OGG)",
            type=["wav", "flac", "mp3", "m4a", "ogg"],
            key="inference_file_uploader",
            help="推論対象の音声ファイルをアップロードしてください"
        )
        if uploaded is not None:
            st.audio(uploaded, format="audio/wav")
            st.success(f"ファイルがアップロードされました: {uploaded.name}")

    with inf_col2:
        st.subheader("🚀 推論実行")
        inference_disabled = uploaded is None or selected_inference_model is None
        if st.button(
            "推論を実行",
            disabled=inference_disabled,
            type="primary",
            key="inference_execute_button",
            use_container_width=True
        ):
            if uploaded is None:
                st.warning("音声ファイルを選択してください")
            elif selected_inference_model is None:
                st.warning("推論用モデルを選択してください")
            else:
                with st.spinner("推論を実行中..."):
                    result = run_inference(uploaded.getvalue(), uploaded.name, selected_inference_model)
                    transcription = result.get("transcription", "")
                    first_token_ms = result.get("first_token_time_ms")
                    inference_ms = result.get("inference_time_ms")
                    total_ms = result.get("total_time_ms")

                    # 推論が完了した場合（空の結果も含む）
                    st.success("推論完了")

                    # 使用したモデル情報を表示
                    st.info(f"使用モデル: **{selected_inference_model}**")

                    # 3種類の時間を表示
                    st.subheader("⏱️ パフォーマンス情報")
                    col_time1, col_time2, col_time3 = st.columns(3)
                    with col_time1:
                        if first_token_ms is not None:
                            st.metric(label="最初の出力まで", value=f"{first_token_ms:.0f} ms")
                    with col_time2:
                        if inference_ms is not None:
                            st.metric(label="推論時間", value=f"{inference_ms:.0f} ms")
                    with col_time3:
                        if total_ms is not None:
                            st.metric(label="総時間", value=f"{total_ms:.0f} ms")

                    # 文字起こし結果
                    st.subheader("📝 文字起こし結果")
                    if transcription:
                        # 正常な文字起こし結果がある場合
                        st.text_area(
                            "文字起こし結果",
                            value=transcription,
                            height=120,
                            key="inference_result_text",
                            help="音声から認識されたテキストが表示されます"
                        )

                        # 結果のコピーボタン
                        if st.button("📋 結果をコピー", key="copy_result_button"):
                            st.write("結果をクリップボードにコピーしました（手動でコピーしてください）")
                    else:
                        # 空の推論結果の場合
                        st.warning("⚠️ 推論結果が空です")
                        st.text_area(
                            "文字起こし結果",
                            value="（音声から認識されたテキストがありません）",
                            height=120,
                            key="inference_result_text_empty",
                            help="音声から認識されたテキストがありません。音声の品質やモデルの学習状況を確認してください。"
                        )
                        st.info("💡 **推奨事項**: 音声の品質を確認するか、別のモデルで試してみてください。")

    # 上部メトリクス表示（学習中のみ）
    if st.session_state.is_training:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Epoch", value=f"{st.session_state.current_epoch}/{st.session_state.total_epochs}")
        with m2:
            st.metric(label="Step", value=f"{st.session_state.current_step}/{st.session_state.total_steps}")
        with m3:
            # 学習再開情報を表示
            if "resume_info" in st.session_state:
                st.metric(label="再開元", value=st.session_state.resume_info)
            else:
                st.metric(label="学習状態", value="実行中")

    with col1:
        st.header("学習ロス")
        if not st.session_state.progress_df.empty:
            loss_data = st.session_state.progress_df.rename(columns={"loss": "train_loss"})
            if not st.session_state.validation_df.empty:
                # エポックの最後のステップに検証ロスを紐付ける
                last_step_per_epoch = loss_data.groupby("epoch")["step"].max().reset_index()
                merged_val = pd.merge(st.session_state.validation_df, last_step_per_epoch, on="epoch")
                loss_data = pd.merge(loss_data, merged_val, on=["epoch", "step"], how="left")

            # 存在する列のみを描画対象にする
            plot_cols = [c for c in ["train_loss", "val_loss"] if c in loss_data.columns]
            st.line_chart(loss_data.set_index("step")[plot_cols])
        else:
            st.info("学習データがありません。学習を開始するとグラフが表示されます。")

    with col2:
        st.header("学習率")
        if not st.session_state.lr_df.empty:
            st.line_chart(st.session_state.lr_df.set_index("step")["learning_rate"])
        else:
            st.info("学習率データがありません。学習を開始するとグラフが表示されます。")

    # ログ表示
    st.header("ログ")
    log_container = st.container()
    with log_container:
        for log in st.session_state.logs[-50:]:  # 最新50件を表示
            st.text(log)

# 学習中の進捗更新
# Note: リアルタイムストリーミング開始中または実行中は自動ポーリングを停止（WebRTCの安定性のため）
if st.session_state.is_training and not st.session_state.get("realtime_running", False) and not st.session_state.get("_should_start_realtime", False):
    # 直近のポーリング時刻を表示（デバッグ/可視化）
    import time
    last_polled = st.session_state.get("last_poll_at")
    if last_polled:
        st.caption(f"最終ポーリング: {time.strftime('%H:%M:%S', time.localtime(last_polled))}")
    # 進捗更新の頻度を制限（1秒ごと）
    import time
    current_time = time.time()
    if "last_progress_update" not in st.session_state:
        st.session_state.last_progress_update = 0

    # 進捗更新の実行
    progress_updated = False
    if current_time - st.session_state.last_progress_update >= 1:
        progress_updated = update_progress_from_backend()
        st.session_state.last_progress_update = current_time

    # 確実な1秒ごとのポーリング（スリープ→再実行）
    # Note: リアルタイムストリーミング中は自動ポーリングを停止（WebRTCの安定性のため）
    time.sleep(1)
    st.rerun()

# --- リアルタイム推論（マイク入力） ---
if current_page == "main":
    st.header("リアルタイム推論（マイク入力）")

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frame_queue = None  # type: queue.Queue
        self.logger = logging.getLogger("ui-rt")
        self.msg_queue = None  # optional queue to report stats
        self._frames_sent = 0

        self.logger.info("MicAudioProcessor initialized",
                        extra={"extra_fields": {"component": "audio_processor", "action": "init"}})

    def recv_audio(self, frames, **kwargs):
        # frames: list of av.AudioFrame
        self.logger.debug("Audio frames received",
                         extra={"extra_fields": {"component": "audio_processor", "action": "frames_received",
                                               "frame_count": len(frames), "has_queue": self.frame_queue is not None}})

        if self.frame_queue is None:
            self.logger.debug("Frame queue is None, returning frames without processing",
                             extra={"extra_fields": {"component": "audio_processor", "action": "no_queue"}})
            return frames

        for frame in frames:
            # 32-bit float PCM, shape: (channels, samples)
            pcm = frame.to_ndarray(format="flt")

            # モノラル化
            if pcm.ndim == 2 and pcm.shape[0] > 1:
                pcm_mono = pcm.mean(axis=0)
            else:
                pcm_mono = pcm[0] if pcm.ndim == 2 else pcm

            # 送信は float32 little-endian bytes（サーバは f32 をサポート）
            pcm_f32 = pcm_mono.astype(np.float32)

            try:
                self.frame_queue.put(pcm_f32.tobytes(), timeout=0.1)
                self._frames_sent += 1

                # 音声フレーム処理（ログ削除）

                if self.msg_queue and (self._frames_sent % 25 == 0):
                    # おおよそ定期的に統計を送る
                    self.msg_queue.put({"type": "stats", "payload": {"frames_sent": self._frames_sent}})
            except queue.Full:
                self.logger.warning("Frame queue is full, dropping audio chunk",
                                  extra={"extra_fields": {"component": "audio_processor", "action": "queue_full"}})
        return frames

async def stream_audio_to_ws(q: "queue.Queue[bytes]", model_name: str, sample_rate: int, running_event: threading.Event, msg_queue_ref=None):
    import websockets
    logger = logging.getLogger("websocket_sender")

    # 接続リトライ（コンテキストマネージャを正しく使用）
    retries = 0
    while True:
        try:
            async with websockets.connect(
                WEBSOCKET_URL,
                ping_interval=30,
                ping_timeout=30,
                open_timeout=10,
                close_timeout=10,
            ) as ws:
                # 接続開始メッセージを送信
                start_msg = {"type": "start", "model_name": model_name, "sample_rate": sample_rate, "format": "f32"}
                await ws.send(json.dumps(start_msg))

                logger.info("WebSocket start message sent",
                           extra={"extra_fields": {"component": "websocket", "action": "start_sent",
                                                 "model_name": model_name, "sample_rate": sample_rate}})

                # 受信タスク
                async def receiver():
                    try:
                        while True:
                            msg = await ws.recv()
                            try:
                                data = json.loads(msg)
                                logger.debug("WebSocket message received",
                                           extra={"extra_fields": {"component": "websocket", "action": "message_received",
                                                                 "message_type": data.get("type")}})
                                # メインスレッドで処理するため、ローカル参照キューに積む
                                try:
                                    if msg_queue_ref is not None:
                                        msg_queue_ref.put(data)
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.error("Error parsing WebSocket message",
                                           extra={"extra_fields": {"component": "websocket", "action": "parse_error",
                                                                 "error": str(e), "message": msg}})
                                try:
                                    if msg_queue_ref is not None:
                                        msg_queue_ref.put({"type": "error", "payload": {"message": f"invalid message: {msg}"}})
                                except Exception:
                                    pass
                                pass
                    except Exception as e:
                        logger.error("WebSocket receiver error",
                                   extra={"extra_fields": {"component": "websocket", "action": "receiver_error",
                                                         "error": str(e)}})
                        return

                recv_task = asyncio.create_task(receiver())

                try:
                    while running_event.is_set():
                        try:
                            chunk = q.get(timeout=0.1)  # タイムアウトを短縮
                            # 音声チャンク送信（ログ削除）
                            await ws.send(chunk)
                            # 音声チャンク送信完了（ログ削除）
                            # 送信カウンタのUI更新はスレッド外で実施するため、ここではログのみ
                        except queue.Empty:
                            # サイレント時も接続維持
                            await asyncio.sleep(0.01)  # 待機時間を短縮
                            continue
                        except Exception as e:
                            logger.error("Error sending audio chunk",
                                       extra={"extra_fields": {"component": "websocket", "action": "chunk_send_error",
                                                             "error": str(e)}})
                            break
                except asyncio.CancelledError:
                    pass
                finally:
                    try:
                        await ws.send(json.dumps({"type": "stop"}))
                        logger.info("WebSocket stop message sent",
                                   extra={"extra_fields": {"component": "websocket", "action": "stop_sent"}})
                    except Exception:
                        pass
                    recv_task.cancel()
                    with contextlib.suppress(Exception):
                        await recv_task
                return
        except Exception as e:
            retries += 1
            logger.error("WebSocket connection error",
                        extra={"extra_fields": {"component": "websocket", "action": "connection_error",
                                              "retry_count": retries, "error": str(e)}})
            try:
                if msg_queue_ref is not None:
                    msg_queue_ref.put({"type": "error", "payload": {"message": f"ws session error (retry {retries}): {e}"}})
            except Exception:
                pass
            if retries >= 5:
                logger.error("Max retries reached, giving up",
                            extra={"extra_fields": {"component": "websocket", "action": "max_retries_reached"}})
                return
            await asyncio.sleep(min(1.0 * retries, 5.0))

import contextlib

st.session_state.setdefault("realtime_running", False)
st.session_state.setdefault("realtime_partial", "")
st.session_state.setdefault("realtime_final", "")
st.session_state.setdefault("realtime_status", {})
st.session_state.setdefault("realtime_error", "")
st.session_state.setdefault("realtime_msg_queue", queue.Queue())

col_rt1, col_rt2 = st.columns([2, 1])
with col_rt1:
    # WebRTC設定 - ICE接続の安定性を向上
    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
        ],
        "iceTransportPolicy": "all",  # すべてのICE候補を使用
    }

    # CRITICAL: Cache the webrtc context in session state to avoid recreating it
    # Recreating webrtc_streamer on every rerun closes the ICE connection
    if "_rtc_ctx" not in st.session_state or not st.session_state.get("realtime_running", False):
        # Only create/update webrtc_streamer when NOT actively streaming
        rtc_ctx = webrtc_streamer(
            key="asr-audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=2048,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            rtc_configuration=rtc_configuration,  # ICE設定を追加
        )
        st.session_state["_rtc_ctx"] = rtc_ctx
    else:
        # During streaming, reuse the cached context - DON'T call webrtc_streamer again!
        rtc_ctx = st.session_state["_rtc_ctx"]

with col_rt2:
    st.subheader("🎯 リアルタイム推論設定")

    # リアルタイム用モデル選択
    if st.session_state.available_models:
        selected_realtime_model = st.selectbox(
            "リアルタイム用モデル",
            st.session_state.available_models,
            index=0,
            key="realtime_model_selector",
            help="リアルタイム推論に使用するモデルを選択してください"
        )
        st.info(f"選択されたモデル: **{selected_realtime_model}**")
    else:
        st.warning("利用可能なモデルがありません")
        selected_realtime_model = None

    # 音声設定
    st.subheader("🔊 音声設定")
    sample_rate = st.number_input(
        "送信サンプルレート",
        min_value=16000,
        max_value=48000,
        value=48000,
        step=1000,
        key="realtime_sample_rate",
        help="マイクから送信する音声のサンプルレート"
    )

    # 制御ボタン
    st.subheader("🎮 制御")

    # ボタンを有効にする条件をより厳密に
    can_start = (
        not st.session_state.get("realtime_running", False) and
        rtc_ctx.state.playing and
        rtc_ctx.audio_receiver is not None and
        selected_realtime_model is not None
    )

    start_btn = st.button(
        "リアルタイム開始",
        disabled=not can_start,
        key="realtime_start_button",
        use_container_width=True,
        help="マイクを有効にしてモデルを選択してからクリックしてください"
    )
    stop_btn = st.button(
        "リアルタイム停止",
        disabled=not st.session_state.get("realtime_running", False),
        key="realtime_stop_button",
        use_container_width=True
    )

    # デバッグ情報を表示
    with st.expander("🔍 デバッグ情報", expanded=False):
        st.write(f"- start_btn: {start_btn}")
        st.write(f"- rtc_ctx: {rtc_ctx is not None}")
        st.write(f"- rtc_ctx.state.playing: {rtc_ctx.state.playing if rtc_ctx else 'N/A'}")
        st.write(f"- rtc_ctx.audio_receiver: {rtc_ctx.audio_receiver is not None if rtc_ctx else 'N/A'}")
        st.write(f"- realtime_running: {st.session_state.get('realtime_running', False)}")

    # 重要な注意事項を表示
    if not st.session_state.get('realtime_running', False):
        if rtc_ctx.state.playing:
            st.success("✅ マイクが有効です。'リアルタイム開始'ボタンをクリックしてください。")
            st.info("💡 **重要:** モデルの読み込みに約15秒かかります。その間マイクをONのままにしてください。")
        else:
            st.warning("⚠️ 先に上の'START'ボタンをクリックしてマイクを有効にしてください。")

    # 既に実行中の場合の警告と自動リセット
    if st.session_state.get('realtime_running', False):
        if not rtc_ctx.state.playing:
            st.error("⚠️ ストリーミング中にWebRTCが停止しました。自動的にリセットします...")
            # 自動リセット
            st.session_state["realtime_running"] = False
            # Clear the running event to stop threads
            running_event = st.session_state.get("_running_event")
            if running_event:
                running_event.clear()
            st.info("💡 'START'ボタンをもう一度クリックしてから、'リアルタイム開始'をクリックしてください。")
            st.rerun()

    if start_btn:
        if not rtc_ctx:
            st.error("❌ WebRTCコンテキストが初期化されていません")
        elif not rtc_ctx.state.playing:
            st.error("❌ マイクが有効になっていません。上の'START'ボタンを先にクリックしてください。")
        elif not rtc_ctx.audio_receiver:
            st.error("❌ オーディオレシーバーが初期化されていません。ページを再読み込みしてください。")
        elif st.session_state.get("realtime_running", False):
            st.warning("⚠️ すでにストリーミング中です。")
        else:
            # START THREADS IMMEDIATELY - don't defer to next rerun!
            # 送信キューとスレッド/タスクの初期化（キューサイズを大幅に増加）
            send_queue = queue.Queue(maxsize=1000)
            # 先にランニングフラグとカウンタを立ててからスレッドを開始
            st.session_state["realtime_running"] = True
            st.session_state["_rt_chunks_sent"] = 0
            st.session_state["_model_loading_start_time"] = __import__('time').time()
            # 既存のエラーメッセージ/ステータスをリセット
            st.session_state["realtime_error"] = ""
            st.session_state["realtime_partial"] = ""
            st.session_state["realtime_final"] = ""

            # スレッド間で共有するためのローカル変数
            msg_queue = st.session_state["realtime_msg_queue"]
            # threading.Eventを使用してスレッド間でrunning状態を共有
            running_event = threading.Event()
            running_event.set()  # Start as running
            st.session_state["_running_event"] = running_event

            # audio_receiverから直接フレームを取得するスレッド
            def pull_audio_frames():
                import time as _time
                import logging
                import sys
                frames_sent = 0

                # ログ設定
                logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
                logger = logging.getLogger("audio_puller")

                # Force flush to stderr
                sys.stderr.write("[FRONTEND] 🎙️ Audio puller thread started!\n")
                sys.stderr.write(f"[FRONTEND] WebRTC state: {rtc_ctx.state if rtc_ctx else 'No context'}\n")
                sys.stderr.write(f"[FRONTEND] Has receiver: {rtc_ctx.audio_receiver is not None if rtc_ctx else False}\n")
                sys.stderr.write(f"[FRONTEND] Running event is set: {running_event.is_set()}\n")
                sys.stderr.flush()

                logger.info("Starting audio puller thread",
                           extra={"extra_fields": {"component": "audio_puller", "action": "thread_start"}})
                logger.info("Audio puller context info",
                           extra={"extra_fields": {"component": "audio_puller", "action": "context_info",
                                                 "rtc_state": str(rtc_ctx.state), "has_receiver": rtc_ctx.audio_receiver is not None}})

                consecutive_errors = 0
                max_consecutive_errors = 50  # 50回連続エラーで停止
                loop_count = 0

                sys.stderr.write("[FRONTEND] 🔄 Entering while loop...\n")
                sys.stderr.flush()

                while running_event.is_set():
                    if loop_count == 0:  # First iteration
                        sys.stderr.write(f"[FRONTEND] 🔄 First loop iteration! rtc_ctx.state.playing = {rtc_ctx.state.playing}\n")
                        sys.stderr.flush()

                    loop_count += 1
                    if loop_count % 100 == 0:  # Every 100 iterations
                        sys.stderr.write(f"[FRONTEND] Audio puller loop iteration {loop_count}\n")
                        sys.stderr.flush()

                    # WebRTC接続状態をチェック
                    if not rtc_ctx.state.playing:
                        sys.stderr.write(f"[FRONTEND] ❌ WebRTC NOT playing! Stopping puller. (iteration {loop_count})\n")
                        sys.stderr.flush()
                        logger.warning("WebRTC connection lost (not playing), stopping puller",
                                     extra={"extra_fields": {"component": "audio_puller", "action": "connection_lost"}})
                        if msg_queue:
                            try:
                                msg_queue.put({"type": "error", "payload": {"message": "WebRTC接続が切断されました"}})
                            except Exception:
                                pass
                        break

                    if rtc_ctx.audio_receiver:
                        try:
                            # streamlit-webrtcの正しい使用方法に修正
                            frames = []
                            try:
                                # streamlit-webrtcでは引数なしでget_frames()を呼び出す
                                frames = rtc_ctx.audio_receiver.get_frames()
                                if frames:
                                    sys.stderr.write(f"[FRONTEND] 🎤 Got {len(frames)} frames from WebRTC!\n")
                                    sys.stderr.flush()
                                    logger.info(f"🎤 Got {len(frames)} audio frames from WebRTC",
                                              extra={"extra_fields": {"component": "audio_puller", "action": "frames_received",
                                                                    "frame_count": len(frames)}})
                                elif loop_count % 500 == 0:  # Log every 500 empty iterations
                                    sys.stderr.write(f"[FRONTEND] ⚠️ get_frames() returned empty (iteration {loop_count})\n")
                                    sys.stderr.flush()
                            except Exception as get_frames_error:
                                # ログレベルを下げて、頻繁なエラーログを抑制
                                logger.debug("get_frames() failed, trying alternative approach",
                                             extra={"extra_fields": {"component": "audio_puller", "action": "get_frames_alternative",
                                                                   "error": str(get_frames_error), "error_type": type(get_frames_error).__name__}})
                                # 代替方法: 引数なしで再試行
                                try:
                                    frames = rtc_ctx.audio_receiver.get_frames()
                                except Exception as alt_error:
                                    logger.debug("Alternative frame getting failed",
                                               extra={"extra_fields": {"component": "audio_puller", "action": "alt_get_frames_error",
                                                                     "error": str(alt_error), "error_type": type(alt_error).__name__}})
                                    _time.sleep(0.1)
                                    continue
                            if frames:
                                # フレーム受信（ログ削除）

                                for frame in frames:
                                    try:
                                        pcm = frame.to_ndarray(format="flt")

                                        if pcm.ndim == 2 and pcm.shape[0] > 1:
                                            pcm_mono = pcm.mean(axis=0)
                                        else:
                                            pcm_mono = pcm[0] if pcm.ndim == 2 else pcm

                                        pcm_f32 = pcm_mono.astype(np.float32)

                                        # 音声フレーム処理（ログ削除）
                                        # キューサイズをチェックして、満杯の場合は古いフレームをドロップ
                                        if send_queue.qsize() > 800:  # 80%以上の場合
                                            try:
                                                send_queue.get_nowait()  # 古いフレームを1つ削除
                                                # 古いフレームドロップ（ログ削除）
                                            except queue.Empty:
                                                pass

                                        # 送信キューに積む（満杯時は例外処理）
                                        try:
                                            send_queue.put(pcm_f32.tobytes(), timeout=0.05)  # タイムアウトを短縮
                                            frames_sent += 1
                                            consecutive_errors = 0  # 成功したらエラーカウントリセット
                                        except queue.Full:
                                            logger.warning("Send queue is full, dropping frame",
                                                         extra={"extra_fields": {"component": "audio_puller", "action": "queue_full",
                                                                               "queue_size": send_queue.qsize()}})
                                            consecutive_errors += 1

                                        # 統計情報をメッセージキューに送信
                                        try:
                                            if msg_queue is not None:
                                                msg_queue.put({"type": "stats", "payload": {
                                                    "frames_sent": frames_sent,
                                                    "queue_size": send_queue.qsize(),
                                                    "queue_capacity": send_queue.maxsize
                                                }})
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        consecutive_errors += 1
                                        logger.debug("Error processing audio frame",
                                                   extra={"extra_fields": {"component": "audio_puller", "action": "frame_error",
                                                                         "error": str(e), "consecutive_errors": consecutive_errors}})

                                        # 連続エラーが多すぎる場合は停止
                                        if consecutive_errors >= max_consecutive_errors:
                                            logger.error("Too many consecutive errors, stopping puller",
                                                       extra={"extra_fields": {"component": "audio_puller", "action": "error_threshold",
                                                                             "consecutive_errors": consecutive_errors}})
                                            if msg_queue:
                                                try:
                                                    msg_queue.put({"type": "error", "payload": {"message": "音声処理エラーが多発しています"}})
                                                except Exception:
                                                    pass
                                            break
                                        continue
                            else:
                                # フレームが取得できない場合は短い間隔で待機
                                _time.sleep(0.02)  # 待機時間を少し長くして負荷軽減
                        except Exception as e:
                            # ログレベルを下げて、頻繁なエラーログを抑制
                            logger.debug("Error getting frames from audio_receiver",
                                       extra={"extra_fields": {"component": "audio_puller", "action": "get_frames_error",
                                                             "error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}})
                            _time.sleep(0.1)
                    else:
                        logger.debug("No audio_receiver available, waiting",
                                   extra={"extra_fields": {"component": "audio_puller", "action": "no_receiver"}})
                        _time.sleep(0.05)

            # WebSocket sender thread - use asyncio.run() instead of managing event loop manually
            def run_websocket_sender():
                import logging
                logger = logging.getLogger("websocket_loop")

                try:
                    logger.info("Starting WebSocket loop",
                               extra={"extra_fields": {"component": "websocket_loop", "action": "loop_start",
                                                     "model": selected_realtime_model or "conformer", "sample_rate": int(sample_rate)}})
                    # Use asyncio.run() which properly manages the event loop
                    asyncio.run(stream_audio_to_ws(send_queue, selected_realtime_model or "conformer", int(sample_rate), running_event, msg_queue))
                except Exception as e:
                    logger.error("WebSocket loop error",
                               extra={"extra_fields": {"component": "websocket_loop", "action": "loop_error",
                                                     "error": str(e), "traceback": traceback.format_exc()}})

            # Start threads WITHOUT triggering Streamlit reruns (no st.write after starting threads!)
            t = threading.Thread(target=run_websocket_sender, daemon=True)
            p = threading.Thread(target=pull_audio_frames, daemon=True)

            # Start both threads
            t.start()
            p.start()

            # Save to session state (note: we don't save 'loop' anymore since asyncio.run manages it)
            st.session_state["realtime_thread"] = t
            st.session_state["realtime_puller"] = p

            # Set a flag to indicate threads just started - DON'T check WebRTC state yet
            st.session_state["_threads_just_started"] = True

            # CRITICAL: After starting threads, rerun immediately to refresh UI
            # This prevents the status monitoring code below from running with stale rtc_ctx
            st.rerun()

    # Show streaming status
    if st.session_state.get("realtime_running", False):
        # Clear the "just started" flag after first status display
        if st.session_state.pop("_threads_just_started", False):
            # Threads just started - show status but DON'T check WebRTC state yet
            st.info("🎙️ リアルタイムストリーミング開始しました！モデルを読み込み中...")
            # Don't check rtc_ctx.state yet - it might be stale
        else:
            # Normal status monitoring
            st.info("🎙️ リアルタイムストリーミング実行中... 話してください！")

            # WebRTC接続状態を監視 (only after threads have settled)
            if rtc_ctx and rtc_ctx.state:
                if not rtc_ctx.state.playing:
                    st.warning("⚠️ WebRTC接続が切断されました。STARTボタンを押して再接続してください。")
                    # 自動的にストリーミングを停止
                    if st.session_state.get("realtime_running"):
                        logger = logging.getLogger("ui-rt")
                        logger.warning("WebRTC disconnected, stopping streaming",
                                     extra={"extra_fields": {"component": "ui", "action": "webrtc_disconnect_stop"}})
                        st.session_state["realtime_running"] = False
                        # Clear the threading event
                        running_event = st.session_state.get("_running_event")
                        if running_event:
                            running_event.clear()
                        loop = st.session_state.get("realtime_loop")
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(loop.stop)
                elif rtc_ctx.audio_receiver is None:
                    st.warning("⚠️ オーディオレシーバーが利用できません。")
    elif st.session_state.get("_should_start_realtime", False):
        # Model loading phase - show progress
        if "_model_loading_start_time" in st.session_state:
            elapsed = __import__('time').time() - st.session_state["_model_loading_start_time"]
            progress = min(elapsed / 15.0, 1.0)  # 15 seconds expected

            st.progress(progress)
            st.info(f"⏳ モデル読み込み中... {elapsed:.1f}秒 / ~15秒 ({progress*100:.0f}%)")
            st.warning("💡 **重要:** マイクをONのまま待ってください！")

            # Check if WebRTC is still alive
            if not rtc_ctx.state.playing:
                st.error(f"❌ マイクが {elapsed:.1f}秒後に切断されました！")
                st.info("💡 'START'ボタンをもう一度クリックして、マイクを再度ONにしてください。")

    if stop_btn and st.session_state.get("realtime_running", False):
        # 送信停止: スレッドとループを停止
        logger = logging.getLogger("ui-rt")
        logger.info("Stopping realtime streaming",
                   extra={"extra_fields": {"component": "ui", "action": "stop_streaming"}})

        st.session_state["realtime_running"] = False

        # Clear the threading event to stop threads
        running_event = st.session_state.get("_running_event")
        if running_event:
            running_event.clear()

        # スレッドの停止を待つ
        loop = st.session_state.get("realtime_loop")
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        # puller スレッドはフラグで停止。追加の操作は不要。
        st.success("✅ リアルタイム推論を停止しました。")
        st.info("💡 最終結果が下に表示されます。")

# メインスレッドでメッセージキューをドレインし、UI状態を更新
while not st.session_state["realtime_msg_queue"].empty():
    try:
        data = st.session_state["realtime_msg_queue"].get_nowait()
    except Exception:
        break
    if data.get("type") == "partial":
        st.session_state["realtime_partial"] = data["payload"].get("text", "")
    elif data.get("type") == "final":
        st.session_state["realtime_final"] = data["payload"].get("text", "")
    elif data.get("type") == "status":
        st.session_state["realtime_status"] = data.get("payload", {})
    elif data.get("type") == "error":
        st.session_state["realtime_error"] = data.get("payload", {}).get("message", "error")
    elif data.get("type") == "stats":
        st.session_state["realtime_stats"] = data.get("payload", {})

st.text_area("部分結果", value=st.session_state.get("realtime_partial", ""), height=80)
st.text_area("最終結果", value=st.session_state.get("realtime_final", ""), height=80)

stats = st.session_state.get("realtime_stats", {})
if stats:
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("frames_sent", value=f"{stats.get('frames_sent', 0)}")
    with col_s2:
        st.metric("chunks_sent", value=f"{stats.get('chunks_sent', st.session_state.get('_rt_chunks_sent', 0))}")
    with col_s3:
        # キューサイズの表示（デバッグ用）
        queue_size = stats.get('queue_size', 0)
        queue_status = "🟢 Normal" if queue_size < 500 else "🟡 High" if queue_size < 800 else "🔴 Critical"
        st.metric("Queue Size", value=f"{queue_size}/1000", help=f"Status: {queue_status}")

    if st.session_state.get("realtime_error"):
        st.error(st.session_state.get("realtime_error"))

# --- チェックポイント管理セクション ---
current_page = st.session_state.get("current_page", "main")
if current_page == "checkpoint_management":
    st.markdown("---")
    st.header("📂 チェックポイント管理")

    # 説明
    st.markdown("""
    このページでは、学習チェックポイントの一覧表示と管理を行うことができます。
    チェックポイントは学習中に自動的に保存され、学習の再開に使用できます。
    """)

    # フィルタリングオプション
    st.subheader("🔍 フィルタリング")
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        filter_model = st.selectbox(
            "モデル名でフィルタ",
            ["全て"] + st.session_state.available_models,
            key="checkpoint_filter_model"
        )

    with col_filter2:
        filter_dataset = st.selectbox(
            "データセット名でフィルタ",
            ["全て"] + st.session_state.available_datasets,
            key="checkpoint_filter_dataset"
        )

    # チェックポイント一覧の取得
    if st.button("🔄 チェックポイント一覧を更新", type="primary", key="refresh_checkpoints_main"):
        st.rerun()

    # フィルタリングパラメータを設定
    model_filter = filter_model if filter_model != "全て" else None
    dataset_filter = filter_dataset if filter_dataset != "全て" else None

    checkpoints = get_checkpoints(model_filter, dataset_filter)

    if not checkpoints:
        st.info("チェックポイントが見つかりません。学習を実行してチェックポイントを作成してください。")
    else:
        st.success(f"{len(checkpoints)}個のチェックポイントが見つかりました。")

        # チェックポイント一覧をテーブル形式で表示
        st.subheader("📋 チェックポイント一覧")

        # データフレーム用のデータを準備
        checkpoint_data = []
        for checkpoint in checkpoints:
            checkpoint_data.append({
                "チェックポイント名": checkpoint["name"],
                "モデル": checkpoint["model_name"],
                "データセット": checkpoint["dataset_name"],
                "エポック": checkpoint["epoch"],
                "サイズ": format_file_size(checkpoint["size_mb"]),
                "ファイル数": checkpoint["file_count"],
                "作成日時": format_timestamp(checkpoint["created_at"])
            })

        # テーブル表示
        st.dataframe(
            checkpoint_data,
            use_container_width=True,
            hide_index=True
        )

        # チェックポイント詳細と学習再開機能
        st.subheader("🔄 学習再開")
        st.info("💡 チェックポイントを選択して学習を再開できます。")

        # チェックポイント選択
        checkpoint_names = [cp["name"] for cp in checkpoints]
        selected_checkpoint = st.selectbox(
            "学習再開に使用するチェックポイントを選択してください:",
            checkpoint_names,
            index=None,
            placeholder="チェックポイントを選択...",
            key="checkpoint_selector"
        )

        if selected_checkpoint:
            # 選択されたチェックポイントの詳細を表示
            selected_checkpoint_info = next((cp for cp in checkpoints if cp["name"] == selected_checkpoint), None)
            if selected_checkpoint_info:
                st.markdown("### 選択されたチェックポイントの詳細")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("チェックポイント名", selected_checkpoint_info["name"])
                    st.metric("モデル", selected_checkpoint_info["model_name"])

                with col2:
                    st.metric("データセット", selected_checkpoint_info["dataset_name"])
                    st.metric("エポック", selected_checkpoint_info["epoch"])

                with col3:
                    st.metric("サイズ", format_file_size(selected_checkpoint_info["size_mb"]))
                    st.metric("ファイル数", selected_checkpoint_info["file_count"])

                # ファイル一覧
                st.markdown("#### 含まれるファイル:")
                for file_name in selected_checkpoint_info["files"]:
                    st.text(f"• {file_name}")

                # 学習再開パラメータ
                st.markdown("### 学習再開パラメータ")
                col_param1, col_param2 = st.columns(2)

                with col_param1:
                    resume_epochs = st.number_input("追加エポック数", min_value=1, value=5, key="resume_epochs")
                    resume_batch_size = st.number_input("バッチサイズ", min_value=1, value=4, key="resume_batch_size")

                with col_param2:
                    resume_lightweight = st.checkbox("軽量モード", value=True, key="resume_lightweight")
                    resume_limit_samples = st.number_input("サンプル数制限", min_value=0, value=0, key="resume_limit_samples")

                # 学習再開ボタン
                if st.button("🔄 このチェックポイントから学習を再開", type="primary", disabled=st.session_state.is_training, key="resume_from_checkpoint_button"):
                    if not st.session_state.is_training:
                        with st.spinner("学習を再開中..."):
                            success = resume_training(
                                selected_checkpoint_info["model_name"],
                                selected_checkpoint_info["dataset_name"],
                                resume_epochs,
                                resume_batch_size,
                                specific_checkpoint=selected_checkpoint,
                                lightweight=resume_lightweight,
                                limit_samples=resume_limit_samples
                            )

                            if success:
                                st.success("学習を再開しました！")
                                st.balloons()
                                # メインダッシュボードに戻る
                                st.session_state.current_page = "main"
                                st.rerun()
                            else:
                                st.error("学習の再開に失敗しました。ログを確認してください。")
                    else:
                        st.error("既に学習が実行中です。現在の学習を停止してから再開してください。")

    # フッター
    st.markdown("---")
    st.markdown("💡 **ヒント**: チェックポイントから学習を再開することで、学習時間を短縮できます。")

# --- モデル管理セクション ---
elif current_page == "model_management":
    st.markdown("---")
    st.header("🤖 学習済みモデル管理")

    # 説明
    st.markdown("""
    このページでは、学習済みモデルの一覧表示と削除を行うことができます。
    モデルは学習完了時に自動的に保存され、ここで管理できます。
    """)

    # モデル一覧の取得
    if st.button("🔄 モデル一覧を更新", type="primary", key="refresh_models_main"):
        st.rerun()

    models = get_models()

    if not models:
        st.info("学習済みモデルが見つかりません。学習を実行してモデルを作成してください。")
    else:
        st.success(f"{len(models)}個の学習済みモデルが見つかりました。")

        # モデル一覧をテーブル形式で表示
        st.subheader("📋 モデル一覧")

        # データフレーム用のデータを準備
        model_data = []
        for model in models:
            model_data.append({
                "モデル名": model["name"],
                "エポック": model["epoch"] if model["epoch"] else "不明",
                "サイズ": format_file_size(model["size_mb"]),
                "ファイル数": model["file_count"],
                "作成日時": format_timestamp(model["created_at"]),
                "パス": model["path"]
            })

        # テーブル表示
        st.dataframe(
            model_data,
            use_container_width=True,
            hide_index=True
        )

        # モデル詳細と削除機能
        st.subheader("🗑️ モデル削除")
        st.warning("⚠️ モデルを削除すると復元できません。削除前に十分確認してください。")

        # モデル選択
        model_names = [model["name"] for model in models]
        selected_model = st.selectbox(
            "削除するモデルを選択してください:",
            model_names,
            index=None,
            placeholder="モデルを選択...",
            key="model_selector"
        )

        if selected_model:
            # 選択されたモデルの詳細を表示
            selected_model_info = next((m for m in models if m["name"] == selected_model), None)
            if selected_model_info:
                st.markdown("### 選択されたモデルの詳細")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("モデル名", selected_model_info["name"])
                    st.metric("エポック", selected_model_info["epoch"] if selected_model_info["epoch"] else "不明")

                with col2:
                    st.metric("サイズ", format_file_size(selected_model_info["size_mb"]))
                    st.metric("ファイル数", selected_model_info["file_count"])

                with col3:
                    st.metric("作成日時", format_timestamp(selected_model_info["created_at"]))

                # ファイル一覧
                st.markdown("#### 含まれるファイル:")
                for file_name in selected_model_info["files"]:
                    st.text(f"• {file_name}")

                # 削除確認
                st.markdown("### 削除確認")
                confirm_text = st.text_input(
                    f"削除を確認するために、モデル名 '{selected_model}' を入力してください:",
                    placeholder="モデル名を入力...",
                    key="confirm_delete"
                )

                # 削除ボタン
                if st.button("🗑️ モデルを削除", type="secondary", disabled=confirm_text != selected_model, key="delete_model_button_main"):
                    if confirm_text == selected_model:
                        with st.spinner("モデルを削除中..."):
                            success, message = delete_model(selected_model)

                            if success:
                                st.success(message)
                                st.balloons()
                                # 削除後、ページを更新
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.error("モデル名が一致しません。正確に入力してください。")

    # フッター
    st.markdown("---")
    st.markdown("💡 **ヒント**: モデルを削除する前に、必要に応じてバックアップを取ることをお勧めします。")
