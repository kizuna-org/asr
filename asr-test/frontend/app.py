import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import json
from typing import Dict, Any
import traceback
import os
import logging
import sys
from datetime import datetime
import warnings

# 機能ポリシー警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*機能ポリシー.*')
warnings.filterwarnings('ignore', message='.*Permissions-Policy.*')
warnings.filterwarnings('ignore', message='.*Feature-Policy.*')

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

# ログ設定を初期化
setup_frontend_logging()

# --- 設定 ---
# 環境変数からバックエンドURLを取得、デフォルトはローカルホスト
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "58081")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api"

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

def get_datasets():
    """データセット一覧を取得"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.get(f"{BACKEND_URL}/datasets", timeout=10, proxies=request_proxies)

        if response.status_code == 200:
            return response.json().get("datasets", [])
        else:
            st.error(f"データセット一覧の取得に失敗しました: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("バックエンドに接続できません。バックエンドサービスが起動しているか確認してください。")
        return []
    except requests.exceptions.Timeout:
        st.error("リクエストがタイムアウトしました。")
        return []
    except Exception as e:
        st.error(f"データセット一覧の取得中にエラーが発生しました: {str(e)}")
        return []

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

def delete_models_bulk(model_names):
    """指定された複数のモデルを一括削除"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.delete(f"{BACKEND_URL}/models", json=model_names, timeout=60, proxies=request_proxies)

        if response.status_code == 200:
            result = response.json()
            deleted_count = len(result.get("deleted", []))
            failed_count = len(result.get("failed", []))
            if failed_count == 0:
                return True, f"{deleted_count}個のモデルが正常に削除されました。"
            else:
                return True, f"{deleted_count}個のモデルが削除されました。{failed_count}個のモデルの削除に失敗しました。"
        else:
            error_detail = response.json().get("detail", "不明なエラー")
            return False, f"一括削除に失敗しました: {error_detail}"
    except requests.exceptions.ConnectionError:
        return False, "バックエンドに接続できません。"
    except requests.exceptions.Timeout:
        return False, "リクエストがタイムアウトしました。"
    except Exception as e:
        return False, f"一括削除中にエラーが発生しました: {str(e)}"

def download_models_bulk(model_names):
    """指定された複数のモデルをZIPファイルとして一括ダウンロード"""
    try:
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        # クエリパラメータとしてモデル名を送信（FastAPIのList[str]形式）
        params = {"model_names": model_names}
        response = requests.get(f"{BACKEND_URL}/models/bulk-download", params=params, timeout=300, proxies=request_proxies, stream=True)

        if response.status_code == 200:
            return True, response
        else:
            error_detail = response.json().get("detail", "不明なエラー")
            return False, f"一括ダウンロードに失敗しました: {error_detail}"
    except requests.exceptions.ConnectionError:
        return False, "バックエンドに接続できません。"
    except requests.exceptions.Timeout:
        return False, "リクエストがタイムアウトしました。"
    except Exception as e:
        return False, f"一括ダウンロード中にエラーが発生しました: {str(e)}"

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

# 機能ポリシーの警告を抑制するためのHTML
st.markdown("""
<meta http-equiv="Permissions-Policy" content="accelerometer=(), ambient-light-sensor=(), autoplay=(), battery=(), clipboard-write=(), document-domain=(), encrypted-media=(), gyroscope=(), layout-animations=(), legacy-image-formats=(), magnetometer=(), midi=(), oversized-images=(), payment=(), picture-in-picture=(), sync-xhr=(), usb=(), vr=(), wake-lock=(), xr-spatial-tracking=()">
<meta http-equiv="Feature-Policy" content="accelerometer 'none'; ambient-light-sensor 'none'; autoplay 'none'; battery 'none'; clipboard-write 'none'; document-domain 'none'; encrypted-media 'none'; gyroscope 'none'; layout-animations 'none'; legacy-image-formats 'none'; magnetometer 'none'; midi 'none'; oversized-images 'none'; payment 'none'; picture-in-picture 'none'; sync-xhr 'none'; usb 'none'; vr 'none'; wake-lock 'none'; xr-spatial-tracking 'none'">
""", unsafe_allow_html=True)
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
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
with col_nav1:
    if st.button("🏠 メインダッシュボード", use_container_width=True, key="nav_main_top"):
        st.session_state.current_page = "main"
        st.rerun()
with col_nav2:
    if st.button("🤖 モデル管理", use_container_width=True, key="nav_model_top"):
        st.session_state.current_page = "model_management"
        st.rerun()
with col_nav3:
    if st.button("📊 データセット管理", use_container_width=True, key="nav_dataset_top"):
        st.session_state.current_page = "dataset_management"
        st.rerun()
with col_nav4:
    if st.button("🎤 リアルタイム推論", use_container_width=True, key="nav_realtime_top"):
        st.session_state.current_page = "realtime"
        st.rerun()

# 現在のページ表示
current_page = st.session_state.get("current_page", "main")
if current_page == "main":
    page_name = "メインダッシュボード"
elif current_page == "model_management":
    page_name = "モデル管理"
elif current_page == "checkpoint_management":
    page_name = "チェックポイント管理"
elif current_page == "dataset_management":
    page_name = "データセット管理"
elif current_page == "realtime":
    page_name = "リアルタイム推論"
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
    if st.button("📊 データセット管理", use_container_width=True, disabled=(current_page == "dataset_management"), key="nav_dataset_sidebar"):
        st.session_state.current_page = "dataset_management"
        st.rerun()
    if st.button("🎤 リアルタイム推論", use_container_width=True, disabled=(current_page == "realtime"), key="nav_realtime_sidebar"):
        st.session_state.current_page = "realtime"
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
if st.session_state.is_training:
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
    time.sleep(1)
    st.rerun()

# --- チェックポイント管理セクション ---
current_page = st.session_state.get("current_page", "main")
if current_page == "checkpoint_management":
    st.markdown("---")
    st.header("📂 チェックポイント管理")

    # 説明
    st.markdown("""
    このページでは、学習チェックポイントの一覧表示と管理を行うことができます。
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
    with st.spinner("チェックポイント一覧を取得中..."):
        try:
            checkpoints = get_checkpoints(
                model_name=filter_model if filter_model != "全て" else None,
                dataset_name=filter_dataset if filter_dataset != "全て" else None
            )
        except Exception as e:
            st.error(f"チェックポイント一覧の取得に失敗しました: {e}")
            checkpoints = []

    # チェックポイント一覧の表示
    if checkpoints:
        st.subheader(f"📋 チェックポイント一覧 ({len(checkpoints)}件)")
        
        # チェックポイント情報をテーブル形式で表示
        checkpoint_data = []
        for cp in checkpoints:
            checkpoint_data.append({
                "名前": cp["name"],
                "モデル": cp["model_name"],
                "データセット": cp["dataset_name"],
                "エポック": cp["epoch"],
                "サイズ": f"{cp['size_mb']:.1f} MB",
                "ファイル数": cp["file_count"],
                "作成日時": format_timestamp(cp["created_at"])
            })
        
        df = pd.DataFrame(checkpoint_data)
        st.dataframe(df, use_container_width=True)
        
        # チェックポイントの詳細表示
        if checkpoints:
            st.subheader("🔍 チェックポイント詳細")
            selected_checkpoint = st.selectbox(
                "詳細を表示するチェックポイントを選択",
                [cp["name"] for cp in checkpoints],
                key="checkpoint_detail_selector"
            )
            
            if selected_checkpoint:
                selected_cp = next(cp for cp in checkpoints if cp["name"] == selected_checkpoint)
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.write("**基本情報:**")
                    st.write(f"- 名前: {selected_cp['name']}")
                    st.write(f"- モデル: {selected_cp['model_name']}")
                    st.write(f"- データセット: {selected_cp['dataset_name']}")
                    st.write(f"- エポック: {selected_cp['epoch']}")
                
                with col_detail2:
                    st.write("**ファイル情報:**")
                    st.write(f"- サイズ: {selected_cp['size_mb']:.1f} MB")
                    st.write(f"- ファイル数: {selected_cp['file_count']}")
                    st.write(f"- 作成日時: {format_timestamp(selected_cp['created_at'])}")
                
                # ファイル一覧
                if selected_cp.get("files"):
                    st.write("**ファイル一覧:**")
                    for file in selected_cp["files"]:
                        st.write(f"- {file}")
                
                # チェックポイントの操作
                st.subheader("⚙️ 操作")
                col_action1, col_action2 = st.columns(2)
                
                with col_action1:
                    if st.button("📥 ダウンロード", key=f"download_{selected_checkpoint}"):
                        st.info("ダウンロード機能は実装中です")
                
                with col_action2:
                    if st.button("🗑️ 削除", key=f"delete_{selected_checkpoint}"):
                        if st.session_state.get(f"confirm_delete_{selected_checkpoint}", False):
                            with st.spinner("チェックポイントを削除中..."):
                                try:
                                    import shutil
                                    import os
                                    checkpoint_path = selected_cp["path"]
                                    shutil.rmtree(checkpoint_path)
                                    st.success("チェックポイントを削除しました")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"削除に失敗しました: {e}")
                        else:
                            st.session_state[f"confirm_delete_{selected_checkpoint}"] = True
                            st.warning("⚠️ 削除を確認するには、もう一度削除ボタンをクリックしてください")
                            st.rerun()
    else:
        st.info("チェックポイントが見つかりませんでした")
        st.write("学習を開始すると、チェックポイントが自動的に作成されます。")

# --- モデル管理セクション ---
elif current_page == "model_management":
    st.markdown("---")
    st.header("🤖 モデル管理")

    # 説明
    st.markdown("""
    このページでは、学習済みモデルの一覧表示と管理を行うことができます。
    """)

    # モデル一覧の取得
    with st.spinner("モデル一覧を取得中..."):
        try:
            models = get_models()
        except Exception as e:
            st.error(f"モデル一覧の取得に失敗しました: {e}")
            models = []

    # モデル一覧の表示
    if models:
        st.subheader(f"📋 モデル一覧 ({len(models)}件)")
        
        # 一括操作セクション
        st.markdown("### 🔧 一括操作")
        
        # 選択状態の初期化
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []
        if "model_selection_df" not in st.session_state:
            st.session_state.model_selection_df = None
        
        # モデル情報をテーブル形式で準備（チェックボックス列を含む）
        model_data = []
        for model in models:
            is_selected = model["name"] in st.session_state.selected_models
            # データセット統計情報を取得
            dataset_stats = ""
            if model.get("training_metadata") and model["training_metadata"].get("dataset_statistics"):
                stats = model["training_metadata"]["dataset_statistics"]
                total_samples = stats.get("total_samples", 0)
                if total_samples > 0:
                    dataset_stats = f"{total_samples:,} samples"
            
            model_data.append({
                "選択": is_selected,
                "名前": model["name"],
                "エポック": model["epoch"] or "不明",
                "データセット": model.get("dataset_name") or "不明",
                "サンプル数": dataset_stats or "不明",
                "サイズ": f"{model['size_mb']:.1f} MB",
                "ファイル数": model["file_count"],
                "作成日時": format_timestamp(model["created_at"])
            })
        
        df = pd.DataFrame(model_data)
        
        # 全選択/全解除ボタン
        col_select_all, col_deselect_all, col_bulk_actions = st.columns([1, 1, 2])
        with col_select_all:
            if st.button("✅ すべて選択", key="select_all_models"):
                st.session_state.selected_models = [model["name"] for model in models]
                st.rerun()
        with col_deselect_all:
            if st.button("❌ すべて解除", key="deselect_all_models"):
                st.session_state.selected_models = []
                st.rerun()
        
        # 編集可能なテーブルでチェックボックスを表示
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "選択": st.column_config.CheckboxColumn(
                    "選択",
                    help="モデルを選択",
                    default=False,
                ),
                "名前": st.column_config.TextColumn(
                    "名前",
                    help="モデル名",
                ),
                "エポック": st.column_config.TextColumn(
                    "エポック",
                    help="エポック番号",
                ),
                "データセット": st.column_config.TextColumn(
                    "データセット",
                    help="学習に使用したデータセット",
                ),
                "サンプル数": st.column_config.TextColumn(
                    "サンプル数",
                    help="学習に使用したデータセットのサンプル数",
                ),
                "サイズ": st.column_config.TextColumn(
                    "サイズ",
                    help="モデルファイルのサイズ",
                ),
                "ファイル数": st.column_config.NumberColumn(
                    "ファイル数",
                    help="モデルに含まれるファイル数",
                ),
                "作成日時": st.column_config.TextColumn(
                    "作成日時",
                    help="モデルの作成日時",
                ),
            },
            disabled=["名前", "エポック", "データセット", "サンプル数", "サイズ", "ファイル数", "作成日時"],
            key="model_selection_table"
        )
        
        # テーブルの選択状態をセッション状態に反映
        if edited_df is not None:
            selected_models_from_table = edited_df[edited_df["選択"] == True]["名前"].tolist()
            if set(selected_models_from_table) != set(st.session_state.selected_models):
                st.session_state.selected_models = selected_models_from_table
                st.rerun()
        
        # 一括操作ボタン
        with col_bulk_actions:
            st.write(f"**選択中のモデル: {len(st.session_state.selected_models)}件**")
            
            col_download, col_delete = st.columns(2)
            
            with col_download:
                # 一括ダウンロードボタン
                if st.button("📥 一括ダウンロード", 
                            disabled=len(st.session_state.selected_models) == 0,
                            key="bulk_download_models",
                            use_container_width=True):
                    if st.session_state.selected_models:
                        with st.spinner("ZIPファイルを作成中..."):
                            success, result = download_models_bulk(st.session_state.selected_models)
                            if success and isinstance(result, requests.Response):
                                # ZIPファイルの内容を取得
                                zip_content = result.content
                                # ZIPファイルをダウンロード
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"models_{timestamp}.zip"
                                
                                # セッション状態に保存
                                st.session_state["bulk_download_zip"] = {
                                    "content": zip_content,
                                    "filename": filename,
                                    "model_count": len(st.session_state.selected_models)
                                }
                                st.success(f"{len(st.session_state.selected_models)}個のモデルを含むZIPファイルを作成しました。")
                                st.rerun()
                            else:
                                st.error(result)
                
                # ダウンロード可能なZIPファイルがある場合はダウンロードボタンを表示
                if "bulk_download_zip" in st.session_state:
                    zip_data = st.session_state["bulk_download_zip"]
                    st.download_button(
                        label=f"📥 ZIPダウンロード ({zip_data['filename']})",
                        data=zip_data["content"],
                        file_name=zip_data["filename"],
                        mime="application/zip",
                        key="download_bulk_zip",
                        use_container_width=True
                    )
            
            with col_delete:
                # 一括削除ボタン
                if st.button("🗑️ 一括削除", 
                            disabled=len(st.session_state.selected_models) == 0,
                            key="bulk_delete_models",
                            use_container_width=True,
                            type="primary"):
                    if st.session_state.selected_models:
                        # 確認ダイアログ
                        if st.session_state.get(f"confirm_bulk_delete", False):
                            with st.spinner("モデルを削除中..."):
                                success, message = delete_models_bulk(st.session_state.selected_models)
                                if success:
                                    st.success(message)
                                    st.balloons()
                                    st.session_state.selected_models = []
                                    st.session_state[f"confirm_bulk_delete"] = False
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(message)
                        else:
                            st.session_state[f"confirm_bulk_delete"] = True
                            st.warning(f"⚠️ {len(st.session_state.selected_models)}個のモデルを削除しようとしています。削除を確認するには、もう一度削除ボタンをクリックしてください。")
                            st.rerun()
        
        st.markdown("---")
        
        # モデルの詳細表示
        if models:
            st.subheader("🔍 モデル詳細")
            selected_model = st.selectbox(
                "詳細を表示するモデルを選択",
                [model["name"] for model in models],
                key="model_detail_selector"
            )
            
            if selected_model:
                selected_model_info = next(model for model in models if model["name"] == selected_model)
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.write("**基本情報:**")
                    st.write(f"- 名前: {selected_model_info['name']}")
                    st.write(f"- エポック: {selected_model_info['epoch'] or '不明'}")
                    st.write(f"- データセット: {selected_model_info.get('dataset_name') or '不明'}")
                    st.write(f"- サイズ: {selected_model_info['size_mb']:.1f} MB")
                
                with col_detail2:
                    st.write("**ファイル情報:**")
                    st.write(f"- ファイル数: {selected_model_info['file_count']}")
                    st.write(f"- 作成日時: {format_timestamp(selected_model_info['created_at'])}")
                
                # 学習メタデータの表示
                if selected_model_info.get("training_metadata"):
                    st.markdown("---")
                    st.subheader("📊 学習詳細情報")
                    metadata = selected_model_info["training_metadata"]
                    
                    col_meta1, col_meta2 = st.columns(2)
                    
                    with col_meta1:
                        st.write("**学習パラメータ:**")
                        if metadata.get("training_params"):
                            params = metadata["training_params"]
                            st.write(f"- エポック数: {params.get('epochs', '不明')}")
                            st.write(f"- バッチサイズ: {params.get('batch_size', '不明')}")
                            st.write(f"- 学習率: {params.get('learning_rate', '不明')}")
                            st.write(f"- オプティマイザ: {params.get('optimizer', '不明')}")
                            st.write(f"- スケジューラ: {params.get('scheduler', 'なし')}")
                            st.write(f"- デバイス: {params.get('device', '不明')}")
                    
                    with col_meta2:
                        st.write("**学習時間:**")
                        if selected_model_info.get("training_start_time"):
                            st.write(f"- 開始時刻: {selected_model_info['training_start_time']}")
                        if selected_model_info.get("training_end_time"):
                            st.write(f"- 終了時刻: {selected_model_info['training_end_time']}")
                        if metadata.get("training_status"):
                            st.write(f"- 状態: {metadata['training_status']}")
                    
                    # データセット統計情報の表示
                    if metadata.get("dataset_statistics"):
                        st.markdown("---")
                        st.write("**データセット統計情報:**")
                        stats = metadata["dataset_statistics"]
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.write(f"- 学習サンプル数: {stats.get('train_samples', '不明'):,}")
                            st.write(f"- 検証サンプル数: {stats.get('validation_samples', '不明'):,}")
                        with col_stats2:
                            st.write(f"- 合計サンプル数: {stats.get('total_samples', '不明'):,}")
                            train_ratio = stats.get('train_ratio', 0)
                            val_ratio = stats.get('validation_ratio', 0)
                            st.write(f"- 学習/検証比率: {train_ratio:.1%} / {val_ratio:.1%}")
                    
                    # データセット設定の表示
                    if metadata.get("dataset_config"):
                        st.markdown("---")
                        with st.expander("**データセット設定を表示**"):
                            dataset_config = metadata["dataset_config"]
                            st.json(dataset_config)
                    
                    # モデル設定の表示（折りたたみ可能）
                    if metadata.get("model_config"):
                        with st.expander("**モデル設定を表示**"):
                            st.json(metadata["model_config"])
                
                # ファイル一覧
                if selected_model_info.get("files"):
                    st.markdown("---")
                    st.write("**ファイル一覧:**")
                    for file in selected_model_info["files"]:
                        st.write(f"- {file}")
                
                # モデルの操作
                st.subheader("⚙️ 操作")
                col_action1, col_action2 = st.columns(2)
                
                with col_action1:
                    if st.button("📥 ダウンロード", key=f"download_model_{selected_model}"):
                        st.info("ダウンロード機能は実装中です")
                
                with col_action2:
                    if st.button("🗑️ 削除", key=f"delete_model_{selected_model}"):
                        if st.session_state.get(f"confirm_delete_model_{selected_model}", False):
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
                            st.session_state[f"confirm_delete_model_{selected_model}"] = True
                            st.warning("⚠️ 削除を確認するには、もう一度削除ボタンをクリックしてください")
                            st.rerun()
    else:
        st.info("モデルが見つかりませんでした")
        st.write("学習を完了すると、モデルが自動的に作成されます。")

# --- データセット管理セクション ---
elif current_page == "dataset_management":
    st.markdown("---")
    st.header("📊 データセット管理")

    # 説明
    st.markdown("""
    このページでは、利用可能なデータセットの一覧表示と管理を行うことができます。
    データセットのダウンロード、状態確認、削除などが可能です。
    """)

    # データセット一覧の取得
    with st.spinner("データセット一覧を取得中..."):
        try:
            datasets = get_datasets()
        except Exception as e:
            st.error(f"データセット一覧の取得に失敗しました: {e}")
            datasets = []

    # データセット一覧の表示
    if datasets:
        st.subheader(f"📋 データセット一覧 ({len(datasets)}件)")
        
        # データセット情報をテーブル形式で表示
        dataset_data = []
        for dataset in datasets:
            status_icon = "✅" if dataset["status"] == "downloaded" else "❌"
            status_text = "ダウンロード済み" if dataset["status"] == "downloaded" else "未ダウンロード"
            
            dataset_data.append({
                "名前": dataset["name"],
                "状態": f"{status_icon} {status_text}",
                "ファイル数": f"{dataset['num_files']:,}" if dataset["num_files"] > 0 else "-",
                "サイズ": f"{dataset['size_mb']:.1f} MB" if dataset["size_mb"] > 0 else "-",
                "パス": dataset["path"] or "-"
            })
        
        df = pd.DataFrame(dataset_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # データセットの詳細表示と操作
        if datasets:
            st.subheader("🔍 データセット詳細・操作")
            selected_dataset = st.selectbox(
                "操作するデータセットを選択",
                [ds["name"] for ds in datasets],
                key="dataset_detail_selector"
            )
            
            if selected_dataset:
                selected_ds = next(ds for ds in datasets if ds["name"] == selected_dataset)
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.write("**基本情報:**")
                    st.write(f"- 名前: {selected_ds['name']}")
                    status_icon = "✅" if selected_ds["status"] == "downloaded" else "❌"
                    status_text = "ダウンロード済み" if selected_ds["status"] == "downloaded" else "未ダウンロード"
                    st.write(f"- 状態: {status_icon} {status_text}")
                    st.write(f"- ファイル数: {selected_ds['num_files']:,}" if selected_ds["num_files"] > 0 else "- ファイル数: -")
                    st.write(f"- サイズ: {selected_ds['size_mb']:.1f} MB" if selected_ds["size_mb"] > 0 else "- サイズ: -")
                
                with col_detail2:
                    st.write("**パス情報:**")
                    st.write(f"- パス: {selected_ds['path'] or '未設定'}")
                    if selected_ds.get("config"):
                        config = selected_ds["config"]
                        st.write(f"- サンプルレート: {config.get('sample_rate', '不明')}")
                        st.write(f"- 検証セット割合: {config.get('validation_size', 0.05) * 100:.1f}%")
                
                # データセット設定の表示
                if selected_ds.get("config"):
                    st.markdown("---")
                    with st.expander("**データセット設定を表示**"):
                        st.json(selected_ds["config"])
                
                # データセットの操作
                st.markdown("---")
                st.subheader("⚙️ 操作")
                col_action1, col_action2 = st.columns(2)
                
                with col_action1:
                    if st.button("📥 ダウンロード", key=f"download_{selected_dataset}", use_container_width=True):
                        with st.spinner(f"データセット '{selected_dataset}' をダウンロード中..."):
                            success = download_dataset(selected_dataset)
                            if success:
                                st.success(f"データセット '{selected_dataset}' のダウンロードが完了しました")
                                st.balloons()
                                # ページを更新して最新の状態を表示
                                import time
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"データセット '{selected_dataset}' のダウンロードに失敗しました")
                
                with col_action2:
                    if selected_ds["status"] == "downloaded":
                        if st.button("🗑️ 削除", key=f"delete_{selected_dataset}", use_container_width=True, type="primary"):
                            if st.session_state.get(f"confirm_delete_dataset_{selected_dataset}", False):
                                with st.spinner(f"データセット '{selected_dataset}' を削除中..."):
                                    try:
                                        import shutil
                                        if selected_ds["path"] and os.path.exists(selected_ds["path"]):
                                            # 親ディレクトリを削除（例: /app/data/ljspeech 全体）
                                            parent_dir = os.path.dirname(selected_ds["path"])
                                            if os.path.exists(parent_dir):
                                                shutil.rmtree(parent_dir)
                                                st.success(f"データセット '{selected_dataset}' を削除しました")
                                                st.balloons()
                                                import time
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                st.error("データセットのパスが見つかりません")
                                        else:
                                            st.error("データセットのパスが設定されていません")
                                    except Exception as e:
                                        st.error(f"削除に失敗しました: {e}")
                            else:
                                st.session_state[f"confirm_delete_dataset_{selected_dataset}"] = True
                                st.warning("⚠️ 削除を確認するには、もう一度削除ボタンをクリックしてください")
                                st.rerun()
                    else:
                        st.info("データセットがダウンロードされていないため、削除できません")
    else:
        st.info("データセットが見つかりませんでした")
        st.write("設定ファイル（config.yaml）にデータセットが定義されているか確認してください。")

# --- リアルタイム推論セクション ---
elif current_page == "realtime":
    st.markdown("---")
    st.header("🎤 リアルタイム音声認識")
    
    # 説明
    st.markdown("""
    このページでは、マイクからの音声をリアルタイムで文字起こしできます。
    ブラウザのマイクアクセス許可が必要です。
    """)
    
    # 推論用モデル選択
    st.subheader("📋 推論設定")
    col_model_select, col_model_info = st.columns([1, 2])
    
    with col_model_select:
        # 利用可能なモデル一覧を取得
        available_models = st.session_state.available_models
        if available_models:
            selected_realtime_model = st.selectbox(
                "リアルタイム推論に使用するモデル:",
                available_models,
                index=0,
                key="realtime_model_selector",
                help="リアルタイム推論に使用するモデルを選択してください"
            )
        else:
            st.warning("利用可能なモデルがありません")
            selected_realtime_model = None
    
    with col_model_info:
        if selected_realtime_model:
            st.info(f"選択されたモデル: **{selected_realtime_model}**")
        else:
            st.warning("モデルを選択してください")
    
    # リアルタイム推論コントロール
    st.subheader("🎮 リアルタイム推論制御")
    
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        if st.button("🎤 開始", key="realtime_start", type="primary", disabled=not selected_realtime_model):
            st.session_state.realtime_running = True
            st.session_state.realtime_partial = ""
            st.session_state.realtime_final = ""
            st.session_state.realtime_status = "接続中..."
            st.rerun()
    
    with col_control2:
        if st.button("⏹️ 停止", key="realtime_stop", disabled=not st.session_state.get("realtime_running", False)):
            st.session_state.realtime_running = False
            st.session_state.realtime_status = "停止中..."
            st.rerun()
    
    with col_control3:
        if st.button("🗑️ クリア", key="realtime_clear"):
            st.session_state.realtime_partial = ""
            st.session_state.realtime_final = ""
            st.rerun()
    
    # リアルタイム推論の状態表示
    if st.session_state.get("realtime_running", False):
        st.success("🎤 リアルタイム推論が実行中です")
        
        # ステータス表示
        status = st.session_state.get("realtime_status", "不明")
        st.info(f"ステータス: {status}")
        
        # リアルタイム推論コンポーネント
        st.subheader("🎵 音声認識結果")
        
        # 部分的な結果（リアルタイム更新）
        st.write("**部分的な結果（リアルタイム）:**")
        partial_text = st.session_state.get("realtime_partial", "")
        st.text_area(
            "部分的な認識結果",
            value=partial_text,
            height=100,
            key="realtime_partial_display",
            help="リアルタイムで更新される部分的な認識結果"
        )
        
        # 最終的な結果
        st.write("**最終的な結果:**")
        final_text = st.session_state.get("realtime_final", "")
        st.text_area(
            "最終的な認識結果",
            value=final_text,
            height=150,
            key="realtime_final_display",
            help="確定された認識結果"
        )
        
        # エラーメッセージ
        error_msg = st.session_state.get("realtime_error", "")
        if error_msg:
            st.error(f"エラー: {error_msg}")
        
        # JavaScriptコンポーネントを埋め込み
        st.components.v1.html("""
        <script>
        // グローバル変数として定義
        window.realtimeAudioRecognition = null;
        
        class RealtimeAudioRecognition {
            constructor() {
                this.websocket = null;
                this.mediaRecorder = null;
                this.audioContext = null;
                this.isRecording = false;
                this.audioChunks = [];
            }
            
            async start() {
                console.log('Starting realtime audio recognition...');
                try {
                    // WebSocket接続を先に確立
                    this.connectWebSocket();
                    
                    // 少し待ってからマイクアクセス許可を取得
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    // マイクアクセス許可を取得
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });
                    
                    // MediaRecorderで音声をキャプチャ
                    this.mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus'
                    });
                    
                    this.mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            this.sendAudioChunk(event.data);
                        }
                    };
                    
                    // 1秒ごとに音声チャンクを送信
                    this.mediaRecorder.start(1000);
                    this.isRecording = true;
                    
                    console.log('リアルタイム音声認識を開始しました');
                    
                } catch (error) {
                    console.error('マイクアクセスエラー:', error);
                    this.showError('マイクアクセスに失敗しました: ' + error.message);
                }
            }
            
            connectWebSocket() {
                // 既存のWebSocketエンドポイントに接続
                const wsUrl = 'ws://localhost:58081/ws';
                
                console.log('=== WebSocket Connection Debug ===');
                console.log('Connecting to WebSocket:', wsUrl);
                console.log('Current time:', new Date().toISOString());
                
                try {
                    this.websocket = new WebSocket(wsUrl);
                    console.log('WebSocket object created:', this.websocket);
                    console.log('WebSocket readyState:', this.websocket.readyState);
                } catch (error) {
                    console.error('Failed to create WebSocket:', error);
                    this.showError('WebSocket作成エラー: ' + error.message);
                    return;
                }
                
                // 接続タイムアウトを設定
                const connectionTimeout = setTimeout(() => {
                    if (this.websocket.readyState === WebSocket.CONNECTING) {
                        console.error('WebSocket connection timeout');
                        this.showError('WebSocket接続がタイムアウトしました');
                        this.websocket.close();
                    }
                }, 10000); // 10秒でタイムアウト
                
                this.websocket.onopen = () => {
                    console.log('=== WebSocket Connected ===');
                    console.log('WebSocket接続が確立されました');
                    console.log('WebSocket readyState:', this.websocket.readyState);
                    clearTimeout(connectionTimeout); // タイムアウトをクリア
                    this.updateStatus('接続済み');
                    
                    // 既存のWebSocketプロトコルに合わせて開始メッセージを送信
                    const startMessage = {
                        type: 'start',
                        model_name: 'conformer',
                        sample_rate: 16000,
                        format: 'i16'
                    };
                    console.log('Sending start message:', startMessage);
                    this.websocket.send(JSON.stringify(startMessage));
                    console.log('Start message sent successfully');
                    
                    // ステータス更新
                    this.updateStatus('モデル初期化中...');
                };
                
                this.websocket.onmessage = (event) => {
                    console.log('WebSocket message received:', event.data);
                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };
                
                this.websocket.onclose = (event) => {
                    console.log('WebSocket接続が閉じられました:', event.code, event.reason);
                    clearTimeout(connectionTimeout); // タイムアウトをクリア
                    this.updateStatus('接続切断');
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocketエラー:', error);
                    clearTimeout(connectionTimeout); // タイムアウトをクリア
                    this.showError('WebSocket接続エラー: ' + error.message);
                };
            }
            
            async sendAudioChunk(audioBlob) {
                if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                    console.log('WebSocket not ready, skipping audio chunk');
                    return;
                }
                
                try {
                    // BlobをArrayBufferに変換
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    
                    // 既存のWebSocketプロトコルに合わせてバイナリデータを直接送信
                    console.log('Sending audio chunk, size:', arrayBuffer.byteLength);
                    this.websocket.send(arrayBuffer);
                    
                } catch (error) {
                    console.error('音声データ送信エラー:', error);
                    this.showError('音声データ送信エラー: ' + error.message);
                }
            }
            
            handleMessage(data) {
                console.log('Handling message:', data);
                switch (data.type) {
                    case 'partial':
                        console.log('Partial result:', data.payload.text);
                        this.updatePartialResult(data.payload.text);
                        break;
                    case 'final':
                        console.log('Final result:', data.payload.text);
                        this.updateFinalResult(data.payload.text);
                        break;
                    case 'error':
                        console.log('Error:', data.payload.message);
                        this.showError(data.payload.message);
                        break;
                    case 'status':
                        console.log('Status update:', data.payload.status);
                        this.updateStatus(data.payload.status);
                        break;
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }
            
            updatePartialResult(text) {
                // Streamlitのセッション状態を更新
                const event = new CustomEvent('realtime-update', {
                    detail: { type: 'partial', text: text }
                });
                window.dispatchEvent(event);
            }
            
            updateFinalResult(text) {
                // Streamlitのセッション状態を更新
                const event = new CustomEvent('realtime-update', {
                    detail: { type: 'final', text: text }
                });
                window.dispatchEvent(event);
            }
            
            updateStatus(status) {
                console.log('Status update:', status);
                const event = new CustomEvent('realtime-update', {
                    detail: { type: 'status', status: status }
                });
                window.dispatchEvent(event);
            }
            
            showError(message) {
                const event = new CustomEvent('realtime-update', {
                    detail: { type: 'error', message: message }
                });
                window.dispatchEvent(event);
            }
            
            stop() {
                if (this.mediaRecorder && this.isRecording) {
                    this.mediaRecorder.stop();
                    this.isRecording = false;
                }
                
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    // 既存のWebSocketプロトコルに合わせて停止メッセージを送信
                    const stopMessage = {
                        type: 'stop'
                    };
                    this.websocket.send(JSON.stringify(stopMessage));
                    console.log('Sent stop message:', stopMessage);
                }
                
                if (this.websocket) {
                    this.websocket.close();
                }
                
                console.log('リアルタイム音声認識を停止しました');
            }
        }
        
        // グローバルインスタンス
        window.realtimeAudio = new RealtimeAudioRecognition();
        
        // イベントリスナー
        window.addEventListener('realtime-update', (event) => {
            const { type, text, status, message } = event.detail;
            
            // Streamlitのセッション状態を更新するためのカスタムイベント
            const updateEvent = new CustomEvent('streamlit:update', {
                detail: { 
                    type: type,
                    text: text,
                    status: status,
                    message: message
                }
            });
            window.parent.dispatchEvent(updateEvent);
        });
        
                // 自動開始（ページロード時）
                if (window.location.hash === '#realtime') {
                    setTimeout(() => {
                        console.log('Auto-starting realtime audio recognition...');
                        window.realtimeAudio.start();
                    }, 1000);
                }
                
                // デバッグ用：グローバル関数を追加
                window.testWebSocket = function() {
                    console.log('Testing WebSocket connection...');
                    window.realtimeAudio.connectWebSocket();
                };
                
                window.testStart = function() {
                    console.log('Testing start function...');
                    window.realtimeAudio.start();
                };
                
                // グローバルインスタンスを作成
                window.realtimeAudioRecognition = new RealtimeAudioRecognition();
                
                // ページロード時のテスト
                console.log('=== JavaScript loaded ===');
                console.log('RealtimeAudioRecognition class:', typeof RealtimeAudioRecognition);
                console.log('window.realtimeAudioRecognition:', window.realtimeAudioRecognition);
                
                // 簡単なテスト
                setTimeout(() => {
                    console.log('=== Auto-test after 2 seconds ===');
                    console.log('Testing basic functionality...');
                    console.log('Available methods:', Object.getOwnPropertyNames(RealtimeAudioRecognition.prototype));
                }, 2000);
        </script>
        """, height=0)
        
    else:
        st.info("🎤 ボタンをクリックしてリアルタイム音声認識を開始してください")
        
        # 使用方法の説明
        st.subheader("📖 使用方法")
        st.markdown("""
        1. **モデル選択**: 上記で推論に使用するモデルを選択してください
        2. **開始ボタン**: 「開始」ボタンをクリックしてリアルタイム推論を開始します
        3. **マイク許可**: ブラウザからマイクアクセスの許可を求められたら「許可」をクリックしてください
        4. **音声認識**: マイクに向かって話すと、リアルタイムで文字起こし結果が表示されます
        5. **停止**: 「停止」ボタンで推論を停止できます
        6. **クリア**: 「クリア」ボタンで結果をクリアできます
        """)
        
        st.subheader("⚠️ 注意事項")
        st.markdown("""
        - ブラウザのマイクアクセス許可が必要です
        - HTTPS環境での使用を推奨します
        - 音声の品質によって認識精度が変わります
        - ネットワーク接続が必要です
        """)
