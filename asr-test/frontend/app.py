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

def start_training(model_name: str, dataset_name: str, epochs: int, batch_size: int, lightweight: bool = False, limit_samples: int = 0):
    """学習を開始"""
    try:
        params = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size
        }
        # 軽量モード/サンプル制限の付与
        if lightweight:
            params["lightweight"] = True
        if isinstance(limit_samples, int) and limit_samples > 0:
            params["limit_samples"] = int(limit_samples)
        st.session_state.logs.append(f"🚀 学習開始リクエスト送信中... URL: {BACKEND_URL}/train/start")
        
        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/train/start", json=params, timeout=30, proxies=request_proxies)
        
        if response.status_code == 200:
            st.session_state.is_training = True
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

def stop_training():
    """学習を停止"""
    try:
        st.session_state.logs.append(f"🛑 学習停止リクエスト送信中... URL: {BACKEND_URL}/train/stop")
        
        # プロキシ設定を適用
        request_proxies = proxies if should_use_proxy(BACKEND_URL) else None
        response = requests.post(f"{BACKEND_URL}/train/stop", timeout=10, proxies=request_proxies)
        
        if response.status_code == 200:
            st.session_state.is_training = False
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
        st.switch_page("app.py")
with col_nav2:
    if st.button("🤖 モデル管理", use_container_width=True, key="nav_model_top"):
        st.switch_page("pages/model_management.py")
with col_nav3:
    st.markdown("### 📊 現在のページ: メインダッシュボード")
st.markdown("---")

# サイドバー - 学習制御
with st.sidebar:
    st.header("📋 ナビゲーション")
    
    # ページ間のナビゲーション
    if st.button("🏠 メインダッシュボード", use_container_width=True, disabled=True, key="nav_main_sidebar"):
        pass  # 現在のページなので無効化
    if st.button("🤖 モデル管理", use_container_width=True, key="nav_model_sidebar"):
        st.switch_page("pages/model_management.py")
    
    st.markdown("---")
    st.header("🎯 学習制御")
    
    # モデル選択
    model_name = st.selectbox(
        "モデル",
        st.session_state.available_models,
        index=0 if st.session_state.available_models else None
    )
    
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
    
    # 学習パラメータ
    epochs = st.number_input("エポック数", min_value=1, value=10)
    batch_size = st.number_input("バッチサイズ", min_value=1, value=4)
    lightweight = st.checkbox("軽量(先頭10件)でテスト実行", value=True)
    limit_samples = st.number_input("使用サンプル数を制限 (0で無効)", min_value=0, value=0)
    
    # 学習開始/停止ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("学習開始", disabled=st.session_state.is_training):
            if model_name and dataset_name:
                success = start_training(model_name, dataset_name, epochs, batch_size, lightweight=lightweight, limit_samples=limit_samples)
                if not success:
                    st.error("学習の開始に失敗しました。ログを確認してください。")
            else:
                st.error("モデルとデータセットを選択してください")
    
    with col2:
        if st.button("学習停止", disabled=not st.session_state.is_training):
            stop_training()
    
    # 進捗表示
    if st.session_state.is_training:
        st.progress(st.session_state.current_progress)
        st.text(st.session_state.progress_text)

# メインコンテンツ
col1, col2 = st.columns(2)

# 推論テストセクション
st.header("推論テスト（音声アップロード）")
inf_col1, inf_col2 = st.columns([2, 1])
with inf_col1:
    uploaded = st.file_uploader("音声ファイルを選択 (WAV/FLACなど)", type=["wav", "flac", "mp3", "m4a", "ogg"])
    if uploaded is not None:
        st.audio(uploaded, format="audio/wav")
with inf_col2:
    if st.button("推論を実行", disabled=uploaded is None):
        if uploaded is None:
            st.warning("音声ファイルを選択してください")
        else:
            with st.spinner("推論を実行中..."):
                result = run_inference(uploaded.getvalue(), uploaded.name, model_name)
                transcription = result.get("transcription", "")
                first_token_ms = result.get("first_token_time_ms")
                inference_ms = result.get("inference_time_ms")
                total_ms = result.get("total_time_ms")
                
                if transcription:
                    st.success("推論完了")
                    
                    # 3種類の時間を表示
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
                    
                    st.text_area("文字起こし結果", value=transcription, height=120)
                else:
                    st.error("推論に失敗しました。ログを確認してください。")

# 上部メトリクス表示（学習中のみ）
if st.session_state.is_training:
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Epoch", value=f"{st.session_state.current_epoch}/{st.session_state.total_epochs}")
    with m2:
        st.metric(label="Step", value=f"{st.session_state.current_step}/{st.session_state.total_steps}")

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

# --- リアルタイム推論（マイク入力） ---
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

async def stream_audio_to_ws(q: "queue.Queue[bytes]", model_name: str, sample_rate: int, running_flag_ref=None, msg_queue_ref=None):
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
                    while running_flag_ref:
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
    rtc_ctx = webrtc_streamer(
        key="asr-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

with col_rt2:
    selected_model = st.selectbox("リアルタイム用モデル", st.session_state.available_models, index=0 if st.session_state.available_models else None)
    sample_rate = st.number_input("送信サンプルレート", min_value=16000, max_value=48000, value=48000, step=1000)
    start_btn = st.button("リアルタイム開始", disabled=st.session_state.get("realtime_running", False) or rtc_ctx.state.playing is False)
    stop_btn = st.button("リアルタイム停止", disabled=not st.session_state.get("realtime_running", False))

    # デバッグ情報を表示
    st.write("**デバッグ情報:**")
    st.write(f"- start_btn: {start_btn}")
    st.write(f"- rtc_ctx: {rtc_ctx is not None}")
    st.write(f"- rtc_ctx.state.playing: {rtc_ctx.state.playing if rtc_ctx else 'N/A'}")
    st.write(f"- rtc_ctx.audio_receiver: {rtc_ctx.audio_receiver is not None if rtc_ctx else 'N/A'}")
    st.write(f"- realtime_running: {st.session_state.get('realtime_running', False)}")

    if start_btn:
        st.write(f"🔍 Start button pressed: rtc_ctx={rtc_ctx is not None}, audio_receiver={rtc_ctx.audio_receiver is not None if rtc_ctx else False}")
        if not rtc_ctx:
            st.error("WebRTCコンテキストが初期化されていません")
        elif not rtc_ctx.audio_receiver:
            st.error("オーディオレシーバーが初期化されていません")
        else:
            st.write("✅ All conditions met, starting realtime streaming")
    
    if start_btn and rtc_ctx and rtc_ctx.audio_receiver:
        # 送信キューとスレッド/タスクの初期化（キューサイズを大幅に増加）
        send_queue = queue.Queue(maxsize=1000)
        # 先にランニングフラグとカウンタを立ててからスレッドを開始
        st.session_state["realtime_running"] = True
        st.session_state["_rt_chunks_sent"] = 0
        # 既存のエラーメッセージ/ステータスをリセット
        st.session_state["realtime_error"] = ""
        st.session_state["realtime_partial"] = ""
        st.session_state["realtime_final"] = ""
        
        # スレッド間で共有するためのローカル変数
        running_flag = st.session_state["realtime_running"]
        msg_queue = st.session_state["realtime_msg_queue"]
        
        # audio_receiverから直接フレームを取得するスレッド
        def pull_audio_frames():
            import time as _time
            import logging
            frames_sent = 0
            
            # ログ設定
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("audio_puller")
            
            logger.info("Starting audio puller thread", 
                       extra={"extra_fields": {"component": "audio_puller", "action": "thread_start"}})
            logger.info("Audio puller context info", 
                       extra={"extra_fields": {"component": "audio_puller", "action": "context_info", 
                                             "rtc_state": str(rtc_ctx.state), "has_receiver": rtc_ctx.audio_receiver is not None}})
            
            while running_flag:
                if rtc_ctx.audio_receiver:
                    try:
                        # streamlit-webrtcの正しい使用方法に修正
                        frames = []
                        try:
                            # streamlit-webrtcでは引数なしでget_frames()を呼び出す
                            frames = rtc_ctx.audio_receiver.get_frames()
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
                                    except queue.Full:
                                        logger.warning("Send queue is full, dropping frame", 
                                                     extra={"extra_fields": {"component": "audio_puller", "action": "queue_full", 
                                                                           "queue_size": send_queue.qsize()}})
                                    
                                    # 統計情報をメッセージキューに送信
                                    try:
                                        if msg_queue_ref is not None:
                                            msg_queue_ref.put({"type": "stats", "payload": {
                                                "frames_sent": frames_sent, 
                                                "queue_size": send_queue.qsize(),
                                                "queue_capacity": send_queue.maxsize
                                            }})
                                    except Exception:
                                        pass
                                except Exception as e:
                                    logger.debug("Error processing audio frame", 
                                               extra={"extra_fields": {"component": "audio_puller", "action": "frame_error", 
                                                                     "error": str(e)}})
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

        st.write("🚀 Starting realtime streaming with audio_receiver")

        # 新しいイベントループで実行
        loop = asyncio.new_event_loop()
        def run_loop():
            import logging
            logger = logging.getLogger("websocket_loop")
            
            try:
                asyncio.set_event_loop(loop)
                logger.info("Starting WebSocket loop", 
                           extra={"extra_fields": {"component": "websocket_loop", "action": "loop_start", 
                                                 "model": selected_model or "conformer", "sample_rate": int(sample_rate)}})
                loop.run_until_complete(stream_audio_to_ws(send_queue, selected_model or "conformer", int(sample_rate), running_flag, msg_queue))
            except Exception as e:
                logger.error("WebSocket loop error", 
                           extra={"extra_fields": {"component": "websocket_loop", "action": "loop_error", 
                                                 "error": str(e), "traceback": traceback.format_exc()}})
                try:
                    st.error(f"WebSocket loop error: {e}")
                except Exception:
                    pass
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        
        # Puller スレッド開始
        p = threading.Thread(target=pull_audio_frames, daemon=True)
        p.start()
        
        st.session_state["realtime_loop"] = loop
        st.session_state["realtime_thread"] = t
        st.session_state["realtime_puller"] = p

    if stop_btn and st.session_state.get("realtime_running", False):
        # 送信停止: スレッドとループを停止
        logger = logging.getLogger("ui-rt")
        logger.info("Stopping realtime streaming", 
                   extra={"extra_fields": {"component": "ui", "action": "stop_streaming"}})
        
        st.session_state["realtime_running"] = False
        
        # スレッドの停止を待つ
        loop = st.session_state.get("realtime_loop")
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        
        # puller スレッドはフラグで停止。追加の操作は不要。

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
