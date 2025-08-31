import streamlit as st
import pandas as pd
import requests
import asyncio
import websockets
import json
from typing import Dict, Any
import traceback
import os

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
        "websocket_task": None,
        "initial_load": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
        else:
            log_detailed_error("ステータス取得", Exception(f"HTTP {response.status_code}"), response)
            
    except requests.exceptions.ConnectionError as e:
        log_detailed_error("ステータス取得", e)
    except requests.exceptions.Timeout as e:
        log_detailed_error("ステータス取得", e)
    except requests.exceptions.RequestException as e:
        log_detailed_error("ステータス取得", e)
    except Exception as e:
        log_detailed_error("ステータス取得", e)

def start_training(model_name: str, dataset_name: str, epochs: int, batch_size: int):
    """学習を開始"""
    try:
        params = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size
        }
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

# --- WebSocketリスナー ---
async def websocket_listener():
    """WebSocketでリアルタイム更新を受信"""
    try:
        st.session_state.logs.append(f"🔌 WebSocket接続試行中... URL: {WEBSOCKET_URL}")
        
        # WebSocket接続の設定
        # プロキシ経由でWebSocketに接続する必要がある場合は、
        # websocketsライブラリのプロキシサポートを確認する必要があります
        # 現在は直接接続を試行
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            st.session_state.logs.append("✅ WebSocket接続確立")
            async for message in websocket:
                data = json.loads(message)
                handle_ws_message(data)
    except Exception as e:
        error_msg = f"❌ WebSocket接続エラー:"
        error_msg += f"\n   - エラータイプ: {type(e).__name__}"
        error_msg += f"\n   - エラーメッセージ: {str(e)}"
        error_msg += f"\n   - 接続先: {WEBSOCKET_URL}"
        error_msg += f"\n   - スタックトレース: {traceback.format_exc()}"
        st.session_state.logs.append(error_msg)

def handle_ws_message(data: Dict[str, Any]):
    type = data.get("type")
    payload = data.get("payload", {})
    if type == "log":
        st.session_state.logs.append(f"[BACKEND] {payload.get('message')}")
    elif type == "progress":
        st.session_state.progress_df.loc[len(st.session_state.progress_df)] = {"epoch": payload["epoch"], "step": payload["step"], "loss": payload["loss"]}
        st.session_state.lr_df.loc[len(st.session_state.lr_df)] = {"step": payload["step"], "learning_rate": payload["learning_rate"]}
        st.session_state.current_progress = payload['step'] / payload['total_steps']
        st.session_state.progress_text = f"Epoch {payload['epoch']}/{payload['total_epochs']}, Step {payload['step']}/{payload['total_steps']}"
    elif type == "validation_result":
        st.session_state.validation_df.loc[len(st.session_state.validation_df)] = payload

# --- UI描画 ---
st.set_page_config(layout="wide")
init_session_state()

if not st.session_state.initial_load:
    get_config()
    get_status()
    st.session_state.initial_load = True

# タイトル
st.title("ASR 学習ダッシュボード")

# サイドバー - 学習制御
with st.sidebar:
    st.header("学習制御")
    
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
    
    # 学習パラメータ
    epochs = st.number_input("エポック数", min_value=1, value=10)
    batch_size = st.number_input("バッチサイズ", min_value=1, value=32)
    
    # 学習開始/停止ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("学習開始", disabled=st.session_state.is_training):
            if model_name and dataset_name:
                start_training(model_name, dataset_name, epochs, batch_size)
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

with col1:
    st.header("学習ロス")
    if not st.session_state.progress_df.empty:
        loss_data = st.session_state.progress_df.rename(columns={"loss": "train_loss"})
        if not st.session_state.validation_df.empty:
            # エポックの最後のステップに検証ロスを紐付ける
            last_step_per_epoch = loss_data.groupby("epoch")["step"].max().reset_index()
            merged_val = pd.merge(st.session_state.validation_df, last_step_per_epoch, on="epoch")
            loss_data = pd.merge(loss_data, merged_val, on=["epoch", "step"], how="left")
        
        st.line_chart(loss_data.set_index("step")[["train_loss", "val_loss"]])
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

# 自動更新
if st.session_state.is_training:
    st.rerun()
